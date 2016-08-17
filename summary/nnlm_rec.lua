require('nn')
require('nngraph')
--require('fbnn')
require('cunn')
require('sys')
require('rnn')

local nnlm_rec = {}

function nnlm_rec.addOpts()
   cmd:option('-epochs',         5, "Number of epochs to train.")
   cmd:option('-miniBatchSize', 64, "Size of training minibatch.")
   cmd:option('-printEvery', 10000,  "How often to print during training.")
   cmd:option('-modelFilename', '', "File for saving loading/model.")
   cmd:option('-window',         5, "Size of NNLM window.")
   cmd:option('-embeddingDim',  50, "Size of NNLM embeddings.")
   cmd:option('-hiddenSize',   100, "Size of NNLM hiddent layer.")
   cmd:option('-learningRate', 0.1, "SGD learning rate.")
end

function nnlm_rec.get_start()
    local table = nn.Identity()()
    local word_embedding = nn.SelectTable(3)(table)
    local mean = nn.Mean(1)(word_embedding)
    local zeros = nn.MulConstant(0)(mean)
    local start = nn.gModule({table}, {zeros, mean})
    return start
end

function nnlm_rec.get_input(D, N, V)
   local sentence = nn.Identity()()
   local positions = nn.Identity()()
   local y = nn.Identity()()
   -- Word and local embeddings
   local word_embedding = nn.LookupTable(V, D)(sentence)
   -- 1000 is the number of words max in the input
   local position_embedding = nn.LookupTable(1000, D)(positions)
   -- Sum embedding of the words and the position
   local sum = nn.CAddTable()({word_embedding, position_embedding})
   -- Convolution over the text
   local view = nn.View(1, -1, D)(sum)
   --local conv = nn.SpatialConvolution(1, 1, 1, N, 1, 1, 0, (N - 1) / 2)(view)
   local conv = cunn.SpatialConvolution(1, 1, 1, N, 1, 1, 0, (N - 1) / 2)(view)
   local post_conv = nn.View(D, -1)(conv)
   local transp = nn.Transpose({1,2})(post_conv)
   local input = nn.gModule({sentence, positions, y}, {transp, y, word_embedding})
   return input
end

function nnlm_rec.get_transfert()
    local h = nn.Identity()()
    local c = nn.Identity()()
    local sig = nn.Sigmoid()(h)
    local transfert = nn.gModule({h, c}, {sig, c})
    return transfert
end

function nnlm_rec.get_feedback()
    local h = nn.Identity()()
    local c = nn.Identity()()
    local out = nn.SelectTable(1)({h,c})
    local feedback = nn.gModule({h, c}, {out})
    return feedback
end

function nnlm_rec.get_merge(D, N, V)
    local t1 = nn.Identity()()
    local hidden = nn.Identity()()
    local transp = nn.SelectTable(1)(t1)
    local y_prev = nn.SelectTable(2)(t1)
    local word_embedding = nn.SelectTable(3)(t1)
    -- Weights
    local view_hidden = nn.View(-1, 1)(hidden)
    local mult = nn.MM()({transp, view_hidden})
    local reduce = nn.View(-1)(mult)
    local alpha = nn.SoftMax()(reduce)
    -- Context vector
    local c = nn.MixtureTable()({alpha, word_embedding})
    local y_embedding = nn.LookupTable(V, D)(y_prev)
    local y_lin = nn.Linear(D, D)(y_embedding)
    local h_lin = nn.Linear(D, D)(hidden)
    local c_lin = nn.Linear(D, D)(c)
    local h_sum = nn.CAddTable()({y_lin, h_lin, c_lin})
    local merge = nn.gModule({t1, hidden}, {h_sum, c})
    return merge
end

function nnlm_rec.recurrent_node(D, N, V)
    local start = nnlm_rec.get_start()
    local input = nnlm_rec.get_input(D, N, V)
    local transfert = nnlm_rec.get_transfert()
    local feedback = nnlm_rec.get_feedback()
    local merge = nnlm_rec.get_merge(D,N,V)
    return nn.Recurrent(start, input, feedback, transfert, 100, merge)
end

function nnlm_rec.output_node(D, N, V)
    local h = nn.Identity()()
    local c = nn.Identity()()
    local h_lin = nn.Linear(D, V)(h)
    local c_lin = nn.Linear(D, V)(c)
    local p_sum = nn.CAddTable()({h_lin, c_lin})
    local p = nn.Tanh()(p_sum)
    local soft_max = nn.LogSoftMax()(p)
    return nn.gModule({h, c}, {soft_max})
end

function nnlm_rec.getCriterion()
   local criterion = nn.ClassNLLCriterion()
   return criterion
end

function nnlm_rec.full_rnn_network(D, N, V)
    local rec = nnlm_rec.recurrent_node(D, N, V)
    local out = nnlm_rec.output_node(D, N, V)
    local rnn = nn.Sequential()
        :add(rec)
        :add(out)
    return nn.Recursor(rnn, 100)
end

function nnlm_rec.train_rnn(sentence, position, target, rnn, criterion, learning_rate)
   local outputs, err = {}, 0
   local y0 = (torch.Tensor{1}):cuda()
   rnn:forget() -- forget all past time-steps
   -- forward
   for step=1, target:size()[1] do
      outputs[step] = rnn:forward({sentence, position, y0})
      y0[1] = target[step]
      err = err + criterion:forward(outputs[step], y0)
   end
   -- backward
   print(string.format("Iteration %d ; NLL err = %f ", 0, err))
   -- 3. backward sequence through rnn (i.e. backprop through time)
   local gradOutputs, gradInputs = {}, {}
   for step=target:size()[1],1,-1 do -- reverse order of forward calls
      gradOutputs[step] = criterion:backward(outputs[step], y0)
      if step == 1 then
          y0[1] = 1
          gradInputs[step] = rnn:backward({sentence, position, y0}, gradOutputs[step])
      else
          y0[1] = target[step-1]
          gradInputs[step] = rnn:backward({sentence, position, y0}, gradOutputs[step])
      end
   end
   -- 4. update
   --rnn:updateParameters(learning_rate)
   return err
end

function nnlm_rec.create_lm(opt, dict)
   local new_mlp = {}
   setmetatable(new_mlp, { __index = nnlm_rec })
   new_mlp.opt = opt
   new_mlp.dict = dict
   new_mlp.window = opt.window
   new_mlp:build_mlp()
   return new_mlp
end


function nnlm_rec:build_mlp()
   -- Set constants
   local D = self.opt.embeddingDim
   local N = self.opt.window
   local H = self.opt.hiddenSize
   local V = #self.dict.index_to_symbol
   local P = encoder_size
   print(H, P)

   self.mlp = self.full_rnn_network(D, N, V)

   self.criterion = self.getCriterion()
   self.mlp:cuda()
   self.criterion:cuda()
   collectgarbage()
end

-- Run validation
function nnlm_rec:validation(valid_data)
   print("[Running Validation]")

   local offset = 1000
   local loss = 0
   local total = 0

   valid_data:reset()
   while not valid_data:is_done() do
       local sentence, position, target = valid_data:next_batch(offset)
       local outputs, err = {}, 0
       local y0 = (torch.Tensor{1}):cuda()
       self.mlp:zeroGradParameters()
       self.mlp:forget() -- forget all past time-steps
       -- forward
       for step=1, target:size()[1] do
          outputs[step] = self.mlp:forward({sentence, position, y0})
          y0[1] = target[step]
          err = err + self.criterion:forward(outputs[step], y0)
       end
      -- Augment counters.
      loss = loss + err
      total = total + target:size(1)
   end
   print(string.format("[perp: %f validation: %f total: %d]",
                       math.exp(loss/total),
                       loss/total, total))
   return loss / total
end


function nnlm_rec:run_valid(valid_data)
   -- Run validation.
   if valid_data ~= nil then
      local cur_valid_loss = self:validation(valid_data)
      -- If valid loss does not improve drop learning rate.
      if cur_valid_loss > self.last_valid_loss then
         self.opt.learningRate = self.opt.learningRate / 2
      end
      self.last_valid_loss = cur_valid_loss
   end

   -- Save the model.
   self:save(self.opt.modelFilename)
end

function nnlm_rec:train(data, data1, valid_data)
   -- Best loss seen yet.
   self.last_valid_loss = 1e9
   -- Train
   for epoch = 1, self.opt.epochs do
      data:reset()
      data1:reset()
      self:run_valid(valid_data)

      -- Loss for the epoch.
      local epoch_loss = 0
      local batch = 1
      local last_batch = 1
      local total = 0
      local loss = 0

      local batch_size = self.opt.miniBatchSize

      local count = 1

      sys.tic()
      while not data:is_done() do
         local sentence, position, target = data:next_batch()

         if batch == 1 then
             self.mlp:zeroGradParameters()
         end

         local err = self.train_rnn(sentence, position, target, self.mlp, self.criterion, self.opt.learningRate)
         epoch_loss = epoch_loss + err

         if batch_size % batch_size then
             self.mlp:updateParameters(learning_rate)
         end

         -- Logging
         if batch % self.opt.printEvery == 1 then
            print(string.format(
                     "[Loss: %f Epoch: %d Position: %d Rate: %f Time: %f]",
                     loss / ((batch - last_batch) * self.opt.miniBatchSize),
                     epoch,
                     batch * self.opt.miniBatchSize,
                     self.opt.learningRate,
                     sys.toc()
            ))
            sys.tic()
            last_batch = batch
            loss = 0
         end

         batch = batch + 1
         total = total + target:size(1)
      end


      while not data1:is_done() do
         local sentence, position, target = data1:next_batch()

         local err = self.train_rnn(sentence, position, target, self.mlp, self.criterion, self.opt.learningRate)
         epoch_loss = epoch_loss + err

         if batch_size % batch_size then
             self.mlp:updateParameters(learning_rate)
         end

         -- Logging
         if batch % self.opt.printEvery == 1 then
            print(string.format(
                     "[Loss: %f Epoch: %d Position: %d Rate: %f Time: %f]",
                     loss / ((batch - last_batch) * self.opt.miniBatchSize),
                     epoch,
                     batch * self.opt.miniBatchSize,
                     self.opt.learningRate,
                     sys.toc()
            ))
            sys.tic()
            last_batch = batch
            loss = 0
         end

         batch = batch + 1
         total = total + target:size(1)
      end

      print(string.format("[EPOCH : %d LOSS: %f TOTAL: %d BATCHES: %d]",
                          epoch, epoch_loss / total, total, batch))
   end
end

function nnlm_rec:save(fname)
    print("[saving mlp: " .. fname .. "]")
    torch.save(fname, self)
    return true
end


function nnlm_rec:load(fname)
    local new_self = torch.load(fname)
    for k, v in pairs(new_self) do
       if k ~= 'opt' then
          self[k] = v
       end
    end
    return true
end


return nnlm_rec
