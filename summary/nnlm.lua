--
--  Copyright (c) 2015, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Author: Alexander M Rush <srush@seas.harvard.edu>
--          Sumit Chopra <spchopra@fb.com>
--          Jason Weston <jase@fb.com>

-- Ngram neural language model with auxiliary model
require('nn')
require('nngraph')
--require('fbnn')
require('cunn')
require('sys')
local utils = require('summary.util')

local nnlm = {}

function nnlm.addOpts()
   cmd:option('-epochs',         5, "Number of epochs to train.")
   cmd:option('-miniBatchSize', 64, "Size of training minibatch.")
   cmd:option('-printEvery', 10000,  "How often to print during training.")
   cmd:option('-modelFilename', '', "File for saving loading/model.")
   cmd:option('-window',         5, "Size of NNLM window.")
   cmd:option('-embeddingDim',  50, "Size of NNLM embeddings.")
   cmd:option('-hiddenSize',   100, "Size of NNLM hiddent layer.")
   cmd:option('-learningRate', 0.1, "SGD learning rate.")
end


function nnlm.create_lm(opt, dict, encoder, encoder_size, encoder_dict)
   local new_mlp = {}
   setmetatable(new_mlp, { __index = nnlm })
   new_mlp.opt = opt
   new_mlp.dict = dict
   new_mlp.encoder_dict = encoder_dict
   new_mlp.encoder_model = encoder
   new_mlp.window = opt.window
   if encoder ~= nil then
      new_mlp:build_mlp(encoder, encoder_size)
   end
   return new_mlp
end


function nnlm:build_mlp(encoder, encoder_size)
   -- Set constants
   local D = self.opt.embeddingDim
   local N = self.opt.window
   local H = self.opt.hiddenSize
   local V = #self.dict.index_to_symbol
   local P = encoder_size
   print(H, P)

   -- Input
   local context_input = nn.Identity()()
   local encoder_input = nn.Identity()()
   local position_input = nn.Identity()()

   local lookup = nn.LookupTable(V, D)(context_input)
   local encoder_node = encoder({encoder_input, position_input, context_input})

   -- tanh W (E y)
   local lm_mlp = nn.Tanh()(nn.Linear(D * N, H)(nn.View(D * N)(lookup)))

   -- Second layer: takes LM and encoder model.
   local mlp = nn.Linear(H + P, V)(nn.View(H + P)(nn.JoinTable(2)(
                                                     {lm_mlp, encoder_node})))
   self.soft_max = nn.LogSoftMax()(mlp)

   -- Input is conditional context and ngram context.
   self.mlp = nn.gModule({encoder_input, position_input, context_input},
      {self.soft_max})

   self.criterion = nn.ClassNLLCriterion()
   self.lookup = lookup.data.module
   self.mlp:cuda()
   self.criterion:cuda()
   collectgarbage()
end

function nnlm:get_rnn_mlp()
   -- Set constants
   local D = self.opt.embeddingDim
   local N = self.opt.window
   local V = #self.dict.index_to_symbol

   -- Input
   local y_prev = nn.Identity()()
   local h_prev = nn.Identity()()
   local c = nn.Identity()()

   local y_embedding = nn.LookupTable(V, D)(y_prev)

   local y_lin = nn.Linear(D, D)(y_embedding)
   local h_lin = nn.Linear(D, D)(h_prev)
   local c_lin = nn.Linear(D, D)(c)

   local h_sum = nn.CAddTable()({y_lin, h_lin, c_lin})

   local h = nn.Sigmoid()(h_sum)

   local mlp = nn.gModule({y_prev, h_prev, c},
      {h})

   local h_id = nn.Identity()()
   local c_id = nn.Identity()()

   local h_lin2 = nn.Linear(D, V)(h_id)
   local c_lin2 = nn.Linear(D, V)(c_id)

   local p_sum = nn.CAddTable()({h_lin2, c_lin2})

   local p = nn.Tanh()(p_sum)

   local soft_max = nn.LogSoftMax()(p)

   local softmax = nn.gModule({h_id, c_id},{soft_max})

   return mlp, softmax
end

function nnlm:clone_many_times(net, T)
    local clones = {}

    local params, gradParams
    if net.parameters then
        params, gradParams = net:parameters()
        if params == nil then
            params = {}
        end
    end

    local paramsNoGrad
    if net.parametersNoGrad then
        paramsNoGrad = net:parametersNoGrad()
    end

    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(net)

    for t = 1, T do
        -- We need to use a new reader for each clone.
        -- We don't want to use the pointers to already read objects.
        local reader = torch.MemoryFile(mem:storage(), "r"):binary()
        local clone = reader:readObject()
        reader:close()

        if net.parameters then
            local cloneParams, cloneGradParams = clone:parameters()
            local cloneParamsNoGrad
            for i = 1, #params do
                cloneParams[i]:set(params[i])
                cloneGradParams[i]:set(gradParams[i])
            end
            if paramsNoGrad then
                cloneParamsNoGrad = clone:parametersNoGrad()
                for i =1,#paramsNoGrad do
                    cloneParamsNoGrad[i]:set(paramsNoGrad[i])
                end
            end
        end

        clones[t] = clone
        collectgarbage()
    end

    mem:close()
    return clones
end

function nnlm:build_rnn_mlp(encoder)
   local H = self.opt.hiddenSize

   self.encoders = self.clone_many_times(encoder, H)

   local mlps = {}
   local softmaxs = {}
   local mlp
   local softmax
   mlp, softmax = self.get_rnn_mlp()
   mlps = self.clone_many_times(mlp, H)
   softmaxs = self.clone_many_times(softmax, H)

   self.mlps = mlps
   self.softmaxs = softmaxs

   self.criterion = nn.ClassNLLCriterion()
   self.mlp:cuda()
   self.criterion:cuda()
   collectgarbage()
end

-- Run validation
function nnlm:validation(valid_data)
   print("[Running Validation]")

   local offset = 1000
   local loss = 0
   local total = 0

   valid_data:reset()
   while not valid_data:is_done() do
      local input, target = valid_data:next_batch(offset)
      local out = self.mlp:forward(input)
      local err = self.criterion:forward(out, target) * target:size(1)

      -- Augment counters.
      loss = loss + err
      total = total + target:size(1)
   end
   print(string.format("[perp: %f validation: %f total: %d]",
                       math.exp(loss/total),
                       loss/total, total))
   return loss / total
end

-- Run validation RNN
function nnlm:validation_rnn(valid_data)
   print("[Running Validation RNN]")

   local loss = 0
   local total = 0
   local D = self.opt.embeddingDim

   valid_data:reset()
   while not valid_data:is_done() do
      local input, position, target = valid_data:next_example()

      local hidden = torch.Tensor(D):zero()
      local y_prev = torch.Tensor{1}
      local err    = 0

      for i=1,target:size()[1] do
          local c = self.encoders[i]:forward({input, position, hidden})
          local h = self.mlps[i]:forward({y_prev, hidden, c})
          local soft = self.softmaxs:forward({h, c})
          y_prev = target[i]
          hidden = h
          err = err + self.criterion:forward(soft, target[i])
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

function nnlm:renorm(data, th)
    local size = data:size(1)
    for i = 1, size do
        local norm = data[i]:norm()
        if norm > th then
            data[i]:div(norm/th)
        end
    end
end


function nnlm:renorm_tables()
    -- Renormalize the lookup tables.
    if self.lookup ~= nil then
        print(self.lookup.weight:size())
        print(self.lookup.weight:type())
        self:renorm(self.lookup.weight, 1)
    end
    if self.encoder_model.lookup ~= nil then
        self:renorm(self.encoder_model.lookup.weight, 1)
        if self.encoder_model.title_lookup ~= nil then
            self:renorm(self.encoder_model.title_lookup.weight, 1)
        end
    end
    if self.encoder_model.lookups ~= nil then
        for i = 1, #self.encoder_model.lookups do
            self:renorm(self.encoder_model.lookups[i].weight, 1)
        end
    end
end


function nnlm:run_valid(valid_data)
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

function nnlm:run_valid_rnn(valid_data)
   -- Run validation.
   if valid_data ~= nil then
      local cur_valid_loss = self:validation_rnn(valid_data)
      -- If valid loss does not improve drop learning rate.
      if cur_valid_loss > self.last_valid_loss then
         self.opt.learningRate = self.opt.learningRate / 2
      end
      self.last_valid_loss = cur_valid_loss
   end

   -- Save the model.
   self:save(self.opt.modelFilename)
end

function nnlm:train(data, valid_data)
   -- Best loss seen yet.
   self.last_valid_loss = 1e9
   -- Train
   for epoch = 1, self.opt.epochs do
      data:reset()
      self:renorm_tables()
      self:run_valid(valid_data)

      -- Loss for the epoch.
      local epoch_loss = 0
      local batch = 1
      local last_batch = 1
      local total = 0
      local loss = 0

      sys.tic()
      while not data:is_done() do
         local input, target = data:next_batch(self.opt.miniBatchSize)
         if data:is_done() then break end

         local out = self.mlp:forward(input)
         local err = self.criterion:forward(out, target) * target:size(1)
         local deriv = self.criterion:backward(out, target)

         if not utils.isnan(err) then
            loss = loss + err
            epoch_loss = epoch_loss + err

            self.mlp:zeroGradParameters()
            self.mlp:backward(input, deriv)
            self.mlp:updateParameters(self.opt.learningRate)
         else
            print("NaN")
            print(input)
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

         if batch % 10000000 == 0 then
	   self:save(self.opt.modelFilename)
         end

         batch = batch + 1
         total = total + input[1]:size(1)
      end
      print(string.format("[EPOCH : %d LOSS: %f TOTAL: %d BATCHES: %d]",
                          epoch, epoch_loss / total, total, batch))
   end
end

function nnlm:train_rnn(data, valid_data)
   -- Best loss seen yet.
   self.last_valid_loss = 1e9
   -- Train
   for epoch = 1, self.opt.epochs do
      data:reset()
      self:renorm_tables()
      self:run_valid_rnn(valid_data)

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
         local input, position, target = data:next_example()
         if data:is_done() then break end

         local hidden0 = torch.Tensor(D):zero()
         local h = {}
         h[1] = hidden0
         local y = {}
         local y_prev = torch.Tensor{1}
         y[1] = y_prev
         local err = 0

         local c = {}

         local doutput_t = {}

         for i=1,target:size()[1] do
             c[i] = self.encoders[i]:forward({input, position, h[i]})
             h[i+1] = self.mlps[i]:forward({y[i], h[i], c[i]})
             local soft = self.softmaxs[i]:forward({h[i+1], c[i]})
             y[i+1] = target[i]
             err = err + self.criterion:forward(soft, target[i])
             doutput[i] = self.criterion:backward(soft, target[i])
         end

         local drnn_c = {}
         local drnn_h = {}
         local drnn_y = {}

         if not utils.isnan(err) then
            loss = loss + err
            epoch_loss = epoch_loss + err

            for i=target:size()[1],1,-1 do
                if (i == target:size()[1]) then
                    drnn_h[i], drnn_c[i] = self.softmaxs[i]:backward({h[i+1], c[i]}, doutput[i])
                else
                    local drnn_htemp
                    drnn_htemp, drnn_c[i] = unpack(self.softmaxs[i]:backward({h[i+1], c[i]}, doutput[i]))
                    drnn_h[i]:add(drnn_htemp)
                end
                local drnn_ctemp
                drnn_y[i-1], drnn_h[i-1], drnn_ctemp = unpack(self.mlps[i]:backward({y[i], h[i], c[i]},
                                                                             drnn_h[i]))
                drnn_c[i]:add(drnn_ctemp)
                local drnn_htemp
                local din
                local dl
                din, dl, drnn_htemp = unpack(self.encoders[i]:backward({input, position, h[i]},
                                                                       drnn_c[i]))
                drnn_h[i-1]:add(drnn_htemp)
            end
            if count%batch_size == 0 then
                self.mlps[1]:updateParameters(self.opt.learningRate)
                self.encoders[1]:updateParameters(self.opt.learningRate)
                self.softmaxs[1]:updateParameters(self.opt.learningRate)
                self.softmaxs[1]:zeroGradParameters()
                self.mlps[1]:zeroGradParameters()
                self.encoders[1]:zeroGradParameters()
            end
         else
            print("NaN")
            print(input)
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
         total = total + input[1]:size(1)
      end
      print(string.format("[EPOCH : %d LOSS: %f TOTAL: %d BATCHES: %d]",
                          epoch, epoch_loss / total, total, batch))
   end
end

function nnlm:save(fname)
    print("[saving mlp: " .. fname .. "]")
    torch.save(fname, self)
    return true
end


function nnlm:load(fname)
    local new_self = torch.load(fname)
    for k, v in pairs(new_self) do
       if k ~= 'opt' then
          self[k] = v
       end
    end
    return true
end


return nnlm
