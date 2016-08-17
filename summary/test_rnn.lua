function clone_many_times(net, T)
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

function build_conv_rec(D, N, V)
   -- Takes the sentence, the positions of the words
   -- and the previous hidden state
   local sentence = nn.Identity()()
   local positions = nn.Identity()()
   local hidden = nn.Identity()()
   -- Word and local embeddings
   local word_embedding = nn.LookupTable(V, D)(sentence)
   local position_embedding = nn.LookupTable(1000, D)(positions)
   -- Sum embedding of the words and the position
   local sum = nn.CAddTable()({word_embedding, position_embedding})
   -- Convolution over the text
   local view = nn.View(1, -1, D)(sum)
   local conv = nn.SpatialConvolution(1, 1, 1, N, 1, 1, 0, (N - 1) / 2)(view)
   local post_conv = nn.View(D, -1)(conv)
   local transp = nn.Transpose({1,2})(post_conv)
   -- Weights
   local view_hidden = nn.View(-1, 1)(hidden)
   local mult = nn.MM()({transp, view_hidden})
   local reduce = nn.View(-1)(mult)
   local alpha = nn.SoftMax()(reduce)
   -- Context vector
   local c = nn.MixtureTable()({alpha, word_embedding})
   local encoder_mlp = nn.gModule({sentence, positions, hidden}, {c})
   return encoder_mlp
end

function get_start()
    local table = nn.Identity()()
    local word_embedding = nn.SelectTable(3)(table)
    local mean = nn.Mean(2)(word_embedding)
    local zeros = nn.MulConstant(0)(mean)
    local start = nn.gModule({table}, {zeros, mean})
    return start
end

function get_input(D, N, V)
   local sentence = nn.Identity()()
   local positions = nn.Identity()()
   local y = nn.Identity()()
   -- Word and local embeddings
   local word_embedding = nn.LookupTable(V, D)(sentence)
   local position_embedding = nn.LookupTable(1000, D)(positions)
   -- Sum embedding of the words and the position
   local sum = nn.CAddTable()({word_embedding, position_embedding})
   -- Convolution over the text
   local view = nn.View(1, -1, D)(sum)
   local conv = nn.SpatialConvolution(1, 1, 1, N, 1, 1, 0, (N - 1) / 2)(view)
   local post_conv = nn.View(D, -1)(conv)
   local transp = nn.Transpose({1,2})(post_conv)
   local input = nn.gModule({sentence, positions, y}, {transp, y, word_embedding})
   return input
end

function get_transfert()
    local h = nn.Identity()()
    local c = nn.Identity()()
    local sig = nn.Sigmoid()(h)
    local transfert = nn.gModule({h, c}, {sig, c})
    return transfert
end

function get_feedback()
    local h = nn.Identity()()
    local c = nn.Identity()()
    local out = nn.SelectTable(1)({h,c})
    local feedback = nn.gModule({h, c}, {out})
    return feedback
end

function get_merge(D, N, V)
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

function recurrent_node(D, N, V)
    local start = get_start()
    local input = get_input(D, N, V)
    local transfert = get_transfert()
    local feedback = get_feedback()
    local merge = get_merge(D,N,V)
    return nn.Recurrent(start, input, feedback, transfert, 100, merge)
end

function output_node(D, N, V)
    local h = nn.Identity()()
    local c = nn.Identity()()
    local h_lin = nn.Linear(D, V)(h)
    local c_lin = nn.Linear(D, V)(c)
    local p_sum = nn.CAddTable()({h_lin, c_lin})
    local p = nn.Tanh()(p_sum)
    local soft_max = nn.LogSoftMax()(p)
    return nn.gModule({h, c}, {soft_max})
end

function getCriterion()
   local criterion = nn.ClassNLLCriterion()
   return criterion
end

function full_rnn_network(D, N, V)
    local rec = recurrent_node(D, N, V)
    local out = output_node(D, N, V)
    local rnn = nn.Sequential()
        :add(rec)
        :add(out)
    return nn.Recursor(rnn, 100)
end

function train_rnn(sentence, position, target, rnn, criterion, learning_rate)
   local outputs, err = {}, 0
   local y0 = torch.Tensor{1}
   rnn:zeroGradParameters()
   rnn:forget() -- forget all past time-steps
   -- forward
   for step=1, target:size()[1] do
      if step == 1 then
          outputs[step] = rnn:forward({sentence, position, y0})
      else
          outputs[step] = rnn:forward({sentence, position, target[step-1]})
      end
      err = err + criterion:forward(outputs[step], target[step])
   end
   -- backward
   print(string.format("Iteration %d ; NLL err = %f ", 0, err))
   -- 3. backward sequence through rnn (i.e. backprop through time)
   local gradOutputs, gradInputs = {}, {}
   for step=target:size()[1],1,-1 do -- reverse order of forward calls
      gradOutputs[step] = criterion:backward(outputs[step], target[step])
      if step == 1 then
          gradInputs[step] = rnn:backward({sentence, position, y0}, gradOutputs[step])
      else
          gradInputs[step] = rnn:backward({sentence, position, target[step-1]}, gradOutputs[step])
      end
   end
   -- 4. update
   rnn:updateParameters(learning_rate)
end

function build_rnn_mlp(D, N, V)
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
   local h_lin2 = nn.Linear(D, V)(h_prev)
   local c_lin2 = nn.Linear(D, V)(c)
   local p_sum = nn.CAddTable()({h_lin2, c_lin2})
   local p = nn.Tanh()(p_sum)
   local soft_max = nn.LogSoftMax()(p)
   -- Input is conditional context and ngram context.
   local mlp = nn.gModule({y_prev, h_prev, c},
      {soft_max, h})
   return mlp
end

function build_network(D, N, V, H)
    local encoders = {}
    local mlps = {}
    local encoder
    local mlp
    encoder = build_conv_rec(D, N, V)
    mlp = build_rnn_mlp(D, N, V)
    encoders = clone_many_times(encoder, H)
    mlps = clone_many_times(mlp, H)
    return encoders, mlps
end

function validation_rnn(D, input, position, target, encoders, mlps, criterion)
   print("[Running Validation RNN]")
   local loss = 0
   local total = 0
    local hidden = torch.Tensor(D):zero()
    local y_prev = torch.Tensor{1}
    local err    = 0
    for i=1,target:size()[1] do
        local c = encoders[i]:forward({input, position, hidden})
        local out = mlps[i]:forward({y_prev, hidden, c})
        soft = out[1]
        h = out[2]
        y_prev[1] = target[i]
        hidden = h
        err = err + criterion:forward(soft, target[i])
    end
    -- Augment counters.
    loss = loss + err
    total = total + target:size(1)
    print(string.format("[perp: %f validation: %f total: %d]",
                       math.exp(loss/total),
                       loss/total, total))
   return loss / total
end
