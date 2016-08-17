local data_rnn = {}

function data_rnn.add_opts(cmd)
   cmd:option('-articleDir', 'working_dir/processed/train/article/',
              'Directory containing article training matrices.')
   cmd:option('-titleDir', 'working_dir/processed/train/title/',
              'Directory containing title training matrices.')
   cmd:option('-validArticleDir', 'working_dir/processed/valid.filter/article/',
              'Directory containing article matricess for validation.')
   cmd:option('-validTitleDir', 'working_dir/processed/valid.filter/title/',
              'Directory containing title matrices for validation.')
end

function data_rnn.load(article_dir, title_dir)
   return data_rnn.init()
end

function data_rnn.init(title_data, article_data)
   local new_data = {}
   setmetatable(new_data, { __index = data_rnn })
   new_data.title_data = title_data
   new_data.article_data = article_data
   new_data.total_size = #title_data.words
   new_data:reset()
   print(article_data.words[1])
   print(title_data.words[1])
   return new_data
end

function data_rnn:reset()
    self.current_index = 0
end

function data_rnn:is_done()
   return self.total_size == self.current_index
end

function data_rnn:next_batch(max_size)
   local diff = self.total_size - self.current_index
   local offset
   if self.current_index + max_size > self.total_size then
      offset = self.total_size - self.current_index
   else
      offset = max_size
   end

   self.current_index = self.current_index + 1
   local sentence = self.article_data.words[offset]
   local target = self.title_data.words[offset]
   local position = torch.range(1, sentence:size(1))

   return sentence, position, target
end

function data_rnn.make_input(article, context, K)
   local bucket = article:size(1)
   local aux_sentence = article:view(bucket, 1)
      :expand(article:size(1), K):t():contiguous()
   local positions = torch.range(1, bucket):view(bucket, 1)
      :expand(bucket, K):t():contiguous() + (200 * bucket)
   return {aux_sentence, positions, context}
end

function data_rnn.load_title_dict(dname)
   return torch.load(dname .. 'dict')
end

function data_rnn.load_title(dname, number, use_dict)
   local input_words = torch.load(dname .. 'word_rnn' .. (number or '')  .. '.mat.torch')
   -- local offsets = torch.load(dname .. 'offset.mat.torch')

   local dict = use_dict or torch.load(dname .. 'dict')
   for length, mat in pairs(input_words) do
      input_words[length] = mat:float()
   end
   local title_data = {words = input_words, dict = dict}
   return title_data
end

function data_rnn.load_article(dname, number, use_dict)
   local input_words = torch.load(dname .. 'word_rnn' .. (number or '') .. '.mat.torch')
   -- local offsets = torch.load(dname .. 'offset.mat.torch')

   local dict = use_dict or torch.load(dname .. 'dict')
   for length, mat in pairs(input_words) do
      input_words[length] = mat:float()
   end
   local article_data = {words = input_words, dict = dict}
   return article_data
end

return data_rnn
