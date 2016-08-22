require('torch')
require('nngraph')

local nnlm = require('summary.nnlm_rec')
local data = require('summary.data_rnn')

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Train a summarization model.')
cmd:text()

data.add_opts(cmd)
nnlm.addOpts(cmd)

opt = cmd:parse(arg)

local function main()
   -- Load in the data.
   local tdata = data.load_title(opt.titleDir, "2")
   local article_data = data.load_article(opt.articleDir, "2")
   --local tdata1 = data.load_title(opt.titleDir, "2")
   --local article_data1 = data.load_article(opt.articleDir, "2")

   local valid_data = data.load_title(opt.validTitleDir, "", tdata.dict)
   local valid_article_data =
      data.load_article(opt.validArticleDir, "", article_data.dict)

   -- Make main LM
   local train_data = data.init(tdata, article_data)
   --local train_data2 = data.init(tdata1, article_data1)
   local valid = data.init(valid_data, valid_article_data)
   local mlp = nnlm.create_lm(opt, tdata.dict)

   mlp:train(train_data, train_data, valid)
end

main()
