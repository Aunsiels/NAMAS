require("torch")
data = require("summary.data_nocuda")

cmd = torch.CmdLine()                                                                                    
cmd:text()                                                                                               
cmd:text()                                                                                               
cmd:text('Train a summarization model.')                                                                 
cmd:text()                                                                                               
                                                                                                         
data.add_opts(cmd)                                                                                       
                                                                                                         
opt = cmd:parse(arg)    

function translate(vector, dict)
    print(#vector)
    for i=1, (#vector)[1] do
        print(dict[vector[i]])
    end
end

data.add_opts(cmd)

tdata = data.load_title(opt.titleDir, false)

article_data = data.load_article(opt.articleDir)

train_data = data.init(tdata, article_data)

dict = train_data.title_data.dict.index_to_symbol
dict2 = train_data.article_data.dict.index_to_symbol

input, target = train_data:next_example()

translate(input[1], dict2)

print("target")

translate(target, dict)

print(target)
