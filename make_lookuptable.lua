------------------------------------------------------------------------------
--  Hadamard Product for Low-rank Bilinear Pooling
--  Jin-Hwa Kim, Kyoung-Woon On, Jeonghee Kim, Jung-Woo Ha, Byoung-Tak Zhang 
--  https://arxiv.org/abs/1610.04325
--
--  Porting Skip-thought Vectors in Torch7
--  https://github.com/HyeonwooNoh/DPPnet
-----------------------------------------------------------------------------

require 'hdf5'
cjson=require('cjson') 

cmd = torch.CmdLine()
cmd:option('-input_path','data_train-val_test-dev_2k')
cmd:option('-input_json','data_prepro.json')
cmd:option('-input_ques_h5','data_prepro.h5')
cmd:option('-output_vocab','vocab_2k.txt')
opt = cmd:parse(arg)
print(opt)

------------------------------------------------------------------------------
file = io.open(paths.concat(opt.input_path, opt.input_json), 'r')
text = file:read()
file:close()
json_file = cjson.decode(text)

vocab_size = 0
for i, w in pairs(json_file['ix_to_word']) do vocab_size = vocab_size + 1 end

vocab = io.open(paths.concat(opt.input_path, opt.output_vocab), 'w')
for i=1,vocab_size do
   vocab:write(json_file.ix_to_word[tostring(i)] .. '\n')
end
vocab:close()
------------------------------------------------------------------------------
-- If you need word frequencies, use this code.
-- h5_file = hdf5.open(paths.concat(opt.input_path, opt.input_ques_h5), 'r')
-- frequencies = h5_file:read('/frequencies'):all()
------------------------------------------------------------------------------
-- @TODO
-- cp vocab_2k.txt ../DPPnet/003_skipthoughts_porting/data/skipthoughts_porting
-- vi ../DPPnet/003_skipthoughts_porting/002_write_vocab_table_vqa.py
-- [change filename as `vocab_2k.txt` in line 57]
-- cd ../DPPnet/003_skipthoughts_porting/
-- python 002_write_vocab_table_vqa.py
-- th 004_save_params_in_torch_file_vqa.lua
-- cd -
-- ls -l ../DPPnet/003_skipthoughts_porting/data/skipthoughts_params/vqa_uni_gru_word2vec.t7
------------------------------------------------------------------------------
require 'rnn'
tmp=torch.load('../DPPnet/003_skipthoughts_porting/data/skipthoughts_params/vqa_uni_gru_word2vec.t7')
lookup = nn.LookupTableMaskZero(vocab_size,620)
assert(lookup.weight:size(1) == vocab_size+1)
assert(lookup.weight:size(2) == 620)
assert(tmp:size(2) == 620)
-- weight[1] is zero, tmp's last row is for <eos>.
lookup.weight[1]:zero()
lookup.weight:narrow(1,2,vocab_size):copy(tmp:narrow(1,1,vocab_size));
torch.save('lookup_2k.t7', lookup)
