------------------------------------------------------------------------------
--  Hadamard Product for Low-rank Bilinear Pooling
--  Jin-Hwa Kim, Kyoung-Woon On, Woosang Lim, Jeonghee Kim, Jung-Woo Ha, Byoung-Tak Zhang 
--  https://arxiv.org/abs/1610.04325
--
--  This code is based on 
--    https://github.com/VT-vision-lab/VQA_LSTM_CNN/blob/master/eval.lua
-----------------------------------------------------------------------------

require 'nn'
require 'rnn'
require 'cutorch'
require 'cunn'
require 'optim'
require 'hdf5'
cjson=require('cjson');
require 'xlua'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Test the Visual Question Answering model')
cmd:text()
cmd:text('Options')
-- Data input settings
cmd:option('-input_img_h5','data_train-val_test-dev_2k/data_res.h5','path to the h5file containing the image feature')
cmd:option('-input_ques_h5','data_train-val_test-dev_2k/data_prepro.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','data_train-val_test-dev_2k/data_prepro.json','path to the json file containing additional info and vocab')
cmd:option('-model_path', 'model/mrn2k.t7', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')
cmd:option('-out_path', 'result/', 'path to save output json file')
cmd:option('-out_prob', false, 'save prediction probability matrix as `model_name.t7`')
cmd:option('-type', 'test-dev2015', 'evaluation set')

-- Model parameter settings (shoud be the same with the training)
cmd:option('-backend', 'nn', 'nn|cudnn')
cmd:option('-batch_size', 200,'batch_size for each iterations')
cmd:option('-rnn_model', 'GRU', 'question embedding model')
cmd:option('-input_encoding_size', 620, 'he encoding size of each token in the vocabulary')
cmd:option('-rnn_size',2400,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-common_embedding_size', 1200, 'size of the common embedding vector')
cmd:option('-num_output', 2000, 'number of output answers')
cmd:option('-model_name', 'MLB', 'model name')
cmd:option('-label','','model label')
cmd:option('-num_layers', 1, '# of layers of Multimodal Residual Networks')
cmd:option('-priming',false,'priming with generated caption')
cmd:option('-glimpse', 2, '# of glimpses')

cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')

opt = cmd:parse(arg)
print(opt)

torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

require 'misc.RNNUtils'
if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.setDevice(opt.gpuid + 1)
end

------------------------------------------------------------------------
-- Setting the parameters
------------------------------------------------------------------------
local model_name = opt.model_name..opt.label..'_L'..opt.num_layers
local model_path = opt.model_path
local num_layers = opt.num_layers
local batch_size=opt.batch_size
local embedding_size_q=opt.input_encoding_size
local rnn_size_q=opt.rnn_size
local common_embedding_size=opt.common_embedding_size
local noutput=opt.num_output
local glimpse=opt.glimpse

------------------------------------------------------------------------
-- Loading Dataset
------------------------------------------------------------------------
print('DataLoader loading h5 file: ', opt.input_json)

local file = io.open(opt.input_json, 'r')
local text = file:read()
file:close()
json_file = cjson.decode(text)

print('DataLoader loading h5 file: ', opt.input_ques_h5)
dataset = {}
local h5_file = hdf5.open(opt.input_ques_h5, 'r')

dataset['question'] = h5_file:read('/ques_test'):all()
dataset['lengths_q'] = h5_file:read('/ques_length_test'):all()
dataset['img_list'] = h5_file:read('/img_pos_test'):all()
dataset['ques_id'] = h5_file:read('/question_id_test'):all()
dataset['MC_ans_test'] = h5_file:read('/MC_ans_test'):all()
h5_file:close()

print('DataLoader loading h5 file: ', opt.input_img_h5)
local h5_file = hdf5.open(opt.input_img_h5, 'r')
local test_list={}
for i,imname in pairs(json_file['unique_img_test']) do
	table.insert(test_list, imname)
end
local nhimage=2048
dataset['question'] = right_align(dataset['question'],dataset['lengths_q'])

local count = 0
for i, w in pairs(json_file['ix_to_word']) do count = count + 1 end
local vocabulary_size_q=count
collectgarbage();

------------------------------------------------------------------------
--Design Parameters and Network Definitions
------------------------------------------------------------------------

-- skip-thought vectors
lookup = nn.LookupTableMaskZero(vocabulary_size_q, embedding_size_q)

if opt.rnn_model == 'GRU' then
   -- Bayesian GRUs have right dropouts
   bgru = nn.GRU(embedding_size_q, rnn_size_q, false, .25, true)
   bgru:trimZero(1)
   --encoder: RNN body
   encoder_net_q=nn.Sequential()
               :add(nn.Sequencer(bgru))
               :add(nn.SelectTable(-1))
elseif opt.rnn_model == 'LSTM' then
   opt.rnn_layers = 2
   local rnn_model = nn.LSTM(embedding_size_q, rnn_size_q, false, nil, .25, true)
   rnn_model:trimZero(1)
   encoder_net_q = nn.Sequential()
         :add(nn.Sequencer(rnn_model))
   for i=2,opt.rnn_layers do
      local rnn_model = nn.LSTM(rnn_size_q, rnn_size_q, false, nil, .25, true)
      rnn_model:trimZero(1)
      encoder_net_q
         :add(nn.ConcatTable()
            :add(nn.SelectTable(-1))
            :add(nn.Sequential()
               :add(nn.Sequencer(rnn_model))
               :add(nn.SelectTable(-1))))
         :add(nn.JoinTable(2))
   end
   rnn_size_q = rnn_size_q*opt.rnn_layers
end

--embedding: word-embedding
embedding_net_q=nn.Sequential()
            :add(lookup)
            :add(nn.SplitTable(2))

require 'netdef.MRN'
multimodal_net=netdef[opt.model_name](rnn_size_q,nhimage,common_embedding_size,dropout,num_layers,noutput,batch_size,glimpse)
print(multimodal_net)

local model = nn.Sequential()
   :add(nn.ParallelTable()
      :add(nn.Sequential()
         :add(embedding_net_q)
         :add(encoder_net_q))
      :add(nn.Identity()))
   :add(multimodal_net)

--criterion
criterion=nn.CrossEntropyCriterion()

if opt.gpuid >= 0 then
   print('shipped data function to cuda...')
   model = model:cuda()
   criterion = criterion:cuda()
end

-- setting to evaluation
model:evaluate()

w,dw=model:getParameters();
print('nParams=', w:size())

-- loading the model
model_param=torch.load(model_path);

-- trying to use the precedding parameters
w:copy(model_param)

------------------------------------------------------------------------
--Grab Next Batch--
------------------------------------------------------------------------
function dataset:next_batch_test(s,e)
	local batch_size=e-s+1;
	local qinds=torch.LongTensor(batch_size):fill(0);
	local iminds=torch.LongTensor(batch_size):fill(0);
	local fv_im=torch.Tensor(batch_size,2048,14,14);
	for i=1,batch_size do
		qinds[i]=s+i-1;
		iminds[i]=dataset['img_list'][qinds[i]];
		fv_im[i]:copy(h5_file:read(paths.basename(test_list[iminds[i]])):all())
	end
	
	local fv_sorted_q=dataset['question']:index(1,qinds) 
	local qids=dataset['ques_id']:index(1,qinds);

	-- ship to gpu
	if opt.gpuid >= 0 then
		fv_sorted_q=fv_sorted_q:cuda() 
		fv_im = fv_im:cuda()
	end
	
	--print(string.format('batch_sort:%f',timer:time().real));
	return fv_sorted_q,fv_im,qids,batch_size;
end

------------------------------------------------------------------------
-- Objective Function and Optimization
------------------------------------------------------------------------
-- duplicate the RNN
function forward(s,e)
	local timer = torch.Timer();
	--grab a batch--
	local fv_sorted_q,fv_im,qids,batch_size=dataset:next_batch_test(s,e);
	
	if batch_size ~= opt.batch_size then
		netdef[opt.model_name..'_updateBatchSize'](multimodal_net,nhimage,common_embedding_size,num_layers,batch_size,glimpse)
	end
	model:cuda()
	local scores = model:forward({fv_sorted_q, fv_im})

	return scores:double(),qids;
end

-----------------------------------------------------------------------
-- Do Prediction
-----------------------------------------------------------------------
nqs=dataset['question']:size(1);
scores=torch.Tensor(nqs,noutput);
qids=torch.LongTensor(nqs);
for i=1,nqs,batch_size do
	xlua.progress(i, nqs);if batch_size>nqs-i then xlua.progress(nqs, nqs) end
	r=math.min(i+batch_size-1,nqs);
	scores[{{i,r},{}}],qids[{{i,r}}]=forward(i,r);
end

if opt.priming then
   -----------------------------------------------------------------------
   -- Caption refinery using a priming vector
   -----------------------------------------------------------------------
   dofile('myutils.lua')
   captions = {}
   captions[1] = readAll('../neuraltalk2/vis/captions_test2015.json')
   priming = torch.Tensor(nqs,noutput):zero()
   exceptions = {'yes','no','on','a'}  -- and numbers
   answers = table.values(json_file['ix_to_ans'])
   lambda = 1.0
   assert(#answers==noutput)
   for i=1,noutput do
      if tonumber(answers[i]) then
         table.insert(exceptions, answers[i])
      end
   end
   function ans_to_ix(answer)
      if not json_file['ans_to_ix'] then
         json_file['ans_to_ix'] = table.inverse(json_file['ix_to_ans'])
      end
      return tonumber(json_file['ans_to_ix'][answer])
   end
   for i=1,nqs do
      imind=dataset['img_list'][i];
      filename='/opt/data/coco/'..json_file.unique_img_test[imind]
      local function fn_to_cap(filename)
         if not _fn_to_cap then
            _fn_to_cap = {}
            for j=1,#captions do
               for k=1,#captions[j] do
                  local key = captions[j][k].file_name
                  if not _fn_to_cap[key] then 
                     _fn_to_cap[key] = captions[j][k].caption
                  else 
                     _fn_to_cap[key] = _fn_to_cap[key]..' '..captions[j][k].caption 
                  end
               end
            end
         end
         return _fn_to_cap[filename]
      end
      caption=fn_to_cap(filename)
      unique_words = table.values(caption:split(' '), true)  -- just single words
      for j=1,#unique_words do
         local idx = ans_to_ix(unique_words[j])
         if idx then
            priming[i][idx]=lambda
         end
         if j<#unique_words then  -- bigram
            local idx = ans_to_ix(unique_words[j]..' '..unique_words[j+1])
            if idx then
               priming[i][idx]=lambda
            end
         end
      end
   end
   for i=1,#exceptions do
      priming[{{},{ans_to_ix(exceptions[i])}}]=lambda
   end
   scores = scores + priming
end

if opt.out_prob then torch.save(model_name..'.t7', scores); return end

tmp,pred=torch.max(scores,2);

------------------------------------------------------------------------
-- Write to json file
------------------------------------------------------------------------
function writeAll(file,data)
	local f = io.open(file, "w")
	f:write(data)
	f:close() 
end

function saveJson(fname,t)
	return writeAll(fname,cjson.encode(t))
end

response={};
for i=1,nqs do
	table.insert(response,{question_id=qids[i],answer=json_file['ix_to_ans'][tostring(pred[{i,1}])]})
end

paths.mkdir(opt.out_path)
saveJson(opt.out_path .. 'vqa_OpenEnded_mscoco_'..opt.type..'_'..model_name..'_results.json',response);

mc_response={};

for i=1,nqs do
	local mc_prob = {}
	local mc_idx = dataset['MC_ans_test'][i]
	local tmp_idx = {}
	for j=1, mc_idx:size()[1] do
		if mc_idx[j] ~= 0 then
			table.insert(mc_prob, scores[{i, mc_idx[j]}])
			table.insert(tmp_idx, mc_idx[j])
		end
	end
	local tmp,tmp2=torch.max(torch.Tensor(mc_prob), 1);
	table.insert(mc_response, {question_id=qids[i],answer=json_file['ix_to_ans'][tostring(tmp_idx[tmp2[1]])]})
end

saveJson(opt.out_path .. 'vqa_MultipleChoice_mscoco_'..opt.type..'_'..model_name..'_results.json',mc_response);

h5_file:close()
