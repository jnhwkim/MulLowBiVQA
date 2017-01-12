------------------------------------------------------------------------------
--  Hadamard Product for Low-rank Bilinear Pooling
--  Jin-Hwa Kim, Kyoung-Woon On, Jeonghee Kim, Jung-Woo Ha, Byoung-Tak Zhang 
--  https://arxiv.org/abs/1610.04325
--
--  This code is based on 
--    https://github.com/VT-vision-lab/VQA_LSTM_CNN/blob/master/train.lua
-----------------------------------------------------------------------------

require 'nn'
require 'rnn'
require 'dp'
require 'torch'
require 'optim'
require 'cutorch'
require 'cunn'
require 'hdf5'
require 'myutils'
mhdf5=require 'misc.mhdf5'
cjson=require('cjson') 

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Visual Question Answering model')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-input_img_h5','data_train-val_test-dev_2k/data_res.h5','path to the h5file containing the image feature')
cmd:option('-input_ques_h5','data_train-val_test-dev_2k/data_prepro.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','data_train-val_test-dev_2k/data_prepro.json','path to the json file containing additional info and vocab')
cmd:option('-input_skip','skipthoughts_model','path to skipthoughts_params')
cmd:option('-mhdf5_size', 10000)

-- Model parameter settings
cmd:option('-batch_size',100,'batch_size for each iterations')
cmd:option('-rnn_model', 'GRU', 'question embedding model')
cmd:option('-input_encoding_size', 620, 'the encoding size of each token in the vocabulary')
cmd:option('-rnn_size',2400,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-common_embedding_size', 1200, 'size of the common embedding vector')
cmd:option('-num_output', 2000, 'number of output answers')
cmd:option('-model_name', 'MLB', 'model name')
cmd:option('-label','','model label')
cmd:option('-num_layers', 1, '# of layers of Multimodal Residual Networks')
cmd:option('-dropout', .5, 'dropout probability for joint functions')
cmd:option('-glimpse', 2, '# of glimpses')
cmd:option('-clipping', 10, 'gradient clipping')

-- Second Answers
cmd:option('-seconds', false, 'usage of second candidate answers')
cmd:option('-input_seconds', 'data_train-val_test-dev_2k/seconds.json')

-- Optimizer parameter settings
cmd:option('-learning_rate',3e-4,'learning rate for rmsprop')
cmd:option('-learning_rate_decay_start', 0, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 100, 'every how many epoch thereafter to drop LR by 0.1?')
cmd:option('-max_iters', 250000, 'max number of iterations to run for ')
cmd:option('-optimizer','rmsprop','opimizer')

--check point
cmd:option('-save_checkpoint_every', 25000, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', 'model/', 'folder to save checkpoints')
cmd:option('-load_checkpoint_path', '', 'path to saved checkpoint')
cmd:option('-previous_iters', 0, 'previous # of iterations to get previous learning rate')
cmd:option('-kick_interval', 50000, 'interval of kicking the learning rate as its double')

-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-seed', 1231, 'random number generator seed to use')

--visualgenome augmentation
cmd:option('-vg', false, 'visual genome augmentation')
cmd:option('-vg_ques_h5', '', 'path to visual genome question h5 file')
cmd:option('-vg_img_h5', '', 'path to visual genome image h5 file')

opt = cmd:parse(arg)
opt.iterPerEpoch = 240000 / opt.batch_size
print(opt)

torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

require 'misc.RNNUtils'
if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1)
end

------------------------------------------------------------------------
-- Setting the parameters
------------------------------------------------------------------------
local model_name = opt.model_name..opt.label..'_L'..opt.num_layers
local num_layers = opt.num_layers
local model_path = opt.checkpoint_path
local batch_size=opt.batch_size
local embedding_size_q=opt.input_encoding_size
local rnn_size_q=opt.rnn_size
local common_embedding_size=opt.common_embedding_size
local noutput=opt.num_output
local dropout=opt.dropout
local glimpse=opt.glimpse
local decay_factor = 0.99997592083  -- math.exp(math.log(0.1)/opt.learning_rate_decay_every/opt.iterPerEpoch)
local question_max_length=26
local seconds=readAll(opt.input_seconds)
paths.mkdir(model_path)

------------------------------------------------------------------------
-- Loading Dataset
------------------------------------------------------------------------
local file = io.open(opt.input_json, 'r')
local text = file:read()
file:close()
json_file = cjson.decode(text)

print('DataLoader loading h5 file: ', opt.input_ques_h5)
dataset = {}
local h5_file = hdf5.open(opt.input_ques_h5, 'r')
local nhimage = 2048

dataset['question'] = h5_file:read('/ques_train'):all()
dataset['question_id'] = h5_file:read('/question_id_train'):all()
dataset['lengths_q'] = h5_file:read('/ques_length_train'):all()
dataset['img_list'] = h5_file:read('/img_pos_train'):all()
dataset['answers'] = h5_file:read('/answers'):all()
h5_file:close()

if opt.vg then
   h5_file = hdf5.open(opt.vg_ques_h5, 'r')
   dataset['question_vg'] = h5_file:read('/ques_train'):all()
   dataset['img_list_vg'] = h5_file:read('/img_id_train'):all()
   dataset['answers_vg'] = h5_file:read('/answers'):all()
   h5_file:close()
end

print('DataLoader loading h5 file: ', opt.input_img_h5)
local h5_file = hdf5.open(opt.input_img_h5, 'r')
local h5_cache = mhdf5(h5_file, {2048,14,14}, opt.mhdf5_size)  -- consumes 48Gb memory
if opt.vg then h5_file_vg = hdf5.open(opt.vg_img_h5, 'r') end
local train_list={}
for i,imname in pairs(json_file['unique_img_train']) do
    table.insert(train_list, imname)
end
dataset['question'] = right_align(dataset['question'],dataset['lengths_q'])

-- Normalize the image feature
if opt.img_norm == 1 then
end

local count = 0
for i, w in pairs(json_file['ix_to_word']) do count = count + 1 end
local vocabulary_size_q=count

collectgarbage() 

------------------------------------------------------------------------
--Design Parameters and Network Definitions
------------------------------------------------------------------------
print('Building the model...')

buffer_size_q=dataset['question']:size()[2]

if opt.rnn_model == 'GRU' then
   -- skip-thought vectors
   -- lookup = nn.LookupTableMaskZero(vocabulary_size_q, embedding_size_q)
   if opt.num_output == 1000 then lookupfile = 'lookup_fix.t7'
   elseif opt.num_output == 2000 then lookupfile = 'lookup_2k.t7' 
   elseif opt.num_output == 3000 then lookupfile = 'lookup_3k.t7' 
   end
   lookup = torch.load(paths.concat(opt.input_skip, lookupfile))
   assert(lookup.weight:size(1)==vocabulary_size_q+1)  -- +1 for zero
   assert(lookup.weight:size(2)==embedding_size_q)
   gru = torch.load(paths.concat(opt.input_skip, 'gru.t7'))
   -- Bayesian GRUs have right dropouts
   rnn_model = nn.GRU(embedding_size_q, rnn_size_q, false, .25, true)
   skip_params = gru:parameters()
   rnn_model:migrate(skip_params)
   rnn_model:trimZero(1)
   gru = nil

   --encoder: RNN body
   encoder_net_q=nn.Sequential()
               :add(nn.Sequencer(rnn_model))
               :add(nn.SelectTable(question_max_length))
   
elseif opt.rnn_model == 'LSTM' then
   lookup = nn.LookupTableMaskZero(vocabulary_size_q, embedding_size_q)
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
   encoder_net_q:getParameters():uniform(-0.08, 0.08) 
end

collectgarbage()
--embedding: word-embedding
embedding_net_q=nn.Sequential()
            :add(lookup)
            :add(nn.SplitTable(2))

require 'netdef.MRN'
if opt.model_name=='MCB' then
   multimodal_net,cbp1,cbp2=netdef[opt.model_name](rnn_size_q,nhimage,common_embedding_size,dropout,num_layers,noutput,batch_size,glimpse)
else
   multimodal_net=netdef[opt.model_name](rnn_size_q,nhimage,common_embedding_size,dropout,num_layers,noutput,batch_size,glimpse)
end
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

local multimodal_w=multimodal_net:getParameters()
multimodal_w:uniform(-0.08, 0.08) 
w,dw=model:getParameters()

if paths.filep(opt.load_checkpoint_path) then
   print('loading checkpoint model...')
   -- loading the model
   model_param=torch.load(opt.load_checkpoint_path);
   -- trying to use the precedding parameters
   w:copy(model_param)
end

-- optimization parameter
local optimize={} 
optimize.maxIter=opt.max_iters 
optimize.learningRate=opt.learning_rate
optimize.update_grad_per_n_batches=1

optimize.winit=w
print('nParams=',optimize.winit:size(1))
print('decay_factor =', decay_factor)

------------------------------------------------------------------------
-- Next batch for train
------------------------------------------------------------------------
function dataset:next_batch(batch_size)
   local qinds=torch.LongTensor(batch_size):fill(0) 
   local iminds=torch.LongTensor(batch_size):fill(0)  
   local nqs=dataset['question']:size(1)
   local fv_im=torch.Tensor(batch_size,2048,14,14)
   -- we use the last val_num data for validation (the data already randomlized when created)
   for i=1,batch_size do
      qinds[i]=torch.random(nqs) 
      iminds[i]=dataset['img_list'][qinds[i]]
      fv_im[i]:copy(h5_cache:get(paths.basename(train_list[iminds[i]])))
      --fv_im[i]:copy(h5_file:read(paths.basename(train_list[iminds[i]])):all())
   end

   local fv_sorted_q=dataset['question']:index(1,qinds) 
   local labels=dataset['answers']:index(1,qinds)

   -- using second candidate answer sampling
   if opt.seconds then
      local sampling=torch.rand(batch_size)
      local qids=dataset['question_id']:index(1,qinds) 
      for i=1,batch_size do
         local second=seconds[tostring(qids[i])]
         if second then
            -- print('seconds hit!')
            if sampling[i]<second.p then
               -- print('seconds sampled! p=', second.p)
               -- print(json_file.ix_to_ans[tostring(labels[i])]..'=>'..
                     -- json_file.ix_to_ans[tostring(second.answer)])
               labels[i]=second.answer
            end
         end
      end
   end
   
   -- ship to gpu
   if opt.gpuid >= 0 then
      fv_sorted_q=fv_sorted_q:cuda() 
      fv_im = fv_im:cuda()
      labels = labels:cuda()
   end
   return fv_sorted_q,fv_im,labels
end

function dataset:next_batch_vg(batch_size)
   local qinds=torch.LongTensor(batch_size):fill(0) 
   local iminds=torch.LongTensor(batch_size):fill(0)  
   local nqs=dataset['question_vg']:size(1)
   local fv_im=torch.Tensor(batch_size,2048,14,14)
   for i=1,batch_size do
      qinds[i]=torch.random(nqs) 
      iminds[i]=dataset['img_list_vg'][qinds[i]]
      fv_im[i]:copy(h5_file_vg:read(iminds[i]..'.jpg'):all())
   end

   local fv_sorted_q=dataset['question_vg']:index(1,qinds) 
   local labels=dataset['answers_vg']:index(1,qinds)
   
   -- ship to gpu
   if opt.gpuid >= 0 then
      fv_sorted_q=fv_sorted_q:cuda() 
      fv_im = fv_im:cuda()
      labels = labels:cuda()
   end
   return fv_sorted_q,fv_im,labels
end

------------------------------------------------------------------------
-- Objective Function and Optimization
------------------------------------------------------------------------
function JdJ(x)
   --clear gradients--
   dw:zero()

   --grab a batch--
   local fv_sorted_q,fv_im,labels
   if not opt.vg then
      fv_sorted_q,fv_im,labels=dataset:next_batch(batch_size)
   else
      fv_sorted_q,fv_im,labels=dataset:next_batch(math.ceil(batch_size/2))
      fv_sorted_q_vg,fv_im_vg,labels_vg=dataset:next_batch_vg(math.floor(batch_size/2))
      local joiner=nn.JoinTable(1):cuda()
      fv_sorted_q=joiner:forward{fv_sorted_q, fv_sorted_q_vg}:clone()
      fv_im=joiner:forward{fv_im, fv_im_vg}:clone()
      labels=joiner:forward{labels, labels_vg}:clone()
   end

   local scores = model:forward({fv_sorted_q, fv_im})
   local f=criterion:forward(scores,labels)
   local dscores=criterion:backward(scores,labels)
   model:backward(fv_sorted_q, dscores)
      
   gradients=dw
   if opt.clipping > 0 then gradients:clamp(-opt.clipping,opt.clipping) end
   if running_avg == nil then
      running_avg = f
   end
   running_avg=running_avg*0.95+f*0.05
   return f,gradients 
end

------------------------------------------------------------------------
-- Training
------------------------------------------------------------------------
local state={}
optimize.learningRate=optimize.learningRate*decay_factor^opt.previous_iters
optimize.learningRate=optimize.learningRate*2^math.min(2, math.floor(opt.previous_iters/opt.kick_interval))
for iter = opt.previous_iters + 1, opt.max_iters do
   if iter%opt.save_checkpoint_every == 0 then
      paths.mkdir(model_path..'save')
      torch.save(string.format(model_path..'save/'..model_name..'_iter%d.t7',iter),w) 
   end
   if iter%100 == 0 then
      print('training loss: ' .. running_avg, 'on iter: ' .. iter .. '/' .. opt.max_iters)
   end
   -- double learning rate at two iteration points
   if iter==opt.kick_interval or iter==opt.kick_interval*2 then
      optimize.learningRate=optimize.learningRate*2
      print('learining rate:', optimize.learningRate)
   end
   if opt.previous_iters == iter-1 then
      print('learining rate:', optimize.learningRate)
   end
   optim[opt.optimizer](JdJ, optimize.winit, optimize, state)
   
   if opt.model_name=='MCB' and iter==opt.previous_iters+1 then
      -- TODO: save only h and s, not entire module having output and tmp
      print('Saving MCB\'s h and s...')
      print(cbp1.h1:sub(1,5));print(cbp2.s2:sub(1,5))  -- sample check
      torch.save('netdef/cbp1.t7',cbp1)
      torch.save('netdef/cbp2.t7',cbp2)
   end
   if iter > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
      optimize.learningRate = optimize.learningRate * decay_factor -- set the decayed rate
   end 
   if iter%1 == 0 then -- change this to smaller value if out of the memory
      collectgarbage()
   end
end

-- Saving the final model
torch.save(string.format(model_path..model_name..'.t7',i),w) 
h5_file:close()
