------------------------------------------------------------------------------
--  Hadamard Product for Low-rank Bilinear Pooling
--  Jin-Hwa Kim, Kyoung-Woon On, Jeonghee Kim, Jung-Woo Ha, Byoung-Tak Zhang 
--  https://arxiv.org/abs/1610.04325
------------------------------------------------------------------------------

require 'hdf5'
require '../myutils'
local py=require('fb.python')
local nltk=py.import('nltk')

-- Command-line Options
cmd=torch.CmdLine()
cmd:text()
cmd:text('Preprocess Visual Genome dataset for VQA augmentation')
cmd:text()
cmd:text('Options')
cmd:option('-input_json', '/mnt/sandisk/visualgenome_v1.2/question_answers.json')
cmd:option('-vqa_json', '../data_train-val_test-dev_2k/data_prepro.json')
cmd:option('-output_h5', 'visg_prepro.h5')
cmd:option('-max_len', 26)
cmd:option('-debug', false)
opt=cmd:parse(arg)

-- Reading json files
local vg=readAll(opt.input_json,true)
local vq=readAll(opt.vqa_json,true)
print('done')

-- Inverse Tables
vq.ans_to_ix=table.inverse(vq.ix_to_ans)
vq.word_to_ix=table.inverse(vq.ix_to_word)

-- One pass: Get statistics
local qa_count=0
local im_count=0
for i=1,#vg do
   if i%100==0 or i==#vg then xlua.progress(i,#vg) end
   local image_id=vg[i].id
   local qas=vg[i].qas
   qa_count=qa_count+#qas
   if #qas>0 then im_count=im_count+1 end
   if i%10000==0 then collectgarbage() end
end
print('# of images=', im_count)  -- 99280
print('# of questions=', qa_count) -- 1445322

-- Vocabulary statistics
local ans_miss_mat={0,0,0}
local ans_cnts_mat={0,0,0}
local word_miss_prob=0

-- Data Tensors
local ques_train=torch.IntTensor(qa_count,opt.max_len):zero()
local ques_length_train=torch.IntTensor(qa_count)
local answers=torch.IntTensor(qa_count)
local question_id_train=torch.IntTensor(qa_count)
local img_id_train=torch.IntTensor(qa_count)
local unique_img_train=torch.IntTensor(im_count)
local qa_idx=0
local im_idx=0

function word2num(w)
   local wordNums={'one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve'}
   wordNums=table.inverse(wordNums)
   return wordNums[w] and tostring(wordNums[w]) or w
end

-- Two pass: Populate data tensors
for i=1,#vg do
   if i%100==0 or i==#vg then xlua.progress(i,#vg) end
   local image_id=vg[i].id
   local qas=vg[i].qas
   if #qas>0 then im_idx=im_idx+1; unique_img_train[im_idx]=image_id end
   for q=1,#qas do
      local qa=qas[q]
      local answer=qa.answer:gsub('%.',''):lower():gsub('^a ',''):gsub('^an ','')
      answer=word2num(answer)
      answer=answer:gsub('^grey$','gray')
      local ques=nltk.word_tokenize(qa.question:lower())
      local isValid=true
      if #ques>opt.max_len then
         if opt.debug then
            print('[TOO LONG] '..table.concat(ques,' '))
         end
         isValid=false
      end
      local ix=#answer:split(' ')
      if not vq.ans_to_ix[answer] then
         if opt.debug then
            print('[NOT FOUND ANS] '..answer)
         end
         isValid=false
         if ix>=3 then ans_miss_mat[3]=ans_miss_mat[3]+1
         else ans_miss_mat[ix]=ans_miss_mat[ix]+1 end
      end
      if isValid then
         qa_idx=qa_idx+1
         local unk=0
         for j=0,#ques-1 do
            local w=tostring(ques[j])
            ques_train[qa_idx][opt.max_len-#ques+j+1]=vq.word_to_ix[w] or vq.word_to_ix['UNK']
            if not vq.word_to_ix[w] then unk=unk+1 end
         end
         ques_length_train[qa_idx]=#ques
         answers[qa_idx]=vq.ans_to_ix[answer]
         question_id_train[qa_idx]=qa.qa_id
         img_id_train[qa_idx]=image_id

         if ix>=3 then ans_cnts_mat[3]=ans_cnts_mat[3]+1
         else ans_cnts_mat[ix]=ans_cnts_mat[ix]+1 end
         word_miss_prob=word_miss_prob+unk/#ques
      end
   end
   if i%1000==0 then collectgarbage() end
end
word_miss_prob=word_miss_prob/qa_idx

print('answer miss=', unpack(ans_miss_mat))
print('valid answer count=', unpack(ans_cnts_mat))
print('word miss prob=', word_miss_prob)
print('valid # ims=', im_idx)
print('valid # qas=', qa_idx)

local f=hdf5.open(opt.output_h5, 'w')
f:write('ques_train',ques_train:narrow(1,1,qa_idx))
f:write('ques_length_train',ques_length_train:narrow(1,1,qa_idx))
f:write('answers',answers:narrow(1,1,qa_idx))
f:write('question_id_train',question_id_train:narrow(1,1,qa_idx))
f:write('img_id_train',img_id_train:narrow(1,1,qa_idx))
f:write('unique_img_train',unique_img_train:narrow(1,1,im_idx))
f:close()
