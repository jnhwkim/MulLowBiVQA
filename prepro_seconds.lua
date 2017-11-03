------------------------------------------------------------------------------
--  Hadamard Product for Low-rank Bilinear Pooling
--  Jin-Hwa Kim, Kyoung-Woon On, Woosang Lim, Jeonghee Kim, Jung-Woo Ha, Byoung-Tak Zhang 
--  https://arxiv.org/abs/1610.04325
------------------------------------------------------------------------------

require 'myutils'

trainset={'train','val'}
a={}
for k,v in pairs(trainset) do
   anno='/opt/data/coco/mscoco_'..v..'2014_annotations.json'
   print('read '..anno)
   local j=readAll(anno)
   a=table.merge(a,j.annotations)
   print(#a,#j.annotations)
   collectgarbage()
end

if not opt then 
   opt={} 
   opt.input_ques_h5='data_train-val_test-dev_2k/data_prepro.json'
end
q=readAll(opt.input_ques_h5)
q.ans_to_ix=table.inverse(q.ix_to_ans)
data_dir=opt.input_ques_h5:gsub('/.+','')
print('data_dir =',data_dir)

nFuzzy={0,0,0}
seconds={}
for i=1,#a do
   if i%1000==0 then
      xlua.progress(i,#a)
      collectgarbage()
   end
   local counts={}
   for j=1,#a[i].answers do
      c=counts[a[i].answers[j].answer]
      counts[a[i].answers[j].answer]=c and c+1 or 1
   end
   nCandidates=0
   for k,v in pairs(counts) do
      if v > 2 then
         nCandidates=nCandidates+1
         if k~=a[i].multiple_choice_answer and q.ans_to_ix[k] then
            local mc=counts[a[i].multiple_choice_answer]
            local ct=10  --v+mc
            seconds[tostring(a[i].question_id)]={answer=tonumber(q.ans_to_ix[k]), p=v/ct}
         end
         if not q.ans_to_ix[k] then
            -- print('not found ', k)
         end
      end
   end
   if nCandidates > 0 then
      nFuzzy[nCandidates]=nFuzzy[nCandidates]+1
   end
end
print(#a, nFuzzy)
-- 369861 
-- {
--   1 : 292764
--   2 : 56604
--   3 : 809
-- }
-- #seconds=49780

saveJson(paths.concat(data_dir,'seconds.json'),seconds)