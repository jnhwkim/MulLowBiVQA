require 'cbp'  -- https://github.com/jnhwkim/cbp

function netdef.MCB(rnn_size_q,nhimage,common_embedding_size,joint_dropout,num_layers,noutput,batch_size,glimpse)
   local p = .5
   local activation = 'ReLU'
   local multimodal_net=nn.Sequential()
   local glimpse=glimpse or 2
   assert(num_layers==1, 'do not support stacked structure')
   print('MCB VER')
   local cbp1,cbp2
   if paths.filep('netdef/cbp1.t7') then
      cbp1 = torch.load('netdef/cbp1.t7')
   else
      cbp1 = nn.CompactBilinearPooling(16000)
   end
   if paths.filep('netdef/cbp2.t7') then
      cbp2 = torch.load('netdef/cbp2.t7')
   else
      cbp2 = nn.CompactBilinearPooling(16000)
   end
   
   local glimpses=nn.ConcatTable()
   for i=1,glimpse do
      local visual_embedding_=nn.Sequential()
            :add(nn.ConcatTable()
               :add(nn.SelectTable(2+i))   -- softmax [3~]
               :add(nn.SelectTable(2)))  -- v
            :add(nn.ParallelTable()
               :add(nn.Identity())
               :add(nn.SplitTable(2)))
            :add(nn.MixtureTable())
      glimpses:add(visual_embedding_)
   end

   local visual_embedding=nn.Sequential()
         :add(glimpses)
         :add(nn.JoinTable(2))

   for i=1,num_layers do
      local reshaper
      if i == 1 then
         reshaper = nn.Sequential()
            :add(nn.Transpose({2,3},{3,4}))
            :add(nn.Reshape(14*14, nhimage))
      else reshaper = nn.Identity() end

      local attention=nn.Sequential()  -- attention networks
            :add(nn.ParallelTable()
               :add(nn.Sequential()
                  :add(nn.Dropout(.3))
                  :add(nn.Replicate(14*14, 2))
                  :add(nn.Reshape(batch_size*14*14, rnn_size_q, false)))
               :add(nn.Sequential()
                  :add(nn.Reshape(batch_size*14*14, nhimage, false))))
            :add(cbp1)
            :add(nn.SignedSquareRoot(true))
            :add(nn.Normalize(2))
            :add(nn.Dropout(.1))
            :add(nn.Reshape(batch_size, 14, 14, 16000, false))
            :add(nn.Transpose({3,4},{2,3}))
            :add(nn.SpatialConvolution(16000,512,1,1,1,1))
            :add(nn[activation](true))
            :add(nn.SpatialConvolution(512,glimpse,1,1,1,1))
            :add(nn.Reshape(batch_size, glimpse, 14*14, false))
            :add(nn.SplitTable(2))

      local para_softmax=nn.ParallelTable()
      for j=1,glimpse do
         para_softmax:add(nn.SoftMax())
      end
      attention:add(para_softmax)

      multimodal_net:add(nn.ParallelTable()
         :add(nn.Identity())
         :add(reshaper)
      ):add(nn.ConcatTable()
         :add(nn.SelectTable(1))  -- q
         :add(nn.SelectTable(2))  -- v1
         :add(attention)  -- second-attention
      ):add(nn.FlattenTable()
      ):add(nn.ConcatTable()
         :add(nn.Sequential()
            :add(nn.SelectTable(1)))
         :add(visual_embedding)  -- if L > 1, do clone
         :add(nn.SelectTable(2))
      ):add(nn.ConcatTable()
         :add(nn.Sequential()
            :add(nn.NarrowTable(1,2))
            :add(cbp2)
            :add(nn.SignedSquareRoot(true))
            :add(nn.Normalize(2))
            :add(nn.Dropout(.1)))
         :add(nn.SelectTable(3))
      )
      rnn_size_q = common_embedding_size
   end
   multimodal_net:add(nn.SelectTable(1))
         :add(nn.Linear(16000,noutput))
   return multimodal_net,cbp1,cbp2
end

function netdef.MCB_updateBatchSize(net,nhimage,common_embedding_size,num_layers,batch_size,glimpse)
   local ngrid=14*14
   local idx=2  -- start idx
   local mstep=5
   local glimpse=glimpse or 2
   for i=0,num_layers-1 do
      local att=net:get(idx+i*mstep)
      local a1=att:get(3):get(1):get(1)
      local a2=att:get(3):get(1):get(2)
      assert(torch.type(a1:get(3)) == 'nn.Reshape')
      assert(torch.type(a2:get(1)) == 'nn.Reshape')
      a1:remove(3)
      a2:remove(1)
      a1:insert(nn.Reshape(batch_size*ngrid,rnn_size_q,false),3)
      a2:insert(nn.Reshape(batch_size,ngrid,nhimage,false),1)
      local a2=att:get(3)
      assert(torch.type(a2:get(9)) == 'nn.Reshape')
      assert(torch.type(a2:get(6)) == 'nn.Reshape')
      a2:remove(9)
      a2:remove(6)
      a2:insert(nn.Reshape(batch_size,14,14,16000,false),6)
      a2:insert(nn.Reshape(batch_size,glimpse,ngrid,false),9)
   end
end
