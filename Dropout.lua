------------------------------------------------------------------------
--[[ Dropout ]]--

-- Implementation of Lazy Dropout. 
-- `lazy` option is used to to only resample after backward is called. 
-- This mechanism is used by Bayesian GRUs to use the same dropout mask 
-- for each sequence, not for each word. 
-- See GRU part in README.md (Ref. E & F)
------------------------------------------------------------------------
local Dropout, Parent = nn.Dropout, nn.Module

function Dropout:__init(p,v1,inplace,lazy,mono,stochastic_inference,func)
   Parent.__init(self)
   self.p = p or 0.5
   self.train = true
   self.inplace = inplace
   self.lazy = lazy or false
   self.mono = mono or false  -- used by trimZero, single sample for a batch
   self.stochastic_inference = stochastic_inference
   self.func = func
   self.flag = true  -- used by lazy noise
   -- version 2 scales output during training instead of evaluation
   self.v2 = not v1
   if self.p >= 1 or self.p < 0 then
      error('<Dropout> illegal percentage, must be 0 <= p < 1')
   end
   self.noise = torch.Tensor()
end

function Dropout:updateOutput(input)
   if self.train or not self.stochastic_inference then
      return self:_updateOutput(input)
   else
      N = 10
      assert(input:dim()==2, 'only support 2d input!')
      if not self.samples then
         self.samples = torch.Tensor(N, input:size(1), self.func:get(1).weight:size(1)):typeAs(input)
      end
      self.samples:zero()
      for i=1,N do
         self.samples[i]:copy(self.func:get(1):forward(self:_updateOutput(input)))
      end
      mu = self.samples:mean(1):squeeze(1)
      std = self.samples:std(1):squeeze(1)
      Z = 10
      self.output:resizeAs(self.samples[1]):zero()
      for z=1,Z do
         u = torch.randn(input:size(1), self.func:get(1).weight:size(1)):typeAs(input):cmul(std):add(mu)
         self.output:add(self.func:get(2):forward(u))
      end
      self.output:div(Z)
      self.train = false
      self.samples = nil
      collectgarbage()
   end
   return self.output
end

function Dropout:_updateOutput(input)
   if self.inplace then
      self.output = input
   else
      self.output:resizeAs(input):copy(input)
   end
   if self.p > 0 then
      if self.train then
         if not self.lazy or self.flag then
            local noiseSize = input:size()
            if self.mono then noiseSize[1] = 1 end
            self.noise:resize(noiseSize)
            self.noise:bernoulli(1-self.p)
            if self.v2 then
               self.noise:div(1-self.p)
            end
            self.flag = false
         end
         if self.mono and self.noise:size(1) ~= input:size(1) then
            self.noise = self.noise:narrow(1,1,1):expandAs(input)
         end
         self.output:cmul(self.noise)
      elseif not self.v2 then
         self.output:mul(1-self.p)
      end
   end
   return self.output
end

function Dropout:updateGradInput(input, gradOutput)
   if self.lazy then
      self.flag = true
   end
   if self.train then
      if self.inplace then
         self.gradInput = gradOutput
      else
         self.gradInput:resizeAs(gradOutput):copy(gradOutput)
      end
      if self.p > 0 then
         self.gradInput:cmul(self.noise) -- simply mask the gradients with the noise vector
      end
   else
      if self.inplace then
         self.gradInput = gradOutput
      else
         self.gradInput:resizeAs(gradOutput):copy(gradOutput)
      end
      if not self.v2 and self.p > 0 then
         self.gradInput:cdiv(1-self.p)
      end
   end
   return self.gradInput
end

function Dropout:__tostring__()
   return string.format('%s(%.1f, %s)', torch.type(self), self.p, self.lazy and 'lazy' or 'busy')
end
