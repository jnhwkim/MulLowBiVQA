--[[

   Regularization of Neural Networks using DropConnect
   Li Wan, Matthew Zeiler, Sixin Zhang, Yann LeCun, Rob Fergus

   Dept. of Computer Science, Courant Institute of Mathematical Science, New York University

   Implemented by John-Alexander M. Assael (www.johnassael.com), 2015

]]--

local LinearDropconnect, parent = torch.class('nn.LinearDropconnect', 'nn.Linear')

function LinearDropconnect:__init(inputSize, outputSize, p, activation)

   self.train = true

   self.p = p or 0.5
   if self.p >= 1 or self.p < 0 then
      error('<LinearDropconnect> illegal percentage, must be 0 <= p < 1')
   end
   self.activation = nn[activation]()

   self.noiseWeight = torch.Tensor(outputSize, inputSize)
   self.noiseBias = torch.Tensor(outputSize)

   parent.__init(self, inputSize, outputSize)
end


function LinearDropconnect:reset(stdv)   
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
         self.bias[i] = torch.uniform(-stdv, stdv)
      end
   else
      self.weight:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv)
   end

   self.noiseWeight:fill(1)
   self.noiseBias:fill(1)

   return self
end

function LinearDropconnect:updateOutput(input)
   if self.train then
      return self:_updateOutput(input)
   else
      N = 20
      assert(input:dim()==2, 'only support 2d input!')
      if not self.samples then
         self.samples = torch.Tensor(N, input:size(1), self.weight:size(1)):typeAs(input)
      end
      self.samples:zero()
      self.train = true
      for i=1,N do
         self.samples[i]:copy(self:_updateOutput(input))
      end
      mu = self.samples:mean(1):squeeze(1)
      std = self.samples:std(1):squeeze(1)
      Z = 20
      self.output:zero()
      for z=1,Z do
         u = torch.randn(input:size(1), self.weight:size(1)):typeAs(input):cmul(std):add(mu)
         self.output:add(self.activation:forward(u))
      end
      self.output:div(Z)
      self.train = false
      self.samples = nil
      collectgarbage()
   end
   return self.output
end

function LinearDropconnect:_updateOutput(input)

   -- Dropconnect
   if self.train then
      self.noiseWeight:bernoulli(1-self.p):cmul(self.weight)
      self.noiseBias:bernoulli(1-self.p):cmul(self.bias)
   end

   if input:dim() == 1 then
      self.output:resize(self.bias:size(1))
      if self.train then
         self.output:copy(self.noiseBias)
         self.output:addmv(1, self.noiseWeight, input)
      else
         self.output:copy(self.bias)
         self.output:addmv(1, self.weight, input)
      end
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nElement = self.output:nElement()
      self.output:resize(nframe, self.bias:size(1))
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end
      self.addBuffer = self.addBuffer or input.new()
      if self.addBuffer:nElement() ~= nframe then
         self.addBuffer:resize(nframe):fill(1)
      end
      if self.train then
         self.output:addmm(0, self.output, 1, input, self.noiseWeight:t())
         self.output:addr(1, self.addBuffer, self.noiseBias)
      else
         self.output:addmm(0, self.output, 1, input, self.weight:t())
         self.output:addr(1, self.addBuffer, self.bias)
      end
   else
      error('input must be vector or matrix')
   end

   return self.output
end

function LinearDropconnect:updateGradInput(input, gradOutput)
   if self.gradInput then

      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      if input:dim() == 1 then
         if self.train then
            self.gradInput:addmv(0, 1, self.noiseWeight:t(), gradOutput)
         else
            self.gradInput:addmv(0, 1, self.weight:t(), gradOutput)
         end
      elseif input:dim() == 2 then
         if self.train then
            self.gradInput:addmm(0, 1, gradOutput, self.noiseWeight)
         else
            self.gradInput:addmm(0, 1, gradOutput, self.weight)
         end
      end

      return self.gradInput
   end
end