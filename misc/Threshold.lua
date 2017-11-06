----------------------------
-- Override functions for Guided-Backpropagation 
----------------------------

function nn.Threshold.updateGradInput(self, input, gradOutput)
   self:validateParameters()
   input.THNN.Threshold_updateGradInput(
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.threshold,
      self.val,
      self.inplace
   )
   return self.gradInput:cmul(torch.gt(gradOutput, 0):float())
end

function cudnn._Pointwise.updateGradInput(self, input, gradOutput)
   if not gradOutput:isContiguous() then
      self._gradOutput = self._gradOutput or gradOutput.new()
      self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
      gradOutput = self._gradOutput
   end
   self:createIODescriptors(input)
   if self.inplace then
      self.output:set(input);
      self.gradInput:set(gradOutput)
   else
      self.gradInput:resizeAs(input)
   end
   cudnn.errcheck('cudnnActivationBackward',
            cudnn.getHandle(), self.activDesc[0],
            cudnn.scalar(input, 1),
            self.iDesc[0], self.output:data(),
            self.iDesc[0], gradOutput:data(),
            self.iDesc[0], input:data(),
            cudnn.scalar(input, 0),
            self.iDesc[0], self.gradInput:data());
   return self.gradInput:cmul(torch.gt(gradOutput, 0):cuda())
end