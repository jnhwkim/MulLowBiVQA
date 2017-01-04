local class = require 'class'

local mhdf5 = class('mhdf5')

function mhdf5:__init(hdf5, dims, N)
   self.hdf5 = hdf5
   self.N = N
   self.data = torch.Tensor(N, unpack(dims))
   self.index = {}
   self.numCached = 0
   self.verbose = false
end

function mhdf5:get(key)
   if not self.index[key] then
      local d = self.hdf5:read(key):all()
      if self.numCached >= self.N then
         if self.verbose then print('missed '..key) end
         return d         
      end
      self.numCached = self.numCached + 1
      self.index[key] = self.numCached
      self.data:select(1, self.numCached):copy(d)
      if self.verbose then print('cached '..key) end
   else
      if self.verbose then print('hit '..key) end
   end
   return self.data[self.index[key]]
end

return mhdf5