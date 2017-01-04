------------------------------------------------------------------------------
--  Hadamard Product for Low-rank Bilinear Pooling
--  Jin-Hwa Kim, Kyoung-Woon On, Jeonghee Kim, Jung-Woo Ha, Byoung-Tak Zhang 
--  https://arxiv.org/abs/1610.04325
-----------------------------------------------------------------------------

if not cjson then
   cjson = require('cjson')
end

------------------------------------------------------------------------
-- Write to / Read from Json file
------------------------------------------------------------------------
function writeAll(file,data)
   local f = io.open(file, "w")
   f:write(data)
   f:close() 
end

function readAll(file,verbose)
   if verbose then print('reading '..file..' ...') end
   local f = io.open(file, 'r')
   local text = f:read()
   f:close()
   local json_file = cjson.decode(text)
   return json_file
end

function saveJson(fname,t)
   return writeAll(fname,cjson.encode(t))
end

function readFileAll(path)
   local f = io.open(path)
   local b = f:read('*all')
   f:close()
   return b
end

function paths.numfiles(s,regex)
   local cnt = 0
   for f in paths.iterfiles(s) do
      if f:find(regex) then
         cnt = cnt + 1
      end
   end
   return cnt
end

function table.keys(tab)
   local keys = {}
   for k,v in pairs(tab) do
      table.insert(keys, k)
   end
   return keys
end

function table.values(tab, unique)
   local values = {}
   if unique then 
      for k,v in pairs(tab) do
         values[v] = 1
      end
      return table.keys(values)
   else
      for k,v in pairs(tab) do
         table.insert(values, v)
      end
      return values
   end
end

function table.intersect(a, b)
   local union = {}
   for i=1,#b do
      for j=1,#a do
         if b[i] == a[j] then
            table.insert(union, a[j])
            break
         end
      end
   end
   return union
end

function table.inverse(tab)
   local _tab = {}
   for k,v in pairs(tab) do
      _tab[v] = k
   end
   return _tab
end

function table.merge(a, b)  -- for lists
   for k,v in pairs(b) do a[#a+1] = v end
   return a
end

function string.trim(self)
   return self:gsub('^%s+', ''):gsub('%s+$', '')
end
