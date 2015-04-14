require 'torch'
require 'nn'
require 'optim'

ffi = require('ffi')

function load_glove(path, inputDim)
    
    local glove_file = io.open(path)
    local glove_table = {}

    local line = glove_file:read("*l")
    while line do
        -- read the GloVe text file one line at a time, break at EOF
        local i = 1
        local word = ""
        for entry in line:gmatch("%S+") do -- split the line at each space
            if i == 1 then
                -- word comes first in each line, so grab it and create new table entry
                word = entry:gsub("%p+", ""):lower() -- remove all punctuation and change to lower case
                if string.len(word) > 0 then
                    glove_table[word] = torch.zeros(inputDim, 1) -- padded with an extra dimension for convolution
                else
                    break
                end
            else
                -- read off and store each word vector element
                glove_table[word][i-1] = tonumber(entry)
            end
            i = i+1
        end
        line = glove_file:read("*l")
    end
    
    return glove_table
end

cmd = torch.CmdLine()
cmd:option('-input', " ")
params = cmd:parse(arg)

glovePath = "/scratch/mu388/glove.twitter.27B.25d.txt" -- path to raw glove data .txt file
model = torch.load('model.net', 'ascii')
glove_table = load_glove(glovePath, 25)
data = torch.zeros(100, 25)
document = params.input
-- break each review into words and compute the document average
local l = 1
for word in document:gmatch("%S+") do
    if glove_table[word:gsub("%p+", "")] and l<100 then
        data[l] = glove_table[word:gsub("%p+", "")]
        l = l + 1
    end
end
data = data:resize(1,100,25)
pred = model:forward(data)
-- print(pred)
p = pred:add(.5):floor()
if p[1][1]<1 then
    p[1][1] = 1
end
if p[1][1]>5 then
    p[1][1] = 5
end

print(p[1][1])