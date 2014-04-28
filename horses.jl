using Datetime
using Devectorize
using MAT
using Optim
using OptionsMod

include("types.jl")
include("util.jl")
include("features.jl")
include("inference.jl")
include("likelihood.jl")
include("test.jl")

cd("/Users/john/Dropbox/Machine\ Learning/Tutorials/Horses/")
srand(1)