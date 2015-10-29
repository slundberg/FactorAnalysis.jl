module FactorAnalysis

using Distributions

export CFADistribution, rand, loglikelihood, dloglikelihood, fit_mle, fit_map


include("sigma.jl")
include("theta.jl")
include("em.jl")
include("utils.jl")
#include("pgradient.jl")
#include("em.jl")

end # module
