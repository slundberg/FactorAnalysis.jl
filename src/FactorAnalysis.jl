module FactorAnalysis

using Distributions

export CFADistribution, rand, loglikelihood, dloglikelihood, fit_mle, fit_map

"Confirmatory factor analysis distribution object."
type CFADistribution
    Theta_X::SparseMatrixCSC
    A::SparseMatrixCSC
    Theta_L::AbstractMatrix
end
Base.length(d::CFADistribution) = Int(size(d.A)[1] + nnz(d.A) + (size(d.A)[2]-1)*size(d.A)[2]/2)

"Likelihood of model given then data represented by the covariance S from N samples"
function Distributions.loglikelihood(d::CFADistribution, S::AbstractMatrix, N::Int64)
    @assert minimum(d.Theta_X.nzval) > 0.0 "Invalid CFADistribution! (Theta_X has elements <= 0)"
    @assert minimum(eig(d.Theta_L)[1]) > 0.0 "Invalid CFADistribution! (Theta_L has eigen value <= 0)"

    # Woodbury transformation of orignal form allows us to only compute a KxK inversion
    Theta = d.Theta_X - d.Theta_X*d.A*inv(d.Theta_L + d.A'*d.Theta_X*d.A)*d.A'*d.Theta_X

    v = logdet(Theta) - trace(S*Theta)
    #println("v :", v, " ", minimum(d.Theta_X.nzval), " ", minimum(eig(Theta)[1]), " ", logdet(Theta))
    v
end

include("gradient.jl")
include("utils.jl")
#include("pgradient.jl")
#include("em.jl")

end # module
