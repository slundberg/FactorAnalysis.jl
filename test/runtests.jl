using FactorAnalysis
using Base.Test

# loglikelihood
srand(1)
P = 20
K = 10
Theta_X = spdiagm(1 ./ (rand(P) .+ 0.2))
A = sparse(1:P, Int64[ceil(i/2) for i in 1:P], ones(P))
Theta_L = inv(FactorAnalysis.randcor(K, 0.2))
d = CFADistribution(Theta_X, A, Theta_L)
N = 1000
X = rand(d, N)
S = X*X' / N
Base.cov2cor!(S, sqrt(diag(S)))
function ll_test(d, S, N)
    Theta = inv(inv(full(d.Theta_X)) + d.A*inv(d.Theta_L)*d.A')
    logdet(Theta) - trace(S*Theta)
end
@test abs(loglikelihood(d, S, N) - ll_test(d, S, N)) < 1e-8

include("utils.jl")
include("gradient.jl")
#include("pgradient.jl")
