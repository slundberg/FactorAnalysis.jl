using Distributions

# make sure randcor works
@test sum(abs(inv(FactorAnalysis.randcor(40, 0.2))) .>= 1e-8) < 40*40/2


# samples
P = 10
K = 5
N = 100
Sigma_X = spdiagm(rand(P) .+ 0.2)
A = sparse(1:P, Int64[ceil(i/2) for i in 1:P], ones(P))
Sigma_L = FactorAnalysis.randcor(K, 0.2)
@test size(FactorAnalysis.samples(Sigma_X, A, Sigma_L, N)) == (10,N)
