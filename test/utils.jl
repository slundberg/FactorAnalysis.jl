using Distributions

srand(1)

# make sure randcor works
@test sum(abs(inv(FactorAnalysis.randcor(40, 0.2))) .>= 1e-8) < 40*40/2


# samples
P = 10
K = 5
N = 100
Theta_X = spdiagm(1 ./ (rand(P) .+ 0.2))
A = sparse(1:P, Int64[ceil(i/2) for i in 1:P], ones(P))
Theta_L = inv(FactorAnalysis.randcor(K, 0.2))
d = CFADistribution(Theta_X, A, Theta_L)
@test size(rand(d, N)) == (10,N)
