
# println("Testing vec2state! and state2vec!...")
# srand(1)
# P = 20
# K = 10
# Sigma_X = spdiagm(rand(P) .+ 0.2)
# A = sparse(1:P, Int64[ceil(i/2) for i in 1:P], ones(P))
# Sigma_L = FactorAnalysis.randcor(K, 0.2)
# d = CFADistribution(Sigma_X, A, Sigma_L)
# N = 1000
# X = rand(d, N)
# S = X*X' / N
# Base.cov2cor!(S, sqrt(diag(S)))
# function ll_test(d, S, N)
#     Theta = inv(d.Sigma_X + d.A*d.Sigma_L*d.A')
#     logdet(Theta) - trace(S*Theta)
# end
# @test abs(loglikelihood(d, S, N) - ll_test(d, S, N)) < 1e-8

println("Testing CFADistributionSigma: vec2state! and state2vec!...")
srand(10)
P = 20
K = 10
Sigma_X = spdiagm(rand(P) .+ 0.2)
A = sparse(1:P, Int64[ceil(i/2) for i in 1:P], ones(P))
Sigma_L = FactorAnalysis.randcor(K, 0.2)

d = CFADistributionSigma(Sigma_X, A, Sigma_L)
x = zeros(P + length(A.nzval) + div((K+1)*K,2))
FactorAnalysis.state2vec!(x, d)
x_copy = copy(x)
FactorAnalysis.vec2state!(d, x)
FactorAnalysis.state2vec!(x, d)
@test maximum(abs(x .- x_copy)) <= 1e-8

g = FactorAnalysis.CFADistributionSigmaGradient(Sigma_X, A, Sigma_L)
x = zeros(P + length(A.nzval) + div((K+1)*K,2))
FactorAnalysis.state2vec!(x, g)
x_copy = copy(x)
FactorAnalysis.vec2state!(g, x)
FactorAnalysis.state2vec!(x, g)
@test maximum(abs(x .- x_copy)) <= 1e-8


println("Testing CFADistributionSigma: loglikelihood...")
N = 100000
X = rand(d, N)
S = X*X' / N
Base.cov2cor!(S, sqrt(diag(S)))
@test loglikelihood(d, S, N) < 0.0
@test loglikelihood(d, S, N) > loglikelihood(CFADistributionSigma(Sigma_X, A, Sigma_L .+ eye(K)*10), S, N)
function ll_test(d, S, N)
    Theta = inv(d.Sigma_X + d.A*d.Sigma_L*d.A')
    logdet(Theta) - trace(S*Theta)
end
@test abs(loglikelihood(d, S, N) - ll_test(d, S, N)) < 1e-8


println("Testing CFADistributionSigma: rand...")
d = CFADistributionSigma(Sigma_X, A, Sigma_L)
@test size(rand(d, N)) == (P,N)


println("Testing CFADistributionSigma: gradient (compare to finite difference)...")
g = dloglikelihood(d, S, N)
eps = 1e-8
tol = 1e-4
for i in 1:P
    tmp = Sigma_X[i,i]
    Sigma_X[i,i] = tmp + eps
    l1 = loglikelihood(CFADistributionSigma(Sigma_X, A, Sigma_L), S, N)
    Sigma_X[i,i] = tmp - eps
    l2 = loglikelihood(CFADistributionSigma(Sigma_X, A, Sigma_L), S, N)
    Sigma_X[i,i] = tmp
    @test abs(g.Sigma_X[i,i] - (l1-l2)/(2*eps)) < tol
    #@test abs(dSigma_X[i,i] - (l1-l2)/(2*eps)) < tol
end
for i in 1:K, j in i+1:K
    tmp = Sigma_L[j,i]
    Sigma_L[j,i] = Sigma_L[i,j] = tmp + eps
    l1 = loglikelihood(CFADistributionSigma(Sigma_X, A, Sigma_L), S, N)
    Sigma_L[j,i] = Sigma_L[i,j] = tmp - eps
    l2 = loglikelihood(CFADistributionSigma(Sigma_X, A, Sigma_L), S, N)
    Sigma_L[j,i] = Sigma_L[i,j] = tmp
    @test abs(g.Sigma_L[j,i] - (l1-l2)/(2*eps)) < tol
end
rows = rowvals(A)
vals = nonzeros(A)
for col = 1:K
    for j in nzrange(A, col)
        tmp = vals[j]
        vals[j] = tmp + eps
        l1 = loglikelihood(CFADistributionSigma(Sigma_X, A, Sigma_L), S, N)
        vals[j] = tmp - eps
        l2 = loglikelihood(CFADistributionSigma(Sigma_X, A, Sigma_L), S, N)
        vals[j] = tmp
        @test abs(g.A[rows[j],col] - (l1-l2)/(2*eps)) < tol
        #@test abs(dA[rows[j],col] - (l1-l2)/(2*eps)) < tol
    end
end

println("Testing CFADistributionSigma: fit_mle...")
truthLL = loglikelihood(d, S, N)
dopt = fit_mle(CFADistributionSigma, spones(A), S, N, show_trace=false, iterations=1000)
@test loglikelihood(dopt, S, N) > truthLL

# println("Testing CFADistributionSigma: fit_map...")
# dopt2 = fit_map(Normal(0, 1.0), CFADistributionSigma, spones(A), S, N, show_trace=true, iterations=1000)
# @test loglikelihood(dopt2, S, N) < loglikelihood(dopt, S, N)
# @test loglikelihood(dopt2, S, N) > truthLL

#display(Sigma_L)
