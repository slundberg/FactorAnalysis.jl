
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
dopt = fit_mle(CFADistributionSigma, ones(K,K), spones(A), S, N, show_trace=false, iterations=1000)
@test loglikelihood(dopt, S, N) > truthLL

# println("Testing CFADistributionSigma: fit_map...")
# dopt2 = fit_map(Normal(0, 1.0), CFADistributionSigma, spones(A), S, N, show_trace=true, iterations=1000)
# @test loglikelihood(dopt2, S, N) < loglikelihood(dopt, S, N)
# @test loglikelihood(dopt2, S, N) > truthLL

println("Testing CFADistributionSigma: fit_mle with a fully connected latent variable...")
srand(10)
P = 40
K = 20
Sigma_X = spdiagm(rand(P) .+ 0.2)
A = sparse(1:P, Int64[ceil(i/2) for i in 1:P], ones(P))
Sigma_L = FactorAnalysis.randcor(K, 0.2)
d = CFADistributionSigma(Sigma_X, A, Sigma_L)
N = 100000
X = rand(d, N)

# add a confounding latent factor
confounder = randn(P)
for i in 1:N
    X[:,i] .+= randn()*confounder
end

S = X*X' / N
Base.cov2cor!(S, sqrt(diag(S)))
truthLL = loglikelihood(d, S, N)

# create an independent component to handle the confounder
maskSigma_L = ones(K+1,K+1)
for i in 1:K
    maskSigma_L[K+1,i] = maskSigma_L[i,K+1] = 0
end
A2 = sparse([1:P; 1:P], [Int64[ceil(i/2) for i in 1:P]; ones(Int64, P)*(K+1)], ones(2*P))

dopt = fit_mle(CFADistributionSigma, maskSigma_L, A2, S, N, show_trace=false, iterations=15000)
Th = inv(dopt.Sigma_X + dopt.A*dopt.Sigma_L*dopt.A')
# display(dopt.A)
# tmp = deepcopy(dopt.Sigma_L)
# dopt.Sigma_L = dopt.Sigma_L * (2*eye(K+1))
# #Base.cov2cor!(dopt.Sigma_L, 2*ones(K+1))
# FactorAnalysis.normalize_Sigma_L!(dopt::CFADistributionSigma)
# display(dopt.A)
# display(tmp)
# println()
# display(dopt.Sigma_L)
# @test maximum(abs(tmp .- dopt.Sigma_L)) < 1e-8
#
# display(dopt.Sigma_X + dopt.A*dopt.Sigma_L*dopt.A')
# println()
# println("GGG2")
# display(S)
# println()
#
# println(logdet(Th) - trace(S*Th))
# println(logdet(inv(S)) - trace(S*inv(S)))
# # display(Sigma_X)
# display(dopt.Sigma_X)
# display(Sigma_L)
# println()
# display(dopt.Sigma_L)
# display(dopt.A)
@test loglikelihood(dopt, S, N) > truthLL
@test FactorAnalysis.area_under_pr(inv(Sigma_L), inv(dopt.Sigma_L[1:K,1:K])) > 0.9


# println(truthLL)
#println(v)
# println()
# v = FactorAnalysis.area_under_pr(abs(FactorAnalysis.upper(inv(Sigma_L))) .> 0.01, abs(FactorAnalysis.upper(inv(dopt.Sigma_L + 0.01*eye(K)))))
# println(v)

#display(Sigma_L)
