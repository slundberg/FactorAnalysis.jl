
# vec2state! and state2vec!
srand(1)
P = 20
K = 10
Sigma_X = spdiagm(rand(P) .+ 0.2)
A = sparse(1:P, Int64[ceil(i/2) for i in 1:P], ones(P))
Sigma_L = FactorAnalysis.randcor(K, 0.2)
x = zeros(P + length(A.nzval) + div((K-1)*K,2))
state2vec!(x, Sigma_X, A, Sigma_L)
x_copy = copy(x)
vec2state!(Sigma_X, A, Sigma_L, x)
state2vec!(x, Sigma_X, A, Sigma_L)
@test maximum(abs(x .- x_copy)) <= 1e-8


# likelihood
N = 1000
X = FactorAnalysis.samples(Sigma_X, A, Sigma_L, N)
S = X*X' / N
Base.cov2cor!(S, sqrt(diag(S)))
@test loglikelihood(S, Sigma_X, A, Sigma_L) < 0.0
@test loglikelihood(S, Sigma_X, A, Sigma_L) > loglikelihood(S, Sigma_X, A, Sigma_L .+ eye(K)*10)
@test abs(loglikelihood(S, Sigma_X, A, Sigma_L) - FactorAnalysis.loglikelihood_slow(S, Sigma_X, A, Sigma_L)) < 1e-5

# gradient (compare to finite difference)
rho = 10.1
dSigma_X, dA, dSigma_L = gradient(S, Sigma_X, A, Sigma_L, rho)
dSigma_X2, dA2, dSigma_L2 = FactorAnalysis.gradient_slow(S, Sigma_X, A, Sigma_L, rho)
eps = 1e-8
tol = 1e-4


for i in 1:P
    tmp = Sigma_X[i,i]
    Sigma_X[i,i] = tmp + eps
    l1 = FactorAnalysis.loglikelihood_slow(S, Sigma_X, A, Sigma_L, rho)
    Sigma_X[i,i] = tmp - eps
    l2 = FactorAnalysis.loglikelihood_slow(S, Sigma_X, A, Sigma_L, rho)
    Sigma_X[i,i] = tmp
    @test abs(dSigma_X2[i,i] - (l1-l2)/(2*eps)) < tol
    #@test abs(dSigma_X[i,i] - (l1-l2)/(2*eps)) < tol
end
for i in 1:K, j in i+1:K,
    tmp = Sigma_L[i,j]
    Sigma_L[i,j] = Sigma_L[j,i] = tmp + eps
    l1 = FactorAnalysis.loglikelihood_slow(S, Sigma_X, A, Sigma_L, rho)
    Sigma_L[i,j] = Sigma_L[j,i] = tmp - eps
    l2 = FactorAnalysis.loglikelihood_slow(S, Sigma_X, A, Sigma_L, rho)
    Sigma_L[i,j] = Sigma_L[j,i] = tmp
    @test abs(dSigma_L2[i,j] - (l1-l2)/(2*eps)) < tol
    #@test abs(dSigma_L[i,j] - (l1-l2)/(2*eps)) < tol
end
rows = rowvals(A)
vals = nonzeros(A)
for col = 1:K
    for j in nzrange(A, col)
        tmp = vals[j]
        vals[j] = tmp + eps
        l1 = FactorAnalysis.loglikelihood_slow(S, Sigma_X, A, Sigma_L, rho)
        vals[j] = tmp - eps
        l2 = FactorAnalysis.loglikelihood_slow(S, Sigma_X, A, Sigma_L, rho)
        vals[j] = tmp
        @test abs(dA2[rows[j],col] - (l1-l2)/(2*eps)) < tol
        #@test abs(dA[rows[j],col] - (l1-l2)/(2*eps)) < tol
    end
end

# @time for i in 1:10000
#     gradient(S, Sigma_X, A, Sigma_L)
# end
# #Profile.print()
# @time for i in 1:10000
#     FactorAnalysis.gradient_slow(S, Sigma_X, A, Sigma_L)
# end
# println()
#
# #Profile.clear()
# @time for i in 1:10000
#     loglikelihood(S, Sigma_X, A, Sigma_L)
# end
# #Profile.print()
# @time for i in 1:10000
#     FactorAnalysis.loglikelihood_slow(S, Sigma_X, A, Sigma_L)
# end

# gradient_optimize
truthLL = loglikelihood(S, Sigma_X, A, Sigma_L)
Sigma_Xopt, Aopt, Sigma_Lopt = gradient_optimize(S, spones(A), show_trace=false, iterations=500, rho=0.01)
@test FactorAnalysis.loglikelihood_slow(S, Sigma_Xopt, Aopt, Sigma_Lopt) > truthLL
@test eigmin(Sigma_Xopt + Aopt*Sigma_Lopt*Aopt') > 0
println(eigmin(Sigma_Lopt))
