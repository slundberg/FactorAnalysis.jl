using MLBase

println("Testing CFADistributionTheta: vec2state! and state2vec!...")
srand(10)
P = 40
K = 20
Theta_X = spdiagm(1 ./ (rand(P) .+ 0.2))
A = sparse(1:P, Int64[ceil(i/2) for i in 1:P], ones(P))
Theta_L = inv(FactorAnalysis.randcor(K, 0.2))

d = CFADistributionTheta(Theta_X, A, Theta_L)
x = zeros(P + length(A.nzval) + div((K+1)*K,2))
FactorAnalysis.state2vec!(x, d)
x_copy = copy(x)
FactorAnalysis.vec2state!(d, x)
FactorAnalysis.state2vec!(x, d)
@test maximum(abs(x .- x_copy)) <= 1e-8

g = FactorAnalysis.CFADistributionThetaGradient(Theta_X, A, Theta_L)
x = zeros(P + length(A.nzval) + div((K+1)*K,2))
FactorAnalysis.state2vec!(x, g)
x_copy = copy(x)
FactorAnalysis.vec2state!(g, x)
FactorAnalysis.state2vec!(x, g)
@test maximum(abs(x .- x_copy)) <= 1e-8

println("Testing CFADistributionTheta: loglikelihood...")
N = 10000
X = rand(d, N)
S = X*X' / N
Base.cov2cor!(S, sqrt(diag(S)))
@test loglikelihood(d, S, N) < 0.0
@test loglikelihood(d, S, N) > loglikelihood(CFADistributionTheta(Theta_X, A, Theta_L .+ eye(K)*10), S, N)
function ll_test(d, S, N)
    Theta = inv(inv(full(d.Theta_X)) + d.A*inv(d.Theta_L)*d.A')
    logdet(Theta) - trace(S*Theta)
end
@test abs(loglikelihood(d, S, N) - ll_test(d, S, N)) < 1e-8

println("Testing CFADistributionTheta: rand...")
d = CFADistributionTheta(Theta_X, A, Theta_L)
@test size(rand(d, N)) == (40,N)

println("Testing CFADistributionTheta: gradient (compare to finite difference)...")
g = dloglikelihood(d, S, N)
eps = 1e-8
tol = 1e-4
for i in 1:P
    tmp = Theta_X[i,i]
    Theta_X[i,i] = tmp + eps
    l1 = loglikelihood(CFADistributionTheta(Theta_X, A, Theta_L), S, N)
    Theta_X[i,i] = tmp - eps
    l2 = loglikelihood(CFADistributionTheta(Theta_X, A, Theta_L), S, N)
    Theta_X[i,i] = tmp
    @test abs(g.Theta_X[i,i] - (l1-l2)/(2*eps)) < tol
    #@test abs(dTheta_X[i,i] - (l1-l2)/(2*eps)) < tol
end
for i in 1:K, j in i+1:K
    tmp = Theta_L[i,j]
    Theta_L[j,i] = Theta_L[i,j] = tmp + eps
    l1 = loglikelihood(CFADistributionTheta(Theta_X, A, Theta_L), S, N)
    Theta_L[j,i] = Theta_L[i,j] = tmp - eps
    l2 = loglikelihood(CFADistributionTheta(Theta_X, A, Theta_L), S, N)
    Theta_L[j,i] = Theta_L[i,j] = tmp
    @test abs(g.Theta_L[j,i] - (l1-l2)/(2*eps)) < tol
end
rows = rowvals(A)
vals = nonzeros(A)
for col = 1:K
    for j in nzrange(A, col)
        tmp = vals[j]
        vals[j] = tmp + eps
        l1 = loglikelihood(CFADistributionTheta(Theta_X, A, Theta_L), S, N)
        vals[j] = tmp - eps
        l2 = loglikelihood(CFADistributionTheta(Theta_X, A, Theta_L), S, N)
        vals[j] = tmp
        @test abs(g.A[rows[j],col] - (l1-l2)/(2*eps)) < tol
        #@test abs(dA[rows[j],col] - (l1-l2)/(2*eps)) < tol
    end
end

# @time for i in 1:10000
#     gradient(S, Theta_X, A, Theta_L)
# end
# #Profile.print()
# @time for i in 1:10000
#     FactorAnalysis.gradient_slow(S, Theta_X, A, Theta_L)
# end
# println()
#
# #Profile.clear()
# @time for i in 1:10000
#     loglikelihood(S, Theta_X, A, Theta_L)
# end
# #Profile.print()
# @time for i in 1:10000
#     FactorAnalysis.loglikelihood_slow(S, Theta_X, A, Theta_L)
# end

println("Testing CFADistributionTheta: fit_mle...")
truthLL = loglikelihood(d, S, N)
dopt = fit_mle(CFADistributionTheta, spones(A), S, N, show_trace=false, iterations=2000)
@test loglikelihood(dopt, S, N) > truthLL

@test FactorAnalysis.area_under_pr(abs(FactorAnalysis.upper(Theta_L)) .> 0.01, abs(FactorAnalysis.upper(dopt.Theta_L))) > 0.9

# a = collect(zip(abs(upper(Theta_L)) .> 0.01, abs(upper(dopt.Theta_L)), 1:length(upper(Theta_L))))
# for v in sort(a, by=x->x[2])
#     println(v)
# end

println("Testing CFADistributionTheta: fit_map...")
dopt2 = fit_map(Normal(0, 1.0), CFADistributionTheta, spones(A), S, N, show_trace=false, iterations=2000)
@test loglikelihood(dopt2, S, N) < loglikelihood(dopt, S, N)
@test loglikelihood(dopt2, S, N) > truthLL

FactorAnalysis.normalize_Sigma_L!(dopt2)
@test FactorAnalysis.area_under_pr(abs(FactorAnalysis.upper(Theta_L)) .> 0.01, abs(FactorAnalysis.upper(dopt2.Theta_L))) > 0.9

# display(Theta_L)
# println()
# display(dopt.Theta_L)
# println()
# display(dopt2.Theta_L)
#
# dopt3 = fit_map(Normal(0, 0.001), CFADistributionTheta, spones(A), S, N, show_trace=true, iterations=20000)
# FactorAnalysis.normalize_Sigma_L!(dopt3)
# println()
# display(dopt3.Theta_L)
# println()
# println(area_under_pr(abs(upper(Theta_L)) .> 0.01, abs(upper(dopt3.Theta_L))))
# upper(upper)
