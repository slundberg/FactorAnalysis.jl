

println("Testing CFADistributionEM: loglikelihood...")
srand(10)
P = 40
K = 20
Sigma_X = spdiagm(rand(P) .+ 0.2)
A = sparse(1:P, Int64[ceil(i/2) for i in 1:P], ones(P))
Sigma_L = FactorAnalysis.randcor(K, 0.2)
d = CFADistributionEM(Sigma_X, A, Sigma_L)
N = 100000
X = rand(d, N)
S = X*X' / N
Base.cov2cor!(S, sqrt(diag(S)))
@test loglikelihood(d, S, N) < 0.0
@test loglikelihood(d, S, N) > loglikelihood(CFADistributionEM(Sigma_X, A, Sigma_L .+ eye(K)*10), S, N)
function ll_test(d, S, N)
    Theta = inv(d.Sigma_X + d.A*d.Sigma_L*d.A')
    logdet(Theta) - trace(S*Theta)
end
@test abs(loglikelihood(d, S, N) - ll_test(d, S, N)) < 1e-8


println("Testing CFADistributionEM: rand...")
d = CFADistributionEM(Sigma_X, A, Sigma_L)
@test size(rand(d, N)) == (P,N)

println("Testing CFADistributionEM: fit_mle...")
truthLL = loglikelihood(d, S, N)
dopt = fit_mle(CFADistributionEM, ones(K,K), spones(A), S, N, show_trace=false, iterations=15000)
@test loglikelihood(dopt, S, N) > truthLL
@test FactorAnalysis.area_under_pr(inv(Sigma_L), inv(dopt.Sigma_L)) > 0.9

# v = FactorAnalysis.area_under_pr(inv(Sigma_L), inv(dopt.Sigma_L))
# println(truthLL)
# println(v)
# a = collect(zip(abs(FactorAnalysis.upper(inv(Sigma_L))) .> 0.01, abs(FactorAnalysis.upper(inv(dopt.Sigma_L))), 1:length(FactorAnalysis.upper(inv(Sigma_L)))))
# for v in sort(a, by=x->x[2])
#     println(v)
# end

dopt2 = fit_mle(CFADistributionSigma, spones(A), S, N, show_trace=false, iterations=1000)
#FactorAnalysis.area_under_pr(abs(FactorAnalysis.upper(inv(Sigma_L))) .> 0.01, abs(FactorAnalysis.upper(inv(dopt2.Sigma_L))))
@assert minimum(eig(dopt2.Sigma_L)[1]) > 0 "CFADistributionSigma found invalid Sigma_L so can't compare to CFADistributionEM"
@assert minimum(dopt2.Sigma_X.nzval) > 0 "CFADistributionSigma found invalid Sigma_X so can't compare to CFADistributionEM"
@test abs(loglikelihood(dopt, S, N) - loglikelihood(dopt2, S, N)) < 1e-6

println("Testing CFADistributionEM: fit_map...")
dopt3 = fit_map(CFADistributionEMRidge(1.1), CFADistributionEM, spones(A), S, N, show_trace=false, iterations=15000)
@test loglikelihood(dopt3, S, N) < loglikelihood(dopt, S, N)
@test loglikelihood(dopt3, S, N) > truthLL
@test FactorAnalysis.area_under_pr(inv(Sigma_L), inv(dopt3.Sigma_L)) > 0.6 # we over regularized here so we don't expect good performance

println("Testing CFADistributionEM: fit_mle with a fully connected latent variable...")
srand(10)
P = 40
K = 20
Sigma_X = spdiagm(rand(P) .+ 0.2)
A = sparse(1:P, Int64[ceil(i/2) for i in 1:P], ones(P))
Sigma_L = FactorAnalysis.randcor(K, 0.2)
d = CFADistributionEM(Sigma_X, A, Sigma_L)
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

dopt = fit_mle(CFADistributionEM, maskSigma_L, A2, S, N, show_trace=false, iterations=15000)
Th = inv(dopt.Sigma_X + dopt.A*dopt.Sigma_L*dopt.A')
# display(dopt.A)
# tmp = deepcopy(dopt.Sigma_L)
# dopt.Sigma_L = dopt.Sigma_L * (2*eye(K+1))
# #Base.cov2cor!(dopt.Sigma_L, 2*ones(K+1))
# FactorAnalysis.normalize_Sigma_L!(dopt::CFADistributionEM)
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
