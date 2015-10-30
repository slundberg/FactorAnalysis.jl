

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
dopt = fit_mle(CFADistributionEM, spones(A), S, N, show_trace=false, iterations=15000)
@test loglikelihood(dopt, S, N) > truthLL

# v = FactorAnalysis.area_under_pr(abs(FactorAnalysis.upper(inv(Sigma_L))) .> 0.01, abs(FactorAnalysis.upper(inv(dopt.Sigma_L))))
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
dopt3 = fit_map(CFADistributionEMRidge(0.01), CFADistributionEM, spones(A), S, N, show_trace=false, iterations=15000)
@test loglikelihood(dopt3, S, N) < loglikelihood(dopt, S, N)
@test loglikelihood(dopt3, S, N) > truthLL

# v = FactorAnalysis.area_under_pr(abs(FactorAnalysis.upper(inv(Sigma_L))) .> 0.01, abs(FactorAnalysis.upper(inv(dopt3.Sigma_L))))
# println(truthLL)
# println(v)
# println()
# v = FactorAnalysis.area_under_pr(abs(FactorAnalysis.upper(inv(Sigma_L))) .> 0.01, abs(FactorAnalysis.upper(inv(dopt.Sigma_L + 0.01*eye(K)))))
# println(v)

#display(Sigma_L)
