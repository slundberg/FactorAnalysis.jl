using MLBase

# vec2state! and state2vec!
srand(10)
P = 40
K = 20
Theta_X = spdiagm(1 ./ (rand(P) .+ 0.2))
A = sparse(1:P, Int64[ceil(i/2) for i in 1:P], ones(P))
Theta_L = inv(FactorAnalysis.randcor(K, 0.2))

d = CFADistribution(Theta_X, A, Theta_L)
x = zeros(P + length(A.nzval) + div((K+1)*K,2))
FactorAnalysis.state2vec!(x, d)
x_copy = copy(x)
FactorAnalysis.vec2state!(d, x)
FactorAnalysis.state2vec!(x, d)
@test maximum(abs(x .- x_copy)) <= 1e-8

g = FactorAnalysis.CFADistributionGradient(Theta_X, A, Theta_L)
x = zeros(P + length(A.nzval) + div((K+1)*K,2))
FactorAnalysis.state2vec!(x, g)
x_copy = copy(x)
FactorAnalysis.vec2state!(g, x)
FactorAnalysis.state2vec!(x, g)
@test maximum(abs(x .- x_copy)) <= 1e-8


# likelihood
N = 10000
X = rand(d, N)
S = X*X' / N
Base.cov2cor!(S, sqrt(diag(S)))
@test loglikelihood(d, S, N) < 0.0
@test loglikelihood(d, S, N) > loglikelihood(CFADistribution(Theta_X, A, Theta_L .+ eye(K)*10), S, N)
#@test abs(loglikelihood(d, S, Theta_X, A, Theta_L) - FactorAnalysis.loglikelihood_slow(S, Theta_X, A, Theta_L)) < 1e-5

# gradient (compare to finite difference)
g = dloglikelihood(d, S, N)
eps = 1e-8
tol = 1e-4


for i in 1:P
    tmp = Theta_X[i,i]
    Theta_X[i,i] = tmp + eps
    l1 = loglikelihood(CFADistribution(Theta_X, A, Theta_L), S, N)
    Theta_X[i,i] = tmp - eps
    l2 = loglikelihood(CFADistribution(Theta_X, A, Theta_L), S, N)
    Theta_X[i,i] = tmp
    @test abs(g.Theta_X[i,i] - (l1-l2)/(2*eps)) < tol
    #@test abs(dTheta_X[i,i] - (l1-l2)/(2*eps)) < tol
end
for i in 1:K, j in i+1:K
    tmp = Theta_L[i,j]
    Theta_L[j,i] = Theta_L[i,j] = tmp + eps
    l1 = loglikelihood(CFADistribution(Theta_X, A, Theta_L), S, N)
    Theta_L[j,i] = Theta_L[i,j] = tmp - eps
    l2 = loglikelihood(CFADistribution(Theta_X, A, Theta_L), S, N)
    Theta_L[j,i] = Theta_L[i,j] = tmp
    @test abs(g.Theta_L[j,i] - (l1-l2)/(2*eps)) < tol
end
rows = rowvals(A)
vals = nonzeros(A)
for col = 1:K
    for j in nzrange(A, col)
        tmp = vals[j]
        vals[j] = tmp + eps
        l1 = loglikelihood(CFADistribution(Theta_X, A, Theta_L), S, N)
        vals[j] = tmp - eps
        l2 = loglikelihood(CFADistribution(Theta_X, A, Theta_L), S, N)
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

# gradient_optimize
truthLL = loglikelihood(d, S, N)
dopt = fit_mle(CFADistribution, spones(A), S, N, show_trace=false, iterations=2000)
@test loglikelihood(dopt, S, N) > truthLL

function upper(X::AbstractMatrix)
    x = Float64[]
    for i in 1:size(X)[1], j in i+1:size(X)[2]
        push!(x, X[i,j])
    end
    x
end
function area_under_pr(truth::AbstractVector, predictor::AbstractVector; resolution=4000)
    rocData = MLBase.roc(round(Int64, truth), float(invperm(sortperm(predictor))), resolution)
    vals = collect(map(x->(recall(x), -precision(x)), rocData))
    sort!(vals)
    xvals = map(x->x[1], vals)
    yvals = map(x->-x[2], vals)
    area_under_curve(xvals, yvals)
end
function area_under_curve(x, y) # must be sorted by increasing x
    area = 0.0
    lastVal = NaN
    for i in 2:length(x)
        v = (y[i-1]+y[i])/2 * (x[i]-x[i-1])
        if !isnan(v)
            area += v
            lastVal = v
        elseif !isnan(lastVal)
            area += lastVal
        end
    end
    area
end

@test area_under_pr(abs(upper(Theta_L)) .> 0.01, abs(upper(dopt.Theta_L))) > 0.9

# a = collect(zip(abs(upper(Theta_L)) .> 0.01, abs(upper(dopt.Theta_L)), 1:length(upper(Theta_L))))
# for v in sort(a, by=x->x[2])
#     println(v)
# end

dopt2 = fit_map(Normal(0, 1.0), CFADistribution, spones(A), S, N, show_trace=false, iterations=2000)
@test loglikelihood(dopt2, S, N) < loglikelihood(dopt, S, N)
@test loglikelihood(dopt2, S, N) > truthLL

FactorAnalysis.normalize_Sigma_L!(dopt2)
@test area_under_pr(abs(upper(Theta_L)) .> 0.01, abs(upper(dopt2.Theta_L))) > 0.9

# display(Theta_L)
# println()
# display(dopt.Theta_L)
# println()
# display(dopt2.Theta_L)
#
# dopt3 = fit_map(Normal(0, 0.001), CFADistribution, spones(A), S, N, show_trace=true, iterations=20000)
# FactorAnalysis.normalize_Sigma_L!(dopt3)
# println()
# display(dopt3.Theta_L)
# println()
# println(area_under_pr(abs(upper(Theta_L)) .> 0.01, abs(upper(dopt3.Theta_L))))
# upper(upper)
