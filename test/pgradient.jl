

# generate some test data
srand(1)
P = 40
K = 20
Sigma_X = spdiagm(rand(P) .+ 0.2)
A = sparse(1:P, Int64[ceil(i/2) for i in 1:P], ones(P))
Sigma_L = FactorAnalysis.randcor(K, 0.2)
N = 1000
X = FactorAnalysis.samples(Sigma_X, A, Sigma_L, N)
S = X*X' / N
Base.cov2cor!(S, sqrt(diag(S)))

# mle_Sigma_L!
Sigma_Xest = spdiagm(ones(P))
Aest = spones(A)
Sigma_Lest = eye(K)
ll = FactorAnalysis.loglikelihood(S, Sigma_Xest, Aest, Sigma_Lest)
FactorAnalysis.mle_Sigma_L!(Sigma_Lest::AbstractMatrix, S::AbstractMatrix, Sigma_Xest::SparseMatrixCSC, Aest::SparseMatrixCSC)
@test ll < FactorAnalysis.loglikelihood(S, Sigma_Xest, Aest, Sigma_Lest)

# # mle_Sigma_X!
# ll = FactorAnalysis.loglikelihood_noerror(S, Sigma_Xest, Aest, Sigma_Lest)
# FactorAnalysis.mle_Sigma_X!(Sigma_Xest::AbstractMatrix, S::AbstractMatrix, Aest::SparseMatrixCSC, Sigma_Lest::AbstractMatrix)
# show(Sigma_Lest)
# show(Sigma_Xest)
# show(Aest)
#
# @test ll < FactorAnalysis.loglikelihood_noerror(S, Sigma_Xest, Aest, Sigma_Lest)
#
# # mle_Sigma_XandL!
# ll = FactorAnalysis.loglikelihood_noerror(S, Sigma_Xest, Aest, Sigma_Lest)
# FactorAnalysis.mle_Sigma_XandL!(Sigma_Xest::AbstractMatrix, Sigma_Lest::AbstractMatrix, S::AbstractMatrix, Aest::SparseMatrixCSC)
# @test ll < FactorAnalysis.loglikelihood_noerror(S, Sigma_Xest, Aest, Sigma_Lest)
