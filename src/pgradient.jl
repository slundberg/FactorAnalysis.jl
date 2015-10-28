function mle_Sigma_L!(Sigma_L::AbstractMatrix, S::AbstractMatrix, Sigma_X::SparseMatrixCSC, A::SparseMatrixCSC)

    # invert Sigma_X efficiently
    Theta_X = deepcopy(Sigma_X)
    Theta_X.nzval[:] = 1 ./ Theta_X.nzval

    B = A'*Theta_X*S*Theta_X*A
    C = A'Theta_X*A
    Cinv = deepcopy(C)
    Cinv.nzval[:] = 1 ./ Cinv.nzval # assumes that C is diagonal!!
    Sigma_L[:,:] = Cinv*(B - C)*Cinv
end

# not correct! you can't just take the diagonal to get the optimal Sigma_X
# function mle_Sigma_X!(Sigma_X::SparseMatrixCSC, S::AbstractMatrix, A::SparseMatrixCSC, Sigma_L::AbstractMatrix)
#     Sigma_X.nzval[:] = diag(S - A*Sigma_L*A')
# end
# function mle_Sigma_XandL!(Sigma_X::SparseMatrixCSC, Sigma_L::AbstractMatrix, S::AbstractMatrix, A::SparseMatrixCSC; iterations=1000)
#     lastll = -Inf
#     for i in 1:iterations
#         #println(ll_score(S, Sigma_Xest, Sigma_Lest, Aest))
#         Sigma_Lest[:,:] = mle_Sigma_L(S, Sigma_Xest, Aest)
#         #println("md ", minimum(diag(Sigma_Lest)))
#         Sigma_Xest[:,:] = mle_Sigma_X(S, Sigma_Lest, Aest)
#         #println("md ", minimum(diag(Sigma_Xest)))
#         ll = loglikelihood(S, Sigma_X, A, Sigma_L)
#         @assert ll > lastll
#         if ll - lastll < 1e-8 break end
#         println(ll)
#     end
# end
