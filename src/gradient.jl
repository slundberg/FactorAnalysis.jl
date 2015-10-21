using Optim
using Distributions

import Base.gradient
import Distributions.loglikelihood

export vec2state!, state2vec!, loglikelihood, gradient, gradient_optimize

"""
Pulls values from a flattened vector into the full matrix state.
"""
function vec2state!(Sigma_X::SparseMatrixCSC, A::SparseMatrixCSC, Sigma_L::AbstractMatrix, x::AbstractVector)
    P = length(Sigma_X.nzval)
    Sigma_X.nzval[:] = x[1:P]

    offset = length(A.nzval)
    A.nzval[:] = x[P+1:P+offset]

    pos = P+offset+1
    K = size(Sigma_L)[1]
    for i in 1:K, j in i+1:K
        Sigma_L[i,j] = Sigma_L[j,i] = x[pos]
        pos += 1
    end
end

"""
Pulls values from the full matrix state into a flattened vector.
"""
function state2vec!(x::AbstractVector, Sigma_X::SparseMatrixCSC, A::SparseMatrixCSC, Sigma_L::AbstractMatrix)
    P = length(Sigma_X.nzval)
    x[1:P] = Sigma_X.nzval

    offest = length(A.nzval)
    x[P+1:P+offest] = A.nzval

    pos = P+offest+1
    K = size(Sigma_L)[1]
    for i in 1:K, j in i+1:K
        x[pos] = Sigma_L[i,j]
        pos += 1
    end
end


"""
Evaluates the loglikelihood of the confirmatory model.

S - The sample covariance matrix.

"""
function loglikelihood(S::AbstractMatrix, Sigma_X::SparseMatrixCSC, A::SparseMatrixCSC, Sigma_L::AbstractMatrix)
    Sigma = Sigma_X + A*Sigma_L*A'
    Theta = inv(Sigma)
    if minimum(real(eigvals(Theta))) < 0.0
        return -1e16
    end
    logdet(Theta) - trace(S*Theta)
end
function loglikelihood(S::AbstractMatrix, Sigma_X::DenseMatrix, A::DenseMatrix, Sigma_L::AbstractMatrix)
    Sigma = Sigma_X + A*Sigma_L*A'
    Theta = inv(Sigma)
    if minimum(eigvals(Theta)) < 0.0
        return -1e16
    end
    # rather than throwing exceptions we treat invalid parameters as very unlikely
    try
        return logdet(Theta) - trace(S*Theta)
    catch e
        return -1e16
    end
end

function gradient(S::AbstractMatrix, Sigma_X::SparseMatrixCSC, A::SparseMatrixCSC, Sigma_L::AbstractMatrix)
    Sigma = Sigma_X + A*Sigma_L*A'
    Theta = inv(Sigma)

    tmp = Theta*(S - Sigma)*Theta
    dA_dense = 2*tmp*A*Sigma_L
    dSigma_L = 2*A'*tmp*A
    dSigma_X = spdiagm(diag(tmp))

    dA = deepcopy(A)
    rows = rowvals(dA)
    vals = nonzeros(dA)
    K = size(dA)[2]
    for col = 1:K
        for j in nzrange(dA, col)
            vals[j] = dA_dense[rows[j],col]
        end
    end

    dSigma_X, dA, dSigma_L
end

function gradient(S::AbstractMatrix, Sigma_X::DenseMatrix, A::DenseMatrix, Sigma_L::AbstractMatrix)
    Sigma = Sigma_X + A*Sigma_L*A'
    Theta = inv(Sigma)

    tmp = Theta*(S - Sigma)*Theta
    dA = 2*tmp*A*Sigma_L
    dSigma_L = A'*tmp*A
    dSigma_X = diagm(diag(tmp))

    dSigma_X, dA, dSigma_L
end

function gradient_optimize(S, Ain; iterations=1000, rho=0.0, show_trace=true)
    P,K = size(Ain)

    # init our parameters
    Sigma_X = speye(P)
    A = deepcopy(Ain)
    Sigma_L = eye(K)
    x_init = zeros(Int(P + nnz(A) + (K-1)*K/2))
    state2vec!(x_init, Sigma_X, A, Sigma_L)

    function f(x)
        vec2state!(Sigma_X, A, Sigma_L, x)
        -loglikelihood(S, Sigma_X, A, Sigma_L)
    end

    function df!(x, g)
        vec2state!(Sigma_X, A, Sigma_L, x)
        dSigma_X, dA, dSigma_L = gradient(S, Sigma_X, A, Sigma_L)
        state2vec!(g, dSigma_X, dA, dSigma_L)
        g[:] = -g
    end

    res = optimize(f, df!,
        x_init,
        method = :l_bfgs,
        show_trace = show_trace,
        iterations = iterations
    )

    vec2state!(Sigma_X, A, Sigma_L, res.minimum)
    Sigma_X,A,Sigma_L
end
