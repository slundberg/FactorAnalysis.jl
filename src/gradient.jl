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
function loglikelihood(S::AbstractMatrix, Sigma_X::SparseMatrixCSC, A::SparseMatrixCSC, Sigma_L::AbstractMatrix, rho=0.0)

    # invert Sigma_X efficiently
    #if minimum(Sigma_X.nzval) <= 0.0 return -1e16 end
    Sigma_Xinv = deepcopy(Sigma_X)
    Sigma_Xinv.nzval[:] = 1 ./ Sigma_Xinv.nzval

    Gamma = A'*Sigma_Xinv*A
    tmp = eye(size(Gamma)[1]) + Sigma_L*Gamma
    R = inv(tmp)
    #if eigmin(tmp) <= 0.0 return -1e16 end
    logdetSigma = nothing
    try
        logdetSigma = sum(log(Sigma_X.nzval)) + logdet(tmp)
    catch
        return -1e16
    end
    B = R*Sigma_L
    C = Sigma_Xinv - Sigma_Xinv*A*B*A'*Sigma_Xinv
    -logdetSigma - trace(S*C) - rho*trace(Sigma_L*Sigma_L)
end
function loglikelihood_slow(S::AbstractMatrix, Sigma_X::SparseMatrixCSC, A::SparseMatrixCSC, Sigma_L::AbstractMatrix, rho=0.0)
    Sigma = Sigma_X + A*Sigma_L*A'
    Theta = inv(Sigma)
    if minimum(real(eigvals(Theta))) < 0.0
        return -1e16
    end
    logdet(Theta) - trace(S*Theta) - rho*trace(Sigma_L*Sigma_L)
end

function gradient(S::AbstractMatrix, Sigma_X::SparseMatrixCSC, A::SparseMatrixCSC, Sigma_L::AbstractMatrix, rho=0.0)

    # invert Sigma_X efficiently
    Sigma_Xtmp = deepcopy(Sigma_X)
    Sigma_Xtmp.nzval[:] = 1 ./ Sigma_Xtmp.nzval

    Gamma = A'*Sigma_Xtmp*A
    tmp = eye(size(Gamma)[1]) + Sigma_L*Gamma
    R = inv(tmp)
    B = R*Sigma_L
    C = Sigma_Xtmp - Sigma_Xtmp*A*B*A'*Sigma_Xtmp
    D = S*C
    E = C - C*D
    G = E*A
    dA_dense = -2*G*Sigma_L
    dSigma_L = -2*A'*G - 4*rho*Sigma_L
    dSigma_X = Sigma_Xtmp
    dSigma_X.nzval[:] = -diag(E)

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
function gradient_slow(S::AbstractMatrix, Sigma_X::SparseMatrixCSC, A::SparseMatrixCSC, Sigma_L::AbstractMatrix, rho=0.0)
    Sigma = Sigma_X + A*Sigma_L*A'
    Theta = inv(Sigma)

    tmp = Theta*(S - Sigma)*Theta
    dA_dense = 2*tmp*A*Sigma_L
    dSigma_L = 2*A'*tmp*A - 4*rho*Sigma_L
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
        -loglikelihood_slow(S, Sigma_X, A, Sigma_L, rho)
    end

    function df!(x, g)
        vec2state!(Sigma_X, A, Sigma_L, x)
        dSigma_X, dA, dSigma_L = gradient_slow(S, Sigma_X, A, Sigma_L, rho)
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
