using Optim

export CFADistributionSigma

"Confirmatory factor analysis distribution object with a Sigma formulation."
type CFADistributionSigma
    Sigma_X::SparseMatrixCSC
    A::SparseMatrixCSC
    Sigma_L::AbstractMatrix
end
Base.length(d::CFADistributionSigma) = Int(size(d.A)[1] + nnz(d.A) + (size(d.A)[2]-1)*size(d.A)[2]/2)

function Base.rand(d::CFADistributionSigma, N::Int)
    P,K = size(d.A)
    distL = MvNormal(zeros(K), d.Sigma_L)
    L = rand(distL, N)
    X = d.A*L
    for i in 1:P
        X[i,:] .+= transpose(rand(Normal(0, d.Sigma_X.nzval[i]), N))
    end
    X
end
Base.rand(d::CFADistributionSigma) = rand(d::CFADistributionSigma, 1)

"Likelihood of model given then data represented by the covariance S from N samples"
function Distributions.loglikelihood(d::CFADistributionSigma, S::AbstractMatrix, N::Int64)
    #@assert minimum(d.Sigma_X.nzval) > 0.0 "Invalid CFADistributionSigma! (Theta_X has elements <= 0)"
    #@assert minimum(eig(d.Sigma_L)[1]) > 0.0 "Invalid CFADistributionSigma! (Theta_L has eigen value <= 0)"

    Theta_X = deepcopy(d.Sigma_X)
    Theta_X.nzval[:] = 1 ./ d.Sigma_X.nzval
    Theta_L = inv(d.Sigma_L)

    # Woodbury transformation of original form allows us to only compute a KxK inversion
    Theta = Theta_X - Theta_X*d.A*inv(Theta_L + d.A'*Theta_X*d.A)*d.A'*Theta_X

    @assert minimum(eig(Theta)[1]) > 0.0 "Invalid CFADistributionSigma! (Theta has eigen value <= 0)"

    v = logdet(Theta) - trace(S*Theta)
    #println("v :", v, " ", minimum(Theta_X.nzval), " ", minimum(eig(Theta)[1]), " ", logdet(Theta))
    v
end

"Gradient of the CFA model."
type CFADistributionSigmaGradient
    Sigma_X::SparseMatrixCSC
    A::SparseMatrixCSC
    Sigma_L::AbstractMatrix
end


"""
Pulls values from a flattened vector into the full matrix state.
"""
function vec2state!(d::Union{CFADistributionSigma,CFADistributionSigmaGradient}, x::AbstractVector)
    P = length(d.Sigma_X.nzval)
    d.Sigma_X.nzval[:] = x[1:P]

    offset = length(d.A.nzval)
    d.A.nzval[:] = x[P+1:P+offset]

    pos = P+offset+1
    K = size(d.Sigma_L)[1]
    for i in 1:K, j in i+1:K
        d.Sigma_L[i,j] = d.Sigma_L[j,i] = x[pos]
        pos += 1
    end
end
# function vec2state!(Sigma_X::SparseMatrixCSC, A::SparseMatrixCSC, Sigma_L::AbstractMatrix, x::AbstractVector)
#     P = length(Sigma_X.nzval)
#     Sigma_X.nzval[:] = x[1:P]
#
#     offset = length(A.nzval)
#     A.nzval[:] = x[P+1:P+offset]
#
#     pos = P+offset+1
#     K = size(Sigma_L)[1]
#     for i in 1:K, j in i+1:K
#         Sigma_L[i,j] = Sigma_L[j,i] = x[pos]
#         pos += 1
#     end
# end

"""
Pulls values from the full matrix state into a flattened vector.
"""
function state2vec!(x::AbstractVector, d::Union{CFADistributionSigma,CFADistributionSigmaGradient})
    P = length(d.Sigma_X.nzval)
    x[1:P] = d.Sigma_X.nzval

    offest = length(d.A.nzval)
    x[P+1:P+offest] = d.A.nzval

    pos = P+offest+1
    K = size(d.Sigma_L)[1]
    for i in 1:K, j in i+1:K
        x[pos] = d.Sigma_L[i,j]
        pos += 1
    end
end
# function state2vec!(x::AbstractVector, Sigma_X::SparseMatrixCSC, A::SparseMatrixCSC, Sigma_L::AbstractMatrix)
#     P = length(Sigma_X.nzval)
#     x[1:P] = Sigma_X.nzval
#
#     offest = length(A.nzval)
#     x[P+1:P+offest] = A.nzval
#
#     pos = P+offest+1
#     K = size(Sigma_L)[1]
#     for i in 1:K, j in i+1:K
#         x[pos] = Sigma_L[i,j]
#         pos += 1
#     end
# end



# function loglikelihood_fast(S::AbstractMatrix, Sigma_X::SparseMatrixCSC, A::SparseMatrixCSC, Sigma_L::AbstractMatrix, rho=0.0)
#
#     # invert Sigma_X efficiently
#     #if minimum(Sigma_X.nzval) <= 0.0 return -1e16 end
#     Sigma_Xinv = deepcopy(Sigma_X)
#     Sigma_Xinv.nzval[:] = 1 ./ Sigma_Xinv.nzval
#
#     Gamma = A'*Sigma_Xinv*A
#     tmp = eye(size(Gamma)[1]) + Sigma_L*Gamma
#     R = inv(tmp)
#     #if eigmin(tmp) <= 0.0 return -1e16 end
#     logdetSigma = nothing
#     try
#         logdetSigma = sum(log(Sigma_X.nzval)) + logdet(tmp)
#     catch
#         return -1e16
#     end
#     B = R*Sigma_L
#     C = Sigma_Xinv - Sigma_Xinv*A*B*A'*Sigma_Xinv
#     -logdetSigma - trace(S*C) - rho*trace(Sigma_L*Sigma_L)
# end


# function gradient_fast(S::AbstractMatrix, Sigma_X::SparseMatrixCSC, A::SparseMatrixCSC, Sigma_L::AbstractMatrix, rho=0.0)
#
#     # invert Sigma_X efficiently
#     Sigma_Xtmp = deepcopy(Sigma_X)
#     Sigma_Xtmp.nzval[:] = 1 ./ Sigma_Xtmp.nzval
#
#     Gamma = A'*Sigma_Xtmp*A
#     tmp = eye(size(Gamma)[1]) + Sigma_L*Gamma
#     R = inv(tmp)
#     B = R*Sigma_L
#     C = Sigma_Xtmp - Sigma_Xtmp*A*B*A'*Sigma_Xtmp
#     D = S*C
#     E = C - C*D
#     G = E*A
#     dA_dense = -2*G*Sigma_L
#     dSigma_L = -2*A'*G - 4*rho*Sigma_L
#     dSigma_X = Sigma_Xtmp
#     dSigma_X.nzval[:] = -diag(E)
#
#     dA = deepcopy(A)
#     rows = rowvals(dA)
#     vals = nonzeros(dA)
#     K = size(dA)[2]
#     for col = 1:K
#         for j in nzrange(dA, col)
#             vals[j] = dA_dense[rows[j],col]
#         end
#     end
#
#     dSigma_X, dA, dSigma_L
# end
function dloglikelihood(d::CFADistributionSigma, S::AbstractMatrix, N::Int64)
    P,K = size(d.A)

    Theta_X = deepcopy(d.Sigma_X)
    Theta_X.nzval[:] = 1 ./ Theta_X.nzval
    Theta_L = inv(d.Sigma_L)

    # Woodbury transformation of orignal form allows us to only compute a KxK inversion
    Theta = Theta_X - Theta_X*d.A*inv(Theta_L + d.A'*Theta_X*d.A)*d.A'*Theta_X

    commonPart = Theta*S*Theta - Theta

    dSigma_X = spdiagm(diag(commonPart))

    dSigma_L = 2*d.A'*commonPart*d.A # diagonals should not be times 2 but we ignore them
    #dTheta_Linv = 2*d.A'*commonPart*d.A
    #for i in 1:K dTheta_Linv[i,i] = 0.0 end # we shouldn't change the diagonal of the covariance matrix
    #dTheta_L = Theta_Linv*dTheta_Linv*Theta_Linv
    #for i in 1:size(dTheta_L)[1] dTheta_L[i,i] /= 2 end # diagonals should not be times 2

    dA_dense = 2*commonPart*d.A*d.Sigma_L
    dA = deepcopy(d.A)
    rows = rowvals(dA)
    vals = nonzeros(dA)
    K = size(dA)[2]
    for col = 1:K
        for j in nzrange(dA, col)
            vals[j] = dA_dense[rows[j],col]
        end
    end

    CFADistributionSigmaGradient(dSigma_X, dA, dSigma_L)
end

function Distributions.fit_mle(::Type{CFADistributionSigma}, A::SparseMatrixCSC, S::AbstractMatrix, N::Int64; iterations=1000, show_trace=true)
    P,K = size(A)

    # init our parameters
    d = CFADistributionSigma(speye(P), deepcopy(A), eye(K))
    x_init = zeros(Int(P + nnz(A) + (K-1)*K/2))
    state2vec!(x_init, d)

    function f(x)
        vec2state!(d, x)
        try
            return -loglikelihood(d, S, N)
        catch
            1e16 # invalid set of parameters
        end
    end

    function df!(x, g)
        vec2state!(d, x)
        gState = dloglikelihood(d, S, N)
        state2vec!(g, gState)
        g[:] = -g
    end

    res = optimize(f, df!,
        x_init,
        method = :l_bfgs,
        show_trace = show_trace,
        iterations = iterations
    )

    vec2state!(d, res.minimum)
    d
end

function fit_map(pri::Normal, ::Type{CFADistributionSigma}, A::SparseMatrixCSC, S::AbstractMatrix, N::Int64; iterations=1000, show_trace=true)
    warning("fit_map for CFADistributionSigma is not working well yet!")
    @assert mean(pri) == 0.0 "Prior must be zero mean!"
    rho = 1/(std(pri)*N)
    P,K = size(A)

    # init our parameters
    d = CFADistributionSigma(speye(P), deepcopy(A), eye(K))
    x_init = zeros(Int(P + nnz(A) + (K-1)*K/2))
    state2vec!(x_init, d)

    function f(x)
        vec2state!(d, x)

        # get likelihood
        ll = 0.0
        try
            ll = loglikelihood(d, S, N)
        catch
            return 1e16 # invalid set of parameters
        end

        # account for prior
        Theta_X = deepcopy(d.Sigma_X)
        Theta_X.nzval[:] = 1 ./ Theta_X.nzval
        Theta_L = inv(d.Sigma_L)
        #for i in 1:K Theta_L[i,i] = 0.0 end # don't penalize the diagonal

        ll -= rho*sum((1 ./ Theta_X.nzval).^2)
        ll -= rho*sum(d.A.^2)
        ll -= rho*sum(Theta_L.^2)

        -ll
    end

    function df!(x, g)
        vec2state!(d, x)

        # get gradient of likelihood
        gState = dloglikelihood(d, S, N)

        # account for prior
        Theta_X = deepcopy(d.Sigma_X)
        Theta_X.nzval[:] = 1 ./ Theta_X.nzval
        Theta_L = inv(d.Sigma_L)
        gState.Sigma_X .-= 2*rho*Theta_X.^3
        gState.A .-= 2*rho*d.A
        gState.Sigma_L .-= 2*rho*Theta_L^3

        state2vec!(g, gState)
        g[:] = -g
    end

    res = optimize(f, df!,
        x_init,
        method = :l_bfgs,
        show_trace = show_trace,
        iterations = iterations
    )

    vec2state!(d, res.minimum)
    d
end
