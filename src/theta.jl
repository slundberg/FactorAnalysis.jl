using Optim

export CFADistributionTheta

"Confirmatory factor analysis distribution object."
type CFADistributionTheta
    Theta_X::SparseMatrixCSC
    A::SparseMatrixCSC
    Theta_L::AbstractMatrix
end
Base.length(d::CFADistributionTheta) = Int(size(d.A)[1] + nnz(d.A) + (size(d.A)[2]-1)*size(d.A)[2]/2)

"Likelihood of model given the data represented by the covariance S from N samples"
function Distributions.loglikelihood(d::CFADistributionTheta, S::AbstractMatrix, N::Int64)
    @assert minimum(d.Theta_X.nzval) > 0.0 "Invalid CFADistributionTheta! (Theta_X has elements <= 0)"
    @assert minimum(eig(d.Theta_L)[1]) > 0.0 "Invalid CFADistributionTheta! (Theta_L has eigen value <= 0)"

    # Woodbury transformation of original form allows us to only compute a KxK inversion
    Theta = d.Theta_X - d.Theta_X*d.A*inv(d.Theta_L + d.A'*d.Theta_X*d.A)*d.A'*d.Theta_X

    v = logdet(Theta) - trace(S*Theta)
    #println("v :", v, " ", minimum(d.Theta_X.nzval), " ", minimum(eig(Theta)[1]), " ", logdet(Theta))
    v
end

function Base.rand(d::CFADistributionTheta, N::Int)
    P,K = size(d.A)
    distL = MvNormal(zeros(K), inv(d.Theta_L))
    L = rand(distL, N)
    X = d.A*L
    for i in 1:P
        X[i,:] .+= transpose(rand(Normal(0, 1 / d.Theta_X.nzval[i]), N))
    end
    X
end
Base.rand(d::CFADistributionTheta) = rand(d::CFADistributionTheta, 1)

function normalize_Sigma_L!(d::CFADistributionTheta)
    Sigma_L = inv(d.Theta_L)
    scaling = sqrt(diag(Sigma_L))
    d.Theta_L[:,:] = inv(Base.cov2cor!(Sigma_L, scaling))
    vals = nonzeros(d.A)
    K = size(d.A)[2]
    for col = 1:K
        for j in nzrange(d.A, col)
            vals[j] *= scaling[col]
        end
    end
end


"Gradient of the CFA model."
type CFADistributionThetaGradient
    Theta_X::SparseMatrixCSC
    A::SparseMatrixCSC
    Theta_L::AbstractMatrix
end


"""
Pulls values from a flattened vector into the full matrix state.
"""
function vec2state!(d::Union{CFADistributionTheta,CFADistributionThetaGradient}, x::AbstractVector)
    P = length(d.Theta_X.nzval)
    d.Theta_X.nzval[:] = x[1:P]

    offset = length(d.A.nzval)
    d.A.nzval[:] = x[P+1:P+offset]

    pos = P+offset+1
    K = size(d.Theta_L)[1]
    for i in 1:K, j in i:K
        d.Theta_L[i,j] = d.Theta_L[j,i] = x[pos]
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
function state2vec!(x::AbstractVector, d::Union{CFADistributionTheta,CFADistributionThetaGradient})
    P = length(d.Theta_X.nzval)
    x[1:P] = d.Theta_X.nzval

    offest = length(d.A.nzval)
    x[P+1:P+offest] = d.A.nzval

    pos = P+offest+1
    K = size(d.Theta_L)[1]
    for i in 1:K, j in i:K
        x[pos] = d.Theta_L[i,j]
        pos += 1
    end
end


function dloglikelihood(d::CFADistributionTheta, S::AbstractMatrix, N::Int64)
    P,K = size(d.A)

    # Woodbury transformation of orignal form allows us to only compute a KxK inversion
    Theta = d.Theta_X - d.Theta_X*d.A*inv(d.Theta_L + d.A'*d.Theta_X*d.A)*d.A'*d.Theta_X

    commonPart = Theta - Theta*S*Theta

    # invert Sigma_X efficiently then compute gradient
    Theta_Xinv = deepcopy(d.Theta_X)
    Theta_Xinv.nzval[:] = 1 ./ Theta_Xinv.nzval
    dTheta_X = spdiagm(diag(Theta_Xinv*commonPart*Theta_Xinv))

    Theta_Linv = inv(d.Theta_L)
    dTheta_L = 2*Theta_Linv*d.A'*commonPart*d.A*Theta_Linv
    #dTheta_Linv = 2*d.A'*commonPart*d.A
    #for i in 1:K dTheta_Linv[i,i] = 0.0 end # we shouldn't change the diagonal of the covariance matrix
    #dTheta_L = Theta_Linv*dTheta_Linv*Theta_Linv
    for i in 1:size(dTheta_L)[1] dTheta_L[i,i] /= 2 end # diagonals should not be times 2

    dA_dense = -2*commonPart*d.A*Theta_Linv
    dA = deepcopy(d.A)
    rows = rowvals(dA)
    vals = nonzeros(dA)
    for col = 1:K
        for j in nzrange(dA, col)
            vals[j] = dA_dense[rows[j],col]
        end
    end

    CFADistributionThetaGradient(dTheta_X, dA, dTheta_L)
end

function Distributions.fit_mle(::Type{CFADistributionTheta}, A::SparseMatrixCSC, S::AbstractMatrix, N::Int64; iterations=1000, show_trace=true)
    P,K = size(A)

    # init our parameters
    d = CFADistributionTheta(speye(P), deepcopy(A), eye(K))
    x_init = zeros(Int(P + nnz(A) + (K+1)*K/2))
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

    # rescale X and Theta_L so that Sigma_L is a correlation matrix
    normalize_Sigma_L!(d)
    d
end

function fit_map(pri::Normal, ::Type{CFADistributionTheta}, A::SparseMatrixCSC, S::AbstractMatrix, N::Int64; iterations=1000, show_trace=true)
    @assert mean(pri) == 0.0 "Prior must be zero mean!"
    rho = 1/(std(pri)*N)
    P,K = size(A)

    # init our parameters
    d = CFADistributionTheta(speye(P), deepcopy(A), eye(K))
    x_init = zeros(Int(P + nnz(A) + (K+1)*K/2))
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
        #ll -= rho*sum(d.Theta_X.^2)
        ll -= rho*sum(d.A.^2)
        ll -= rho*sum(d.Theta_L.^2)

        -ll
    end

    function df!(x, g)
        vec2state!(d, x)

        # get gradient of likelihood
        gState = dloglikelihood(d, S, N)

        # account for prior
        #gState.Theta_X .-= 2*rho*d.Theta_X
        gState.A .-= 2*rho*d.A
        gState.Theta_L .-= 2*rho*d.Theta_L

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
