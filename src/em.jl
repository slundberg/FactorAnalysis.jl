using Optim

export CFADistributionEM

"Confirmatory factor analysis distribution object with a Sigma formulation."
type CFADistributionEM
    Sigma_X::SparseMatrixCSC
    A::SparseMatrixCSC
    Sigma_L::AbstractMatrix
end
Base.length(d::CFADistributionEM) = Int(size(d.A)[1] + nnz(d.A) + (size(d.A)[2]-1)*size(d.A)[2]/2)

function Base.rand(d::CFADistributionEM, N::Int)
    P,K = size(d.A)
    distL = MvNormal(zeros(K), d.Sigma_L)
    L = rand(distL, N)
    X = d.A*L
    for i in 1:P
        X[i,:] .+= transpose(rand(Normal(0, d.Sigma_X.nzval[i]), N))
    end
    X
end
Base.rand(d::CFADistributionEM) = rand(d::CFADistributionEM, 1)

"Likelihood of model given the data represented by the covariance S from N samples"
function Distributions.loglikelihood(d::CFADistributionEM, S::AbstractMatrix, N::Int64)
    #@assert minimum(d.Sigma_X.nzval) > 0.0 "Invalid CFADistributionEM! (Theta_X has elements <= 0)"
    #@assert minimum(eig(d.Sigma_L)[1]) > 0.0 "Invalid CFADistributionEM! (Theta_L has eigen value <= 0)"

    Theta_X = deepcopy(d.Sigma_X)
    Theta_X.nzval[:] = 1 ./ d.Sigma_X.nzval
    Theta_L = inv(d.Sigma_L)

    # Woodbury transformation of original form allows us to only compute a KxK inversion
    Theta = Theta_X - Theta_X*d.A*inv(Theta_L + d.A'*Theta_X*d.A)*d.A'*Theta_X

    @assert minimum(eig(Theta)[1]) > 0.0 "Invalid CFADistributionEM! (Theta has eigen value <= 0)"

    logdet(Theta) - trace(S*Theta)
end

function normalize_Sigma_L!(d::CFADistributionEM)
    scaling = sqrt(diag(d.Sigma_L))
    Base.cov2cor!(d.Sigma_L, scaling)
    vals = nonzeros(d.A)
    K = size(d.A)[2]
    for col = 1:K
        for j in nzrange(d.A, col)
            vals[j] *= scaling[col]
        end
    end
end

" Run the EM algorithm to find the MLE fit."
function Distributions.fit_mle(::Type{CFADistributionEM}, A::SparseMatrixCSC, S::AbstractMatrix, N::Int64; iterations=1000, show_trace=true, ftol=1e-10)
    P,K = size(A)

    # init our parameters
    d = CFADistributionEM(speye(P), deepcopy(A), eye(K))

    Theta_X = deepcopy(d.Sigma_X)
    lastScore = -Inf
    timeFindT = 0.0
    timeFindW = 0.0
    timeFindQ = 0.0
    timeUpdateSigma_X = 0.0
    timeUpdateA = 0.0
    timeUpdateSigma_L = 0.0
    for count in 1:iterations

        ## E-step

        t = time()
        for i in 1:P
            Theta_X.nzval[i] = 1/d.Sigma_X.nzval[i]
        end
        Theta_L = inv(d.Sigma_L)
        Theta = Theta_X - Theta_X*d.A*inv(Theta_L + d.A'*Theta_X*d.A)*d.A'*Theta_X
        #Theta = inv(Sigma_X + A*Sigma_L*A')
        T = Theta*d.A*d.Sigma_L
        timeFindT += time() - t

        # find W
        t = time()
        W = d.Sigma_L - d.Sigma_L*d.A'*Theta*d.A*d.Sigma_L
        timeFindW += time() - t

        # find Q (the expected covariance of the hidden variables)
        t = time()
        Q = T'*S*T + W
        #Base.cov2cor!(Q, sqrt(diag(Q)))
        timeFindQ += time() - t

        ## M-step

        # update A and Sigma_X
        t = time()
        tmpTS = T'*S
        # for i in 1:P
        #     gi = grouping[i]
        #     d.A[i,gi] = tmpTS[gi,i]/Q[gi,gi]
        #     d.Sigma_X[i,i] = S[i,i] - (tmpTS[gi,i]^2)/Q[gi,gi]
        # end
        rows = rowvals(d.A)
        vals = nonzeros(d.A)
        for col = 1:K
            for j in nzrange(d.A, col)
                row = rows[j]
                d.Sigma_X.nzval[row] = S[row,row] - (tmpTS[col,row]^2)/Q[col,col]
                vals[j] = tmpTS[col,row]/Q[col,col]
            end
        end
        timeUpdateA += time() - t

        # update Sigma_L
        t = time()
        d.Sigma_L[:,:] = Q

        timeUpdateSigma_L += time() - t


        # print our progress
        if count % 1000 == 0 && show_trace
            s = loglikelihood(d, S, N)
            println("likelihood $count = ", s, " ", s - lastScore)

            println("timeFindT = $timeFindT")
            println("timeFindW = $timeFindW")
            println("timeFindQ = $timeFindQ")
            println("timeUpdateSigma_X = $timeUpdateSigma_X")
            println("timeUpdateA = $timeUpdateA")
            println("timeUpdateSigma_L = $timeUpdateSigma_L")
            println()

            if s - lastScore < ftol
                break
            end
            lastScore = s
        end
    end
    normalize_Sigma_L!(d)
    d
end

function fit_map(pri::Normal, ::Type{CFADistributionEM}, A::SparseMatrixCSC, S::AbstractMatrix, N::Int64; iterations=1000, show_trace=true)
    warning("fit_map for CFADistributionEM is not working well yet!")
    @assert mean(pri) == 0.0 "Prior must be zero mean!"
    rho = 1/(std(pri)*N)
    P,K = size(A)

    # init our parameters
    d = CFADistributionEM(speye(P), deepcopy(A), eye(K))
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
