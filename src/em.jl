using Optim

export CFADistributionEM, CFADistributionEMRidge

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
function Distributions.fit_mle(::Type{CFADistributionEM}, maskSigma_L::AbstractMatrix, A::SparseMatrixCSC, S::AbstractMatrix, N::Int64; iterations=1000, show_trace=true, ftol=1e-10)
    P,K = size(A)
    @assert size(maskSigma_L) == (K,K) "Sigma_L mask must be of size K x K (where K = size(A)[2])"
    #@assert size(maskL) == (K,K) "Sigma_L mask must be of size K x K (where K = size(A)[2])"

    # just to make sure it is a nice float data type inside our loop
    maskL = zeros(Float64, K, K)
    copy!(maskL, maskSigma_L)

    # init our parameters
    d = CFADistributionEM(speye(P), deepcopy(A), eye(K))

    # find the latent vars each observed is attached to
    latentInds = Array{Int64}[find(d.A[i,:]) for i in 1:P]

    Theta_X = deepcopy(d.Sigma_X)
    lastScore = -Inf
    timeFindT = 0.0
    timeFindW = 0.0
    timeFindQ = 0.0
    timeUpdateSigma_X = 0.0
    timeUpdateA = 0.0
    timeUpdateSigma_L = 0.0
    stopCount = 0
    for count in 1:iterations

        ## E-step

        t = time()
        for i in 1:P
            Theta_X.nzval[i] = 1/d.Sigma_X.nzval[i]
        end
        Theta_L = inv(d.Sigma_L)
        Theta = Theta_X - Theta_X*d.A*inv(Theta_L + d.A'*Theta_X*d.A)*d.A'*Theta_X
        #@assert maximum(abs(Theta .- inv(d.Sigma_X + d.A*d.Sigma_L*d.A'))) < 1e-6
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
        #At = d.A'
        #rows = rowvals(At)
        #vals = nonzeros(At)
        for row = 1:P

            # update Sigma_X
            inds = latentInds[row]
            tmp = tmpTS[inds,row]
            #println(inds, " ", collect(nzrange(At, col)))
            newLoadings = inv(Q[inds,inds])*tmp
            #println(S[col,col] - tmpTS[inds,col]'*newLoadings)
            d.Sigma_X.nzval[row] = (S[row,row] - tmp'*newLoadings)[1]

            # Update A
            for (i,ind) in enumerate(inds)
                d.A[row,ind] = newLoadings[i]
            end
        end
        #d.A = At'
        timeUpdateA += time() - t

        # update Sigma_L
        t = time()
        d.Sigma_L[:,:] = Q .* maskL # note we enforce the Sigma_L mask here

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
            println(d.Sigma_L[1,1], " ", d.Sigma_L[1,2])
            println()

            if s - lastScore < ftol
                stopCount += 1
            end
            if stopCount > 5
                break
            end
            lastScore = s
        end
    end
    normalize_Sigma_L!(d)
    d
end

type CFADistributionEMRidge
    rho::Float64
end

" Run the EM algorithm to find the MAP fit."
function fit_map(prior::CFADistributionEMRidge, ::Type{CFADistributionEM}, A::SparseMatrixCSC, S::AbstractMatrix, N::Int64; iterations=1000, show_trace=true, ftol=1e-10)
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
    stopCount = 0
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
        timeFindQ += time() - t

        ## M-step

        # update A and Sigma_X
        t = time()
        tmpTS = T'*S
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

        Base.cov2cor!(Q, sqrt(diag(Q)))
        d.Sigma_L[:,:] = Q + prior.rho*eye(K)
        Base.cov2cor!(d.Sigma_L, sqrt(diag(d.Sigma_L)))
        #Base.cov2cor!(d.Sigma_L, sqrt(prior.rho ./ (diag(Q) + prior.rho)))
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
            println(d.Sigma_L[1,1], " ", d.Sigma_L[1,2])
            println()

            if abs(s - lastScore) < ftol
                stopCount += 1
            end
            if stopCount > 5
                break
            end
            lastScore = s
        end
    end
    normalize_Sigma_L!(d)
    d
end
