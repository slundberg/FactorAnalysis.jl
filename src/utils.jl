using Distributions

"Generate a correlation matrix of a given size and density of network edges."
function randcor(K, netDensity)
    IC = diagm(abs(randn(K)))
    for i in 1:K, j in 1:i-1
        IC[i,j] = rand() < netDensity ? randn()-1 : 0.0
        IC[j,i] = IC[i,j]
    end
    mineval = minimum(eig(IC)[1])
    if mineval < 0.01
        IC -= eye(K)*mineval*1.1
    end
    C = inv(IC)
    Base.cov2cor!(C, sqrt(diag(C)))
end

function samples(Sigma_X::SparseMatrixCSC, A::SparseMatrixCSC, Sigma_L::AbstractMatrix, N::Int)
    P,K = size(A)
    distL = MvNormal(zeros(K), Sigma_L)
    L = rand(distL, N)
    X = A*L
    for i in 1:P
        X[i,:] .+= transpose(rand(Normal(0, Sigma_X.nzval[i]), N))
    end
    X
end
