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

function Base.rand(d::CFADistribution, N::Int)
    P,K = size(d.A)
    distL = MvNormal(zeros(K), inv(d.Theta_L))
    L = rand(distL, N)
    X = d.A*L
    for i in 1:P
        X[i,:] .+= transpose(rand(Normal(0, 1 / d.Theta_X.nzval[i]), N))
    end
    X
end
Base.rand(d::CFADistribution) = rand(d::CFADistribution, 1)

function normalize_Sigma_L!(d::CFADistribution)
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
