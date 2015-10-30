using Distributions
using MLBase

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

function upper(X::AbstractMatrix)
    x = Float64[]
    for i in 1:size(X)[1], j in i+1:size(X)[2]
        push!(x, X[i,j])
    end
    x
end
function area_under_pr(truth::AbstractVector, predictor::AbstractVector; resolution=4000)
    rocData = MLBase.roc(round(Int64, truth), float(invperm(sortperm(predictor))), resolution)
    vals = collect(map(x->(recall(x), -precision(x)), rocData))
    sort!(vals)
    xvals = map(x->x[1], vals)
    yvals = map(x->-x[2], vals)
    # println(xvals)
    # println(yvals)
    area_under_curve([0.0; xvals], [yvals[1]; yvals]) # make sure we extend all the way to zero
end
function area_under_curve(x, y) # must be sorted by increasing x
    area = 0.0
    lastVal = NaN
    for i in 2:length(x)
        v = (y[i-1]+y[i])/2 * (x[i]-x[i-1])
        if !isnan(v)
            area += v
            lastVal = v
        elseif !isnan(lastVal)
            area += lastVal
        end
    end
    area
end
function area_under_pr(truth::AbstractMatrix, predictor::AbstractMatrix)
    area_under_pr(abs(upper(truth)) .> 0.01, abs(upper(predictor)))
end
