using Distributions

srand(1)

# make sure randcor works
@test sum(abs(inv(FactorAnalysis.randcor(40, 0.2))) .>= 1e-8) < 40*40/2
