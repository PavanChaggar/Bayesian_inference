# Setting up Bayesian inference, plotting distributions and
# Bayesian linear regression
using Plots
using Distributions

n = 10
μ = 5.0 * ones(n)
σ = 4.0 * ones(n)
ran = rand(Normal(1,1),n)

y = μ .+ (σ .* ran)

plot(y)
