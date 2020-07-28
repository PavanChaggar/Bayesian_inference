# Setting up Bayesian inference, plotting distributions and
# Bayesian linear regression
using Plots
using Distributions


μ = 7.0
σ = 2.0
n = 1000
lim = 10
x = range(0, lim, length = n)

f(μ, σ, x) = μ .+ (σ * rand(Normal(0,1),lim))

y = f(μ, σ, x)

function likelihood!(y, n, sd)
    yy = repeat(y, outer=(1,n))
    ran = range(0, stop = length(y), length = n)
    xx = repeat(ran, outer=(1, length(y)))'
    likelihood = exp.(-0.5 * sum((yy .- xx ).^2,dims=1) / σ^2)
    norm_likelihood = likelihood / sum(likelihood)
    return likelihood
end

likelihood = likelihood!(y, n, σ)

plot(x,likelihood[1,:])
scatter!(y, zeros(n))
