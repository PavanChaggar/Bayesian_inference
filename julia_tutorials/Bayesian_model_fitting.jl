# Setting up Bayesian inference, plotting distributions and
# Bayesian linear regression
using Plots
using Distributions


μ = 5.0
σ = 4.0
n = 1000
lim = 10
x = range(0, stop = lim, length=1000)

f(μ, σ, x) = μ .+ (σ * rand(Normal(0,1),lim))

y = f(μ, σ, x)

function likelihood!(y, n, sd)
    yy = repeat(y, outer=(1,n))
    ran = range(0, stop = length(y), length = n)
    xx = repeat(ran, outer=(1, length(y)))'
    likelihood = exp.(-0.5 * sum((yy .- xx ).^2,dims=1) / σ^2)
    norm_likelihood = likelihood / sum(likelihood)
    return norm_likelihood
end

likelihood = likelihood!(y, n, σ)

ρ = exp.((-0.5 .* (x .- 0 ).^2) / 5^2)
ρ = ρ/sum(ρ)

plot(x, likelihood[1,:])
plot!(x, ρ)
scatter!(y, zeros(n))

#= Linear regression 
Use linear regression to estimate the posterior mean and std 
y = B*x + e
e = N(0,s^2)

y = N(B^T * x, s^2 * I)
=#

beta1 = 1 / σ^2
beta2 = 1 / 5^2 
 
post_σ = 1 / √(lim * beta1 + beta2)
post_μ = (lim * beta1 * mean(y) + beta2 * 0) / (lim * beta1 + beta2)

posterior = pdf(Normal(post_μ, post_σ), x)
posterior = posterior/sum(posterior)

plot!(x, posterior)