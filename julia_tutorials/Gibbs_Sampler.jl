using Plots
using Distributions

#= 
implement function to perform Gibbs sampling for for a linear simple
linear model y = ax + e 
=#

a = 2.0
σ = 5.0
t = range(1, stop = 10, length=20)

f(a, σ, t) = a .* t .+ (rand(Normal(0,σ),20))

y = f(a, σ, t)

scatter(t, y)

function Gibbs_Sampler!(y, σ, t, n, r_a, r_s)

    aₗ, aᵤ = r_a
    r_a = range(aₗ, stop = aᵤ, length = n)

    lₗ, lᵤ = r_s
    r_s = range(lₗ, stop = lᵤ, length = n)

    posterior = zeros(n, n)
    for i = 1:n
        for j = 1:n
            x = r_a[j] .* t
            likelihood = r_s[i]^(-length(t)) .* exp.(-sum((y .- x ).^2,dims=1)/2/r_s[i]^2)
            prior = 1/r_s[i] * pdf(Normal(r_a[j],1.0), 0.0)
            posterior[i,j] = likelihood[1] * prior
            posterior = posterior/sum(posterior)
            return likelihood, prior, posterior
        end
    end
end

n = 100 
r_a = 0.0, 5.0
r_s = 0.0, 10

likelihood, prior, posterior = Gibbs_Sampler!(y, σ, t, n, r_a, r_s)

heatmap(posterior)            
