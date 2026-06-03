using DelimitedFiles
using Plots

# Read file, skipping header line
data = readdlm("out/steadystate_residual.dat", skipstart=1)

timestep = data[:, 1]

rho   = data[:, 3]
rhov1 = data[:, 4]
rhov2 = data[:, 5]
rhoE  = data[:, 6]

plot(
    timestep, rho,
    label = "ρ",
    lw = 2,
    xlabel = "Timestep",
    ylabel = "log₁₀(L₂ residual)",
    legend = :topright
)

plot!(timestep, rhov1, label = "ρv₁", lw = 2)
plot!(timestep, rhov2, label = "ρv₂", lw = 2)
plot!(timestep, rhoE,  label = "ρE",  lw = 2)

#savefig("residual_plot.pdf")