using Trixi
using Plots
using DelimitedFiles
using HDF5
using LaTeXStrings

##Plotting the residuals from the steady state callback

# Read file, skipping header line
data = readdlm("out/steadystate_residual.dat", skipstart=1)

timestep = data[:, 1]

rho   = data[:, 3]
rhov1 = data[:, 4]
rhov2 = data[:, 5]
rhoE  = data[:, 6]

p = plot(
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

savefig(p, "residual_plot.pdf")

## Plotting quantity of interest at the boundary
analysis_dict = read(h5open("out/extra_analysis_final_00006535.h5"))##change this filename

x = analysis_dict["point_coordinates"][:,1]
y = analysis_dict["point_coordinates"][:,2]


t = analysis_dict["time"]
@show t
pressure = analysis_dict["point_data"][:,1]

 
x_ref = [0.32667596544503014, 0.7492560343877438, 1.040605510096269, 2.055081959736543, 3.0380781173394347, 4.05038062914977, 5.075935800423386, 5.975820642527848, 5.999692152160446, 6.612575394216261, 7.833784970184366, 9.052904221315993]
pressure_ref_shockexpansiontheory = [0.10166769224351019, 0.10088338669809452, 0.3092666951742264, 0.30832594570352834, 0.30881613666941315, 0.3075604134291466, 0.30754998383412774, 0.30754998383412774, 0.025333486300726946, 0.028005548544550018, 0.026203314525296985, 0.024778631845725433]

pressure_ref_shockexpansiontheory = pressure_ref_shockexpansiontheory * 1.0e6

pp = sortperm(x)

x_sorted   = x[pp]
y_sorted   = y[pp]
pressure_sorted = pressure[pp]

# numerical data degrez cf
p = plot(x_sorted, pressure_sorted,label="N=3 ESDG", linecolor = "magenta", linewidth=1, xticks = 0:11)
scatter!(x_ref, pressure_ref_shockexpansiontheory, marker = :circle, ms=4, markercolor = :white, markerstrokecolor = :black, markerstrokewidth = 2, label = "ref-shock expansion wave theory")
title!("Pressure on top of inlet")
xlabel!("Length (cm)")
ylabel!("Pressure")

savefig(p, "top_pressure_plot.png")