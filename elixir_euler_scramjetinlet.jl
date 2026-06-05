include("my_steadystate_callback_vL2.jl")
using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

###############################################################################
# NASA TP-1940: Model inlet, INVISCID (Euler only)
# M1=5.0, p1=101325 Pa, T1=293 K

# Initial condition — Mach 5 uniform flow


@inline function initial_condition_mach5(x, t,
                                         equations::CompressibleEulerEquations2D)
    R_air = 287.0
    T1    = 293.0
    p1    = 101325.0
    M1    = 5.0

    rho = p1 / (R_air * T1)
    a   = sqrt(equations.gamma * p1 / rho)
    v1  = M1 * a
    v2  = 0.0

    return prim2cons(SVector(rho, v1, v2, p1), equations)
end

initial_condition = initial_condition_mach5

# Boundary conditions

boundary_condition_inflow = BoundaryConditionDirichlet(initial_condition_mach5)

@inline function boundary_condition_outflow(u_inner,
                                            normal_direction::AbstractVector,
                                            x, t,
                                            surface_flux_function,
                                            equations::CompressibleEulerEquations2D)
    return flux(u_inner, normal_direction, equations)
end

#basic setup
polydeg = 3
basis = LobattoLegendreBasis(polydeg)
volume_flux = flux_chandrashekar
surface_flux = FluxLaxFriedrichs(max_abs_speed)

shock_indicator = IndicatorHennemannGassner(equations, basis,
                                            alpha_max = 0.5,
                                            alpha_min = 0.001,
                                            alpha_smooth = true,
                                            variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(shock_indicator;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)
solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

#mesh
mesh_file = joinpath(@__DIR__, "no_strut_scramjet1_newname.inp")

boundary_symbols = [:Left,:TopInlet, :Right, :Top, :Bottom]

mesh = P4estMesh{2}(mesh_file, polydeg = 1, boundary_symbols = boundary_symbols)

# boundary_conditions = Dict(:Left   => boundary_condition_inflow,
#                             :TopInlet => boundary_condition_inflow,
#                             :Right  => boundary_condition_outflow,
#                             :Bottom => boundary_condition_slip_wall,
#                             :Top    => boundary_condition_slip_wall)

boundary_conditions = (; Left   = boundary_condition_inflow,
                        TopInlet = boundary_condition_inflow,
                        Right  = boundary_condition_outflow,
                        Bottom = boundary_condition_slip_wall,
                        Top    = boundary_condition_slip_wall)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)
                    
###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 10.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

residual_l2_tol = 172.6 #2.0e-8 #172.6

Iters_threshold = 1000 #stop after 1000 iters if not converged to avoid long runs. 1000 is just a trial

boundary_symbol_names = (:TopInlet,:Top,)#top

steady_state_callback = mySteadyStateCallback(abstol = residual_l2_tol, reltol = 0,
                                            output_directory = "out", save_extra_analysis = true, 
                                            boundary_symbols = boundary_symbol_names, 
                                            stop_after_NIters = true, Iters_threshold = Iters_threshold)

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                        analysis_errors = Symbol[])
                                     

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 1000,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

#amr (only if necesary, otherwise comment)
amr_indicator = IndicatorLöhner(semi, variable = density)

amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = 0,
                                      med_level = 1, med_threshold = 0.05,
                                      max_level = 3, max_threshold = 0.1)

amr_callback = AMRCallback(semi, amr_controller,
                           interval = 5,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)                                     

stepsize_callback = StepsizeCallback(cfl = 0.9)


callbacks = CallbackSet(summary_callback, steady_state_callback,
                        analysis_callback, alive_callback,
                        save_solution, amr_callback,
                        stepsize_callback)
                        

# positivity limiter necessary for this example with strong shocks
stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds = (5.0e-6, 5.0e-6),
                                                     variables = (Trixi.density, pressure))                        
###############################################################################
sol = solve(ode, SSPRK54(; stage_limiter!); dt = 0.5e-3,
            #maxiters = 999999, 
            adaptive=false ,ode_default_options()...,
            callback = callbacks);
summary_callback() # print the timer summary
