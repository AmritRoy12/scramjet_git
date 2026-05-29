using OrdinaryDiffEqSSPRK
using Trixi

include("my_steadystate_callback.jl")
###############################################################################
# NASA TP-1940: Model inlet, INVISCID (Euler only)
# M1=5.0, p1=101325 Pa, T1=293 K
###############################################################################

const gamma_air = 1.4

equations = CompressibleEulerEquations2D(gamma_air)

###############################################################################
# Initial condition — Mach 5 uniform flow
###############################################################################

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

###############################################################################
# Boundary conditions
###############################################################################

boundary_condition_inflow = BoundaryConditionDirichlet(initial_condition_mach5)

@inline function boundary_condition_outflow(u_inner,
                                            normal_direction::AbstractVector,
                                            x, t,
                                            surface_flux_function,
                                            equations::CompressibleEulerEquations2D)
    return flux(u_inner, normal_direction, equations)
end

boundary_conditions = Dict(
    :Left   => boundary_condition_inflow,
    :Right  => boundary_condition_outflow,
    :Bottom => boundary_condition_slip_wall,
    :Top    => boundary_condition_slip_wall
)

###############################################################################
# Solver
###############################################################################

polydeg      = 3
basis        = LobattoLegendreBasis(polydeg)
volume_flux  = flux_ranocha
surface_flux = FluxLaxFriedrichs(max_abs_speed_naive)

shock_indicator = IndicatorHennemannGassner(equations, basis;
                                            alpha_max    = 0.8,
                                            alpha_min    = 0.001,
                                            alpha_smooth = true,
                                            variable     = density_pressure)

volume_integral = VolumeIntegralShockCapturingHG(shock_indicator;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

solver = DGSEM(polydeg         = polydeg,
               surface_flux    = surface_flux,
               volume_integral = volume_integral)

###############################################################################
# Mesh
###############################################################################

mesh_file = joinpath(@__DIR__, "no_strut_scramjet1.inp")

boundary_symbols = [:Left, :Right, :Top, :Bottom]

mesh = P4estMesh{2}(mesh_file;
                    polydeg          = polydeg,
                    boundary_symbols = boundary_symbols)

###############################################################################
# Semidiscretization
###############################################################################

semi = SemidiscretizationHyperbolic(
    mesh,
    equations,
    initial_condition,
    solver;
    boundary_conditions = boundary_conditions
)

###############################################################################
# ODE + callbacks
###############################################################################

tspan = (0.0, 5.0)
ode   = semidiscretize(semi, tspan)

summary_callback  = SummaryCallback()
analysis_interval = 100

analysis_callback = AnalysisCallback(semi;
                                     interval                 = analysis_interval,
                                     extra_analysis_integrals = (entropy,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(dt                    = 5.0e-4,
                                     save_initial_solution = true,
                                     save_final_solution   = true,
                                     solution_variables    = cons2prim)
                                     
steady_state_callback = mySteadyStateCallback(abstol = 1.0e-6, reltol = 0)

stepsize_callback = StepsizeCallback(cfl = 0.5)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        stepsize_callback,
                        steady_state_callback)

stage_limiter! = PositivityPreservingLimiterZhangShu(
    thresholds = (1.0e-4, 1.0e-4),
    variables  = (Trixi.density, pressure)
)

###############################################################################
# Solve
###############################################################################

sol = solve(ode, SSPRK43(stage_limiter!);
            dt       = 1.0e-6,
            adaptive = false,
            ode_default_options()...,
            callback = callbacks)
