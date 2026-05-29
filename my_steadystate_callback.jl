using Trixi
using LinearAlgebra
using Trixi: @muladd,summary_box, mpi_isroot, mpi_isparallel, wrap_array,@batch

using OrdinaryDiffEqSSPRK: DiscreteCallback

# Author Kedar Wagh
# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    mySteadyStateCallback(; abstol=1.0e-8, reltol=1.0e-6)

Terminates the integration when the [`residual_steady_state(du, equations)`](@ref)
falls below the threshold specified by `abstol, reltol`.
"""
mutable struct mySteadyStateCallback{RealT <: Real}
    abstol::RealT
    reltol::RealT
    output_directory::String
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:mySteadyStateCallback})
    @nospecialize cb # reduce precompilation time

    steady_state_callback = cb.affect!
    print(io, "mySteadyStateCallback(abstol=", steady_state_callback.abstol, ", ",
          "reltol=", steady_state_callback.reltol, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:mySteadyStateCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        steady_state_callback = cb.affect!

        setup = [
            "absolute tolerance" => steady_state_callback.abstol,
            "relative tolerance" => steady_state_callback.reltol,
            "output_directory" => abspath(normpath(steady_state_callback.output_directory))
        ]
        summary_box(io, "mySteadyStateCallback", setup)
    end
end

function mySteadyStateCallback(; abstol = 1.0e-8, reltol = 1.0e-6, output_directory = "out")
    abstol, reltol = promote(abstol, reltol)
    steady_state_callback = mySteadyStateCallback(abstol, reltol,output_directory )

    DiscreteCallback(steady_state_callback, steady_state_callback,
                     save_positions = (false, false),
                     initialize = initialize_ss_cb!)
end

function initialize_ss_cb!(cb, u, t, integrator)
    # The mySteadyStateCallback is either cb.affect! (with DiscreteCallback)
    # or cb.affect!.affect! (with PeriodicCallback).
    # Let recursive dispatch handle this.
    initialize_ss_cb!(cb.affect!, u, t, integrator)
end

function initialize_ss_cb!(ss_callback::mySteadyStateCallback, u, t, integrator)
    mpi_isroot() && mkpath(ss_callback.output_directory)
    mkpath(ss_callback.output_directory)
    # write header of output file
    open(joinpath(ss_callback.output_directory,"steadystate_du_Linf.dat"), "w")  do io
        print(io, "#timestep ")
        print(io, "time ")
        print(io, "du_Linf ")
        println(io)
    end
    return nothing
end

# affect!
(::mySteadyStateCallback)(integrator) = terminate!(integrator)

function save_du_max_domain(du,dg,cache,equations,t, iter, output_directory)
   max_du_Linf =  nextfloat(zero(t))
    @batch reduction=(max, max_du_Linf) for element in eachelement(dg, cache)
    max_du_Linf_ij = zero(max_du_Linf)
        for j in eachnode(dg), i in eachnode(dg)
            du_local = get_node_vars(du, equations, dg, i, j, element)
            du_local_Linf = norm(du_local,Inf)
            max_du_Linf_ij = max(max_du_Linf_ij,du_local_Linf)
        end
        max_du_Linf = max(max_du_Linf,max_du_Linf_ij)
    end
    #append data
        open(joinpath(output_directory,"steadystate_du_Linf.dat"), "a")  do io
            print(io, iter)
            print(io, " ", t)
            print(io, " ", max_du_Linf)
            println(io) # This writes newline to the file
        end

end

# the condition
function (steady_state_callback::mySteadyStateCallback)(u_ode, t, integrator)
    @unpack abstol, reltol, output_directory = steady_state_callback
    semi = integrator.p
    equations = semi.equations
    dg = semi.solver
    cache = semi.cache
    iter = integrator.stats.naccept

    u = wrap_array(u_ode, semi)
    du = wrap_array(get_du(integrator), semi)

    #save the Linf of du residual vector
    save_du_max_domain(du,dg,cache,equations,t, iter, output_directory)

    terminate = true
    for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            u_local = get_node_vars(u, equations, dg, i, j, element)
            du_local = get_node_vars(du, equations, dg, i, j, element)
            threshold = abstol + reltol * rho_residual_steady_state(u_local, equations)
            terminate = terminate &&
                        rho_residual_steady_state(du_local, equations) <= threshold
        end
    end

    if mpi_isparallel()
        # MPI.jl doesn't seem to have MPI_C_BOOL
        terminate_integer = Int(terminate)
        terminate = !iszero(MPI.Allreduce!(Ref(terminate_integer), +, mpi_comm())[])
    end
    if mpi_isroot() && terminate
        @info "  Steady state tolerance reached" steady_state_callback t
    end
    return terminate
end

function rho_residual_steady_state(du, ::CompressibleEulerEquations2D)
    abs(du[1])
end

end # @muladd






