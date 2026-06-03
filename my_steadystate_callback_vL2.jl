using Trixi
using LinearAlgebra
using Trixi: @muladd,summary_box, mpi_isroot, mpi_isparallel, wrap_array,@batch

using Trixi: TreeMesh, StructuredMesh, StructuredMeshView, UnstructuredMesh2D,
            P4estMesh, P4estMeshView, T8codeMesh

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
    open(joinpath(ss_callback.output_directory,"steadystate_residual.dat"), "w")  do io
        print(io, "#timestep ")
        print(io, "time ")
        print(io, "log10_du_rho_L2 ")
        print(io, "log10_du_rhov1_L2 ")
        print(io, "log10_du_rhov2_L2 ")
        print(io, "log10_du_rhoE_L2 ")
        println(io)
    end
    return nothing
end

# affect!
(::mySteadyStateCallback)(integrator) = terminate!(integrator)


#for any mesh of structured, unstructured, p4est
function save_du_l2_domain(du,dg,cache,mesh::Union{StructuredMesh{2}, StructuredMeshView{2},
                                      UnstructuredMesh2D,
                                      P4estMesh{2}, P4estMeshView{2},
                                      T8codeMesh{2}}, 
                            equations,t, iter, output_directory)
    @unpack weights = dg.basis
    du_l2norm = zeros(4) #compute l2 error for each variable in residual du
    log10_du_l2norm = zeros(4)
    total_volume = 0.0

    # Use quadrature to numerically integrate over entire domain
    for element in eachelement(dg, cache)

        for j in eachnode(dg), i in eachnode(dg)
            volume_jacobian = abs(inv(cache.elements.inverse_jacobian[i, j, element]))
            du_local = get_node_vars(du, equations, dg, i, j, element)
            du_l2norm += volume_jacobian * weights[i] * weights[j] * du_local.^2
            total_volume += volume_jacobian * weights[i] * weights[j]
        end
    end

    du_l2norm = @. sqrt(du_l2norm / total_volume)
    log10_du_l2norm = @. log10(du_l2norm)

    #append data after every 100  iterations# can change frequency here
    if iter % 100 == 0
        open(joinpath(output_directory,"steadystate_residual.dat"), "a")  do io
            print(io, iter)
            print(io, " ", t)
            print(io, " ", log10_du_l2norm[1])
            print(io, " ", log10_du_l2norm[2])
            print(io, " ", log10_du_l2norm[3])
            print(io, " ", log10_du_l2norm[4])
            println(io) # This writes newline to the file
        end
    end

    return du_l2norm
end


#for tree mesh
function save_du_l2_domain(du,dg,cache,mesh::TreeMesh{2}, equations,t, iter, output_directory)
    @unpack weights = dg.basis
    du_l2norm = zeros(4) #compute l2 error for each variable in residual du
    log10_du_l2norm = zeros(4)


    # Use quadrature to numerically integrate over entire domain
    for element in eachelement(dg, cache)
        du_l2norm_ = zeros(4)
        volume_jacobian_ = volume_jacobian(element, mesh, cache)
        for j in eachnode(dg), i in eachnode(dg)
            du_local = get_node_vars(du, equations, dg, i, j, element)
            du_l2norm_ += volume_jacobian_ * weights[i] * weights[j] * du_local.^2
        end
        du_l2norm += du_l2norm_
    end

    total_volume_ = total_volume(mesh)
    du_l2norm = @. sqrt(du_l2norm / total_volume_)
    log10_du_l2norm = @. log10(du_l2norm)

    #append data after every 100  iterations# can change frequency here
    if iter % 100 == 0
        open(joinpath(output_directory,"steadystate_residual.dat"), "a")  do io
            print(io, iter)
            print(io, " ", t)
            print(io, " ", log10_du_l2norm[1])
            print(io, " ", log10_du_l2norm[2])
            print(io, " ", log10_du_l2norm[3])
            print(io, " ", log10_du_l2norm[4])
            println(io) # This writes newline to the file
        end
    end

    return du_l2norm
end

# the condition
function (steady_state_callback::mySteadyStateCallback)(u_ode, t, integrator)
    @unpack abstol, reltol, output_directory = steady_state_callback
    semi = integrator.p
    equations = semi.equations
    dg = semi.solver
    cache = semi.cache
    mesh = semi.mesh
    iter = integrator.stats.naccept

    u = wrap_array(u_ode, semi)
    du = wrap_array(get_du(integrator), semi)

    #save the log10 of L2 of du residual in domain
    l2_du = save_du_l2_domain(du,dg,cache,mesh, equations,t, iter, output_directory)
    l2_du_rho = l2_du[1]

    threshold_L2 = abstol
    
    terminate = false
    if l2_du_rho <= threshold_L2
        terminate = true
    else
        terminate = false
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

end # @muladd