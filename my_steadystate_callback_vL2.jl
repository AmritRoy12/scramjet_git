using Trixi
using LinearAlgebra
using Trixi: @muladd,summary_box, mpi_isroot, mpi_isparallel, wrap_array,@batch,
            indices2direction, index_to_start_step_2d, get_node_coords, get_node_vars,
             HDF5, h5open, attributes

using Trixi: TreeMesh, StructuredMesh, StructuredMeshView, UnstructuredMesh2D,
            P4estMesh, P4estMeshView, T8codeMesh

using OrdinaryDiffEqSSPRK: DiscreteCallback

using Printf: @sprintf

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
falls below the threshold specified by `abstol, reltol`. Also stops the code after NIters.
The callback writes the log10 of L2 norm of du residual in the domain to a file.
Termination criteria is when L2 of density is below the threshold.

If stop_after_NIters is true then dont give tolerence or give 1e-14 which is default.

If tolerance is given the set stop_after_NIters to false. Default iters is 1000.
"""
mutable struct mySteadyStateCallback{RealT <: Real, NBoundaries}
    abstol::RealT
    reltol::RealT
    output_directory::String
    save_extra_analysis::Bool
    boundary_symbols::NTuple{NBoundaries, Symbol}
    stop_after_NIters::Bool
    Iters_threshold::Int
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:mySteadyStateCallback})
    @nospecialize cb # reduce precompilation time

    steady_state_callback = cb.affect!
    print(io, "mySteadyStateCallback(abstol=", steady_state_callback.abstol, ", ",
          "reltol=", steady_state_callback.reltol, "save_extra_analysis=", steady_state_callback.save_extra_analysis, 
          "output_directory=", steady_state_callback.output_directory, "stop_after_NIters=", steady_state_callback.stop_after_NIters, "Iters_threshold=", steady_state_callback.Iters_threshold, ")")
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
            "output_directory" => abspath(normpath(steady_state_callback.output_directory)),
            "save extra analysis" => steady_state_callback.save_extra_analysis ?
                                     "yes" : "no",
            "stop after N iters" => steady_state_callback.stop_after_NIters ? "yes" : "no",
            "N iters threshold" => steady_state_callback.Iters_threshold,
        ]
        summary_box(io, "mySteadyStateCallback", setup)
    end
end

function mySteadyStateCallback(; abstol = 1.0e-14, reltol = 0.0, output_directory = "out", save_extra_analysis = false, 
                                boundary_symbols, stop_after_NIters = false, Iters_threshold = 1000)
    abstol, reltol = promote(abstol, reltol)
    steady_state_callback = mySteadyStateCallback(abstol, reltol,output_directory, 
    save_extra_analysis, boundary_symbols, stop_after_NIters, Iters_threshold)

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

# Helper function that saves a space-dependent quantity `values`
# at every solution/quadrature point `coordinates` at 
# time `t` and iteration `iter` to disk.
# The file is written to the `output_directory` with name `varname` in HDF5 (.h5) format.
# The latter two are retrieved from the `surface_variable`,
# an instantiation of `AnalysisSurfacePointwise`.
function save_pointwise_file(output_directory, varname, coordinates, values, t, iter)
    n_points = length(values)

    filename = joinpath(output_directory, varname) * @sprintf("_%08d.h5", iter)

    h5open(filename, "w") do file
        # Add context information as attributes
        attributes(file)["n_points"] = n_points
        attributes(file)["variable_name"] = varname

        file["time"] = t
        file["timestep"] = iter
        file["point_coordinates"] = coordinates
        file["point_data"] = values
    end
end

#to save whatever results at boundary after steady state is reached
# example of pressure is implemented here, can be changed to any other variable or combination of variables as needed
function save_extraresults( boundary_symbols, integrator, semi, dg, iter, t, output_directory )
    
    @unpack boundaries = semi.cache
    @unpack node_coordinates, contravariant_vectors = semi.cache.elements
    @unpack boundary_symbol_indices = semi.boundary_conditions

    #indices = semi.boundary_conditions.boundary_symbol_indices[boundary_symbols]

    indices = Int[]
    for name in boundary_symbols
        append!(indices, boundary_symbol_indices[name])
    end
    sort!(indices) # Try to achieve some data locality by sorting

    dim = ndims(semi.mesh)
    n_nodes = nnodes(dg)
    n_elements = length(indices)

    coordinates = Matrix{real(dg)}(undef, n_elements * n_nodes, dim) # physical coordinates of indices
    # variable values at indices, change the no here
    # 1 for dv1/dy
    values = Matrix{real(dg)}(undef, n_elements * n_nodes, 1) 

    index_range = eachnode(dg)
    global_node_counter = 1 # Keeps track of solution point number on the surface

    for boundary in indices
    element = boundaries.neighbor_ids[boundary]
    node_indices = boundaries.node_indices[boundary]
    
    i_node_start, i_node_step = index_to_start_step_2d(node_indices[1], index_range)
    j_node_start, j_node_step = index_to_start_step_2d(node_indices[2], index_range)

    i_node = i_node_start
    j_node = j_node_start
    for node_index in index_range
        u_node = Trixi.get_node_vars(semi.cache.boundaries.u, semi.equations, dg, node_index,
                                        boundary)

        x = get_node_coords(node_coordinates, semi.equations, dg, i_node, j_node,
                            element)

        p_node = Trixi.pressure(u_node, semi.equations)

        coordinates[global_node_counter, 1] = x[1]
        coordinates[global_node_counter, 2] = x[2]
        values[global_node_counter, 1] = p_node

        i_node += i_node_step
        j_node += j_node_step
        global_node_counter += 1
        end
    end

    # Save to disk
    save_pointwise_file(output_directory, "extra_analysis_final",
                        coordinates, values, t, iter)


    # avoid re-evaluating possible FSAL stages
    u_modified!(integrator, false)
    return nothing
end


# the condition
function (steady_state_callback::mySteadyStateCallback)(u_ode, t, integrator)
    @unpack abstol, reltol, output_directory, save_extra_analysis = steady_state_callback
    @unpack stop_after_NIters, Iters_threshold = steady_state_callback


    if save_extra_analysis == true
      @unpack boundary_symbols = steady_state_callback  
    end

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
    
    terminate = false

    if stop_after_NIters == true

        if iter >= Iters_threshold
            terminate = true

            if save_extra_analysis == true
                save_extraresults( boundary_symbols, integrator, semi, dg, iter, t, output_directory )
            end
        else
            terminate = false
        end

    else

        l2_du_rho = l2_du[1]
        threshold_L2 = abstol
        
        if l2_du_rho <= threshold_L2
            terminate = true

            if save_extra_analysis == true
                save_extraresults( boundary_symbols, integrator, semi, dg, iter, t, output_directory )
            end
        else
            terminate = false
        end        
    end


    if mpi_isparallel()
        # MPI.jl doesn't seem to have MPI_C_BOOL
        terminate_integer = Int(terminate)
        terminate = !iszero(MPI.Allreduce!(Ref(terminate_integer), +, mpi_comm())[])
    end
    if mpi_isroot() && terminate
        @info "  Steady state tolerance/Iterations reached" steady_state_callback t
    end
    return terminate
end

end # @muladd