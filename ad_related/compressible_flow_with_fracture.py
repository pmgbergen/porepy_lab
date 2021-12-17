"""
Solve a compressibe flow problem in an unfractured domain, using AD.
"""

"""

"""

import porepy as pp
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

#%% Callables

def rho(p):
    """
    Constitutive relationship between density, rho, and pressure, p
    """
    c = 1e-5 # If this is 0 (i.e. incompressible flow), 
             # but the ad-equations are interpret as compressible, 
             # the Jacobin becomes singular.
    p_ref = 1e0
    rho_ref = 1e0
    
    if isinstance(p, np.ndarray):
        rho = rho_ref * np.exp(c * (p-p_ref))
    else:
        rho = rho_ref * pp.ad.exp(c * (p-p_ref))
    return rho

def update_darcy(gb, dof_manager):
    """
    Update the darcy flux
    """
    
    # Get the Ad-fluxes
    gb_2d = gb.grids_of_dimension(gb.dim_max())[0]
    data = gb.node_props(gb_2d)
    full_flux = data[pp.PARAMETERS]["flow"]["AD_flux"]
    
    # Convert to numerical values
    num_flux = pp.ad.Expression(full_flux, dof_manager).to_ad(gb).val 
    
    # Get the signs
    sign_flux = np.sign(num_flux)
    
    # Finally, loop over the gb and return the signs of the darcy fluxes
    val = 0
    for g,d in gb:
        inds = np.arange(val, val+g.num_faces) 
        d[pp.PARAMETERS]["flow"]["darcy_flux"] = np.abs(sign_flux[inds])
        val += g.num_faces
    # end g,d-loop
    
    if gb.dim_max() > gb.dim_min():
        
        # the flux
        edge_flux = data[pp.PARAMETERS]["flow"]["AD_lam_flux"]
        num_edge_flux = pp.ad.Expression(edge_flux, dof_manager).to_ad(gb).val
        sign_flux = np.sign(num_edge_flux) 
        
        val = 0
        for e,d in gb.edges():
            inds = np.arange(val, val+d["mortar_grid"].num_cells)
            d[pp.PARAMETERS]["flow"]["darcy_flux"] = sign_flux[inds]
            val += d["mortar_grid"].num_cells
        # end e,d-loop
    
    return

def abs_ad(v, gb, dof_manager, matrix=False):
    """Return the absolute value of an Ad-vector v"""
    num_v = pp.ad.Expression(v, dof_manager).to_ad(gb).val
    if matrix is False :
       abs_v = np.abs(num_v)
       ad_abs = pp.ad.Array(abs_v)
    
    else:
       abs_v = sps.dia_matrix((np.abs(num_v), 0), 
                              shape = (gb.num_mortar_cells(),  gb.num_mortar_cells()))
       ad_abs = pp.ad.Matrix(abs_v)
    
    return ad_abs

#%% Define a grid and initialize parameters


# Cartesian grid, and wrap it in a GB
# g = pp.CartGrid([11,11])
# g.compute_geometry()
# gb = pp.GridBucket() 
# gb.add_nodes(g)

frac_channel = np.array([
    [3, 3],
    [0, 3] 
    ]) # The fracture is from 0.2 to 0.7 in y-direction and fixed at y=0.5
gb = pp.meshing.cart_grid([frac_channel], nx=[6, 6])

# String variables
pressure_variable = "pressure"
mortar_variable = "mortar_variable"
parameter_keyword = "flow" # Parameter keyword

# Loop over the gb, define cell centered variables,
# provide some initial values and problem data
for g, d in gb:
    
    d[pp.PRIMARY_VARIABLES] = {pressure_variable: {"cells": 1}}
    pp.set_state(d)
    # Important: Think about the initial values of the pressure
    # and the boundary values. If they are "not in accordance"  
    # we may not have the left-to-right flow
    d[pp.STATE] = {pressure_variable : 1 * np.ones(g.num_cells), 
                   pp.ITERATE :{ pressure_variable: 1 * np.ones(g.num_cells) }}
    
    if g.dim == 2:
        K = 1e0 * np.ones(g.num_cells)
        perm = pp.SecondOrderTensor(K)
        
        boundary_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        boundary_face_centers = g.face_centers[:, boundary_faces]
        
        # Right and left side of the domain
        left =  np.abs(boundary_face_centers[0] - gb.bounding_box()[0][0]) < 1e-4
        right = np.abs(boundary_face_centers[0] - gb.bounding_box()[1][0]) < 1e-4
        
        # Labels
        bc_cond = np.array(['neu'] * boundary_faces.size) 
        bc_cond[left] = "dir"
        bc_cond[right] = "dir"
        bc = pp.BoundaryCondition(g, boundary_faces, bc_cond)
        
        bc_val = np.zeros(g.num_faces)
        bc_val[boundary_faces[left]] = 1 #-1e1 * g.face_areas[boundary_faces[left]]
        bc_val[boundary_faces[right]] = 0
        
        # Source function S. 
        # We assume it is constants, so that the finite-volume approximation becomes
        # \int_Omega S dx = S * vol(Omega_i)
        S = np.zeros(g.num_cells) 
        S[12] = 0
        
        init_darcy_flux = np.zeros(g.num_faces)
        #init_darcy_flux[np.arange(0, int(g.num_faces/2))] = 1
        
        specified_data = {"second_order_tensor": perm,
                          "permeability": K,
                          "bc": bc,
                          "bc_values": bc_val,
                          "mass_weight": 0.2 * np.ones(g.num_cells),
                          "source": S * g.cell_volumes,
                          "darcy_flux": init_darcy_flux,
                          "time_step": 0.02}
        
        d = pp.initialize_data(g, d, parameter_keyword, specified_data)
    
    else:
        
        specific_vol = 1e-3 
        
        K = 1e2 * np.ones(g.num_cells) * specific_vol
        perm = pp.SecondOrderTensor(K)
        
        # In the lower dimansions, no-flow (zero Neumann) conditions
        bc_val = np.zeros(g.num_faces)
        bc = pp.BoundaryCondition(g)        
        
        # No sources
        S= np.zeros(g.num_cells)
        
        specified_data = {"second_order_tensor": perm,
                          "permeability": K,
                          "bc": bc,
                          "bc_values": bc_val,
                          "mass_weight": 1.0 * specific_vol * np.ones(g.num_cells), # scaled porosiy by specific volume
                          "source": S * g.cell_volumes,
                          "darcy_flux": np.zeros(g.num_faces)}
        d = pp.initialize_data(g, d, parameter_keyword, specified_data)
    # end if
    
# end g,d-loop

for e,d in gb.edges():
    
    # Initialize the primary variables and the state in the dictionary
    d[pp.PRIMARY_VARIABLES] = {mortar_variable: {"cells": 1}}    
    pp.set_state(d)
    
    # Number of mortar cells
    mg_num_cells = d["mortar_grid"].num_cells
    # State variables
    d[pp.STATE].update({
        mortar_variable: 1 * np.ones(mg_num_cells),
        pp.ITERATE: {
            mortar_variable: 1 * np.ones(mg_num_cells),
          } 
    })
    
    nd = 1e2 * np.ones(mg_num_cells) 
    d= pp.initialize_data(e, d, parameter_keyword, 
                          {"normal_diffusivity": nd, 
                           "darcy_flux": np.ones(mg_num_cells),
                              })
    
# end e,d-loop

#%% Residual equations

def equation(gb, dof_manager, equation_manager, iterate = False):
    
    data = gb.node_props(gb.grids_of_dimension(2)[0])

    grid_list = [g for g,_ in gb ]
    edge_list = [e for e,_ in gb.edges()]
    
    div = pp.ad.Divergence(grid_list) # The divergence 
    
    bound_ad = pp.ad.BoundaryCondition(parameter_keyword, grids=grid_list)# Boundary condionts
    
    # Wrap the density-pressure function into AD
    rho_ad = pp.ad.Function(rho, "")
    
    # Pressure, in AD-sense
    p = equation_manager.merge_variables([(g, pressure_variable) for g in grid_list])
    if len(edge_list) > 0:
        lam = equation_manager.merge_variables([(e, mortar_variable) for e in edge_list])
    # end if
    
    mpfa = pp.ad.MpfaAd(keyword=parameter_keyword, grids=grid_list) # AD version of mpfa 
    upwind = pp.ad.UpwindAd(keyword=parameter_keyword, grids=grid_list)  
    mass = pp.ad.MassMatrixAd(keyword=parameter_keyword, grids=grid_list)
    
    source = pp.ad.ParameterArray(parameter_keyword, "source", grid_list)
    
    rho_on_face = (
        upwind.upwind * rho_ad(p) 
        + upwind.rhs * rho_ad(bound_ad) 
        + upwind.outflow_neumann * rho_ad(p)
        ) 
    
    interior_flux = mpfa.flux * p # the flux on the subdomains
    bound_flux = mpfa.bound_flux * bound_ad
    
    flux = interior_flux + bound_flux
    full_flux = rho_on_face * flux # The full flux, weighted by densities 
      
    data[pp.PARAMETERS][parameter_keyword].update({"AD_flux": flux})
    if len(edge_list) > 0:
        data[pp.PARAMETERS][parameter_keyword].update({"AD_lam_flux": lam})
    # end if
         
    # Add the fluxes from the interface to higher dimension,
    # the source accounting for fluxes to the lower dimansions
    # and multiply by the divergence
    if len(edge_list) > 0:
        
        mortar_projection = pp.ad.MortarProjections(gb, grids=grid_list, edges=edge_list)
        
        # The boundary term
        full_flux += mpfa.bound_flux * mortar_projection.mortar_to_primary_int * lam
        
        # Tools to include the density in the source term 
        upwind_coupling_weight = pp.ad.UpwindCouplingAd(keyword=parameter_keyword, edges=edge_list)
        trace = pp.ad.Trace(gb, grid_list)
              
        up_weight_flux = upwind_coupling_weight.flux
        up_weight_primary = upwind_coupling_weight.upwind_primary
        up_weight_secondary = upwind_coupling_weight.upwind_secondary
      
        abs_lam = abs_ad(lam, gb, dof_manager, matrix=True)
        
        # Project the density for \Omega_h and \Omega_l to the interface
        high_to_low = ( 
            #abs_lam * 
            #up_weight_flux *
            up_weight_primary * 
            mortar_projection.primary_to_mortar_int *
            trace.trace * rho_ad(p)
            )
        
        low_to_high = (
            #abs_lam * 
            #up_weight_flux * 
            up_weight_secondary * 
            mortar_projection.secondary_to_mortar_int * 
            rho_ad(p) 
            ) 
    
        # The source term
        sources_from_mortar = (
            mortar_projection.mortar_to_secondary_int * ( 
                (high_to_low + low_to_high) * lam
        ) 
            ) 
        
        conservation = div * full_flux - sources_from_mortar
        
    else :
        conservation = div * full_flux 
    # end if
    
    # AD-equation
    
    if iterate:
        p_prev = data[pp.PARAMETERS][parameter_keyword]["prev_p"]
    else:
        p_prev = p.previous_timestep()
        data[pp.PARAMETERS][parameter_keyword].update({"prev_p": p_prev})
    # end if

    dt = data[pp.PARAMETERS][parameter_keyword]["time_step"]
    density_eq = (
        mass.mass * (rho_ad(p) - rho_ad(p_prev)) / dt
        + conservation
        - source
        )
    
    if len(edge_list)>0:
                 
        pressure_trace_from_high = (
            mortar_projection.primary_to_mortar_avg * mpfa.bound_pressure_cell * p
            + (
            mortar_projection.primary_to_mortar_avg * mpfa.bound_pressure_face *
               mortar_projection.mortar_to_primary_int * lam)
            )
  
        pressure_from_low = mortar_projection.secondary_to_mortar_avg * p   
    
        robin = pp.ad.RobinCouplingAd(parameter_keyword, edge_list)
  
        # The interface flux, lambda
        interface_flux = (
            lam + 
            robin.mortar_scaling * ( 
               pressure_from_low - pressure_trace_from_high
             ) 
      )
  
    # end if
    
    if len(edge_list) > 0:
    
         eq_1 = pp.ad.Expression(density_eq, dof_manager) 
         eq_2 = pp.ad.Expression(interface_flux, dof_manager)
         
         eq_1.discretize(gb)
         eq_2.discretize(gb)
         equation_manager.equations = [eq_1, eq_2]
         
    else:
        eqs = pp.ad.Expression(density_eq, dof_manager)
        eqs.discretize(gb)
        equation_manager.equations = [eqs]
    # end if
    
    update_darcy(gb, dof_manager)

    return equation_manager


#%%  Advance forward in time
# Final time and number of steps
data = gb.node_props(gb.grids_of_dimension(2)[0])
#data_1d=gb.node_props(gb.grids_of_dimension(1)[0])
T_end = 0.2
dt = data[pp.PARAMETERS]["flow"]["time_step"]
n_steps = int(T_end/dt)

# dof_manager and equation manager
dof_manager = pp.DofManager(gb)
equation_manager = pp.ad.EquationManager(gb, dof_manager)


#%% Time-loop
for i in range(10):
      
    equation_manager = equation(gb, dof_manager, equation_manager, False)

     # Get the Jacobian and the rhs
    A, b = equation_manager.assemble_matrix_rhs()

    iter = 0
    init_res = np.linalg.norm(b)
    #res = 1 + init_res
    res = np.linalg.norm(b)
    
    # Run newton. Note, using relative residual error caused the problem
    # of both init_res and res become, say, 1e-13, but their ration is 1
    while (res > 1e-10 and iter < 15) :
        
        # prev sol
        x_prev = dof_manager.assemble_variable(from_iterate=True)
        
        # Solve for pressure
        x = spla.spsolve(A, b)
        
        # updated sol
        x_new = x + x_prev
        
        # Distribute the solution, in an additive manner
        dof_manager.distribute_variable(values=x_new, 
                                        additive=False, to_iterate=True)  
        
        # Rediscretize the equations.
        # Mainly to update the Darcy flux
        equation_manager = equation(gb, dof_manager, equation_manager, True)
        
        A,b = equation_manager.assemble_matrix_rhs()
        
        res = np.linalg.norm(b)
        iter += 1
        
    # end while
    
    # Distribute the solution to the next time step
    x_new = dof_manager.assemble_variable(from_iterate=True)
    dof_manager.distribute_variable(x_new.copy(), to_iterate=False, additive=False)
    
    pp.plot_grid(gb, pressure_variable, figsize=(15,12))
    print(res)

 
    print(f"time step {i}")
    print(f"newton iters {iter}")    
# end i-loop
   
