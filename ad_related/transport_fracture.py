#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 08:31:36 2021

Test script for transport in a domain with a fracture.

IMPORTANT
To be able to run this script, you will need the grid_operator.py and 
hyperbolic_interface_law.py that can be found at my forked porepy_lab.

@author: shin.banshoya@uib.no
"""

#%% Get the neseccary modules
import porepy as pp
import numpy as np
import scipy.sparse as scs
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

#%% The problem data

# Flow
def flow_in_gb(gb, domain_size):
    
    gb.add_node_props(["param", "is_tangential"])
    tol = 1e-5
    a = 1e-4
    
    # The permeability in the fracture, matrix and at the interface
    fracture_permeability = 1e2
    matrix_permeability = 1
    interface_permeability = np.array([1e2])
    
    for g, d in gb:
        
        # Assign aperture. For now we keep the aperture fixed; for the not to far away future, 
        # we will use some thing like described Steefel and Lichtner for changes in aperture 
        a_dim = np.power(a, gb.dim_max() - g.dim)
        aperture = np.ones(g.num_cells) * a_dim
          
        if g.dim == gb.dim_max(): #  we're in 2D
        
            # Set the permeability
            K = np.ones(g.num_cells)
            perm = pp.SecondOrderTensor(kxx=matrix_permeability* K, 
                                        kyy=matrix_permeability* K, 
                                        kzz=1)
      
            # Set boundary condtions. Neumann inflow on the left, Dirichlet outflow on the right
            bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]  
            bound_faces_centers = g.face_centers[:, bound_faces]
             
            # Define the inflow and outflow in the domain
            inflow = bound_faces_centers[0] < domain_size["xmin"] + tol
            outflow = bound_faces_centers[0] > domain_size["xmax"] - tol
            
            # The BC labels
            labels = np.array(["neu"] * bound_faces.size)
            labels[outflow] = "dir"
            #labels[inflow] = "dir"
            
            # Set the BC values
            bc_values = np.zeros(g.num_faces) 
            bc_values[bound_faces[inflow]] = -1 * g.face_areas[bound_faces[inflow]]
            bc_values[bound_faces[outflow]] = 0 
            
            bound = pp.BoundaryCondition(g, faces=bound_faces, cond=labels)
            
            # In order to include the BC info in the dictionary
            specified_param = {"bc_values": bc_values,
                               "bc": bound,
                               "second_order_tensor": perm}
             
            d = pp.initialize_data(g, d, "flow", specified_param)
            
            
        else: # 1 and 0 D
            
            # The permerability
            K = np.ones(g.num_cells) * np.power(fracture_permeability, g.dim < gb.dim_max() ) * aperture
            perm = pp.SecondOrderTensor(fracture_permeability*K)
            
            # No-flow (zero Neumann) conditions
            bc_values = np.zeros(g.num_faces)
            bound = pp.BoundaryCondition(g) 
                 
            # In order to include the BC info in the dictionary
            specified_param = {"bc_values": bc_values,
                               "bc": bound,
                               "second_order_tensor": perm}
             
            d = pp.initialize_data(g, d, "flow", specified_param)
            
        # end if-else
        
        d["is_tangential"] = True
        
    # end g,d-loop
    
    # Finally, look at the interfaces
    for e,d in gb.edges():
        mg = d["mortar_grid"]
        kn = interface_permeability[0] * np.ones(mg.num_cells)
        d[pp.PARAMETERS] = pp.Parameters( mg, ["flow"], [ {"normal_diffusivity": kn} ])
        d[pp.DISCRETIZATION_MATRICES] = {"flow": {}}
    # end e,d loop
    
    return gb

# Transport
def transport_in_gb(gb, num_aq_components, domain, parameter_keyword):
    
    # Loop over the gb
    for g,d in gb:
        
        unity = np.ones(g.num_cells)
        
        if g.dim == 2:
                  
            # Get the number of cells used to discretize in the y-direction. 
            # We need this for the boundary conditions, atleast in 2D
            Ny = g.cart_dims[1]
            
            # Boundary conditions for the advective problem
            b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
            b_face_centers = g.face_centers[:, b_faces]
            labels = np.array( ["neu"]*b_faces.size )
           
            # Dirichlet inflow at left. Zero-Neumann on the rest
            left = np.ravel( np.argwhere( b_face_centers[0, :] < domain["xmin"] + 1e-4 ) )
            right = np.ravel( np.argwhere( b_face_centers[0, :] > domain["xmax"] - 1e-4  ) )
            labels[left] = "dir"
            #labels[right] = "dir"
            bc = pp.BoundaryCondition(g, b_faces, labels)
           
            # Expand bc indicies
            expanded_left = pp.fvutils.expand_indices_nd(b_faces[left], num_aq_components)
                       
            # Set bc values. At each face, we have num_aq_components bc values. 
            extended_bc_values = np.zeros(g.num_faces * num_aq_components)
            extended_bc_values[expanded_left] = np.tile( np.ones(num_aq_components) , Ny)
            #np.array([0.3, 0.3, 0.3, 0])
            
            # Set the porosity, the values are as in the tracer tutorial
            porosity = 0.2 * unity
            aperture = 1
            
            specified_paramters = {"bc": bc,
                                   "bc_values": extended_bc_values,
                                   #"mass_weight": porosity * aperture, 
                                   "num_components": num_aq_components
                                   }
            
            mass_data = { "num_components": num_components,
                         "mass_weight": porosity*aperture                
                }
        else:
            
            # Boundary conditions betweem the fracture and the matrix for the transport.
            # It should be a continuity condition: conc(at fracture) = conc(at matrix) 
            b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
            b_face_centers = g.face_centers[:, b_faces]
            labels = np.array( ["neu"]*b_faces.size )
           
            bc = pp.BoundaryCondition(g)
            porosity = 0.8 * unity 
            aperture = np.power(1e-4, gb.dim_max() - g.dim)
            
            specified_paramters = {"bc": bc,
                                   "bc_values": np.zeros(g.num_faces * num_aq_components),
                                   #"mass_weight": porosity * aperture, 
                                   "num_components": num_aq_components
                                   }
            mass_data = { "num_components": num_components,
                         "mass_weight": porosity*aperture                
                }
            
        # end if-else
        
        # Initialize parameters
        pp.initialize_data(g, d, parameter_keyword , specified_paramters)
        pp.initialize_data(g, d, "mass" , mass_data)
        
        # For visualizations purposes later on
        d[pp.STATE] = {"dimension": g.dim * np.ones(g.num_cells)}
        
        # Finally, loop over the interfaces
        for e,d in gb.edges():
            d[pp.PARAMETERS].update_dictionaries(parameter_keyword,
                                                 {"num_components": num_aq_components } )
            d[pp.DISCRETIZATION_MATRICES][parameter_keyword] = {}
        # end e,d-loop
    
    return gb

#%% Create a fracture and a gb
frac = np.array( [ [1, 1], [0.1, 0.9]   ]) # Changed from[0.75, 0.75], [0.0, 0.9]
frac2 = np.array( [[1.5, 1.5], [0.5, 1] ])
fracs=[frac, ]
for f in fracs:
    is_x_frac = f[1, 0] == f[1, 1]
    is_y_frac = f[0, 0] == f[0, 1]
# end f-loop

gb = pp.meshing.cart_grid( fracs  , 
                          nx=[10,10], 
                          physdims=np.array( [2 , 1 ]) 
                          )
# pp.plot_grid(gb, figsize=(15,12))

# g = pp.CartGrid([5,5])
# g.compute_geometry()
# gb = pp.GridBucket() 
# gb.add_nodes(g)

# Keywords
kw_f = "flow"
kw_t = "transport"

# Set flow data
domain = {"xmin":0 , "xmax": 2, "ymin": 0, "ymax": 1}
gb = flow_in_gb(gb, domain)
# Set transport data
aq_components = np.array([0,1])
num_aq_components = aq_components.size
num_components = num_aq_components + 0 # one fixed, one aqueous
gb = transport_in_gb(gb, num_aq_components, domain, kw_t)
gb_2d = gb.grids_of_dimension(2)[0]

#%% Set the sting variables for the flow problem and initialize them on the gb

grid_variable = "pressure"
mortar_variable = "mortar_variable"

np.random.seed(1234)

# Loop over the gb and its edges to define the cell variables and set some initial values
for g,d in gb:
    d[pp.PRIMARY_VARIABLES] ={grid_variable: {"cells": 1}}
    pp.set_state(d)
    d[pp.STATE][grid_variable] = np.ones( g.num_cells )
# end g,d-loop

for e,d in gb.edges():
    d[pp.PRIMARY_VARIABLES] = {mortar_variable: {"cells" :1} }
    pp.set_state(d)
    d[pp.STATE][mortar_variable] = np.ones( d["mortar_grid"].num_cells ) 
# end e,d-loop


#%% Define grid-related operators

# Lists of grids and edges
grid_list = [g for g, _ in gb ]
edge_list = [e for e, _ in gb.edges() ]   

# The divergence 
div = pp.ad.Divergence(grid_list)

# projection between subdomains and mortar grids
if len(edge_list) > 0:
    mortar_projection = pp.ad.MortarProjections(gb=gb, nd=1)
# end if

# Boundary condionts
bound_ad = pp.ad.BoundaryCondition(kw_f, grids=grid_list)

#%% Set an dof and equation manager to keep track of the equations

dof_manager = pp.DofManager(gb)
equation_manager = pp.ad.EquationManager(gb, dof_manager)

p  = equation_manager.merge_variables( [(g, grid_variable) for g in grid_list ])

if len(edge_list) > 0:
    lam = equation_manager.merge_variables([(e, mortar_variable) for e in edge_list ])
# end if

#%% Discretize the flow problem

# AD version of mpfa 
mpfa = pp.ad.MpfaAd(kw_f, grids=grid_list) 

interior_flux = mpfa.flux * p # the flux on the subdomains

eval_flux = pp.ad.Expression(interior_flux, dof_manager)
eval_flux.discretize(gb) # "standard" discretization
num_flux = eval_flux.to_ad(gb) # Get the numerical values of the flux

# The full flux
full_flux = interior_flux + mpfa.bound_flux * bound_ad 

# Project flux from mortar grids onto higher-dimensional grids
if len(edge_list) > 0:
    full_flux += mpfa.bound_flux * mortar_projection.mortar_to_primary_int * lam 
# end if

# Numerical values of the full flux
vals = pp.ad.Expression(full_flux, dof_manager).to_ad(gb)

if len(edge_list) > 0:
    
    sources_from_mortar = mortar_projection.mortar_to_secondary_int * lam
    
    # Mass conservation, with values from mortars as a source term 
    # \div q - \Xi lamda = 0; compare with the second equation in Eq. (3.4)
    # in th PP-article
    conservation = div * full_flux - sources_from_mortar
else:
    # "Classical" mass conservation, ie. div q
    conservation = div * full_flux
# end if-else    

# The flux of the interfaces
if len(edge_list) > 0:
    
    # The pressure from higher to lower dimensions 
    pressure_trace_from_high = ( 
        mortar_projection.primary_to_mortar_avg * mpfa.bound_pressure_cell * p 
        + ( 
        mortar_projection.primary_to_mortar_avg * mpfa.bound_pressure_face * 
        mortar_projection.mortar_to_primary_int * lam
        )
    )
    
    # Coupling term
    robin = pp.ad.RobinCouplingAd(kw_f, edge_list)

    # Flux over the interface. This is Eq. (3.3) in the PP-article 
    interface_flux_eq = ( 
        robin.mortar_scaling *  (
            pressure_trace_from_high - mortar_projection.secondary_to_mortar_avg * p 
            )
        - robin.mortar_discr * lam
        )
# end if        
        
#%% Give the flow problem as an equation to the AD-machinery

if len(edge_list) > 0:
    eqs = [pp.ad.Expression(conservation, dof_manager), 
           pp.ad.Expression(interface_flux_eq, dof_manager) 
           ]
else:
    eqs =[pp.ad.Expression(conservation, dof_manager)]
# end if-else

equation_manager.equations += eqs

# Discretize, assemble and solve for the pressure
equation_manager.discretize(gb)
A, b = equation_manager.assemble_matrix_rhs()
solution = spla.spsolve(A,b) # Pressure

# Distribute variable (ie. the solution) to local data dictionaries
dof_manager.distribute_variable(solution, )

# Plot the pressure distribution
#pp.plot_grid(gb, grid_variable, figsize=(15,12))

# Compute the Darcy flux
pp.fvutils.compute_darcy_flux(gb, 
                              keyword_store=kw_t, 
                              lam_name=mortar_variable
                              )

#%% Now, lets look at the transport problem

# Initialize the string variables, define them on the gb 
# and provide some initial values
grid_transport = "tracer"
grid_fixed = "tracer2"
mortar_transport = "mortar_tracer"
mortar_fixed = "mortar_fixed"
plot_aqueous_part = "aqueous"


# Loop over the grids and edges
# At each cell, we have num_components 
# In this code, the number of components and aqueous components are the same
for g,d in gb:
    d[pp.PRIMARY_VARIABLES] = {grid_transport: {"cells": num_components }, 
                               #grid_fixed: {"cells": 1}
                               }
    pp.set_state(d)
    
    IC1 = np.zeros(g.num_cells)
    IC2 = np.zeros(g.num_cells) if num_components > 1 else None
    
    combined_IC = np.zeros(IC1.size+IC2.size) if num_components > 1 else np.zeros(IC1.size)
    
    if num_components > 1: 
        combined_IC[::2] = IC1
        combined_IC[1::2] = IC2 
    else:
        combined_IC = IC1.copy()
    # end if-else
    
    d[pp.STATE][grid_transport] = combined_IC
# end g,d-loop

# Only for the aqueous components    
for e,d in gb.edges():
    
    d[pp.PRIMARY_VARIABLES] = {mortar_transport: {"cells": num_aq_components },
                              # mortar_fixed: {"cells": 1}
                                }
    pp.set_state(d)
    
    IC1 = np.zeros(d["mortar_grid"].num_cells)
    IC2 = np.zeros(d["mortar_grid"].num_cells) if num_aq_components > 1 else None
    
    combined_IC = np.zeros(IC1.size+IC2.size) if num_aq_components > 1 else np.zeros(IC1.size)
    
    if num_aq_components > 1: 
        combined_IC[::2] = IC1
        combined_IC[1::2] = IC2 
    else:
        combined_IC = IC1.copy()
    # end if-else
    
    d[pp.STATE][mortar_transport] = combined_IC
# end e,d-loop


# The upwind, both for the grids and coupling, and mass discretization
upwind = pp.ad.UpwindAd(kw_t, grid_list)
upwind_coupling = pp.ad.UpwindCouplingAd(kw_t, edge_list)
mass = pp.ad.MassMatrixAd("mass", grid_list)

# The AD wrapper for the boundary conditions
bc_c = pp.ad.BoundaryCondition(keyword=kw_t, grids=grid_list) 

# Set the transport equation
# Start by defining an DOF manager and an equation manager
dof_manager_for_transport = pp.DofManager(gb)
equation_manager_for_transport = pp.ad.EquationManager(gb, dof_manager_for_transport)

# AD-variables
T = equation_manager_for_transport.merge_variables( 
    [ (g, grid_transport) for g in grid_list ],
    )

if len(edge_list) > 0:
    eta = equation_manager_for_transport.merge_variables(
        [ (e, mortar_transport) for e in edge_list ] 
        )
# end if

# Some variables we need to "redefine" from the flow problem,
# due to different sizes

# The divergence
div = pp.ad.Divergence(grid_list, dim=num_aq_components) 

# projection between subdomains and mortar grids
if len(edge_list) > 0:
    mortar_projection = pp.ad.MortarProjections(gb=gb, nd=num_components)
# end if

# Define the transport equation.
# We need: 
# the mass term
# Advection
# Boundary conditions
# A coupling term (if necessary)

dt = 0.001
T_prev = T.previous_timestep()

cols = np.ravel(
    aq_components.reshape((-1, 1)) + (num_components * np.arange(gb.num_cells() )), "F"
)
sz = num_aq_components * gb.num_cells()
rows = np.arange(sz)
matrix_vals = np.ones(sz)

# Mapping from all to aquatic components. The reverse map is achieved by a transpose.
all_2_aquatic = pp.ad.Matrix(
    scs.coo_matrix(
        (matrix_vals, (rows, cols)),
        shape=(num_aq_components * gb.num_cells(), num_components * gb.num_cells() ),
    ).tocsr()
)

transport = (
    mass.mass * (T-T_prev)/dt
    
    # Advection. 
    + all_2_aquatic.transpose() * div * upwind.upwind * all_2_aquatic * T
    - all_2_aquatic.transpose() * div * upwind.rhs * bc_c
    - all_2_aquatic.transpose() * div * upwind.outflow_neumann * all_2_aquatic * T # 
    )


if len(edge_list) > 0 :
    
    # The trace operator 
    trace = pp.ad.Trace(gb, grid_list, nd= num_components) 

    # # check numerical values
    # trace_expr = pp.ad.Expression(trace.trace, dof_manager_for_transport).to_ad(gb).A
    # #transport_val = pp.ad.Expression(eta, dof_manager_for_transport).to_ad(gb)
    # mortar_eta_val = pp.ad.Expression( mortar_projection.mortar_to_primary_int ,
    #                                   dof_manager_for_transport).to_ad(gb).A

    # Like earlier, we need a mapping between the aquoues and "all" spcecies
    # but now with mortar_cells
    cols2 = np.ravel(
    aq_components.reshape((-1, 1)) + 
    ( num_components * np.arange(gb.num_mortar_cells() )), "F" 
    )
    
    sz2 = num_aq_components * gb.num_mortar_cells()
    rows2 = np.arange(sz2)
    matrix_vals2 = np.ones(sz2)

    # Mapping from all to aquatic components. The reverse map is achieved by a transpose.
    all_2_aquatic2 = pp.ad.Matrix(
        scs.coo_matrix(
            (matrix_vals2, (rows2, cols2)),
            shape=(num_aq_components * gb.num_mortar_cells(), 
                   num_components * gb.num_mortar_cells() )
            ).tocsr()
        ) # size num*aq*comp *num_mortar_cells x num_comp * num_mortar_cells 
    
    # Add the projections from the mortar onto the the lower-dimensional grids
    # I.e. the term -\Xi * eta (see Eq 3.6) in the pp-article
    
    # Also, eta is only posed the aqueous components, so we need to map 
    # them to all species
    
    transport += (
        trace.inv_trace * mortar_projection.mortar_to_primary_int *
        all_2_aquatic2.transpose() *
        eta
        )
    
    transport -= (
        mortar_projection.mortar_to_secondary_int * 
        all_2_aquatic2.transpose() *
        eta
        )
# end if


# Transport over the interfaces
if len(edge_list) > 0:
        
    # Some the tools we need
    up_flux = upwind_coupling.flux
    up_primary = upwind_coupling.upwind_primary
    up_secondary = upwind_coupling.upwind_secondary

    # The upstream-type approach, based on the Darcy interface flux.
    
    # First we project the concentrations from higher onto lower dimensions. 
    # That means the interface flux (lambda) is positive.
    
    # We need to project the (trace of) concentration onto the lower dimensions 
    # Move the concentration T from the cell center to the edge (boundary), 
    # using the trace operator
    trace_of_conc = trace.trace * T
    
    # Project the concentration from a higher dim on a lower dim. 
    # Similar to line 181 in hyperbolic interface code 
    high_to_low = ( 
        #all_2_aquatic2.transpose() *
        up_flux * up_primary * # Scale values
        all_2_aquatic2 *
        mortar_projection.primary_to_mortar_avg * # Project onto mortar grid
        trace_of_conc  # The trace of concentrations
                ) 
   
    # Next, project lower dimension concentration onto higher dimension
    # In this case, the flux should be negative 
    
    # Line 187 in hyperbolic interface code
    low_to_high = (
        #all_2_aquatic2.transpose() *
        up_flux * up_secondary * # Scale
        all_2_aquatic2 *
        mortar_projection.secondary_to_mortar_avg * # Projection from lower to mortar grids
        T # The concentration
        )
    
    # The coupling term
    transport_over_interface = ( 
        #all_2_aquatic2.transpose() *
        upwind_coupling.mortar_discr *
        #all_2_aquatic2 * 
        eta 

        - (high_to_low + low_to_high ) 
        
        )
# end if

# Define the transport equation as an expression
if len(edge_list) > 0:
    transport_eq = [
        pp.ad.Expression(transport, dof_manager_for_transport),
        pp.ad.Expression(transport_over_interface, 
                         dof_manager_for_transport,)
        ]    
else:
    transport_eq = [ pp.ad.Expression(transport, dof_manager_for_transport) ]
# end if-else

# Give it to the equation manager
equation_manager_for_transport.equations += transport_eq

# Discretize the equation
equation_manager_for_transport.discretize(gb)

data_2d = gb.node_props(gb.grids_of_dimension(2)[0] )
#data_1d = gb.node_props(gb.grids_of_dimension(1)[0] )


A_tr,_=equation_manager_for_transport.assemble_matrix_rhs()
A_tr[np.abs(A_tr<1e-10)]=0
plt.spy(A_tr)


#%%  Advance forward in time
# Final time and number of steps
T_end = 0.25
n_steps = int(T_end/dt)

# Time-loop
for i in range(n_steps):
    
    # An instability occurs at time step 0.26/dt at the right boundary  
    # Most likely wrong BC
    
    # Get the Jacobian and the rhs
    A_tracer, b_tracer = equation_manager_for_transport.assemble_matrix_rhs()
    
    # Solve in order to advance forward in time.
    # Think of this as the "dt*stuff" in "sol = sol + dt*stuff"
    x = spla.spsolve(A_tracer, b_tracer)

    # Distribute (in an additive manner, ie update) the solution
    dof_manager_for_transport.distribute_variable(x, additive=True)  
    
# end i-loop


#%% Plot solutions 

# This cannot be an efficient way 
for _, d in gb:
    d[pp.STATE][plot_aqueous_part] = d[pp.STATE][grid_transport][0::2]
    
for _, d in gb.edges():
    d[pp.STATE][plot_aqueous_part] = d[pp.STATE][mortar_transport][0::2]



# Plot the solution (in a fractued domain). 
pp.plot_grid(gb, plot_aqueous_part, figsize=(15,12))  
#sol1 = data_2d[pp.STATE]["tracer"][0::2]
#pp.plot_grid(gb_2d, sol1, figsize=(15,12))


# Plot the solution (the aqueous component for a Cartesian 
# grid without fracture, hidden in a gb)
#pp.plot_grid(gb_2d, x[0:50:2], figsize=(15,12))  


#%%
# check solutions, if necessary
if num_aq_components > 1:
    sol1 = data_2d[pp.STATE]["tracer"][0::2]
    sol2 = data_2d[pp.STATE]["tracer"][1::2]
    sols_ok = np.abs(np.sum(sol1)-np.sum(sol2)) < 1e-10