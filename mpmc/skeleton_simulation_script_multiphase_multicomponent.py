"""This is a unfinished runscript that explores how to set up a simulation
for multiphase multicomponent systems (geochemistry to be included).

The implementation is an attempt to follow

    Kala and Voskov, Element balance formulation in reactive compositional flow
    and transport with parameterization technique, Comput. Geosci. 24, 609-624,
    2020.

However, in its current state this is surely not a full implementation of that
model. Moreover, somewhere in the process, I got derailed and kind of lost contact
with this script. Thus, chances are, it won't run as it is now, but it should
indicate how a multiphase multicomponent transport simulator can be implemented
in PorePy.

An attempt has been made to document properly, sorry for shortcomings.

Main contact person: Eirik.Keilegavlen@uib.no

"""
import numpy as np
import porepy as pp

import scipy.sparse as sps
import scipy.sparse.linalg as spla


# Define grid.
# NOTE: Only a single grid for now
g = pp.CartGrid([3])
g.compute_geometry()
gb = pp.GridBucket()
gb.add_nodes(g)
data = gb.node_props(g)


## Define variables
# The formulation follows the overall molar variable formulation.
# Note that we declear both overall mole fractions (BTW: I will likely abuse
# terminology in the comments, please be generous), saturations (phase mass
# fractions) and component phase molar fractions as variables, but we will end
# up solving a global system for only a subset. That is to say, variables here
# include both primary and secondary.

# Number of components and phases in the system. For now, no explicit conditions are
# set on which components can enter which phases - this should come from the
# equation of state or similar.
num_components = 2
num_fluid_phases = 2

# Pressure
primary_variables = {"p": {"cells": 1}}

# Total molar fraction of each component
primary_variables.update({f"z_{i}": {"cells": 1} for i in range(num_components)})

# Phase mass fractions (essentially saturations)
primary_variables.update({f"v_{i}": {"cells": 1} for i in range(num_fluid_phases)})

# Component phase molar fractions
# Note systematic naming convention: i is always component, j is phase.
for j in range(num_fluid_phases):
    for i in range(num_components):
        primary_variables.update({f"x_{i}_{j}": {"cells": 1}})

#######
# Set parameters
#######

## Set parameters for flow.
flow_keyword = "flow"
transport_keyword_0 = "transport_0"
transport_keyword_1 = "transport_1"
mass_keyword = "mass"

k = 100 * pp.MILLI * pp.DARCY * np.ones(g.num_cells)
perm = pp.SecondOrderTensor(k)


# Boundary conditions
inflow = np.array([0])
outflow = np.array([g.num_faces - 1])

faces = np.array([0, g.num_faces - 1])
cond = ["neu", "dir"]

bc = pp.BoundaryCondition(g, faces, cond)

bc_values = np.zeros(g.num_faces)

# Boundary conditions on flow:
# Inlet has rate of 0.2 m^3 / day
bc_values[inflow] = 0.2
# Outlet has a fixed pressure of 50 bar
bc_values[outflow] = 50

param_flow = {}
param_flow["second_order_tensor"] = perm
param_flow["bc"] = bc
param_flow["bc_values"] = bc_values

data[pp.PARAMETERS] = {flow_keyword: param_flow}

## Set parameters for transport
# One boundary condition per phase in this case, this will depend on the
# problem at hand.
for j in range(num_fluid_phases):
    bc_transport = pp.BoundaryCondition(g, faces, cond)
    bc_values_transport = np.zeros(g.num_faces)

    # Boundary condition on the left. Non-zero only for the
    # second component (CO2)
    if j == 1:
        bc_values_transport[0] = 1

    param_transport = {}
    param_transport["bc"] = bc_transport
    param_transport["bc_values"] = bc_values_transport
    param_transport["darcy_flux"] = np.ones(g.num_faces)
    key = locals()[f"transport_keyword_{j}"]
    data[pp.PARAMETERS][key] = param_transport

mass_param = {"mass_weight": np.ones(g.num_cells)}

data[pp.PARAMETERS][mass_keyword] = mass_param

data[pp.PRIMARY_VARIABLES] = primary_variables

####
# Initialization of variables. How to do this consistently between
# primary and secondary variables is still something of a mystery to me.

set_vals = {
    "z_0": 1 * np.ones(g.num_cells),
    "p": 95 * np.ones(g.num_cells),
    "v_0": np.ones(g.num_cells),
    "x_0_0": np.ones(g.num_cells),
}
# Note that both state and state-iterate must be set.
state = {pp.ITERATE: {}}
for prim_var in primary_variables:
    val = set_vals.get(prim_var, np.zeros(g.num_cells))
    state[prim_var] = val
    state[pp.ITERATE][prim_var] = val


data[pp.STATE] = state
data[pp.DISCRETIZATION_MATRICES] = {
    flow_keyword: {},
    transport_keyword_0: {},
    transport_keyword_1: {},
    mass_keyword: {},
}


#### Spatial discretizations, using the Ad framework.
grids = [g]

mpfa = pp.ad.MpfaAd(flow_keyword, grids=grids)

# One upwind discretization object for each phase.
upwind_0 = pp.ad.UpwindAd(transport_keyword_0, grids=grids)
upwind_1 = pp.ad.UpwindAd(transport_keyword_1, grids=grids)

# Mass matrix.
mass = pp.ad.MassMatrixAd(mass_keyword, grids=grids)

div = pp.ad.Divergence(grids)

## Initialize Dof and Equation managers. These are the main
# engines in the Ad framework (as it is now, more engines will be implemented in due course).
dof_manager = pp.DofManager(gb)
eq_manager = pp.ad.EquationManager(gb, dof_manager)

#### Equations are specified below

# Define Ad variable of all primary variables
for prim_var in primary_variables:
    locals()[prim_var] = eq_manager.variable(g, prim_var)

# Get the Ad-representation of various variables at the previous iteration
# and time step.
for i in range(num_components):
    # Overall molar fractions
    # Note the trick with locals here, this allows us to loop over variables
    # with names that includes an index
    z_i = locals()[f"z_{i}"]
    locals()[f"z_{i}_prev_time"] = z_i.previous_timestep()

# Pressure
p_prev_time = locals()["p"].previous_timestep()

for j in range(num_fluid_phases):
    # Saturations. These may be needed both at previous time step and previous iteration,
    # I got derailed before I understood in detail how to do this.
    v_j = locals()[f"v_{j}"]
    locals()[f"v_{j}_prev_iter"] = v_j.previous_iteration()
    locals()[f"v_{j}_prev_time"] = v_j.previous_iteration()

    for i in range(num_components):
        # Component phase fractions
        x_ij = locals()[f"x_{i}_{j}"]
        locals()[f"x_{i}_{j}_prev_iter"] = x_ij.previous_iteration()

### Constitutive laws.
# The data strucutres here are really ad hoc, we should come up with something
# much better.

# Densities for the two phases. The fluids are considered slightly incompressible,
# and there is no dependency in the density of composition, temperature etc.
rho_0 = pp.ad.Function(lambda p: 1000 * (1 + 1e-6 * (p - 95)), "Liquid density")(p)
rho_1 = pp.ad.Function(lambda p: 100 * (1 + 1e-4 * (p - 95)), "Gas density")(p)
# Densities as functions of the prevous time step. Needed to approximate an
# accumulation term.
rho_0_prev = pp.ad.Function(lambda p: 1000 * (1 + 1e-6 * (p - 95)), "Liquid density")(
    p_prev_time
)
rho_1_prev = pp.ad.Function(lambda p: 100 * (1 + 1e-4 * (p - 95)), "Gas density")(
    p_prev_time
)

# Saturation weighted total density, see Kala and Voskov for details.
rho_tot = 0
rho_tot_prev_time = 0
for j in range(num_fluid_phases):
    rho_tot += locals()[f"v_{j}"] / locals()[f"rho_{j}"]
    rho_tot_prev_time += locals()[f"v_{j}_prev_time"] / locals()[f"rho_{j}_prev"]

# Recover saturations from density-weighted quantities
S_0 = v_0 * rho_1 / (rho_0 + (rho_1 - rho_0) * v_0)
S_1 = pp.ad.Scalar(1) - S_0

# Relative permeabilities. Again, the data structure is really simple.
rel_perm_0 = pp.ad.Function(lambda x: x ** 2, "Rel. perm. liquid")(S_0)
rel_perm_1 = pp.ad.Function(lambda x: x ** 2, "Rel. perm. liquid")(S_1)


### Finally, Set up equations

# The overall components should sum to 1
resid_component_sum = pp.ad.Array(np.ones(g.num_cells))
for i in range(num_components):
    z = locals()[f"z_{i}"]
    resid_component_sum -= z
# eq_manager.equations["resid_component_sum"] = resid_component_sum

# Components in phases should sum to full component
for i in range(num_components):
    z = locals()[f"z_{i}"]
    resid = z

    for j in range(num_fluid_phases):
        resid -= locals()[f"x_{i}_{j}"] * locals()[f"v_{j}"]
    locals()[f"resid_z_{i}"] = resid
    name = f"Total_component_{i}"
    eq_manager.equations[name] = resid

# Volume constraint on phases
denominator = 0
for j in range(num_fluid_phases):
    denominator += locals()[f"S_{j}"] * locals()[f"rho_{j}"]

volume_constraint = pp.ad.Array(np.ones(g.num_cells))
for j in range(num_fluid_phases):
    v, s, rho = [globals()[key] for key in [f"v_{j}", f"S_{j}", f"rho_{j}"]]

    name = f"Definition_v_{j}"
    #    eq_manager.equations[name] = v - s * rho / denominator

    volume_constraint -= v

eq_manager.equations["Volume_constraint"] = volume_constraint

### NOTE: Below follows a confused mess that was my attempt at solving the phase
# equilibrium problem, before I realized I had to do a flash calculation.
# There are attempts at using KKT-conditions, in the style of Lauser, below,
# but I'm not sure this is what we want to do. Don't spend time on trying to
# understand the logic here, I'm pretty sure there is none.

# Constraint on components distribution on all phases
for j in range(num_fluid_phases):
    sum_ij = sum([globals()[f"x_{i}_{j}"] for i in range(num_components)])
    sum_ij_prev = sum(
        [globals()[f"x_{i}_{j}_prev_iter"] for i in range(num_components)]
    )
    arr = pp.ad.Array(np.ones(g.num_cells))
    eq = (arr - sum_ij) * locals()[f"v_{j}_prev_iter"] + (arr - sum_ij_prev) * locals()[
        f"v_{j}"
    ]

    name = f"KKT_phase_{j}"
    eq_manager.equations[name] = eq


def set_KKT(eq_man):
    for j in range(num_fluid_phases):
        sum_ij_prev = sum(
            [globals()[f"x_{i}_{j}_prev_iter"] for i in range(num_components)]
        )
        sat_prev = globals()[f"v_{j}_prev_iter"]
        active = (
            sat_prev - (pp.ad.Array(np.ones(g.num_cells)) - sum_ij_prev)
        ).evaluate(dof_manager) > 0
        sz = active.size
        active_mat = pp.ad.Matrix(
            sps.dia_matrix((active.astype(int), 0), shape=(sz, sz))
        )
        inactive_mat = pp.ad.Matrix(
            sps.dia_matrix((np.logical_not(active).astype(int), 0), shape=(sz, sz))
        )
        arr = pp.ad.Array(np.ones(g.num_cells))
        sum_ij = arr - sum([globals()[f"x_{i}_{j}"] for i in range(num_components)])

        sat = globals()[f"v_{j}"]

        eq = active_mat * sum_ij + inactive_mat * sat

        eq_man.equations[f"KKT_phase_{j}"] = eq


bc_ad = pp.ad.BoundaryCondition(flow_keyword, grids, gb)

total_flux = 0

cols = np.tile(np.arange(g.num_faces), num_components)
rows = np.arange(g.num_faces * num_components)
vals = np.ones(g.num_faces * num_components)

phase_to_component = pp.ad.Matrix(sps.coo_matrix((vals, (rows, cols))).tocsc())


# Phase equilibrium equations
# Set K-values for the equilibrium
K_0 = 0.1
K_1 = 10

# Phase equilibrium
eq_manager.equations["Phase_equilibrium_component_0"] = x_0_0 - K_0 * x_0_1
eq_manager.equations["Phase_equilibrium_component_1"] = x_1_0 - K_1 * x_1_1

# Transport equations
for i in range(num_components):
    locals()[f"flux_{i}"] = 0

# Single phase darcy - not yet scaled with relative permeability
darcy = mpfa.flux * p + mpfa.bound_flux * bc_ad

for j in range(num_fluid_phases):
    # Phase-wise darcy by rel perm scaling
    rp = locals()[f"rel_perm_{j}"]
    upwind = locals()[f"upwind_{j}"]
    darcy_j = (upwind.upwind * rp) * darcy

    # phase denisty
    rho_j = locals()[f"rho_{j}"]

    param_key = upwind.keyword

    for i in range(num_components):
        # Advective flux of component i in phase j
        xij = locals()[f"x_{i}_{j}"]
        locals()[f"flux_{i}"] += darcy_j * (upwind.upwind * (rho_j * xij))


component_mass_balance = []

# Set the full conservation equation for the overall components
dt = 0.01
for i in range(num_components):
    # Boundary conditions
    bc = pp.ad.ParameterArray(
        param_keyword=upwind.keyword, array_keyword="bc_values", grids=[g]
    )
    # The advective flux is the sum of the internal (computed in flux_{i} above)
    # and the boundary condition
    adv_flux = locals()[f"flux_{i}"] + upwind.bound_transport * bc

    # accumulation term
    accum = (
        mass.mass
        * (
            locals()[f"z_{i}"] / rho_tot
            - locals()[f"z_{i}_prev_time"] / rho_tot_prev_time
        )
        / dt
    )

    # Append to set of conservation equations
    component_mass_balance.append(accum + div * adv_flux)

for i, eq in enumerate(component_mass_balance):
    # Discretize equations, add to EquationManager
    eq.discretize(gb)
    name = f"Mass_balance_component_{i}"
    eq_manager.equations[name] = eq

## We are now done with specifying equations. Next step is to prepare for solving them.
# The equations and variables can be divided into primary and secondary, where the latter
# can be solved locally in each cell.
secondary_variable_names = [
    v for v in eq_manager.variables[g] if v[0] not in ["p", "z"]
]
secondary_variable_names.append("z_1")

secondary_variables = [
    eq_manager.variable(g, name) for name in secondary_variable_names
]

eq_names = [
    name for name in list(eq_manager.equations.keys()) if name[:12] != "Mass_balance"
]
# The secondary equation manager has only secondary variables and equations as active.
# The state of the primary variables will be frozen.
secondary_manager = eq_manager.subsystem_equation_manager(eq_names, secondary_variables)


def prolongation_matrix(variables):
    # Construct mappings from subsects of variables to the full set.
    nrows = dof_manager.num_dofs()
    rows = np.unique(
        np.hstack(
            # The use of private variables here indicates that something is wrong
            # with the data structures. Todo..
            [dof_manager.grid_and_variable_to_dofs(s._g, s._name) for s in variables]
        )
    )
    ncols = rows.size
    cols = np.arange(ncols)
    data = np.ones(ncols)

    return sps.coo_matrix((data, (rows, cols)), shape=(nrows, ncols)).tocsr()


# Obtain primary variables and equations.
primary_eqs = list(set(eq_manager.equations.keys()).difference(eq_names))

variable_names = list(primary_variables.keys())
primary_variable_names = list(set(variable_names).difference(secondary_variable_names))
primary_variables = [eq_manager.variable(g, name) for name in primary_variable_names]

eq_manager.discretize(gb)
A, b = eq_manager.assemble()

prolong_from_primary = prolongation_matrix(primary_variables)
prolong_from_secondary = prolongation_matrix(secondary_variables)

num_iter = 0

for _ in range(15):
    # Here's how far I got with the actual time loop - this part of the code is surely
    # not complete. The logic for a single time step is as follows:
    # For each Newton iteration:
    #   1) Given the set of the primary variables, solve the secondary variables.
    #   Here I have been really confused, but this should end up as equilibrium
    #   calculations + something more.
    #   2) Update the state of the secondary variables
    #   3) Obtain Jacobian matrix for the primary variables, using a Schur complement
    #      technique to eliminate the secondary variables.
    #   4) solve for primary variables, update, go back to 1).

    while True:
        # Here's the inner loop, describing the secondary variables. Should be replaced by
        # a flash
        set_KKT(secondary_manager)
        As, bs = secondary_manager.assemble()
        As_0 = As[:: g.num_cells, :: g.num_cells]
        bs_0 = bs[:: g.num_cells]
        if np.linalg.norm(bs) < 1e-10:
            break
        xs = spla.spsolve(As, bs)
        full_xs = prolong_from_secondary * xs
        print(xs[:7])
        dof_manager.distribute_variable(full_xs, additive=True, to_iterate=True)
        num_iter += 1
        if num_iter > 10:
            assert False
            breakpoint()

    # Note to self: Need to make operator.__str__

    # FIXME: Here we need some convergence criterion for the full system.

    set_KKT(eq_manager)
    # Inverter for the Schur complement system. We have an inverter for block
    # diagnoal systems ready but I never got around to actually using it (this
    # is not difficult to do).
    inverter = lambda A: sps.csr_matrix(np.linalg.inv(A.A))

    # This forms the Jacobian matrix and residual for the primary variables,
    # where the secondary variables are first discretized according to their
    # current state, and the eliminated using a Schur complement technique.
    A_red, b_red = eq_manager.assemble_schur_complement_system(
        primary_equations=primary_eqs,
        primary_variables=primary_variables,
        inverter=inverter,
    )
    if np.linalg.norm(b_red) < 1e-10:
        break
    # Solve the linearized system for the primary variables, map back.
    x = spla.spsolve(A_red, b_red)
    dof_manager.distribute_variable(
        prolong_from_primary * x, additive=True, to_iterate=True
    )
    # Done with global computation, time to go back to the secondary equations.
