""" Experimental runscript for an advection-reaction system using the full Ad framework.

The chemical system is modeled as reaching equilibium instantaneously.
There are a number of primary and secondary species, with the consentrations of
secondary species, C_j, related to the consentrations of primary species, X_i,
according to the mass action law:

    C_j = K_j * X_i^S_ji  (summation over i is implied).

Here, K_j are the equilibrium constants for the reactions, while the coefficients
S_ji are the stochiometric coefficients for the reactions. These will be identified
in the runscript below.

The species can further be divided into aqueous and fixed, where the former can
be advected, while the latter are fixed in space. Again, these are identified below.

Parameter values are loosly based on the MOMAS benchmark for reactive transport, however,
the numbers are modified to simplify the system.

Main contact: Eirik.Keilegavlen@uib.no.

"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

import porepy as pp


# Equilibrium constants.
eq_consts = np.array([1e-2, 1e-2, 1, 1e3, 1e5, 1e3, 1])

# Stochiometric matrix for the reactions between primary and secondary species.
# For some reason, we ended up splitting the definition between aqueous and fixed
# components (S and W, respectively), however, it is the combined matrix S_W which
# is used below.

# IMPLEMENTATION NOTE: Because of implementation technicalities in PorePy Ad,
# S_W should be a sparse matrix, not a numpy nd array.
S = sps.csr_matrix(
    np.array(
        [
            [0, -1, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, -1, 0, 1, 0],
            [0, -4, 1, 3, 0],
            [0, 3, 1, 1, 0],
        ]
    )
)
W = sps.csr_matrix(np.array([[0, 3, 1, 0, 1], [0, -3, 0, 1, 2]]))

# The number of primary and secondary species can be inferred from the number of
# columns and rows, respectively, of S_W.
# The indices of which primary species are aqueous are specified below.
# Note that the aqueous secondary species should have no connection with fixed
# primary species (the column of S should be all zeros).
S_W = sps.bmat([[S], [W]])


# Bookkeeping
# Index of aquatic and fixed components, referring to the cell-wise T vector
aq_components = np.array([0, 1, 2, 3])
fixed_components = np.array([4])
num_aq_components = aq_components.size
num_fixed_components = fixed_components.size
num_components = num_aq_components + num_fixed_components
num_secondary_components = S_W.shape[0]


def secondary_from_primary(primary):
    # Helper function to get the secondary concentrations from primary ones
    # by use of the mass action law. Not sure if we ever need it (note that
    # in the script below, we mainly use the logarithm of concentrations,
    # which gives a different expression for the secondary concentrations).
    num_sec, num_prim = S_W.shape
    secondary = np.zeros(num_sec)

    # Implementation is cumbersome due to restrictions on how powers can be
    # computed in numpy.
    for row in range(num_sec):
        factor = 1
        for col in range(num_prim):
            exp = S_W[row, col]
            if exp > 0:
                factor *= primary[col] ** exp
            elif exp < 0:
                # Negative powers are not popular.
                factor /= primary[col] ** (-exp)
        secondary[row] = eq_consts[row] * factor

    return secondary


def equilibration_func(log_x: np.ndarray, T_val: np.ndarray) -> pp.ad.Ad_array:
    # Function to compute the residual and Jacobian matrix of the equilibrium
    # system. Uses the forward mode Ad in PorePy, but not the higher-level
    # abstractions exploited below.
    #
    # The equilibrium equation is defined as
    #
    #     T_i = X_i + SW_ij * C_j (summation over j)
    #
    # where C_j contains both aqueous and fixed primary species.
    #
    # We use log(X_i) as primary variable instead of X_i, in which case
    # C_j is related to primary concentrations by
    #
    #     loc(C_j) = log(K_j) + SW_ji * log(X_i) (summation over i)
    #
    # Insertion into the definition of T_i, and then taking exponentials
    # gives the residual equation (on vector form)
    #
    #     exp(X) + SW^T * K * exp(SW * log(X)) - T = residual
    #
    # Here, multiplication with SW corresponds to summation over i,
    # with SW^T gives summation over j.

    # Convert array to Ad object
    ad_logx = pp.ad.initAdArrays(log_x)

    # SW * log_x
    tmp = pp.ad.exp(S_W * ad_logx)

    # Matrix for multiplication with equilibrium constants
    sz = eq_consts.size
    mt = sps.dia_matrix((eq_consts, 0), shape=(sz, sz))

    resid = pp.ad.exp(ad_logx) + S_W.T * mt * tmp - T_val
    return resid


def newton_on_function(f, x0, T_vals):
    # Newton's method on a non-linear function.
    # f is assumed to take two numpy ndarrays (first is initial guess, second is
    # the total concentrations T), and give back a PorePy AdArray

    x_prev = x0
    f_prev = f(x_prev, T_vals)

    norm_orig = np.linalg.norm(f_prev.val)
    #    print(f"Original norm {norm_orig}")

    # Somewhat random number of maximum iterations.
    for i in range(100):
        J = f_prev.jac.A
        #        print(f"condition number {np.linalg.cond(J)}")
        resid = -f_prev.val
        dx = np.linalg.solve(J, resid)

        # Try to enforce reasonable values of the new solution.
        x_new = np.clip(x_prev + dx, a_min=-25, a_max=30)
        f_new = f(x_new, T_vals)
        norm_now = np.linalg.norm(f_new.val)
        #        print(norm_now)
        x_prev = x_new
        f_prev = f_new

        if norm_now < norm_orig * 1e-10:
            return x_new

    # If we get here, convergence has not been achieved. Not sure what to do then.
    # Take a break before continuing?
    breakpoint()
    return x_new


# Define a grid, and wrap it in a GridBucket (this makes the future extension to true
# md problems easier).
g = pp.CartGrid([150])
g.compute_geometry()
gb = pp.GridBucket()
gb.add_nodes(g)

# Use the above functions to find intial composition of the system, based on
# given values for total concentrations T.
# NOTE: For Momas, insert medium B will have a different composition init_T_B.
init_T_A = np.array([0, -2, 0, 2, 0])

# Initial guess.
guess_x0 = np.array([1, -1, 1, 0.3, 0.01])
# Initial values for X_i on log form
init_X_A_log = newton_on_function(equilibration_func, guess_x0, init_T_A)
# .. and on standard form
init_X_A = np.exp(init_X_A_log)

keyword = "chem"
transport_key = "transport"
mass_key = "mass"

# Data dictionary
data = gb.node_props(g)

# Primary variables:
# T: The total concentration of the primary components
# log_c: Logarithm of c
tot_var = "T"
log_var = "log_X"
# Feed to PorePy
data[pp.PRIMARY_VARIABLES] = {
    tot_var: {"cells": num_components},
    log_var: {"cells": num_components},
}

# Initial values, replicated versions of the equilibrium.
# NOTE: Conversion to full Momas requires introduction of medium B here.
init_log_X = np.tile(init_X_A_log.copy(), g.num_cells)
init_T = np.tile(init_T_A.copy(), g.num_cells)


initial_guess_log_X = init_log_X.copy()
initial_guess_log_X[0::num_components] = 0.3
initial_guess_log_X[2::num_components] = 0.3

# Set both initial values, and make these initial guesses for the next
# time step.
data[pp.STATE] = {
    tot_var: init_T,
    log_var: init_log_X,
    pp.ITERATE: {
        tot_var: init_T.copy(),
        log_var: initial_guess_log_X,
    },
}


# The advective problem is posed only for the aqueous primary species.
# This can be done in multiple ways, but essentially we could use separate primary
# variables for aqueous and fixed variables, or we can enforce advection for the
# aqueous subset only. We use the latter approach below.


# Boundary conditions for the advective problem
bc = pp.BoundaryCondition(g, np.array([0]), np.array(["dir"]))
bc_values = np.zeros(g.num_faces * num_aq_components)
bc_values[aq_components] = np.array([0.3, 0.3, 0.3, 0])
transport_data = {
    "darcy_flux": np.ones(g.num_faces),
    "bc": bc,
    "bc_values": bc_values,
    # Only advect the aquatic components
    "num_components": num_aq_components,
}
# Add parameters for the advective subproblem.
data = pp.initialize_data(g, data, transport_key, transport_data)

# Data for the mass term.
# This applies to both aqueous and fixed species.
mass_data = {
    "mass_weight": 0.25 * g.cell_volumes,
    "num_components": num_components,
}
# Technical detail: The mass term needs a different parameter key than the
# advective term, since the number of components are different
data[pp.PARAMETERS][mass_key] = mass_data
data[pp.DISCRETIZATION_MATRICES][mass_key] = {}

# Now we are ready to fire up the Ad machinery.
# First set a DofManager for the GridBucket, and an equation manager.
dof_manager = pp.DofManager(gb)
eq_manager = pp.ad.EquationManager(gb, dof_manager)

# Ad representations of the primary variables.
T = eq_manager.variable(g, tot_var)
log_X = eq_manager.variable(g, log_var)
# Also T at the previous time step - needed for the time derivative.
T_prev = T.previous_timestep()

#### Define equations for Ad

## First chemical equilibrium.
# We already solved this for initialization above, however, to get a system
# that fits for all cells requires some changes.

# Make block matrices of S_W and equilibrium constants, one block per cell.
cell_SW = sps.block_diag([S_W for i in range(g.num_cells)]).tocsr()
cell_equil = sps.dia_matrix(
    (np.hstack([eq_consts for i in range(g.num_cells)]), 0),
    shape=(
        num_secondary_components * g.num_cells,
        num_secondary_components * g.num_cells,
    ),
).tocsr()


def equilibrium_all_cells(total, log_primary):
    # Residual form of the equilibrium problem, posed for all cells together.
    # This is essentially the same as equilibration_func, but is better fit for
    # combination with the full Ad fromework
    secondary_C = pp.ad.exp(cell_SW * log_primary)
    eq = pp.ad.exp(log_primary) + cell_SW.transpose() * cell_equil * secondary_C - total
    return eq


# Wrap the equilibrium residual into an Ad function.
equil_ad = pp.ad.Function(equilibrium_all_cells, "equil")
# And make it into an Expression which can be evaluated.
equilibrium_eq = pp.ad.Expression(equil_ad(T, log_X), dof_manager, "equilibrium")

## Next, the transport equation.
# This consists of terms for accumulation and advection, in addition to boundary conditions.
# There are two additional complications:
# 1) We need mappings between the full set of primary species and the advective subset.
# 2) The primary variable for concentrations is on log-form, while the standard term is
#    advected.
# None of these are difficult to handle, it just requires a bit of work.

# The advection problem is set for the aquatic components only.
# Create mapping between aquatic and all components
cols = np.ravel(
    aq_components.reshape((-1, 1)) + (num_components * np.arange(g.num_cells)), "F"
)
sz = num_aq_components * g.num_cells
rows = np.arange(sz)
matrix_vals = np.ones(sz)

# Mapping from all to aquatic components. The reverse map is achieved by a transpose.
all_2_aquatic = pp.ad.Matrix(
    sps.coo_matrix(
        (matrix_vals, (rows, cols)),
        shape=(num_aq_components * g.num_cells, num_components * g.num_cells),
    ).tocsr()
)

# Mass matrix for accumulation
mass = pp.ad.MassMatrixAd(mass_key, g)

# Upwind discretization for advection.
upwind = pp.ad.UpwindAd(transport_key, g)

# Ad wrapper of boundary conditions
bc_c = pp.ad.BoundaryCondition(keyword=transport_key, grids=[g])

# Time step, consider this random
dt = 10

# Divergence operator. Acts on fluxes of aquatic components only.
div = pp.ad.Divergence([g], dim=num_aq_components)


# Conservation applies to the linear form of the total consentrations.
# Make a function which carries out both conversion and summation over primary
# and secondary species.
def log_to_linear(c: pp.ad.Ad_array) -> pp.ad.Ad_array:
    return pp.ad.exp(c) + cell_SW.T * cell_equil * pp.ad.exp(cell_SW * c)


# Wrap function in an Ad function, ready to be parsed
log2lin = pp.ad.Function(log_to_linear, "")

# Transport equation. Four terms:
# 1) Accumulation
# 2) Advection
# 3) Boundary condition for inlet
# 4) boundary condition for outlet.
transport = (
    mass.mass * (T - T_prev) / dt
    + all_2_aquatic.transpose() * div * upwind.upwind * all_2_aquatic * log2lin(log_X)
    - all_2_aquatic.transpose() * div * upwind.rhs * bc_c
    - all_2_aquatic.transpose() * div * upwind.outflow_neumann * all_2_aquatic * T
)

# Define transport problem as an Expression
transport_eq = pp.ad.Expression(transport, dof_manager, "transport")
# Discretize it
transport_eq.discretize(gb)

# Feed the two equations to the equation manager
eq_manager.equations += [transport_eq, equilibrium_eq]


def _clip_variable(x, target_name, min_val=None, max_val=None):
    # Helper method to cut the values of a target variable.
    # Intended use is the concentration variable.
    dof_ind = np.cumsum(np.hstack((0, dof_manager.full_dof)))

    for key, val in dof_manager.block_dof.items():
        if key[1] == target_name:
            inds = slice(dof_ind[val], dof_ind[val + 1])
            x[inds] = np.clip(x[inds], a_min=min_val, a_max=max_val)
    return x


def newton_gb(equation):
    # Newton's method applied to an equation, pulling values and state from gb.
    # The code should not be considered final, in particular tolerances etc
    # can be considered arbitrary.

    J, resid = equation.assemble_matrix_rhs()
    norm_orig = np.linalg.norm(resid)
    print(norm_orig)

    for i in range(30):
        dx = spla.spsolve(J, resid)

        x_prev = dof_manager.assemble_variable(from_iterate=True)
        x_new = _clip_variable(x_prev + dx, log_var, min_val=-25, max_val=25)
        dof_manager.distribute_variable(x_new, to_iterate=True, additive=False)
        #        breakpoint()
        J, resid = equation.assemble_matrix_rhs()
        norm_now = np.linalg.norm(resid)
        #        print(norm_now)

        if norm_now < norm_orig * 1e-7 or norm_now < 1e-15:
            break

    # NOTE: No special treatment if sufficient residual reduction is not reached.
    # This cannot be the best option.

    # Distribute the variable to the next step
    dof_manager.distribute_variable(x_new, to_iterate=False)
    # Also use as initial guess at the next time step.
    dof_manager.distribute_variable(x_new.copy(), to_iterate=True)

    print(f"Residual reduction {norm_now / norm_orig}")


#    breakpoint()


to_plot = 2

for i in range(100):
    print(f"Time step {i}")
    x = dof_manager.assemble_variable()
    #    print(x[1:250:5])
    newton_gb(eq_manager)

    primary_species = x[g.num_cells * num_components :]

    secondary_species = cell_SW * primary_species

    #    plt.figure()
#    print(x[251::5])
#    print(primary_species[0::num_components])
#    plt.plot(g.cell_centers[0], primary_species[to_plot::num_components])
#    plt.show()
