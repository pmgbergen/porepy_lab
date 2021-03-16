""" Simple example on using Ad with upwind transport. 

Single grid only. The UpwindCouplingAd is not implemented yet
(should be simple), and this example should be updated accordingly.

Contact person: Eirik Keilegavlen

"""
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

import porepy as pp


transport_key = 'transport'

g = pp.CartGrid([3])
g.compute_geometry()
gb = pp.GridBucket()
gb.add_nodes(g)


data = gb.node_props(g)

bc = pp.BoundaryCondition(g, np.array([0]), np.array(['dir']))
bc_values = np.zeros(g.num_faces)
bc_values[0] = 1

# NB: Neumann outflow condition at the right end. This is not properly handled
# by the current code. I'm working on the fix, it is a bit 
transport_data = {'darcy_flux': np.ones(g.num_faces),
                  'bc': bc,
                  'bc_values': bc_values,
                  'mass_weight': 0.25 * np.ones(g.num_cells)
}
data = pp.initialize_data(g, data, transport_key, transport_data)

var = 'c'

data[pp.STATE] = {var: np.zeros(g.num_cells)}

data[pp.PRIMARY_VARIABLES] = {var: {'cells': 1}}

dof_manager = pp.DofManager(gb)
eq_manager = pp.ad.EquationManager(gb, dof_manager)

c = eq_manager.variable(g, var)
c_prev = c.previous_timestep()


upwind = pp.ad.UpwindAd(transport_key, g)
mass = pp.ad.MassMatrixAd(transport_key, g)

bc_c = pp.ad.BoundaryCondition(keyword=transport_key, grids=[g])

dt = 1

div = pp.ad.Divergence([g])

transport = (mass.mass * (c - c_prev) / dt
             + div * upwind.upwind * c
             - div * upwind.rhs * bc_c
             )

transport_eq = pp.ad.Expression(transport, dof_manager)
transport_eq.discretize(gb)

for i in range(10):
    c = data[pp.STATE][var]
    print(c)
    ad = transport_eq.to_ad(gb)
    update = spla.spsolve(ad.jac, -ad.val)
    dof_manager.distribute_variable(update, additive=True)


