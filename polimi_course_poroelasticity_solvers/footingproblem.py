from typing import List
import math
import numpy as np
import porepy as pp
import scipy.sparse as sps

from porepy.numerics.solvers.andersonacceleration import AndersonAcceleration

from footingproblem_model import FootingProblem

############################################################################################3
# Define model parameters
model_params = {
        # DO NOT CHANGE THESE:
        "file_name" : "poroelasticity",   # exporter
        "folder_name" : "out",  # expoter
        "use_ad" : True,
        # FEEL FREE TO CHANGE ANY OF THOSE:
        "lame_mu": 1e8,
        "lame_lambda": 1e7,
        "force": -1e6,
        "kappa": 1e-12,
        "s0": 0.5e-8,
        "alpha": 1.,
        "mesh_resolution": 80,
        "time_step": 0.25,
        "fixed_stress_L": 0.9,
        "aa_depth": 5,
        }

############################################################################################3
# TODO / Task: What would be a natural defintion of the coupling
# strength (think of the theoretical stability results)
def coupling_strength(params) -> float:

    Kdr = params["lame_lambda"] + 2 * params["lame_mu"] / 2.
    alpha = params["alpha"]
    s0 = params["s0"]
    s0inv = 1. / s0 if s0 > 1e-17 else float('inf')
    return alpha**2 / Kdr * s0inv

tau = coupling_strength(model_params)
print(f"Coupling strength: {tau}")

############################################################################################3

# Construct model - FluidFlower benchmark geometry, running a well test
model = FootingProblem(model_params)

## Prepare simulation - create grid, set parameters, assign equations, and init the algebraic structures
model.prepare_simulation()

############################################################################################3
# Some management
first_call = True
time_step = 0

# Container for global increment vector
# TODO move to model
g = [g for g, _ in model.gb.nodes()][0]
increment = np.zeros(model.dof_manager.num_dofs())
residual = np.zeros(model.dof_manager.num_dofs())
flow_dofs = model.dof_manager.grid_and_variable_to_dofs(g, model.scalar_variable)
mech_dofs = model.dof_manager.grid_and_variable_to_dofs(g, model.displacement_variable)
mass_sqrt = 1. / model_params["mesh_resolution"]

aa = AndersonAcceleration(dimension = model.dof_manager.num_dofs(), depth = model_params["aa_depth"])

############################################################################################3
# Time loop
while model.time < model.end_time:
    # Time management
    time_step += 1
    model.time += model.time_step

    print(f"+=====================================+\nTime step {time_step}:") 

    # Reset Anderson acceleration
    aa.reset()

    # Newton loop
    model.before_newton_loop()
    while model.convergence_status == False:

        # Empty method, but include for completeness
        model.before_newton_iteration()

        # # Monolithic solution
        # TODO move to model
        # if first_call:
        #     A, _ = model.assemble_linear_system()
        #     A_splu = sps.linalg.splu(A)
        #     first_call = False
        # _, residual = model.assemble_linear_system()
        # increment = A_splu.solve(residual)
        # model.distribute_solution(increment)

        # # Fixed strain
        # # TODO move to model
        # if first_call:
        #     A_mech, _ = model.assemble_linear_mechanics_system()
        #     A_flow, _ = model.assemble_linear_flow_system()
        #     A_mech_splu = sps.linalg.splu(A_mech)
        #     A_flow_splu = sps.linalg.splu(A_flow)
        #     first_call = False
        # # 1. Step: Solve the flow problem for given displacement
        # _, residual[flow_dofs] = model.assemble_linear_flow_system()
        # increment[flow_dofs] = A_flow_splu.solve(residual[flow_dofs])
        # model.distribute_solution(increment, variables=["p"])
        # # 2. Step: Solve the mechanics problem for updated pressre
        # _, residual[mech_dofs] = model.assemble_linear_mechanics_system()
        # increment[mech_dofs] = A_mech_splu.solve(residual[mech_dofs])
        # model.distribute_solution(increment, variables=["u"])

        ## Fixed stress
        # TODO move to model
        if first_call:
            A_mech, _ = model.assemble_linear_mechanics_system()
            A_flow, _ = model.assemble_linear_flow_system()
            A_flow += model.fixed_stress_stabilization()
            A_mech_splu = sps.linalg.splu(A_mech)
            A_flow_splu = sps.linalg.splu(A_flow)
            first_call = False
        # 1. Step: Solve the flow problem for given displacement
        _, residual[flow_dofs] = model.assemble_linear_flow_system()
        increment[flow_dofs] = A_flow_splu.solve(residual[flow_dofs])
        model.distribute_solution(increment, variables=["p"])
        # 2. Step: Solve the mechanics problem for updated pressre
        _, residual[mech_dofs] = model.assemble_linear_mechanics_system()
        increment[mech_dofs] = A_mech_splu.solve(residual[mech_dofs])
        model.distribute_solution(increment, variables=["u"])

        # Anderson acceleration
        if model.nonlinear_iteration() > 0:
            gk = model.dof_manager.assemble_variable(from_iterate=True) # Application of solver to previous approximation
            fk = increment # Residual in terms of the fixed point iteration, ie the increment
            improved_approximation = aa.apply(gk, fk, model.nonlinear_iteration()-1)
            model.distribute_solution(improved_approximation, overwrite=True)

        # Error management
        increment_norm = np.linalg.norm(mass_sqrt * increment)
        residual_norm = np.linalg.norm(mass_sqrt * residual)

        print("Iter.", model.nonlinear_iteration(), "| Residual", residual_norm)
        
        # Manage the iteration (accumulate to determine the solution)
        model.after_newton_iteration(increment)

        # Check for convergence
        if residual_norm < 1e-8:
            # Finish off
            model.after_newton_convergence()
            model.export(time_step)

    print(f"+=====================================+\n")

print("Finished simulation")
