import math
import numpy as np
import porepy as pp
import scipy.sparse as sps

# Details of class not relevant - needed to setup exercise
class FootingProblem(pp.ContactMechanicsBiot):

    def __init__(self, params, lame_mu=1, lame_lambda=1, force=0, mesh_resolution=4, kappa = 1, s0 = 1, alpha = 1, time_step = 1):
        super().__init__(params)

        self.lame_mu = lame_mu
        self.lame_lambda = lame_lambda
        self.force = force

        self.kappa = kappa
        self.mass_weight = s0

        self.alpha = 1
    
        self.nx = int(mesh_resolution)
        self.ny = int(mesh_resolution)

        self.time_step = time_step

    def prepare_simulation(self) -> None:
        super().prepare_simulation()

        self.viz = pp.Exporter(
            self.gb,
            file_name=self.params["file_name"],
            folder_name=self.params["folder_name"],
        )

    def create_grid(self) -> None:
        """Create the grid bucket.

        A unit square grid with no fractures is assigned by default.

        The method assigns the following attributes to self:
            gb (pp.GridBucket): The produced grid bucket.
            box (dict): The bounding box of the domain, defined through minimum and
                maximum values in each dimension.
        """
        phys_dims = [1, 1]
        n_cells = [self.nx, self.ny]
        self.box: Dict = pp.geometry.bounding_box.from_points(
            np.array([[0, 0], phys_dims])
        )
        g: pp.Grid = pp.CartGrid(n_cells, phys_dims)
        g.compute_geometry()
        self.gb: pp.GridBucket = pp.meshing._assemble_in_bucket([[g]])

    def _set_mechanics_parameters(self) -> None:
        gb = self.gb
        for g, d in gb:
            # Rock parameters
            lam = np.ones(g.num_cells) / self.scalar_scale
            mu = np.ones(g.num_cells) / self.scalar_scale
            C = pp.FourthOrderTensor(mu, lam)

            # Define boundary condition
            bc = self._bc_type_mechanics(g)
            # BC and source values
            bc_val = self._bc_values_mechanics(g)
            source_val = self._source_mechanics(g)

            pp.initialize_data(
                g,
                d,
                self.mechanics_parameter_key,
                {
                    "bc": bc,
                    "bc_values": bc_val,
                    "source": source_val,
                    "fourth_order_tensor": C,
                    "time_step": self.time_step,
                    "biot_alpha": self._biot_alpha(g),
                    "p_reference": np.zeros(g.num_cells),
                },
            )

    def _set_scalar_parameters(self) -> None:
        tensor_scale = self.scalar_scale / self.length_scale ** 2
        kappa = self.kappa * tensor_scale
        mass_weight = self.mass_weight * self.scalar_scale
        for g, d in self.gb:
            bc = self._bc_type_scalar(g)
            bc_values = self._bc_values_scalar(g)
            source_values = self._source_scalar(g)

            specific_volume = self._specific_volume(g)
            diffusivity = pp.SecondOrderTensor(
                kappa * specific_volume * np.ones(g.num_cells)
            )

            alpha = self.alpha * self._biot_alpha(g)
            pp.initialize_data(
                g,
                d,
                self.scalar_parameter_key,
                {
                    "bc": bc,
                    "bc_values": bc_values,
                    "mass_weight": mass_weight * specific_volume,
                    "biot_alpha": alpha,
                    "source": source_values,
                    "second_order_tensor": diffusivity,
                    "time_step": self.time_step,
                    "vector_source": np.zeros(g.num_cells * self.gb.dim_max()),
                    "ambient_dimension": self.gb.dim_max(),
                },
            )

    def _bc_type_mechanics(self, g) -> pp.BoundaryConditionVectorial:
        all_bf = g.get_boundary_faces()
        bc = pp.BoundaryConditionVectorial(g, all_bf, "neu")
        # Apply fix on the bottom - Dirichlet boundary conditions
        bottom_face = np.nonzero(g.face_centers[1,all_bf] < self.box["ymin"] + 1e-6)[0] 
        bc.is_dir[:, bottom_face] = True
        bc.is_neu[:, bottom_face] = False

        return bc

    def _bc_type_scalar(self, g: pp.Grid) -> pp.BoundaryCondition:
        # Define boundary regions
        all_bf, *_ = self._domain_boundary_sides(g)
        # Define boundary condition on faces - Dirichlet per default
        bc = pp.BoundaryCondition(g, all_bf, "dir")
        # No flow on top and bottom
        top_face = np.nonzero(g.face_centers[1,all_bf] > self.box["ymax"] - 1e-6)[0] 
        bottom_face = np.nonzero(g.face_centers[1,all_bf] < self.box["ymin"] + 1e-6)[0] 
        bc.is_dir[top_face] = False
        bc.is_neu[top_face] = True
        bc.is_dir[bottom_face] = False
        bc.is_neu[bottom_face] = True

        return bc

    def _bc_values_mechanics(self, g: pp.Grid) -> np.ndarray:
        """Set homogeneous conditions on all boundary faces."""
        # Values for all Nd components, facewise
        all_bf, *_ = self._domain_boundary_sides(g)
        values = np.zeros((self._Nd, g.num_faces))
        # Downward force on top boundary
        top_face = np.nonzero(g.face_centers[1,all_bf] > self.box["ymax"] - 1e-6)[0] 
        values[1,top_face] = -self.force
        # Reshape according to PorePy convention
        values = values.ravel("F")
        return values

    def _bc_values_scalar(self, g: pp.Grid) -> np.ndarray:
        """
        No flow and homogeneous pressure boundary conditions.
        """
        return np.zeros(g.num_faces)

    def assemble_linear_system(self):

        A, b = self._eq_manager.assemble()

        return A,b

    def assemble_mechanics_linear_system(self):

        A, b = self._eq_manager.assemble()

        return A,b

    def assemble_flow_linear_system(self):

        A, b = self._eq_manager.assemble()

        return A,b

    def after_newton_convergence(self) -> None:
        # Fetch iterate solution, which was updated in after_newton_iteration
        solution = self.dof_manager.assemble_variable(from_iterate=True)
        # Distribute to pp.STATE
        self.dof_manager.distribute_variable(values=solution, additive=False)
        self.convergence_status = True

    def nonlinear_iteration(self) -> int:
        return self._nonlinear_iteration

    def export(self, time_step: int) -> None:
        data = [self.displacement_variable, self.scalar_variable]
        self.viz.write_vtu(data, time_step = time_step)

##########################
# Main starts here
##########################

# Define model parameters
model_params = {
        "file_name" : "poroelasticity",   # exporter
        "folder_name" : "out",  # expoter
        "use_ad" : True,
        }

# Construct model - FluidFlower benchmark geometry, running a well test
model = FootingProblem(model_params)

## Prepare simulation - create grid, set parameters, assign equations, and init the algebraic structures
model.prepare_simulation()

first_call = True
time_step = 0

while model.time < model.end_time:
    # Time management
    time_step += 1
    model.time += model.time_step

    # Newton loop
    model.before_newton_loop()
    while model.convergence_status == False:

        # Empty method, but include for completeness
        model.before_newton_iteration()

        # Assemble and solve for the increment
        if first_call:
            A, _ = model.assemble_linear_system()
            A_splu = sps.linalg.splu(A)
            first_call = False

        _, residual = model.assemble_linear_system()
        increment = A_splu.solve(residual)

        print(increment.shape)

        # Error management
        increment_norm = np.linalg.norm(increment)
        residual_norm = np.linalg.norm(residual)
        if model.nonlinear_iteration() == 0:
            residual_norm_ref = residual_norm
            if residual_norm_ref < 1e-10:
                residual_norm_ref = 1.

        # Manage the iteration (accumulate to determine the solution)
        model.after_newton_iteration(increment)

        print("time step", model.time_step, "current time", model.time, "it count", model.nonlinear_iteration(), residual_norm, residual_norm / residual_norm_ref)
        # Check for convergence
        if residual_norm < 1e-5:
            # Finish off
            model.after_newton_convergence()
            model.export(time_step)

print("Finished simulation")
