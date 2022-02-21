from typing import List
import math
import numpy as np
import porepy as pp
import scipy.sparse as sps

# Details of class not relevant - needed to setup exercise
class FootingProblem(pp.ContactMechanicsBiot):

    def __init__(self, params):
        super().__init__(params)

        self.time_step = self.params.get("time_step", 1.)

    def create_grid(self) -> None:
        """Create the grid bucket.

        A unit square grid with no fractures is assigned by default.

        The method assigns the following attributes to self:
            gb (pp.GridBucket): The produced grid bucket.
            box (dict): The bounding box of the domain, defined through minimum and
                maximum values in each dimension.
        """
        phys_dims = [1, 1]
        N = self.params.get("mesh_resolution", 1)
        n_cells = [N, N]
        self.box: Dict = pp.geometry.bounding_box.from_points(
            np.transpose(np.array([[0, 0], phys_dims]))
        )
        g: pp.Grid = pp.CartGrid(n_cells, phys_dims)
        g.compute_geometry()
        self.gb: pp.GridBucket = pp.meshing._assemble_in_bucket([[g]])

    def _set_mechanics_parameters(self) -> None:
        super()._set_mechanics_parameters()

        gb = self.gb
        for g, d in gb:
            # Rock parameters
            lam = self.params.get("lame_lambda", 1.) * np.ones(g.num_cells) / self.scalar_scale
            mu = self.params.get("lame_mu", 1.) * np.ones(g.num_cells) / self.scalar_scale
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
        super()._set_scalar_parameters()

        tensor_scale = self.scalar_scale / self.length_scale ** 2
        kappa = self.params.get("kappa", 1.) * tensor_scale
        mass_weight = self.params.get("s0", 1.) * self.scalar_scale
        for g, d in self.gb:
            bc = self._bc_type_scalar(g)
            bc_values = self._bc_values_scalar(g)
            source_values = self._source_scalar(g)

            specific_volume = self._specific_volume(g)
            diffusivity = pp.SecondOrderTensor(
                kappa * specific_volume * np.ones(g.num_cells)
            )

            alpha = self.params.get("alpha", 1.) * self._biot_alpha(g)
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
        # Apply fix on the bottom - Dirichlet boundary conditions
        all_bf, *_ = self._domain_boundary_sides(g)
        bottom_face = np.nonzero(g.face_centers[1,:] < self.box["ymin"] + 1e-6)[0]
        return pp.BoundaryConditionVectorial(g, bottom_face, "dir")

    def _bc_type_scalar(self, g: pp.Grid) -> pp.BoundaryCondition:
        # No flow on top and bottom
        all_bf, *_ = self._domain_boundary_sides(g)
        dir_face = np.nonzero(
                np.logical_or(
                    g.face_centers[0,:] > self.box["xmax"] - 1e-6,
                    g.face_centers[0,:] < self.box["xmin"] + 1e-6,
                )
            )[0]
        return pp.BoundaryCondition(g, dir_face, "dir")

    def _bc_values_mechanics(self, g: pp.Grid) -> np.ndarray:
        """Set homogeneous conditions on all boundary faces."""
        # Values for all Nd components, facewise
        all_bf, *_ = self._domain_boundary_sides(g)
        values = np.zeros((self._Nd, g.num_faces))
        # Downward force on top boundary
        half_top_face = np.nonzero(
                np.logical_and(
                    g.face_centers[1,:] > self.box["ymax"] - 1e-6,
                    np.absolute(g.face_centers[0,:] - 0.5 * (self.box["xmin"] + self.box["xmax"])) < 0.25,
                )
            )[0] 
        values[1,half_top_face] = self.params.get("force", 0.)
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

    def assemble_linear_mechanics_system(self):

        u = self._eq_manager.merge_variables(
                [(g, self.displacement_variable) for g,_ in self.gb.nodes()]
            )
        A, b = self._eq_manager.assemble_subsystem(eq_names = ["momentum"], variables = [u])

        return A,b

    def assemble_linear_flow_system(self):

        p = self._eq_manager.merge_variables(
                [(g, self.scalar_variable) for g,_ in self.gb.nodes()]
            )
        A, b = self._eq_manager.assemble_subsystem(eq_names = ["subdomain_flow"], variables = [p])

        return A,b

    def fixed_stress_stabilization(self):

        g = [g for g, _ in self.gb.nodes()][0]
        d = self.gb.node_props(g)

        L = self.params.get("fixed_stress_L", 0.)
        alpha = d[pp.PARAMETERS][self.mechanics_parameter_key]["biot_alpha"]
        lam = self.params.get("lame_lambda", 1.) * np.ones(g.num_cells) / self.scalar_scale
        mu = self.params.get("lame_mu", 1.) * np.ones(g.num_cells) / self.scalar_scale
        Kdr = lam + 2 * mu / d[pp.PARAMETERS][self.scalar_parameter_key]["ambient_dimension"] 

        return sps.diags(L * alpha**2 / Kdr * g.cell_volumes)

    def after_newton_iteration(self, solution_vector: np.ndarray) -> None:
        self._nonlinear_iteration += 1

    def distribute_solution(self, solution_vector: np.ndarray, variables: List[str] = None, overwrite: bool = False) -> None:
        self.dof_manager.distribute_variable(
            values=solution_vector, additive=not overwrite, variables=variables, to_iterate=True
        )

    def after_newton_convergence(self) -> None:
        # Fetch iterate solution, which was updated in after_newton_iteration
        solution = self.dof_manager.assemble_variable(from_iterate=True)
        # Distribute to pp.STATE
        self.dof_manager.distribute_variable(values=solution, additive=False)
        self.convergence_status = True

    def nonlinear_iteration(self) -> int:
        return self._nonlinear_iteration

    def export(self, time_step: int) -> None:
        grid_list = [g for g, _ in self.gb.nodes()]
        g = grid_list[0]
        d = self.gb.node_props(g)

        displacement = np.zeros((3, g.num_cells))
        displacement[:2,:] = np.reshape(d[pp.STATE][self.displacement_variable], (2,g.num_cells), 'F')
        pressure = d[pp.STATE][self.scalar_variable]

        self.viz.write_vtu({"u": displacement, "p": pressure}, time_step = time_step)
