from base import grids
Array = grids.Array
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
Grid = grids.Grid


import jax.numpy as jnp
from typing import Sequence,Union

def FirstOrderEulerTransientTerm(grid: Grid) -> tuple[Array, Array, Array]:
    """
    Assembles the transient flux coefficients based on the grid properties.

    Returns
    -------

    FluxC
        Coefficient for the current time step.

    FluxC_old
        Coefficient for the previous time step.

    FluxV
        Additional transient flux term (zero in this case).
    """

    # Local fluxes
    # reference: 13.3 First Order Transient Schemes

    # Local fluxes
    FluxC = grid.cell_volumes * grid.rho / grid.deltaT
    FluxC_old = -grid.cell_volumes * grid.rho / grid.deltaT
    FluxV = jnp.zeros_like(FluxC)
    return FluxC, FluxC_old, FluxV

def CK(grid: Grid) -> tuple[Array, Array, Array]:
    FluxC = 2*grid.cell_volumes * grid.rho / grid.deltaT
    FluxC_old = -2*grid.cell_volumes * grid.rho / grid.deltaT
    FluxV = jnp.zeros_like(FluxC)
    return FluxC, FluxC_old, FluxV

def SOUE(grid) -> tuple[Array, Array, Array]:
    FluxC = (3 / 2) * grid.cell_volumes * grid.rho / grid.deltaT
    FluxC_old = -2 * grid.cell_volumes * grid.rho / grid.deltaT
    FluxC_old_old = 0.5 * grid.cell_volumes * grid.rho / grid.deltaT
    # FluxV = jnp.zeros_like(FluxC)
    return FluxC, FluxC_old, FluxC_old_old


class ButcherTableau:
    def __init__(self, name: str) -> None:
        """
        Initialize the Butcher Tableau for a given method.

        :param name: The name of the Runge-Kutta method.
        """
        self.name = name
        self.a, self.b, self.c, self.full_name = self.get_coefficients()

    def get_coefficients(self) -> tuple[Sequence[Sequence], Sequence, Sequence, str]:
        """Get the Butcher tableau coefficients for the chosen explicit Runge-Kutta method."""

        a: Sequence[Sequence[Union[float, int]]] = list()
        b: Sequence[Union[float, int]] = list()
        c: Sequence[Union[float, int]] = list()

        if self.name == "forwardEuler":
            # Forward Euler (1-stage)
            a = [[0]]
            b = [1]
            c = [0]
            full_name = "forwardEuler"
        elif self.name == "midpoint":
            # Midpoint (2-stage)
            a = [[0, 0], [0.5, 0]]
            b = [0, 1]
            c = [0, 0.5]
            full_name = "midpoint"
        elif self.name == "heun":
            # Heun's method (2-stage)
            a = [[0, 0], [1, 0]]
            b = [0.5, 0.5]
            c = [0, 1]
            full_name = "Heun"
        elif self.name == "rk3":
            # Third-order Runge-Kutta (3-stage)
            a = [[0, 0, 0], [0.5, 0, 0], [-1, 2, 0]]
            b = [1 / 6, 2 / 3, 1 / 6]
            c = [0, 0.5, 1]
            full_name = "3rd-order Runge-Kutta"
        elif self.name == "rk4":
            # Fourth-order Runge-Kutta (4-stage)
            a = [[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0]]
            b = [1 / 6, 1 / 3, 1 / 3, 1 / 6]
            c = [0, 0.5, 0.5, 1]
            full_name = "4th-order Runge-Kutta"
        elif self.name == "ssprk42":
            # Second-order Strong Stability Preserving Runge-Kutta (4-stage)
            a = [
                [0, 0, 0, 0],
                [1 / 3, 0, 0, 0],
                [1 / 3, 1 / 3, 0, 0],
                [1 / 3, 1 / 3, 1 / 3, 0],
            ]
            b = [0.25, 0.25, 0.25, 0.25]
            c = [0, 1 / 3, 2 / 3, 1]
            full_name = "4-stage, 2nd order SSP Runge-Kutta"
        return a, b, c, full_name     
