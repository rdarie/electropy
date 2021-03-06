from .particle import Particle
import numpy as np
from scipy import constants

# Globals
epsilon_0 = constants.epsilon_0
pi = constants.pi


class Charge(Particle):
    """Base class for a point electric charge

    Attributes:
        position: particle position, 1D numpy array of length 3
        charge: electric charge in Coulomb.

    Methods:
        getPosition(): Returns numpy array
    """

    def __init__(
        self,
        position,
        charge,
        velocity=[0, 0, 0],
        acceleration=[0, 0, 0],
        mass=np.inf,
    ):
        """Charge class initializer

        Args:
            position: position. units: meters. numpy array or a list.
            charge: electric charge. units: Coulombs. float.
        """
        Particle.__init__(self, position, velocity, acceleration, mass)
        self.charge = charge

    @property
    def q(self):
        """Electric charge value in Coulomb"""
        return self.charge

    def field(self, fpos, type="analytical", h=0.001, component=None):
        """Electric field at a given position.

        Args:
            fpos: field position. numpy array or a list.
            type: type of field calculation. 'analytical' (default) or from
                gradient of potential.
            h: potential gradient spatial difference.
            component: 'x', 'y', 'z', or None (default)
        """

        fpos = np.asarray(fpos)
        if not self.__verify3D__(fpos):
            raise TypeError(
                "Initializer argument must be a \
                            1D numpy array or list of length 3"
            )

        if np.array_equal(fpos, self.position):
            electric_field = fpos.astype(float)
            electric_field.fill(np.nan)

            return electric_field

        if type == "analytical":
            displacement = fpos - self.position

            electric_field = (
                self.q
                * (4 * pi * epsilon_0) ** -1
                * displacement
                * np.linalg.norm(displacement) ** -3
            )

        if type == "potential":
            potential_grid = np.empty([3, 3, 3], dtype=object)
            x = np.linspace(fpos[0] - h, fpos[0] + h, 3)
            y = np.linspace(fpos[1] - h, fpos[1] + h, 3)
            z = np.linspace(fpos[2] - h, fpos[2] + h, 3)

            for (i, j, k), _ in np.ndenumerate(potential_grid):
                potential_grid[i][j][k] = self.potential([x[i], y[j], z[k]])

            xgrad, ygrad, zgrad = np.gradient(potential_grid, h)
            grad_potential = np.array(
                [xgrad[1, 1, 1], ygrad[1, 1, 1], zgrad[1, 1, 1]]
            )

            electric_field = -1 * grad_potential

        if component is None:
            return electric_field
        elif component == "x":
            return electric_field[0]
        elif component == "y":
            return electric_field[1]
        elif component == "z":
            return electric_field[2]

    def potential(self, ppos):
        """Electric potential at a given position.

        Args:
            ppos: potential position. numpy array or a list.
        """

        ppos = np.asarray(ppos)
        if not self.__verify3D__(ppos):
            raise TypeError(
                "Initializer argument must be a \
                            1D numpy array or list of length 3"
            )

        if np.array_equal(ppos, self.position):
            return np.nan

        displacement = ppos - self.position

        electric_potential = (
            self.q
            * (4 * pi * epsilon_0) ** -1
            * np.linalg.norm(displacement) ** -1
        )

        return electric_potential
