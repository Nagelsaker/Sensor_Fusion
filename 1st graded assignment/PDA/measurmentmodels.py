# %% Imports
from typing import Any, Dict, Sequence, Optional
from dataclasses import dataclass, field
from typing_extensions import Protocol

import numpy as np

# %% Measurement models interface declaration


class MeasurementModel(Protocol):
    m: int

    def h(self, x: np.ndarray, *,
          sensor_state: Dict[str, Any] = None) -> np.ndarray: ...

    def H(self, x: np.ndarray, *,
          sensor_state: Dict[str, Any] = None) -> np.ndarray: ...

    def R(self, x: np.ndarray, *,
          sensor_state: Dict[str, Any] = None, z: np.ndarray = None) -> np.ndarray: ...

# %% Models


@dataclass
class CartesianPosition:
    sigma: float
    m: int = 2
    state_dim: int = 4

    def h(self,
            x: np.ndarray,
            *,
            sensor_state: Dict[str, Any] = None,
          ) -> np.ndarray:
        """Calculate the noise free measurement location at x in sensor_state."""
        # x[0:2] is position
        # you do not need to care about sensor_state

        z = x[:2]

        return z

    def H(self,
            x: np.ndarray,
            *,
            sensor_state: Dict[str, Any] = None,
          ) -> np.ndarray:
        """Calculate the measurement Jacobian matrix at x in sensor_state."""
        # x[0:2] is position
        # you do not need to care about sensor_state
        # if you need the size of the state dimension it is in self.state_dim

        #h1 = np.eye(2)
        #h2 = np.zeros(2,2)
        #H = np.concatenate((h1, h2), axis=1)
        #more generally:
        H = np.eye(self.m, self.state_dim)

        return H

    def R(self,
            x: np.ndarray,
            *,
            sensor_state: Dict[str, Any] = None,
            z: np.ndarray = None,
          ) -> np.ndarray:
        """Calculate the measurement covariance matrix at x in sensor_state having potentially received measurement z."""
        # you do not need to care about sensor_state
        # sigma is available as self.sigma, and @dataclass makes it available in the init class constructor

        #R = np.zeros(2,2)
        #R[[0,0], [1,1]] = self.sigma**2
        R = self.sigma**2 * np.eye(2)

        return R
