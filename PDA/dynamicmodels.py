#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic models to be used with eg. EKF.

"""
# %%
from typing import Optional, Sequence
from typing_extensions import Final, Protocol
from dataclasses import dataclass, field

import numpy as np

# %% the dynamic models interface declaration


class DynamicModel(Protocol):
    n: int
    def f(self, x: np.ndarray, Ts: float) -> np.ndarray: ...
    def F(self, x: np.ndarray, Ts: float) -> np.ndarray: ...
    def Q(self, x: np.ndarray, Ts: float) -> np.ndarray: ...

# %%


@dataclass
class WhitenoiseAccelleration:
    """
    A white noise accelereation model also known as CV, states are position and speed.

    The model includes the discrete prediction equation f, its Jacobian F, and
    the process noise covariance Q as methods.
    """
    # noise standard deviation
    sigma: float
    # number of dimensions
    dim: int = 2
    # number of states
    n: int = 4

    def f(self,
            x: np.ndarray,
            Ts: float,
          ) -> np.ndarray:
        """
        Calculate the zero noise Ts time units transition from x.

        x[:2] is position, x[2:4] is velocity
        """
        x_p = x
        x_p[:2] += Ts*x[2:]
        #x[:2] er posisjonskoordinatene. x[2:] er hastigheten i m/s, som ganget med Ts blir til posisjon.
        return x_p

    def F(self,
            x: np.ndarray,
            Ts: float,
          ) -> np.ndarray:
        """ Calculate the transition function jacobian for Ts time units at x."""
        #Poenget er å lage F-matrisen (4.64) i boka.

        F = np.eye(4)
        #F[0,2] = Ts
        #F[1,3] = Ts
        F[[0,1],[2,3]] = Ts

        return F

    def Q(self,
            x: np.ndarray,
            Ts: float,
          ) -> np.ndarray:
        """
        Calculate the Ts time units transition Covariance.
        """
        # Hint: sigma can be found as self.sigma, see variable declarations
        # Note the @dataclass decorates this class to create an init function that takes
        # sigma as a parameter, among other things.

        #Q = np.zeros((4,4))
        #Q[[0,0], [1,1]] = (Ts**3/3)*self.sigma**2
        #Q[[0,2], [1,3], [2,0], [3,1]] = (Ts**2/2)*self.sigma**2
        #Q[[2,2], [3,3]] = Ts*self.sigma**2

        Q = np.zeros((4, 4))
        pos_idx = [0, 1]
        vel_idx = [2, 3]
        sigma2 = self.sigma ** 2
        # diags
        Q[pos_idx, pos_idx] = sigma2 * Ts**3 / 3
        Q[vel_idx, vel_idx] = sigma2 * Ts
        # off diags
        Q[pos_idx, vel_idx] = sigma2 * Ts**2 / 2
        Q[vel_idx, pos_idx] = sigma2 * Ts**2 / 2

        return Q
