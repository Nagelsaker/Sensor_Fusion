from typing import TypeVar, Optional, Dict, Any, List, Generic
from dataclasses import dataclass, field
import numpy as np
import scipy
import scipy.special

from estimatorduck import StateEstimator
from mixturedata import MixtureParameters
from gaussparams import GaussParams

ET = TypeVar("ET")


@dataclass
class PDA(Generic[ET]):  # Probabilistic Data Association
    state_filter: StateEstimator[ET]
    clutter_intensity: float
    PD: float
    gate_size: float

    def predict(self, filter_state: ET, Ts: float) -> ET:
        """Predict state estimate Ts time units ahead"""
        return self.state_filter.predict(filter_state, Ts)

    def gate(
        self,
        # measurements of shape=(M, m)=(#measurements, dim)
        Z: np.ndarray,
        # the filter state that should gate the measurements
        filter_state: ET,
        *,
        sensor_state: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:  # gated: shape=(M,), type=bool (gated(j) = true if measurement j is within gate)
        """Gate/validate measurements: (z-h(x))'S^(-1)(z-h(x)) <= g^2."""
        M = Z.shape[0]
        g_squared = self.gate_size ** 2

        #Array with either shape=(M)
        gated = np.ndarray([np.any((self.state_filter.NIS(z, filter_state) <= g_squared for z in Z), axis=0)])
        return gated

    def loglikelihood_ratios( #det er mer numerisk stabilt å regne med logaritmer
        self,  # measurements of shape=(M, m)=(#measurements, dim)
        Z: np.ndarray,
        filter_state: ET,
        *,
        sensor_state: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:  # shape=(M + 1,), first element for no detection
        """ Calculates the posterior event loglikelihood ratios."""

        log_PD = np.log(self.PD)
        log_PND = np.log(1 - self.PD)  # P_ND = 1 - P_D
        log_clutter = np.log(self.clutter_intensity)

        # allocate
        ll = np.empty(Z.shape[0] + 1)

        # calculate log likelihood ratios
        #hentet fra corollary 7.3.3
        ll[0] = log_clutter+log_PND # TODO: missed detection
        # TODO: some for loop over elements of Z using self.state_filter.loglikelihood
        for i in range(Z.shape[0]):
            ll[i+1] = log_PD + self.state_filter.loglikelihood(Z[i], filter_state, sensor_state=sensor_state) #dette er riktig syntaks

        #ll[0:] = np.ndarray([log_PD]) + np.ndarray([self.state_filter.loglikelihood(z, filter_state, sensor_state=sensor_state) for z in Z])
        return ll


    def association_probabilities(
        self,
        # measurements of shape=(M, m)=(#measurements, dim)
        Z: np.ndarray,
        filter_state: ET,
        *,
        sensor_state: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:  # beta, shape=(M + 1,): the association probabilities (normalized likelihood ratios)
        """calculate the poseterior event/association probabilities."""

        # log likelihoods
        lls = self.loglikelihood_ratios(Z, filter_state, sensor_state=sensor_state)

        # probabilities
        beta = np.exp(lls - scipy.special.logsumexp(lls)) #normaliser slik at summen av association probabilities = 1
        return beta

    def conditional_update(
        self,
        # measurements of shape=(M, m)=(#measurements, dim)
        Z: np.ndarray,
        filter_state: ET,
        *,
        sensor_state: Optional[Dict[str, Any]] = None,
    ) -> List[
        ET
    ]:  # Updated filter_state for all association events, first element is no detection
        """Update the state with all possible measurement associations."""

        #conditional_update skal inneholde updates fra predicted position
        conditional_update = []
        conditional_update.append(
            # TODO: missed detection
            filter_state
        )
        for i in range(Z.shape[0]):
            conditional_update.extend(
                # TODO: some loop over Z making a list of updates
                self.state_filter.update(Z[i], filter_state)
            )

        return conditional_update

    def reduce_mixture(
        self, mixture_filter_state: MixtureParameters[ET]
    ) -> ET:  # the two first moments of the mixture
        """Reduce a Gaussian mixture to a single Gaussian."""
        #return  # TODO: utilize self.state_filter to keep this working for both EKF and IMM
        return self.state_filter.reduce_mixture(mixture_filter_state)


    def update(
        self,
        # measurements of shape=(M, m)=(#measurements, dim)
        Z: np.ndarray,
        filter_state: ET,
        *,
        sensor_state: Optional[Dict[str, Any]] = None,
    ) -> ET:  # The filter_state updated by approximating the data association
        """
        Perform the PDA update cycle.

        Gate -> association probabilities -> conditional update -> reduce mixture.
        """
        # remove the not gated measurements from consideration
        gated = self.gate(Z, filter_state) # TODO
        Zg = Z[gated]

        # find association probabilities
        beta = self.association_probabilities(Zg, filter_state, sensor_state=sensor_state)# TODO
        #blir det riktig med z_g ovenfor?

        # find the mixture components
        filter_state_updated_mixture_components = self.conditional_update(Zg, filter_state, sensor_state=sensor_state) # TODO

        # make mixture
        filter_state_update_mixture = MixtureParameters(
            beta, filter_state_updated_mixture_components
        )

        # reduce mixture
        filter_state_updated_reduced = self.reduce_mixture(filter_state_update_mixture) # TODO

        return filter_state_updated_reduced

    def step(
        self,
        # measurements of shape=(M, m)=(#measurements, dim)
        Z: np.ndarray,
        filter_state: ET,
        Ts: float,
        *,
        sensor_state: Optional[Dict[str, Any]] = None,
    ) -> ET:
        """Perform a predict update cycle with Ts time units and measurements Z in sensor_state"""

        filter_state_predicted = self.predict(filter_state, Ts)# TODO
        filter_state_updated = self.update(Z, filter_state_predicted) # TODO
        return filter_state_updated

    def estimate(self, filter_state: ET) -> GaussParams:
        """Get an estimate with its covariance from the filter state."""
        return self.state_filter.estimate(filter_state)
        #return # TODO: remember to use self.state_filter to keep it working for both EKF and IMM

    def init_filter_state(
        self,
        # need to have the type required by the specified state_filter
        init_state: "ET_like",
    ) -> ET:
        """Initialize a filter state to proper form."""
        return self.state_filter.init_filter_state(init_state)
