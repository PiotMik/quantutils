import numpy as np
import matplotlib.pyplot as plt


def gbm(s0: float, r: float, sigma: float, t: float, time_steps: int, n_paths: int):
    """
    Generate Geometric Brownian Motion paths
    :param s0: float, starting price
    :param r: float, risk free rate
    :param sigma: float, volatility
    :param t: float, time horizon
    :param time_steps: int, how many subintervals include from 0 to T
    :param n_paths: int, how many paths to simulate
    :return: paths: ndarray, simulated paths
    """
    dt = float(t) / time_steps
    paths = np.zeros((time_steps + 1, n_paths), np.float64)
    paths[0] = s0
    for t in range(1, time_steps + 1):
        rand = np.random.standard_normal(n_paths)
        paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt +
                                         sigma * np.sqrt(dt) * rand)
    return paths


class BarrierOption:
    """Represents a barrier option"""
    barrier_action_dict = {"U": "Up", "D": "Down",
                           "O": "Out", "I": "In",
                           "A": "and"}
    option_type_dict = {"C": "call",
                        "P": "put"}

    def __init__(self, option_type: str, strike: [float, int], maturity: float,
                 barrier: [float, int], barrier_action: str):
        """
        :param option_type: str, "C" or "P" designating Call or Put option
        :param strike: [float, int], strike of the option
        :param maturity: float, time until the maturity of the option
        :param barrier: [float, int], barrier
        :param barrier_action: str, type of barrier: "UAO", "UAI", "DAO" or "DAI"
        """
        self.barrier = barrier
        self.strike = strike
        self.maturity = maturity
        self.option_type = option_type
        self.barrier_action = barrier_action

    def __str__(self):
        barrier_action_full = str.join("-", [self.barrier_action_dict[letter] for letter in self.barrier_action])
        option_type_full = self.option_type_dict[self._option_type]
        representation = f"{barrier_action_full} {option_type_full} option with barrier @ {self.barrier}"
        return representation

    def __repr__(self):
        return f"BarrierOption(barrier = {self.barrier}, option_type = {self.option_type}, barrier_action = {self.barrier_action})"

    @property
    def barrier(self) -> [float, int]:
        return self._barrier

    @barrier.setter
    def barrier(self, new_barrier: [float, int]):
        if not isinstance(new_barrier, (float, int)):
            raise ValueError(f"Barrier must be float. Got: {new_barrier}")
        self._barrier = new_barrier

    @property
    def strike(self) -> [float, int]:
        return self._strike

    @strike.setter
    def strike(self, new_strike: [float, int]):
        if not isinstance(new_strike, (float, int)):
            raise ValueError(f"Barrier must be float. Got: {new_strike}")
        self._strike = new_strike

    @property
    def option_type(self) -> str:
        return self._option_type

    @option_type.setter
    def option_type(self, new_option_type: str):
        if not isinstance(new_option_type, str):
            raise TypeError(f"Option type needs to be a string. Got {type(new_option_type)}")

        if new_option_type.upper() not in ["C", "P"]:
            raise ValueError(f'Option type must be "C" or "P". Got: {new_option_type}')
        self._option_type = new_option_type.upper()

    @property
    def barrier_action(self) -> str:
        return self._barrier_action

    @barrier_action.setter
    def barrier_action(self, new_barrier_action: str):
        if not isinstance(new_barrier_action, str):
            raise TypeError(f"Barrier action needs to be a string. Got {type(new_barrier_action)}")

        admissible_values = ["UAI", "UAO", "DAO", "DAI"]
        if new_barrier_action.upper() not in admissible_values:
            raise ValueError(f'Option type must be one of: {admissible_values}. Got: {new_barrier_action}')
        self._barrier_action = new_barrier_action.upper()

    def vanilla_payoff(self, paths: np.ndarray) -> np.ndarray:
        """
        Payoff of a vanilla option
        :param paths: np.ndarray, realizations of stock prices
        :return: payoff: np.ndarray, payoff per realization
        """
        if self.option_type == "C":
            payoff = np.maximum(paths[-1, :] - self.strike, 0)
        else:
            payoff = np.maximum(self.strike - paths[-1, :], 0)
        return payoff

    def _barrier_breach(self, paths: np.ndarray):
        """
        Calculates the multiplier of 1 or 0 depending if there's any payoff at maturity
        :param paths:
        :return: multiplier: np.ndarray, 1 or 0 if the option should pay
        """
        if self.barrier_action == "UAI":
            multiplier = np.where(paths.max(axis=0) > self.barrier, 1, 0)
        elif self.barrier_action == "UAO":
            multiplier = np.where(paths.max(axis=0) > self.barrier, 0, 1)
        elif self.barrier_action == "DAI":
            multiplier = np.where(paths.max(axis=0) < self.barrier, 1, 0)
        else:
            multiplier = np.where(paths.max(axis=0) < self.barrier, 0, 1)
        return multiplier

    def payoff(self, paths: np.ndarray):
        """
        Payoff of the barrier option
        :param paths: np.ndarray, realizations of stock prices
        :return: payoff: np.ndarray, payoff per realization
        """
        vanilla_payoff = self.vanilla_payoff(paths)
        barrier_ind = self._barrier_breach(paths)
        return vanilla_payoff * barrier_ind

    def price_mc(self, r: float, paths: np.ndarray):

        expected_payoff = self.payoff(paths).mean()
        return np.exp(-r*self.maturity)*expected_payoff


if __name__ == '__main__':

    gbm_paths = gbm(95, 0.1, 0.25, 1, 365, 3)
    t = np.linspace(0, 5, 365 + 1)
    fig, ax = plt.subplots()
    ax.plot(t, gbm_paths)
    ax.set_xlabel('t')
    ax.set_ylabel('S(t)')
    ax.plot(t, t*0 + 120, color = 'red', linestyle ='dashed')
    plt.title('Barrier vs Stock price')
    plt.xlim([0, 5])
    plt.ylim([50, 300])

    gbm_paths = gbm(95, 0.1, 0.25, 1, 365, 1000)
    fig, ax = plt.subplots()
    barrier = 150
    gbm_paths_filtered = gbm_paths[:, (gbm_paths.max(axis=0) < barrier)]
    ax.plot(t, gbm_paths_filtered, color='blue', alpha=0.1)
    ax.plot(t, t*0 + barrier, color='red')
    ax.legend()
    ax.set_xlabel('t')
    ax.set_ylabel('S(t)')
    plt.title('GBM simulation (filtered by barrier)')
    plt.xlim([0, 5])
    plt.ylim([50, 300])

    b1 = BarrierOption(option_type="P", strike=100.00, maturity=1.0,
                       barrier_action="UAI", barrier=110.0)
    weights = b1.payoff(paths= gbm_paths)
    weights = weights/weights.max()
    b1.price_mc(r=0.1, paths=gbm_paths)

    fig, ax = plt.subplots()
    for i, weight in enumerate(list(weights)):
        ax.plot(t, gbm_paths[:, i], color='blue', alpha=weight)
    ax.plot(t, t*0 + b1.barrier, color='red')
    ax.legend()
    ax.set_xlabel('t')
    ax.set_ylabel('S(t)')
    plt.title('GBM simulation (weighted by payoff)')
    plt.xlim([0, 5])
    plt.ylim([50, 300])