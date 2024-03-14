from qulacs import QuantumState
import qulacs
import numpy as np


def fidelity(state: QuantumState, target_state: QuantumState) -> float:
    fidelity = np.abs(qulacs.state.inner_product(state, target_state)) ** 2
    return fidelity


def log_fidelity(state: QuantumState, target_state: QuantumState) -> float:
    return np.log(fidelity(state, target_state))


def bhattacharyya_coefficient(state: QuantumState, target_state: QuantumState) -> float:
    state_list = []
    for i in range(2 ** state.get_qubit_count()):
        state_list.append(
            list(map(int, list(format(i, "0" + str(state.get_qubit_count()) + "b"))))
        )
    bhattacharyya_coefficient = 0.0
    for state in state_list:
        prob = state.get_marginal_probability(state)
        target_prob = target_state.get_marginal_probability(state)
        bhattacharyya_coefficient += np.sqrt(prob * target_prob)
    return bhattacharyya_coefficient


def log_bhattacharyya_coefficient(
    state: QuantumState, target_state: QuantumState
) -> float:
    return np.log(bhattacharyya_coefficient(state, target_state))
