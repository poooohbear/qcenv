import numpy as np


def complex2real(complex_array):
    b = np.zeros(2 * len(complex_array))
    b[0::2] = complex_array.real
    b[1::2] = complex_array.imag
    return b.astype(np.float32)


def calc_bhattacharyya_coefficient(state1, state2):
    # Bhattacharyya distance
    # Bhattacharrya distanceはself.stateの測定確率*target_stateの測定確率の平方根の総和
    state_list = []
    for i in range(2 ** state1.get_qubit_count()):
        state_list.append(
            list(map(int, list(format(i, "0" + str(state1.get_qubit_count()) + "b"))))
        )
    bhattacharyya_coefficient = 0.0
    for state in state_list:
        prob = state1.get_marginal_probability(state)
        target_prob = state2.get_marginal_probability(state)
        bhattacharyya_coefficient += np.sqrt(prob * target_prob)
    return bhattacharyya_coefficient
