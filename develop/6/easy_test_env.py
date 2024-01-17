"""
Gymnasium形式で簡単な量子コンピューター用強化学習環境を作成する
4つの量子ビットを持つ量子回路を用意
初期状態はすべて|0>で、ゲートはHゲートとCXゲートのみ
正解アルゴリズムはCNOT(0,1)とH(3)
"""

import numpy as np
import gymnasium as gym
import qulacs


def complex2real(complex_array):
    b = np.zeros(2 * len(complex_array))
    b[0::2] = complex_array.real
    b[1::2] = complex_array.imag
    return b.astype(np.float32)


class EasyTestEnv(gym.Env):
    metadata = {"render.modes": ["ansi"]}  # ansi形式とは、文字列を返す形式のこと

    def __init__(self, render_mode=None):
        self.n_qubits = 3
        self.T = 5
        self._n_one_qubit_gates = 1
        self._n_two_qubit_gates = 1
        self.n_gates = (
            self._n_one_qubit_gates * self.n_qubits
            + self._n_two_qubit_gates * self.n_qubits * (self.n_qubits - 1)
        )
        self.action_space = gym.spaces.Discrete(self.n_gates)

        self.observation_size = (self.n_gates, self.T)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=self.observation_size, dtype=np.int32
        )

        assert render_mode is None or render_mode in self.metadata["render.modes"]
        self.render_mode = render_mode

        self.state = qulacs.QuantumState(self.n_qubits)
        self.target_state = qulacs.QuantumState(self.n_qubits)
        self.circuit = self._initialize_circuit(self.n_qubits)

        self.step_count = 0
        self.target_fidelity = 0.99
        self.target_reward = np.log(self.target_fidelity)

        self.observation = np.zeros(self.observation_size, dtype=np.int32)

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.state.set_Haar_random_state()
        self.target_state = self.state.copy()  # これはいわゆるdeepcopy
        self.circuit.update_quantum_state(self.target_state)

        self.step_count = 0

        # observation = complex2real(self.state.get_vector())
        self.observation = np.zeros(self.observation_size, dtype=np.int32)
        info = None
        return self.observation, info

    def step(self, action):
        self.step_count += 1
        gate = self._actionid2gate(action)
        gate.update_quantum_state(self.state)
        # observation = complex2real(self.state.get_vector())
        self.observation[action, self.step_count - 1] = 1
        reward = self._calc_reward()
        terminated = reward >= self.target_reward
        if terminated:  # 変更2
            reward += 1.0
        truncated = self.step_count >= self.T
        info = None
        return self.observation, reward, terminated, truncated, info

    def _initialize_circuit(self, n_qubits):
        circuit = qulacs.QuantumCircuit(n_qubits)
        circuit.add_H_gate(0)
        circuit.add_CNOT_gate(1, 2)
        return circuit

    def _actionid2gate(self, action_id: int):
        # action_idがself._n_one_qubit_gates*self.n_qubits以下なら1量子ビットゲート
        # 作用するqubitはaction_id%self.n_qubits
        # 作用するゲートはaction_id//self.n_qubits
        # 0:H
        if action_id < self._n_one_qubit_gates * self.n_qubits:
            gate_id = action_id // self.n_qubits
            qubit_id = action_id % self.n_qubits
            if gate_id == 0:
                return qulacs.gate.H(qubit_id)
            else:
                raise ValueError("Invalid action_id")
        # action_idがself._n_one_qubit_gates*self.n_qubits以上なら2量子ビットゲート
        # 簡単のため、action_id -= self._n_one_qubit_gates*self.n_qubitsとしておく
        # control_qubitはaction_id%self.n_qubits
        # target_qubitはaction_id//self.n_qubits%(self.n_qubits-1)
        # target_qubit!=control_qubitにするため、target_qubit>=control_qubitならtarget_qubit+=1
        # 作用するゲートはaction_id//(self.n_qubits*(self.n_qubits-1))
        # 0:CNOT
        else:
            action_id -= self._n_one_qubit_gates * self.n_qubits
            control_qubit = action_id % self.n_qubits
            target_qubit = action_id // self.n_qubits % (self.n_qubits - 1)
            if target_qubit >= control_qubit:
                target_qubit += 1
            gate_id = action_id // (self.n_qubits * (self.n_qubits - 1))
            if gate_id == 0:
                return qulacs.gate.CNOT(control_qubit, target_qubit)
            else:
                raise ValueError("Invalid action_id")

    def _calc_reward(self):
        # 今回はfidelityベース
        fidelity = (
            np.abs(qulacs.state.inner_product(self.state, self.target_state)) ** 2
        )
        reward = np.log(fidelity)
        # reward = reward / 60  # 報酬を[-1, 1]にするために60で割る 変更1
        # reward = 0
        # if fidelity > 0.9:
        #     reward = 1.0
        # elif fidelity > 0.8:
        #     reward = 1e-1
        # elif fidelity > 0.7:
        #     reward = 1e-2
        # elif fidelity > 0.6:
        #     reward = 1e-3
        # elif fidelity > 0.5:
        #     reward = 1e-4

        return reward
