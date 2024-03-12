import numpy as np
import gymnasium as gym
import qulacs

from qcenv.utils import ActionFactory

from typing import Callable


class QCEnv(gym.Env):
    metadata = {"render.modes": ["ansi"]}  # ansi形式とは、文字列を返す形式のこと

    def __init__(
        self,
        target_circuit: qulacs.QuantumCircuit,
        reward_function: Callable[[qulacs.QuantumState, qulacs.QuantumState], float],
        target_reward: float,
        action_factory: ActionFactory,
        max_circuit_depth: int,
    ):
        self.target_circuit = target_circuit
        self.n_qubits = target_circuit.get_qubit_count()
        self.action_factory = action_factory
        self.max_circuit_depth = max_circuit_depth
        n_gates = self.action_factory.action_id
        self.action_space = gym.spaces.Discrete(n_gates)
        self.observation_size = (n_gates, self.max_circuit_depth)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=self.observation_size, dtype=np.int32
        )

        self.reward_function = reward_function
        self.target_reward = target_reward

        self.state = qulacs.QuantumState(self.n_qubits)
        self.target_state = qulacs.QuantumState(self.n_qubits)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state.set_Haar_random_state()
        self.target_state = self.state.copy()
        self.target_circuit.update_quantum_state(self.target_state)
        self.step_count = 0
        self.observation = np.zeros(self.observation_size, dtype=np.int32)
        info = {}
        return self.observation, info

    def step(self, action_id: int):  # namedtupleを使うと良い(typing.NamedTuple)
        self.step_count += 1
        action = self.action_factory.sample(action_id)
        action.gate.update_quantum_state(self.state)
        self.observation[action_id, self.step_count - 1] = 1
        reward = self.reward_function(self.state, self.target_state)
        terminated = reward >= self.target_reward
        if terminated:
            reward += 1.0
        truncated = self.step_count >= self.max_circuit_depth
        info = {}
        return self.observation, reward, terminated, truncated, info
