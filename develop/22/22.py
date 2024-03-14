from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import qulacs
from qcenv.environments import QCEnv
from qcenv.utils import log_fidelity, ActionFactory
from functools import partial
from gymnasium.wrappers import NormalizeReward

target_circuit = qulacs.QuantumCircuit(2)
target_circuit.add_H_gate(0)
target_circuit.add_CNOT_gate(0, 1)

action_factory = ActionFactory(2)
action_factory.add_one_qubit_gate(["H"])
action_factory.add_two_qubit_gate(["CNOT"])

env = QCEnv(
    target_circuit=target_circuit,
    reward_function=partial(log_fidelity),
    target_reward=np.log(0.9),
    action_factory=action_factory,
    max_circuit_depth=10,
)
env = NormalizeReward(env)

model = PPO(MlpPolicy, env, verbose=0)
model.learn(total_timesteps=10000)

rewards, length = evaluate_policy(
    model, env, n_eval_episodes=1000, return_episode_rewards=True
)
print(f"mean reward:{np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
print(f"mean length:{np.mean(length):.2f} +/- {np.std(length):.2f}")

# mean reward:0.97 +/- 0.04
# mean length:1.92 +/- 0.26
