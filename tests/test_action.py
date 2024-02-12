from qcenv.utils import Action, ActionFactory
import numpy as np


def test_action():
    action_factory = ActionFactory(3)
    # H gate
    action_factory.add_one_qubit_gate(["H"])
    n_action = action_factory.action_id
    assert n_action == 3
    action = action_factory.sample(n_action - 1)
    assert action.action_id == 2
    assert action.gate_name == "H"
    assert action.gate.get_name() == "H"

    # U1 gate
    angle_list = [np.pi / 4, -np.pi / 4]
    action_factory.add_one_qubit_gate(["U1"], angle_list)
    n_action = action_factory.action_id
    assert n_action == 9  # H,U1(+) and U1(-)がそれぞれ3つずつ
    action = action_factory.sample(n_action - 1)
    assert action.action_id == 8
    assert action.gate_name == "U1"

    # CNOT gate
    action_factory.add_two_qubit_gate(["CNOT"])
    n_action = action_factory.action_id
    assert n_action == 15  # 9+3*2
    action = action_factory.sample(n_action - 1)
    assert action.action_id == 14
    assert action.gate_name == "CNOT"
    assert action.gate.get_name() == "CNOT"
