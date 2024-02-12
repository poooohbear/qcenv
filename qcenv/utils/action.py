import qulacs


class Action:
    def __init__(self, action_id, gate_name, *gate_params):
        self.action_id = action_id
        self.gate_name = gate_name
        self.gate = getattr(qulacs.gate, self.gate_name)(*gate_params)

    def __str__(self):
        return str(self.gate)


class ActionFactory:
    def __init__(self, n_qubit):
        self.n_qubit = n_qubit
        self.action_list = []
        self.action_id = 0

    def sample(self, action_id):
        return self.action_list[action_id]

    def add_one_qubit_gate(self, one_qubit_gate_list, angle_list=None):
        if angle_list is None:
            for one_qubit_gate_name in one_qubit_gate_list:
                for target_qubit in range(self.n_qubit):
                    action = Action(self.action_id, one_qubit_gate_name, target_qubit)
                    self.action_list.append(action)
                    self.action_id += 1
        else:
            for one_qubit_gate_name in one_qubit_gate_list:
                for target_qubit in range(self.n_qubit):
                    for angle in angle_list:
                        action = Action(
                            self.action_id, one_qubit_gate_name, target_qubit, angle
                        )
                        self.action_list.append(action)
                        self.action_id += 1

    def add_two_qubit_gate(self, two_qubit_gate_list):
        for two_qubit_gate_name in two_qubit_gate_list:
            for control_qubit in range(self.n_qubit):
                for target_qubit in range(self.n_qubit):
                    if control_qubit == target_qubit:
                        continue
                    action = Action(
                        self.action_id, two_qubit_gate_name, control_qubit, target_qubit
                    )
                    self.action_list.append(action)
                    self.action_id += 1
