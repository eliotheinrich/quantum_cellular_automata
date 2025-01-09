import numpy as np
from dataframe import Simulator, register_component
from qutils import QuantumCHPState, QuantumCircuit, PauliString, EntropySampler


def cluster_circuit(num_qubits):
    circuit = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        circuit.h(i)

    for i in range(num_qubits):
        circuit.cz(i, (i + 1) % num_qubits)

    return circuit

def modified_cluster_circuit(num_qubits):
    circuit = cluster_circuit(num_qubits)
    for i in range(num_qubits):
        circuit.cx(i, (i + 1) % num_qubits)
    return circuit


class QCASimulator(Simulator):
    def __init__(self, params, num_threads):
        super().__init__(params, num_threads)

        self.state = None
        self.circuit = None

        self.sampler = register_component(EntropySampler, params)

        self.system_size = params["system_size"]
        self.mzr_prob = params["mzr_prob"]
        self.circuit_type = params["circuit_type"]


    def init(self, serialized_data=None):
        self.state = QuantumCHPState(self.system_size)

        if self.circuit_type == "cluster":
            self.circuit = cluster_circuit(self.system_size)
        elif self.circuit_type == "modified_cluster":
            self.circuit = modified_cluster_circuit(self.system_size)

    def timesteps(self, num_steps):
        for _ in range(num_steps):
            self.state.evolve(self.circuit)

            for i in range(self.system_size):
                if np.random.rand() < self.mzr_prob:
                    self.state.mzr(i)

    def take_samples(self):
        samples = self.sampler.take_samples(self.state)

        s = []
        for i in range(0, self.system_size//2):
            sites = list(range(i))
            s.append(self.state.entropy(sites, 2))

        samples["entanglement"] = np.array([s]).T

        return samples

    def serialize(self):
        pass

    def get_texture(self):
        red = [1.0, 0.0, 0.0]
        blue = [0.0, 0.0, 1.0]
        purple = [0.0, 0.0, 1.0]
        print(self.state)
        return self.state.get_texture(red, blue, purple)
