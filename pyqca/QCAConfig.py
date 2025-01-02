import numpy as np
from dataframe import Simulator, ExperimentParams
from qutils import QuantumCHPState, QuantumCircuit, PauliString, EntropySampler

class QCASimulator(Simulator):
    def __init__(self, params, num_threads):
        super().__init__(params, num_threads)

        self.state = None

        self.sampler = EntropySampler(params)

        self.system_size = params["system_size"]
        self.mzr_prob = params["mzr_prob"]

        self.circuit = QuantumCircuit(self.system_size)
        for i in range(self.system_size):
            self.circuit.h(i)

        for i in range(self.system_size):
            self.circuit.cz(i, (i + 1) % self.system_size)


    def init(self, serialized_data=None):
        self.state = QuantumCHPState(self.system_size)

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

        #sites = list(range(self.system_size//2))

        #s = self.state.entropy(sites, 2)

        #samples["entanglement"] = [[s]]

        return samples

    def serialize(self):
        pass

    def get_texture(self):
        red = [1.0, 0.0, 0.0]
        blue = [0.0, 0.0, 1.0]
        purple = [1.0, 0.0, 1.0]
        return self.state.get_texture(red, blue, purple)
