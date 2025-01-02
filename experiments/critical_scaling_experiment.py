import numpy as np
from dataframe import compute, unbundle_param_matrix, SimulatorConfig
from QCAConfig import QCASimulator

if __name__ == "__main__":
    param_matrix = {
        "system_size": 128,
        "mzr_prob": np.linspace(0.0, 0.2, 20),

        "equilibration_timesteps": 500,
        "sampling_timesteps": 500,
        "temporal_avg": True,

        "num_runs": 10
    }
    configs = [SimulatorConfig(p, QCASimulator) for p in unbundle_param_matrix(param_matrix)]

    data = compute(configs, num_threads=4, parallelization_type=1)

    data.write("data_critical.eve")

