import numpy as np
from dataframe import compute, unbundle_param_matrix, SimulatorConfig, load_data
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from QCAConfig import QCASimulator

if __name__ == "__main__":
    param_matrix = {
        "system_size": 16,
        "mzr_prob": 0.5,

        "equilibration_timesteps": 50,
        "sampling_timesteps": 50,
        "temporal_avg": True,

        "sample_entropy": True,
        "sample_mutual_information": True,
        "num_runs": 10
    }
    configs = [SimulatorConfig(p, QCASimulator) for p in unbundle_param_matrix(param_matrix)]
    print(configs)

    data = compute(configs, parallelization_type=1)
    print(data.describe(1))

    data.write("test_data.eve")
    data = load_data("test_data.eve")
    print(data.describe(1))
