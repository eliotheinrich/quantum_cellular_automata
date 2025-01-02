import numpy as np
from dataframe import compute, unbundle_param_matrix, SimulatorConfig
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from QCAConfig import QCASimulator


if __name__ == "__main__":
    L = [8, 16, 32, 64]

    param_matrix = {
        "system_size": L,
        "mzr_prob": 0.05,

        "equilibration_timesteps": 0,
        "sampling_timesteps": 5000,
        "temporal_avg": False,

        "num_runs": 100
    }
    configs = [SimulatorConfig(p, QCASimulator) for p in unbundle_param_matrix(param_matrix)]

    data = compute(configs, num_threads=4, parallelization_type=1)




    data.write("time_data.eve")

