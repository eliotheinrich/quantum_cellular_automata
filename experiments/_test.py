import numpy as np
from dataframe import compute, unbundle_param_matrix, SimulatorConfig
from pysims import RandomCliffordSimulator
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from QCAConfig import QCASimulator

import dill as pkl


if __name__ == "__main__":
    L = [8, 16, 32, 64]

    param_matrix = {
        "system_size": L,
        "mzr_prob": 0.01,

        "equilibration_timesteps": 0,
        "sampling_timesteps": 100,
        "temporal_avg": False,

        "num_runs": 1
    }
    configs = [SimulatorConfig(p, RandomCliffordSimulator) for p in unbundle_param_matrix(param_matrix)]
    data = compute(configs, num_threads=4, parallelization_type=1)


    #data.write("time_data.eve")

