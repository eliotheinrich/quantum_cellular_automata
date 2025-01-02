import numpy as np
from dataframe import compute, unbundle_param_matrix, SimulatorConfig
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from QCAConfig import QCASimulator


if __name__ == "__main__":
    L = [8, 16, 32, 64]
    zparams = [{"system_size": Li, "x1": 0, "x2": Li//8, "x3": Li//2, "x4": Li//2 + Li//8} for Li in L]

    param_matrix = {
        "zparams": zparams,
        "mzr_prob": np.linspace(0, 0.2, 50),

        "equilibration_timesteps": 500,
        "sampling_timesteps": 500,
        "temporal_avg": True,
        "sample_fixed_mutual_information": True,

        "num_runs": 100
    }
    configs = [SimulatorConfig(p, QCASimulator) for p in unbundle_param_matrix(param_matrix)]

    data = compute(configs, num_threads=4, parallelization_type=1)




    data.write("data.eve")

