import numpy as np
from dataframe import compute, unbundle_param_matrix, SimulatorConfig
from qutils import SimulatorAnimator
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from QCAConfig import QCASimulator


if __name__ == "__main__":
    L = 16
    params = {
        "system_size": L,
        "mzr_prob": 0.,
    }

    simulator = QCASimulator(params, 1)
    simulator.init()

    for i in range(L):
        simulator.state.h(i)
        simulator.state.s(i)

    print(simulator.state)

    animator = SimulatorAnimator(simulator, fps=60, steps_per_frame=1)
    animator.start(900, 900)
