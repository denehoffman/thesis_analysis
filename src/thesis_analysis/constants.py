import luigi
import numpy as np

RUN_PERIODS = ['s17', 's18', 'f18', 's20']
GLUEX_PHASE_I = ['s17', 's18', 'f18']
GLUEX_PHASE_II = ['s20']
DATA_TYPES = ['data', 'accmc', 'genmc', 'bkgmc']
CHISQNDF = [3.0, 3.5, 4.0, 4.5, 5.0]
HOSTNAME = 'ernest.phys.cmu.edu'

RUN_RANGES = {
    's17': (30000, 39999),
    's18': (40000, 49999),
    'f18': (50000, 59999),
    's20': (71275, 79999),
}


def get_run_period(run_number: int) -> str | None:
    for rp, (lo, hi) in RUN_RANGES.items():
        if lo <= run_number <= hi:
            return rp
    return None


TRUE_POL_ANGLES = {
    's17': {'0.0': 1.8, '45.0': 47.9, '90.0': 94.5, '135.0': -41.6},
    's18': {'0.0': 4.1, '45.0': 48.5, '90.0': 94.2, '135.0': -42.4},
    'f18': {'0.0': 3.3, '45.0': 48.3, '90.0': 92.9, '135.0': -42.1},
    's20': {'0.0': 0.0, '45.0': 45.0, '90.0': 90.0, '135.0': -45.0},
}


def get_pol_angle(run_period: str | None, angle_deg: str) -> float | None:
    if run_period is None:
        return None
    pol_angle_deg = TRUE_POL_ANGLES[run_period].get(angle_deg)
    if pol_angle_deg is None:
        return None
    return pol_angle_deg * np.pi / 180.0


class global_parameters(luigi.Config):
    username = luigi.Parameter()
    hostname = luigi.Parameter('ernest.phys.cmu.edu')
