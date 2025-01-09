from typing import Callable
import luigi
import numpy as np
from numpy.typing import DTypeLike
from thesis_analysis.root_io import RootBranch

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


def get_branch(branch_name: str, dim: int = 1) -> RootBranch:
    BRANCH_TYPES: dict[str, DTypeLike] = {
        'RunNumber': np.uint32,
        'EventNumber': np.ulonglong,
        'ComboNumber': np.uint32,
        'Weight': np.float32,
        'E_Beam': np.float32,
        'Px_Beam': np.float32,
        'Py_Beam': np.float32,
        'Pz_Beam': np.float32,
        'NumFinalState': np.int32,
        'E_FinalState': np.float32,
        'Px_FinalState': np.float32,
        'Py_FinalState': np.float32,
        'Pz_FinalState': np.float32,
        'HX_CosTheta': np.float32,
        'HX_Phi': np.float32,
        't_meson': np.float32,
        't_baryon': np.float32,
        't_baryon2': np.float32,
        'RFL1': np.float32,
        'RFL2': np.float32,
        'FS1': np.float32,
        'FS2': np.float32,
        'M_Resonance': np.float32,
        'M_PPiP1': np.float32,
        'M_PPiP2': np.float32,
        'M_PPiM1': np.float32,
        'M_PPiM2': np.float32,
        'LogConf': np.float32,
        'ChiSqDOF': np.float32,
        'RF': np.float32,
        'Proton_P': np.float32,
        'Proton_Theta': np.float32,
        'Proton_dEdx_CDC': np.float32,
        'Proton_dEdx_CDC_integral': np.float32,
        'Proton_dEdx_FDC': np.float32,
        'Proton_dEdx_ST': np.float32,
        'Proton_dEdx_TOF': np.float32,
        'Proton_E_BCAL': np.float32,
        'Proton_E_FCAL': np.float32,
        'Proton_DeltaT_BCAL': np.float32,
        'Proton_DeltaT_TOF': np.float32,
        'Proton_DeltaT_FCAL': np.float32,
        'Proton_Beta_BCAL': np.float32,
        'Proton_Beta_TOF': np.float32,
        'Proton_Beta_FCAL': np.float32,
        'PiPlus1_P': np.float32,
        'PiPlus1_Theta': np.float32,
        'PiPlus1_dEdx_CDC': np.float32,
        'PiPlus1_dEdx_CDC_integral': np.float32,
        'PiPlus1_dEdx_FDC': np.float32,
        'PiPlus1_dEdx_ST': np.float32,
        'PiPlus1_dEdx_TOF': np.float32,
        'PiPlus1_E_BCAL': np.float32,
        'PiPlus1_E_FCAL': np.float32,
        'PiPlus1_DeltaT_BCAL': np.float32,
        'PiPlus1_DeltaT_TOF': np.float32,
        'PiPlus1_DeltaT_FCAL': np.float32,
        'PiPlus1_Beta_BCAL': np.float32,
        'PiPlus1_Beta_TOF': np.float32,
        'PiPlus1_Beta_FCAL': np.float32,
        'PiMinus1_P': np.float32,
        'PiMinus1_Theta': np.float32,
        'PiMinus1_dEdx_CDC': np.float32,
        'PiMinus1_dEdx_CDC_integral': np.float32,
        'PiMinus1_dEdx_FDC': np.float32,
        'PiMinus1_dEdx_ST': np.float32,
        'PiMinus1_dEdx_TOF': np.float32,
        'PiMinus1_E_BCAL': np.float32,
        'PiMinus1_E_FCAL': np.float32,
        'PiMinus1_DeltaT_BCAL': np.float32,
        'PiMinus1_DeltaT_TOF': np.float32,
        'PiMinus1_DeltaT_FCAL': np.float32,
        'PiMinus1_Beta_BCAL': np.float32,
        'PiMinus1_Beta_TOF': np.float32,
        'PiMinus1_Beta_FCAL': np.float32,
        'PiPlus2_P': np.float32,
        'PiPlus2_Theta': np.float32,
        'PiPlus2_dEdx_CDC': np.float32,
        'PiPlus2_dEdx_CDC_integral': np.float32,
        'PiPlus2_dEdx_FDC': np.float32,
        'PiPlus2_dEdx_ST': np.float32,
        'PiPlus2_dEdx_TOF': np.float32,
        'PiPlus2_E_BCAL': np.float32,
        'PiPlus2_E_FCAL': np.float32,
        'PiPlus2_DeltaT_BCAL': np.float32,
        'PiPlus2_DeltaT_TOF': np.float32,
        'PiPlus2_DeltaT_FCAL': np.float32,
        'PiPlus2_Beta_BCAL': np.float32,
        'PiPlus2_Beta_TOF': np.float32,
        'PiPlus2_Beta_FCAL': np.float32,
        'PiMinus2_P': np.float32,
        'PiMinus2_Theta': np.float32,
        'PiMinus2_dEdx_CDC': np.float32,
        'PiMinus2_dEdx_CDC_integral': np.float32,
        'PiMinus2_dEdx_FDC': np.float32,
        'PiMinus2_dEdx_ST': np.float32,
        'PiMinus2_dEdx_TOF': np.float32,
        'PiMinus2_E_BCAL': np.float32,
        'PiMinus2_E_FCAL': np.float32,
        'PiMinus2_DeltaT_BCAL': np.float32,
        'PiMinus2_DeltaT_TOF': np.float32,
        'PiMinus2_DeltaT_FCAL': np.float32,
        'PiMinus2_Beta_BCAL': np.float32,
        'PiMinus2_Beta_TOF': np.float32,
        'PiMinus2_Beta_FCAL': np.float32,
        'KShort1_Z': np.float32,
        'KShort2_Z': np.float32,
    }
    return RootBranch(branch_name, BRANCH_TYPES[branch_name], dim=dim)
