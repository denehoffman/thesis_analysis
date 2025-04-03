from dataclasses import dataclass
from typing import TypedDict

import luigi
import numpy as np
from numpy.typing import DTypeLike, NDArray

RUN_PERIODS = ['s17', 's18', 'f18', 's20']
RUN_PERIOD_LABELS = ['Spring 2017', 'Spring 2018', 'Fall 2018', 'Spring 2020']
GLUEX_PHASE_I = ['s17', 's18', 'f18']
GLUEX_PHASE_II = ['s20']
DATA_TYPES = ['data', 'accmc', 'genmc', 'bkgmc']
DATA_TYPE_TO_LATEX = {
    'data': 'Data',
    'accmc': '$K_S^0K_S^0$ MC',
    'genmc': '$K^0_K^0$ MC (generated)',
    'bkgmc': r'$4\pi$ MC',
}
MC_TYPES = ['accmc', 'genmc', 'bkgmc']
CHISQDOF = [3.0, 3.5, 4.0, 4.5, 5.0]
HOSTNAME = 'ernest.phys.cmu.edu'
SPLOT_SB = [(1, 1), (1, 2), (2, 1), (2, 2)]

RUN_RANGES = {
    's17': (30000, 39999),
    's18': (40000, 49999),
    'f18': (50000, 59999),
    's20': (71275, 79999),
}

NUM_THREADS = 8
NBINS = 40
RANGE = (1.0, 2.0)
GUIDED_MAX_STEPS = 300
RFL_RANGE = (0.0, 0.5)
NSIG_BINS_DE = 200

SIG_QUANTILES = [2, 3, 4, 5]


def get_run_period(run_number: int) -> str | None:
    for rp, (lo, hi) in RUN_RANGES.items():
        if lo <= run_number <= hi:
            return rp
    return None


# from GlueX docdb:3977
TRUE_POL_ANGLES = {
    's17': {'0.0': 1.8, '45.0': 47.9, '90.0': 94.5, '135.0': -41.6},
    's18': {'0.0': 4.1, '45.0': 48.5, '90.0': 94.2, '135.0': -42.4},
    'f18': {'0.0': 3.3, '45.0': 48.3, '90.0': 92.9, '135.0': -42.1},
    's20': {'0.0': 1.4, '45.0': 47.1, '90.0': 93.4, '135.0': -42.2},
}


def get_pol_angle(run_period: str | None, angle_deg: str) -> float | None:
    if run_period is None:
        return None
    pol_angle_deg = TRUE_POL_ANGLES[run_period].get(angle_deg)
    if pol_angle_deg is None:
        return None
    return pol_angle_deg * np.pi / 180.0


class global_parameters(luigi.Config):
    username: luigi.Parameter = luigi.Parameter()
    hostname: luigi.Parameter = luigi.Parameter('ernest.phys.cmu.edu')


@dataclass
class RootBranch:
    name: str
    dtype: DTypeLike
    dim: int = 1

    def get_array(self):
        if self.dtype == np.str_:
            return None
        return np.zeros(self.dim, dtype=self.dtype)


class RootBranchDict(TypedDict):
    RunNumber: NDArray[np.uint32]
    EventNumber: NDArray[np.ulonglong]
    ComboNumber: NDArray[np.uint32]
    Weight: NDArray[np.float32]
    E_Beam: NDArray[np.float32]
    Px_Beam: NDArray[np.float32]
    Py_Beam: NDArray[np.float32]
    Pz_Beam: NDArray[np.float32]
    NumFinalState: NDArray[np.int32]
    E_FinalState: NDArray[np.float32]
    Px_FinalState: NDArray[np.float32]
    Py_FinalState: NDArray[np.float32]
    Pz_FinalState: NDArray[np.float32]
    HX_CosTheta: NDArray[np.float32]
    HX_Phi: NDArray[np.float32]
    t_meson: NDArray[np.float32]
    t_baryon: NDArray[np.float32]
    t_baryon2: NDArray[np.float32]
    RFL1: NDArray[np.float32]
    RFL2: NDArray[np.float32]
    FS1: NDArray[np.float32]
    FS2: NDArray[np.float32]
    M_Resonance: NDArray[np.float32]
    M_PPiP1: NDArray[np.float32]
    M_PPiP2: NDArray[np.float32]
    M_PPiM1: NDArray[np.float32]
    M_PPiM2: NDArray[np.float32]
    LogConf: NDArray[np.float32]
    ChiSqDOF: NDArray[np.float32]
    RF: NDArray[np.float32]
    MM2: NDArray[np.float32]
    Proton_Z: NDArray[np.float32]
    Proton_P: NDArray[np.float32]
    Proton_Theta: NDArray[np.float32]
    Proton_dEdx_CDC: NDArray[np.float32]
    Proton_dEdx_CDC_integral: NDArray[np.float32]
    Proton_dEdx_FDC: NDArray[np.float32]
    Proton_dEdx_ST: NDArray[np.float32]
    Proton_dEdx_TOF: NDArray[np.float32]
    Proton_E_BCAL: NDArray[np.float32]
    Proton_E_FCAL: NDArray[np.float32]
    Proton_DeltaT_BCAL: NDArray[np.float32]
    Proton_DeltaT_TOF: NDArray[np.float32]
    Proton_DeltaT_FCAL: NDArray[np.float32]
    Proton_Beta_BCAL: NDArray[np.float32]
    Proton_Beta_TOF: NDArray[np.float32]
    Proton_Beta_FCAL: NDArray[np.float32]
    PiPlus1_P: NDArray[np.float32]
    PiPlus1_Theta: NDArray[np.float32]
    PiPlus1_dEdx_CDC: NDArray[np.float32]
    PiPlus1_dEdx_CDC_integral: NDArray[np.float32]
    PiPlus1_dEdx_FDC: NDArray[np.float32]
    PiPlus1_dEdx_ST: NDArray[np.float32]
    PiPlus1_dEdx_TOF: NDArray[np.float32]
    PiPlus1_E_BCAL: NDArray[np.float32]
    PiPlus1_E_FCAL: NDArray[np.float32]
    PiPlus1_DeltaT_BCAL: NDArray[np.float32]
    PiPlus1_DeltaT_TOF: NDArray[np.float32]
    PiPlus1_DeltaT_FCAL: NDArray[np.float32]
    PiPlus1_Beta_BCAL: NDArray[np.float32]
    PiPlus1_Beta_TOF: NDArray[np.float32]
    PiPlus1_Beta_FCAL: NDArray[np.float32]
    PiMinus1_P: NDArray[np.float32]
    PiMinus1_Theta: NDArray[np.float32]
    PiMinus1_dEdx_CDC: NDArray[np.float32]
    PiMinus1_dEdx_CDC_integral: NDArray[np.float32]
    PiMinus1_dEdx_FDC: NDArray[np.float32]
    PiMinus1_dEdx_ST: NDArray[np.float32]
    PiMinus1_dEdx_TOF: NDArray[np.float32]
    PiMinus1_E_BCAL: NDArray[np.float32]
    PiMinus1_E_FCAL: NDArray[np.float32]
    PiMinus1_DeltaT_BCAL: NDArray[np.float32]
    PiMinus1_DeltaT_TOF: NDArray[np.float32]
    PiMinus1_DeltaT_FCAL: NDArray[np.float32]
    PiMinus1_Beta_BCAL: NDArray[np.float32]
    PiMinus1_Beta_TOF: NDArray[np.float32]
    PiMinus1_Beta_FCAL: NDArray[np.float32]
    PiPlus2_P: NDArray[np.float32]
    PiPlus2_Theta: NDArray[np.float32]
    PiPlus2_dEdx_CDC: NDArray[np.float32]
    PiPlus2_dEdx_CDC_integral: NDArray[np.float32]
    PiPlus2_dEdx_FDC: NDArray[np.float32]
    PiPlus2_dEdx_ST: NDArray[np.float32]
    PiPlus2_dEdx_TOF: NDArray[np.float32]
    PiPlus2_E_BCAL: NDArray[np.float32]
    PiPlus2_E_FCAL: NDArray[np.float32]
    PiPlus2_DeltaT_BCAL: NDArray[np.float32]
    PiPlus2_DeltaT_TOF: NDArray[np.float32]
    PiPlus2_DeltaT_FCAL: NDArray[np.float32]
    PiPlus2_Beta_BCAL: NDArray[np.float32]
    PiPlus2_Beta_TOF: NDArray[np.float32]
    PiPlus2_Beta_FCAL: NDArray[np.float32]
    PiMinus2_P: NDArray[np.float32]
    PiMinus2_Theta: NDArray[np.float32]
    PiMinus2_dEdx_CDC: NDArray[np.float32]
    PiMinus2_dEdx_CDC_integral: NDArray[np.float32]
    PiMinus2_dEdx_FDC: NDArray[np.float32]
    PiMinus2_dEdx_ST: NDArray[np.float32]
    PiMinus2_dEdx_TOF: NDArray[np.float32]
    PiMinus2_E_BCAL: NDArray[np.float32]
    PiMinus2_E_FCAL: NDArray[np.float32]
    PiMinus2_DeltaT_BCAL: NDArray[np.float32]
    PiMinus2_DeltaT_TOF: NDArray[np.float32]
    PiMinus2_DeltaT_FCAL: NDArray[np.float32]
    PiMinus2_Beta_BCAL: NDArray[np.float32]
    PiMinus2_Beta_TOF: NDArray[np.float32]
    PiMinus2_Beta_FCAL: NDArray[np.float32]
    KShort1_Z: NDArray[np.float32]
    KShort2_Z: NDArray[np.float32]
    Topology: NDArray[np.str_]


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
        'MM2': np.float32,
        'Proton_Z': np.float32,
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
        'Topology': np.str_,
    }
    return RootBranch(branch_name, BRANCH_TYPES[branch_name], dim=dim)


SPLOT_CONTROL = 'M_Resonance'
SPLOT_METHODS = ['A', 'B', 'C', 'D', 'E']

BRANCH_NAME_TO_LATEX: dict[str, str] = {
    'RunNumber': 'Run Number',
    'EventNumber': 'Event Number',
    'ComboNumber': 'Combo Number',
    'Weight': 'Event Weight',
    'E_Beam': r'$E_\gamma$',
    'Px_Beam': r'$p_{x,\gamma}$',
    'Py_Beam': r'$p_{y,\gamma}$',
    'Pz_Beam': r'$p_{z,\gamma}$',
    # 'NumFinalState': np.int32,
    # 'E_FinalState': np.float32,
    # 'Px_FinalState': np.float32,
    # 'Py_FinalState': np.float32,
    # 'Pz_FinalState': np.float32,
    'HX_CosTheta': r'$\cos\theta_\text{HX}$',
    'HX_Phi': r'$\varphi_\text{HX}$',
    # 't_meson': np.float32,
    # 't_baryon': np.float32,
    # 't_baryon2': np.float32,
    'RFL1': r'Rest-Frame Lifetime $t_{K_{S,1}^0}$',
    'RFL2': r'Rest-Frame Lifetime $t_{K_{S,2}^0}$',
    # 'FS1': np.float32,
    # 'FS2': np.float32,
    'M_Resonance': '$m(K_S^0K_S^0)$',
    # 'M_PPiP1': np.float32,
    # 'M_PPiP2': np.float32,
    # 'M_PPiM1': np.float32,
    # 'M_PPiM2': np.float32,
    # 'LogConf': np.float32,
    'ChiSqDOF': r'$\chi^2_\nu$',
    # 'RF': np.float32,
    'Proton_Z': r'Proton $z$-vertex',
    # 'Proton_P': np.float32,
    # 'Proton_Theta': np.float32,
    # 'Proton_dEdx_CDC': np.float32,
    # 'Proton_dEdx_CDC_integral': np.float32,
    # 'Proton_dEdx_FDC': np.float32,
    # 'Proton_dEdx_ST': np.float32,
    # 'Proton_dEdx_TOF': np.float32,
    # 'Proton_E_BCAL': np.float32,
    # 'Proton_E_FCAL': np.float32,
    # 'Proton_DeltaT_BCAL': np.float32,
    # 'Proton_DeltaT_TOF': np.float32,
    # 'Proton_DeltaT_FCAL': np.float32,
    # 'Proton_Beta_BCAL': np.float32,
    # 'Proton_Beta_TOF': np.float32,
    # 'Proton_Beta_FCAL': np.float32,
    # 'PiPlus1_P': np.float32,
    # 'PiPlus1_Theta': np.float32,
    # 'PiPlus1_dEdx_CDC': np.float32,
    # 'PiPlus1_dEdx_CDC_integral': np.float32,
    # 'PiPlus1_dEdx_FDC': np.float32,
    # 'PiPlus1_dEdx_ST': np.float32,
    # 'PiPlus1_dEdx_TOF': np.float32,
    # 'PiPlus1_E_BCAL': np.float32,
    # 'PiPlus1_E_FCAL': np.float32,
    # 'PiPlus1_DeltaT_BCAL': np.float32,
    # 'PiPlus1_DeltaT_TOF': np.float32,
    # 'PiPlus1_DeltaT_FCAL': np.float32,
    # 'PiPlus1_Beta_BCAL': np.float32,
    # 'PiPlus1_Beta_TOF': np.float32,
    # 'PiPlus1_Beta_FCAL': np.float32,
    # 'PiMinus1_P': np.float32,
    # 'PiMinus1_Theta': np.float32,
    # 'PiMinus1_dEdx_CDC': np.float32,
    # 'PiMinus1_dEdx_CDC_integral': np.float32,
    # 'PiMinus1_dEdx_FDC': np.float32,
    # 'PiMinus1_dEdx_ST': np.float32,
    # 'PiMinus1_dEdx_TOF': np.float32,
    # 'PiMinus1_E_BCAL': np.float32,
    # 'PiMinus1_E_FCAL': np.float32,
    # 'PiMinus1_DeltaT_BCAL': np.float32,
    # 'PiMinus1_DeltaT_TOF': np.float32,
    # 'PiMinus1_DeltaT_FCAL': np.float32,
    # 'PiMinus1_Beta_BCAL': np.float32,
    # 'PiMinus1_Beta_TOF': np.float32,
    # 'PiMinus1_Beta_FCAL': np.float32,
    # 'PiPlus2_P': np.float32,
    # 'PiPlus2_Theta': np.float32,
    # 'PiPlus2_dEdx_CDC': np.float32,
    # 'PiPlus2_dEdx_CDC_integral': np.float32,
    # 'PiPlus2_dEdx_FDC': np.float32,
    # 'PiPlus2_dEdx_ST': np.float32,
    # 'PiPlus2_dEdx_TOF': np.float32,
    # 'PiPlus2_E_BCAL': np.float32,
    # 'PiPlus2_E_FCAL': np.float32,
    # 'PiPlus2_DeltaT_BCAL': np.float32,
    # 'PiPlus2_DeltaT_TOF': np.float32,
    # 'PiPlus2_DeltaT_FCAL': np.float32,
    # 'PiPlus2_Beta_BCAL': np.float32,
    # 'PiPlus2_Beta_TOF': np.float32,
    # 'PiPlus2_Beta_FCAL': np.float32,
    # 'PiMinus2_P': np.float32,
    # 'PiMinus2_Theta': np.float32,
    # 'PiMinus2_dEdx_CDC': np.float32,
    # 'PiMinus2_dEdx_CDC_integral': np.float32,
    # 'PiMinus2_dEdx_FDC': np.float32,
    # 'PiMinus2_dEdx_ST': np.float32,
    # 'PiMinus2_dEdx_TOF': np.float32,
    # 'PiMinus2_E_BCAL': np.float32,
    # 'PiMinus2_E_FCAL': np.float32,
    # 'PiMinus2_DeltaT_BCAL': np.float32,
    # 'PiMinus2_DeltaT_TOF': np.float32,
    # 'PiMinus2_DeltaT_FCAL': np.float32,
    # 'PiMinus2_Beta_BCAL': np.float32,
    # 'PiMinus2_Beta_TOF': np.float32,
    # 'PiMinus2_Beta_FCAL': np.float32,
    # 'KShort1_Z': np.float32,
    # 'KShort2_Z': np.float32,
}

BRANCH_NAME_TO_LATEX_UNITS: dict[str, str] = {
    # 'RunNumber': 'Run Number',
    # 'EventNumber': 'Event Number',
    # 'ComboNumber': 'Combo Number',
    # 'Weight': 'Event Weight',
    'E_Beam': 'GeV',
    'Px_Beam': 'GeV/$c$',
    'Py_Beam': 'GeV/$c$',
    'Pz_Beam': 'GeV/$c$',
    # 'NumFinalState': np.int32,
    # 'E_FinalState': np.float32,
    # 'Px_FinalState': np.float32,
    # 'Py_FinalState': np.float32,
    # 'Pz_FinalState': np.float32,
    # 'HX_CosTheta': r'\cos\theta_\text{HX}',
    'HX_Phi': 'rad',
    # 't_meson': np.float32,
    # 't_baryon': np.float32,
    # 't_baryon2': np.float32,
    'RFL1': 'ns',
    'RFL2': 'ns',
    # 'FS1': np.float32,
    # 'FS2': np.float32,
    'M_Resonance': 'GeV/$c^{2}$',
    # 'M_PPiP1': np.float32,
    # 'M_PPiP2': np.float32,
    # 'M_PPiM1': np.float32,
    # 'M_PPiM2': np.float32,
    # 'LogConf': np.float32,
    # 'ChiSqDOF': np.float32,
    # 'RF': np.float32,
    'Proton_z': 'cm',
    # 'Proton_P': np.float32,
    # 'Proton_Theta': np.float32,
    # 'Proton_dEdx_CDC': np.float32,
    # 'Proton_dEdx_CDC_integral': np.float32,
    # 'Proton_dEdx_FDC': np.float32,
    # 'Proton_dEdx_ST': np.float32,
    # 'Proton_dEdx_TOF': np.float32,
    # 'Proton_E_BCAL': np.float32,
    # 'Proton_E_FCAL': np.float32,
    # 'Proton_DeltaT_BCAL': np.float32,
    # 'Proton_DeltaT_TOF': np.float32,
    # 'Proton_DeltaT_FCAL': np.float32,
    # 'Proton_Beta_BCAL': np.float32,
    # 'Proton_Beta_TOF': np.float32,
    # 'Proton_Beta_FCAL': np.float32,
    # 'PiPlus1_P': np.float32,
    # 'PiPlus1_Theta': np.float32,
    # 'PiPlus1_dEdx_CDC': np.float32,
    # 'PiPlus1_dEdx_CDC_integral': np.float32,
    # 'PiPlus1_dEdx_FDC': np.float32,
    # 'PiPlus1_dEdx_ST': np.float32,
    # 'PiPlus1_dEdx_TOF': np.float32,
    # 'PiPlus1_E_BCAL': np.float32,
    # 'PiPlus1_E_FCAL': np.float32,
    # 'PiPlus1_DeltaT_BCAL': np.float32,
    # 'PiPlus1_DeltaT_TOF': np.float32,
    # 'PiPlus1_DeltaT_FCAL': np.float32,
    # 'PiPlus1_Beta_BCAL': np.float32,
    # 'PiPlus1_Beta_TOF': np.float32,
    # 'PiPlus1_Beta_FCAL': np.float32,
    # 'PiMinus1_P': np.float32,
    # 'PiMinus1_Theta': np.float32,
    # 'PiMinus1_dEdx_CDC': np.float32,
    # 'PiMinus1_dEdx_CDC_integral': np.float32,
    # 'PiMinus1_dEdx_FDC': np.float32,
    # 'PiMinus1_dEdx_ST': np.float32,
    # 'PiMinus1_dEdx_TOF': np.float32,
    # 'PiMinus1_E_BCAL': np.float32,
    # 'PiMinus1_E_FCAL': np.float32,
    # 'PiMinus1_DeltaT_BCAL': np.float32,
    # 'PiMinus1_DeltaT_TOF': np.float32,
    # 'PiMinus1_DeltaT_FCAL': np.float32,
    # 'PiMinus1_Beta_BCAL': np.float32,
    # 'PiMinus1_Beta_TOF': np.float32,
    # 'PiMinus1_Beta_FCAL': np.float32,
    # 'PiPlus2_P': np.float32,
    # 'PiPlus2_Theta': np.float32,
    # 'PiPlus2_dEdx_CDC': np.float32,
    # 'PiPlus2_dEdx_CDC_integral': np.float32,
    # 'PiPlus2_dEdx_FDC': np.float32,
    # 'PiPlus2_dEdx_ST': np.float32,
    # 'PiPlus2_dEdx_TOF': np.float32,
    # 'PiPlus2_E_BCAL': np.float32,
    # 'PiPlus2_E_FCAL': np.float32,
    # 'PiPlus2_DeltaT_BCAL': np.float32,
    # 'PiPlus2_DeltaT_TOF': np.float32,
    # 'PiPlus2_DeltaT_FCAL': np.float32,
    # 'PiPlus2_Beta_BCAL': np.float32,
    # 'PiPlus2_Beta_TOF': np.float32,
    # 'PiPlus2_Beta_FCAL': np.float32,
    # 'PiMinus2_P': np.float32,
    # 'PiMinus2_Theta': np.float32,
    # 'PiMinus2_dEdx_CDC': np.float32,
    # 'PiMinus2_dEdx_CDC_integral': np.float32,
    # 'PiMinus2_dEdx_FDC': np.float32,
    # 'PiMinus2_dEdx_ST': np.float32,
    # 'PiMinus2_dEdx_TOF': np.float32,
    # 'PiMinus2_E_BCAL': np.float32,
    # 'PiMinus2_E_FCAL': np.float32,
    # 'PiMinus2_DeltaT_BCAL': np.float32,
    # 'PiMinus2_DeltaT_TOF': np.float32,
    # 'PiMinus2_DeltaT_FCAL': np.float32,
    # 'PiMinus2_Beta_BCAL': np.float32,
    # 'PiMinus2_Beta_TOF': np.float32,
    # 'PiMinus2_Beta_FCAL': np.float32,
    # 'KShort1_Z': np.float32,
    # 'KShort2_Z': np.float32,
}
