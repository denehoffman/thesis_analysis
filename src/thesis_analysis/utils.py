from dataclasses import dataclass

import luigi
import numpy as np
from iminuit import Minuit
from numpy.typing import NDArray

from thesis_analysis.constants import RUN_PERIODS


@dataclass
class ScalingFactors:
    hodoscope_hi_factor: float
    hodoscope_lo_factor: float
    microscope_factor: float
    energy_bound_hi: float
    energy_bound_lo: float


class CCDBData:
    def __init__(
        self,
        accidental_scaling_factors: dict[
            int,
            ScalingFactors,
        ],
    ):
        self.accidental_scaling_factors: dict[int, ScalingFactors] = (
            accidental_scaling_factors
        )

    def get_scaling(
        self,
        run_number: int,
        beam_energy: float,
    ) -> float:
        factors = self.accidental_scaling_factors.get(run_number)
        if factors is None:
            return 1.0
        if beam_energy > factors.energy_bound_hi:
            return factors.hodoscope_hi_factor
        if beam_energy > factors.energy_bound_lo:
            return factors.microscope_factor
        return factors.hodoscope_lo_factor

    def get_accidental_weight(
        self,
        run_number: int,
        beam_energy: float,
        rf: float,
        weight: float,
        *,
        is_mc: bool,
    ) -> float:
        relative_beam_bucket = int(np.floor(rf / 4.008016032) + 0.5)
        if abs(relative_beam_bucket) == 1:
            return 0.0
        if abs(relative_beam_bucket) == 0:
            return weight
        scale = (
            1.0
            if is_mc
            else self.get_scaling(
                run_number,
                beam_energy,
            )
        )
        return weight * (-scale / 8.0)


@dataclass
class Histogram:
    counts: NDArray[np.floating]
    bins: NDArray[np.floating]

    @staticmethod
    def sum(histograms: list['Histogram']) -> 'Histogram | None':
        if not histograms:
            return None
        bins = histograms[0].bins
        for histogram in histograms:
            assert histogram.bins == bins
        counts = np.sum(
            np.array([histogram.counts for histogram in histograms]), axis=0
        )
        return Histogram(counts, bins)


class RCDBData:
    def __init__(
        self,
        pol_angles: dict[int, tuple[str, str, float | None]],
        pol_magnitudes: dict[str, dict[str, Histogram]],
    ):
        self.pol_angles: dict[int, tuple[str, str, float | None]] = pol_angles
        self.pol_magnitudes: dict[str, dict[str, Histogram]] = pol_magnitudes

    def get_eps_xy(
        self,
        run_number: int,
        beam_energy: float,
    ) -> tuple[float, float, bool]:
        pol_angle = self.pol_angles.get(run_number)
        if pol_angle is None:
            return (np.nan, np.nan, False)
        run_period, pol_name, angle = pol_angle
        if angle is None:
            return (np.nan, np.nan, False)
        pol_hist = self.pol_magnitudes[run_period][pol_name]
        energy_index = np.digitize(beam_energy, pol_hist.bins)
        if energy_index >= len(pol_hist.counts):
            return (np.nan, np.nan, False)
        magnitude: float = pol_hist.counts[energy_index]
        return magnitude * np.cos(angle), magnitude * np.sin(angle), True


@dataclass
class FitResult:
    n2ll: float
    n_parameters: int
    n_events: int
    values: dict[str, float]
    errors: dict[str, float]

    @staticmethod
    def from_minuit(minuit: Minuit, n_events: int) -> 'FitResult':
        assert minuit.fval is not None
        return FitResult(
            minuit.fval,
            minuit.nfit,
            n_events,
            minuit.values.to_dict(),
            minuit.errors.to_dict(),
        )

    @property
    def aic(self) -> float:
        return 2.0 * self.n_parameters + self.n2ll  # 2k + -2ln(L)

    @property
    def bic(self) -> float:
        return (
            self.n_parameters * np.log(self.n_events) + self.n2ll
        )  # kln(n) + -2ln(L)


def get_plot_requirements(
    data_type,
    original,
    chisqdof,
    ksb_costheta,
    cut_baryons,
    splot_method,
    nsig,
    nbkg,
) -> list[luigi.Task]:
    from thesis_analysis.tasks.accid_and_pol import AccidentalsAndPolarization
    from thesis_analysis.tasks.baryon_cut import BaryonCut
    from thesis_analysis.tasks.chisqdof import ChiSqDOF
    from thesis_analysis.tasks.data import GetData
    from thesis_analysis.tasks.splot_weights import SPlotWeights

    if original:
        return [GetData(data_type, run_period) for run_period in RUN_PERIODS]
    elif chisqdof is None:
        return [
            AccidentalsAndPolarization(data_type, run_period)
            for run_period in RUN_PERIODS
        ]
    elif ksb_costheta is None:
        return [
            ChiSqDOF(data_type, run_period, chisqdof)
            for run_period in RUN_PERIODS
        ]
    elif nsig is None and nbkg is None:
        return [
            BaryonCut(
                data_type,
                run_period,
                chisqdof,
                ksb_costheta,
                cut_baryons,
            )
            for run_period in RUN_PERIODS
        ]
    elif splot_method is not None and nsig is not None and nbkg is not None:
        return [
            SPlotWeights(
                data_type,
                run_period,
                chisqdof,
                ksb_costheta,
                cut_baryons,
                splot_method,
                nsig,
                nbkg,
            )
            for run_period in RUN_PERIODS
        ]
    else:
        raise Exception('Invalid requirements for plotting!')


def get_plot_paths(
    names: list[str],
    data_type,
    original,
    chisqdof,
    ksb_costheta,
    cut_baryons,
    splot_method,
    nsig,
    nbkg,
) -> list[luigi.LocalTarget]:
    from thesis_analysis.paths import Paths

    path = Paths.plots
    if original:
        return [
            luigi.LocalTarget(path / f'{name}_{data_type}.png')
            for name in names
        ]
    elif chisqdof is None:
        return [
            luigi.LocalTarget(path / f'{name}_{data_type}_accpol.png')
            for name in names
        ]
    elif ksb_costheta is None:
        return [
            luigi.LocalTarget(
                path / f'{name}_{data_type}_accpol_chisqdof_{chisqdof:.1f}.png'
            )
            for name in names
        ]
    elif nsig is None and nbkg is None:
        return [
            luigi.LocalTarget(
                path
                / f'{name}_{data_type}_accpol_chisqdof_{chisqdof:.1f}_ksb_costheta_{ksb_costheta:.2f}{"mesons" if cut_baryons else "baryons"}.png'
            )
            for name in names
        ]
    elif splot_method is not None and nsig is not None and nbkg is not None:
        return [
            luigi.LocalTarget(
                path
                / f'{name}_{data_type}_accpol_chisqdof_{chisqdof:.1f}_ksb_costheta_{ksb_costheta:.2f}{"mesons" if cut_baryons else "baryons"}_splot_{splot_method}_{nsig}s_{nbkg}b.png'
            )
            for name in names
        ]
    else:
        raise Exception('Invalid requirements for plotting!')
