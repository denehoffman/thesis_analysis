from abc import ABC, abstractmethod
from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from typing import Literal, override

import laddu as ld
import matplotlib.pyplot as plt
import numpy as np
from corner import corner  # pyright:ignore[reportUnknownVariableType]
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.stats import norm

from thesis_analysis import colors
from thesis_analysis.constants import GUIDED_MAX_STEPS, NUM_THREADS
from thesis_analysis.logger import logger
from thesis_analysis.utils import Histogram
from thesis_analysis.wave import Wave


@dataclass
class Binning:
    bins: int
    range: tuple[float, float]

    @property
    def edges(self) -> NDArray[np.float64]:
        return np.linspace(
            *self.range, self.bins + 1, endpoint=True, dtype=np.float64
        )


class PathSet(ABC):
    @property
    @abstractmethod
    def data_paths(self) -> list[Path]: ...

    @property
    @abstractmethod
    def accmc_paths(self) -> list[Path]: ...

    def get_data_datasets(
        self,
    ) -> list[ld.Dataset]:
        return [
            ld.open_amptools(path, pol_in_beam=True) for path in self.data_paths
        ]

    def get_data_datasets_binned(
        self,
        binning: Binning,
    ) -> list[ld.BinnedDataset]:
        datasets = self.get_data_datasets()
        res_mass = ld.Mass([2, 3])
        return [
            dataset.bin_by(res_mass, binning.bins, binning.range)
            for dataset in datasets
        ]

    def get_accmc_datasets(
        self,
    ) -> list[ld.Dataset]:
        return [
            ld.open_amptools(path, pol_in_beam=True)
            for path in self.accmc_paths
        ]

    def get_accmc_datasets_binned(
        self,
        binning: Binning,
    ) -> list[ld.BinnedDataset]:
        datasets = self.get_accmc_datasets()
        res_mass = ld.Mass([2, 3])
        return [
            dataset.bin_by(res_mass, binning.bins, binning.range)
            for dataset in datasets
        ]


@dataclass
class SinglePathSet(PathSet):
    data: Path
    accmc: Path

    @property
    @override
    def data_paths(self) -> list[Path]:
        return [self.data]

    @property
    @override
    def accmc_paths(self) -> list[Path]:
        return [self.accmc]


@dataclass
class FullPathSet(PathSet):
    s17: SinglePathSet
    s18: SinglePathSet
    f18: SinglePathSet
    s20: SinglePathSet

    @property
    @override
    def data_paths(self) -> list[Path]:
        return (
            self.s17.data_paths
            + self.s18.data_paths
            + self.f18.data_paths
            + self.s20.data_paths
        )

    @property
    @override
    def accmc_paths(self) -> list[Path]:
        return (
            self.s17.accmc_paths
            + self.s18.accmc_paths
            + self.f18.accmc_paths
            + self.s20.accmc_paths
        )


class LoggingObserver(ld.Observer):
    @override
    def callback(self, step: int, status: ld.Status) -> tuple[ld.Status, bool]:
        logger.info(f'Step {step}: {status.fx} {status.x}')
        return status, False


def add_parameter_text(
    fig: Figure,
    param_names: list[str],
    param_values: NDArray[np.float64],
    prev_values: NDArray[np.float64] | None,
):
    num_params = len(param_names)
    min_font_size, max_font_size = 8, 14
    min_spacing, max_spacing = 0.015, 0.04
    font_size = max(min_font_size, min(max_font_size, 14 - 0.2 * num_params))
    y_step = max(min_spacing, min(max_spacing, 0.85 / num_params))
    differences = (
        param_values - prev_values
        if prev_values is not None
        else np.zeros_like(param_values)
    )
    y_start = 0.95
    fig.subplots_adjust(right=0.7)
    ax_text = fig.add_axes((0.72, 0.1, 0.25, 0.8))
    ax_text.axis('off')
    for i, (name, value, diff) in enumerate(
        zip(param_names, param_values, differences)
    ):
        y_pos = y_start - i * y_step
        color = 'blue' if diff > 0 else 'red'
        arrow = '↑' if diff > 0 else '↓'
        ax_text.text(
            0,
            y_pos,
            f'{name:10}',
            fontsize=font_size,
            verticalalignment='center',
        )
        ax_text.text(
            0.3,
            y_pos,
            f'{value:.4f}',
            fontsize=font_size,
            verticalalignment='center',
        )
        ax_text.text(
            0.6,
            y_pos,
            f'{diff:+.4f} {arrow}',
            fontsize=font_size,
            verticalalignment='center',
            color=color,
        )


class GuidedLoggingObserver(ld.Observer):
    def __init__(
        self,
        masses: list[NDArray[np.float64]],
        n_accmc_weighted: list[float],
        nlls: list[ld.NLL],
        wavesets: list[list[Wave]],
        histograms: dict[int, Histogram],
        error_bars: dict[
            int,
            tuple[
                NDArray[np.float64],
                NDArray[np.float64],
                NDArray[np.float64],
            ],
        ],
        *,
        binning: Binning,
        phase_factor: bool,
        ndof: int,
    ):
        self.masses: list[NDArray[np.float64]] = masses
        self.n_accmc_weighted: list[float] = n_accmc_weighted
        self.n_accmc_tot: float = sum(self.n_accmc_weighted)
        self.nlls: list[ld.NLL] = nlls
        self.wavesets: list[list[Wave]] = wavesets
        self.histograms: dict[int, Histogram] = histograms
        self.error_bars: dict[
            int,
            tuple[
                NDArray[np.float64],
                NDArray[np.float64],
                NDArray[np.float64],
            ],
        ] = error_bars
        self.binning: Binning = binning
        self.phase_factor: bool = phase_factor
        self.ndof: int = ndof
        self.previous_x: NDArray[np.float64] | None = None

    @override
    def callback(self, step: int, status: ld.Status) -> tuple[ld.Status, bool]:
        logger.info(f'Step {step}: {status.fx} ({status.fx / self.ndof})')
        fig, ax = plt.subplots(
            nrows=len(self.nlls) + 1,
            ncols=len(self.wavesets),
            sharex=True,
            figsize=(6 * len(self.wavesets) + 6, 4 * len(self.nlls) + 4),
        )
        fit_histograms: dict[int, list[Histogram]] = {}
        for waveset in self.wavesets:
            fit_histograms[Wave.encode_waves(waveset)] = [
                Histogram(
                    *np.histogram(
                        self.masses[i],
                        weights=self.nlls[i].project_with(
                            status.x,
                            Wave.get_waveset_names(
                                waveset,
                                mass_dependent=True,
                                phase_factor=self.phase_factor,
                            ),
                        ),
                        bins=self.binning.edges,
                    )
                )
                for i in range(len(self.nlls))
            ]
        for idataset in range(len(self.nlls)):
            for iwaveset in range(len(self.wavesets)):
                waveset = Wave.encode_waves(self.wavesets[iwaveset])
                fit_hist = fit_histograms[waveset][idataset]
                wave_hist = self.histograms[waveset]
                wave_error_bars = self.error_bars[waveset]
                centers = (wave_hist.bins[1:] + wave_hist.bins[:-1]) / 2
                ax[idataset][iwaveset].plot(
                    centers,
                    wave_hist.counts
                    * self.n_accmc_weighted[idataset]
                    / self.n_accmc_tot,
                    marker='.',
                    linestyle='none',
                    color=colors.black,
                )
                ax[idataset][iwaveset].errorbar(
                    centers,
                    wave_error_bars[1]
                    * self.n_accmc_weighted[idataset]
                    / self.n_accmc_tot,
                    yerr=(
                        wave_error_bars[0]
                        * self.n_accmc_weighted[idataset]
                        / self.n_accmc_tot,
                        wave_error_bars[2]
                        * self.n_accmc_weighted[idataset]
                        / self.n_accmc_tot,
                    ),
                    fmt='none',
                    color=colors.black,
                )
                ax[idataset][iwaveset].stairs(
                    fit_hist.counts,
                    fit_hist.bins,
                    color=colors.black,
                    fill=True,
                    alpha=0.2,
                )
                ax[idataset][iwaveset].stairs(
                    fit_hist.counts,
                    fit_hist.bins,
                    baseline=wave_hist.counts
                    * self.n_accmc_weighted[idataset]
                    / self.n_accmc_tot,
                    fill=True,
                    color=colors.red,
                    alpha=0.2,
                )
        itot = len(self.nlls)
        for iwaveset in range(len(self.wavesets)):
            waveset = Wave.encode_waves(self.wavesets[iwaveset])
            fit_counts = np.sum(
                [
                    fit_histograms[waveset][idataset].counts
                    for idataset in range(len(self.nlls))
                ],
                axis=0,
            )
            fit_bins = fit_histograms[waveset][0].bins
            wave_hist = self.histograms[waveset]
            wave_error_bars = self.error_bars[waveset]
            centers = (wave_hist.bins[1:] + wave_hist.bins[:-1]) / 2
            ax[itot][iwaveset].plot(
                centers,
                wave_hist.counts,
                marker='.',
                linestyle='none',
                color=colors.black,
            )
            ax[itot][iwaveset].errorbar(
                centers,
                wave_error_bars[1],
                yerr=(wave_error_bars[0], wave_error_bars[2]),
                fmt='none',
                color=colors.black,
            )
            ax[itot][iwaveset].stairs(
                fit_counts,
                fit_bins,
                color=colors.black,
                fill=True,
                alpha=0.2,
            )
            ax[itot][iwaveset].stairs(
                fit_counts,
                fit_bins,
                baseline=wave_hist.counts,
                fill=True,
                color=colors.red,
                alpha=0.2,
            )
        add_parameter_text(
            fig, self.nlls[0].parameters, status.x, self.previous_x
        )
        plt.savefig('guided_fit.svg')
        plt.close()
        self.previous_x = status.x
        if status.fx / self.ndof <= 1.0:
            return status, True
        return status, False


@dataclass
class BinnedFitResult:
    statuses: list[ld.Status]
    waves: list[Wave]
    model: ld.Model
    paths: PathSet
    binning: Binning
    phase_factor: bool
    data_hist_cache: Histogram | None = None
    fit_histograms_cache: dict[int, Histogram] | None = None

    def get_data_histogram(self) -> Histogram:
        if data_hist := self.data_hist_cache:
            return data_hist
        data_datasets = self.paths.get_data_datasets()
        res_mass = ld.Mass([2, 3])
        values = np.concatenate(
            [res_mass.value_on(dataset) for dataset in data_datasets]
        )
        weights = np.concatenate([dataset.weights for dataset in data_datasets])
        data_hist = Histogram(
            *np.histogram(
                values,
                bins=self.binning.bins,
                range=self.binning.range,
                weights=weights,
            )
        )
        self.data_hist_cache = data_hist
        return data_hist

    def get_histograms(self) -> dict[int, Histogram]:
        if fit_histograms := self.fit_histograms_cache:
            return fit_histograms
        data_datasets = self.paths.get_data_datasets_binned(self.binning)
        accmc_datasets = self.paths.get_accmc_datasets_binned(self.binning)
        wavesets = Wave.power_set(self.waves)
        counts: dict[int, list[float]] = {
            Wave.encode_waves(waveset): [] for waveset in wavesets
        }
        edges = self.binning.edges
        for ibin, status in enumerate(self.statuses):
            nlls = [
                ld.NLL(
                    self.model,
                    ds_data[ibin],
                    ds_accmc[ibin],
                )
                for ds_data, ds_accmc in zip(data_datasets, accmc_datasets)
            ]
            for waveset in wavesets:
                counts[Wave.encode_waves(waveset)].append(
                    np.sum(
                        np.concatenate(
                            [
                                nll.project_with(
                                    status.x,
                                    Wave.get_waveset_names(
                                        waveset,
                                        mass_dependent=False,
                                        phase_factor=self.phase_factor,
                                    ),
                                )
                                for nll in nlls
                            ]
                        )
                    )
                )
        fit_hists = {
            Wave.encode_waves(waveset): Histogram(
                np.array(counts[Wave.encode_waves(waveset)]), edges
            )
            for waveset in wavesets
        }
        self.fit_histograms_cache = fit_hists
        return fit_hists


def fit_binned(
    waves: list[Wave],
    paths: PathSet,
    binning: Binning,
    *,
    iters: int,
    phase_factor: bool = False,
) -> BinnedFitResult:
    data_datasets = paths.get_data_datasets_binned(binning)
    accmc_datasets = paths.get_accmc_datasets_binned(binning)
    model = Wave.get_model(
        waves, mass_dependent=False, phase_factor=phase_factor
    )
    statuses: list[ld.Status] = []
    for ibin in range(binning.bins):
        manager = ld.LikelihoodManager()
        bin_model = ld.likelihood_sum(
            [
                manager.register(
                    ld.NLL(model, ds_data[ibin], ds_accmc[ibin]).as_term()
                )
                for ds_data, ds_accmc in zip(data_datasets, accmc_datasets)
            ]
        )
        nll = manager.load(bin_model)
        best_nll = np.inf
        best_status = None
        rng = np.random.default_rng(0)
        for iiter in range(iters):
            p_init = rng.uniform(-1000.0, 1000.0, len(nll.parameters))
            status = nll.minimize(
                p_init,
                observers=[LoggingObserver()],
                threads=NUM_THREADS,
                skip_hessian=True,
            )
            if status.converged:
                if status.fx < best_nll:
                    best_nll = status.fx
                    best_status = status
        if best_status is None:
            raise Exception('No fit converged')
        best_status_with_hessian = nll.minimize(
            best_status.x,
            threads=NUM_THREADS,
        )
        statuses.append(best_status_with_hessian)
    return BinnedFitResult(statuses, waves, model, paths, binning, phase_factor)


@dataclass
class UnbinnedFitResult:
    status: ld.Status
    waves: list[Wave]
    model: ld.Model
    paths: PathSet
    phase_factor: bool
    data_hist_cache: Histogram | None = None
    fit_histograms_cache: dict[int, list[Histogram]] | None = None

    def get_data_histogram(self, binning: Binning) -> Histogram:
        if data_hist := self.data_hist_cache:
            return data_hist
        data_datasets = self.paths.get_data_datasets()
        res_mass = ld.Mass([2, 3])
        data_hist = Histogram(
            *np.histogram(
                np.concatenate(
                    [
                        res_mass.value_on(data_dataset)
                        for data_dataset in data_datasets
                    ]
                ),
                weights=np.concatenate(
                    [data_dataset.weights for data_dataset in data_datasets]
                ),
                bins=binning.edges,
            )
        )
        self.data_hist_cache = data_hist
        return data_hist

    def get_histograms_by_run_period(
        self, binning: Binning
    ) -> dict[int, list[Histogram]]:
        if fit_histograms := self.fit_histograms_cache:
            return fit_histograms
        data_datasets = self.paths.get_data_datasets()
        accmc_datasets = self.paths.get_accmc_datasets()
        wavesets = Wave.power_set(self.waves)
        histograms: dict[int, list[Histogram]] = {}
        nlls = [
            ld.NLL(
                self.model,
                ds_data,
                ds_accmc,
            )
            for ds_data, ds_accmc in zip(data_datasets, accmc_datasets)
        ]
        res_mass = ld.Mass([2, 3])
        for waveset in wavesets:
            histograms[Wave.encode_waves(waveset)] = [
                Histogram(
                    *np.histogram(
                        res_mass.value_on(accmc_datasets[i]),
                        weights=nlls[i].project_with(
                            self.status.x,
                            Wave.get_waveset_names(
                                waveset,
                                mass_dependent=True,
                                phase_factor=self.phase_factor,
                            ),
                        ),
                        bins=binning.edges,
                    )
                )
                for i in range(len(accmc_datasets))
            ]
        self.fit_histograms_cache = histograms
        return histograms

    def get_histograms(self, binning: Binning) -> dict[int, Histogram]:
        if fit_histograms := self.fit_histograms_cache:
            hists = {
                wave: Histogram.sum(hists)
                for wave, hists in fit_histograms.items()
            }
            return {
                wave: hist for wave, hist in hists.items() if hist is not None
            }
        data_datasets = self.paths.get_data_datasets()
        accmc_datasets = self.paths.get_accmc_datasets()
        wavesets = Wave.power_set(self.waves)
        histograms: dict[int, Histogram] = {}
        nlls = [
            ld.NLL(
                self.model,
                ds_data,
                ds_accmc,
            )
            for ds_data, ds_accmc in zip(data_datasets, accmc_datasets)
        ]
        res_mass = ld.Mass([2, 3])
        for waveset in wavesets:
            histograms[Wave.encode_waves(waveset)] = Histogram(
                *np.histogram(
                    np.concatenate(
                        [
                            res_mass.value_on(accmc_dataset)
                            for accmc_dataset in accmc_datasets
                        ]
                    ),
                    weights=np.concatenate(
                        [
                            nll.project_with(
                                self.status.x,
                                Wave.get_waveset_names(
                                    waveset,
                                    mass_dependent=True,
                                    phase_factor=self.phase_factor,
                                ),
                            )
                            for nll in nlls
                        ]
                    ),
                    bins=binning.edges,
                )
            )
        return histograms


def fit_unbinned(
    waves: list[Wave],
    paths: PathSet,
    *,
    p0: NDArray[np.float64] | None = None,
    iters: int,
    phase_factor: bool = False,
) -> UnbinnedFitResult:
    data_datasets = paths.get_data_datasets()
    accmc_datasets = paths.get_accmc_datasets()
    model = Wave.get_model(
        waves, mass_dependent=True, phase_factor=phase_factor
    )
    manager = ld.LikelihoodManager()
    likelihood_model = ld.likelihood_sum(
        [
            manager.register(ld.NLL(model, ds_data, ds_accmc).as_term())
            for ds_data, ds_accmc in zip(data_datasets, accmc_datasets)
        ]
    )
    nll = manager.load(likelihood_model)
    best_nll = np.inf
    best_status = None
    rng = np.random.default_rng(0)
    init_mag = 0.0001 if phase_factor else 1000.0
    for iiter in range(iters):
        p_init = (
            p0
            if p0 is not None
            else rng.uniform(-init_mag, init_mag, len(nll.parameters))
        )
        status = nll.minimize(
            p_init,
            observers=[LoggingObserver()],
            threads=NUM_THREADS,
            skip_hessian=True,
        )
        if status.converged:
            if status.fx < best_nll:
                best_nll = status.fx
                best_status = status
    if best_status is None:
        raise Exception('No fit converged')
    best_status_with_hessian = nll.minimize(
        best_status.x,
        threads=NUM_THREADS,
    )
    return UnbinnedFitResult(
        best_status_with_hessian, waves, model, paths, phase_factor
    )


@dataclass
class BinnedFitResultUncertainty:
    samples: list[list[NDArray[np.float64]]]
    fit_result: BinnedFitResult
    _: KW_ONLY
    uncertainty: Literal['sqrt', 'bootstrap', 'mcmc'] = 'sqrt'
    lcu_cache: (
        dict[
            str,
            dict[
                int,
                tuple[
                    NDArray[np.float64],
                    NDArray[np.float64],
                    NDArray[np.float64],
                ],
            ],
        ]
        | None
    ) = None

    def get_error_bars(
        self,
        *,
        bootstrap_mode: Literal['SE', 'CI', 'CI-BC'] | str = 'CI-BC',
        confidence_percent: int = 68,
    ) -> dict[
        int,
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    ]:
        lcu = self.get_lower_center_upper(
            bootstrap_mode=bootstrap_mode, confidence_percent=confidence_percent
        )
        error_bars: dict[
            int,
            tuple[
                NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
            ],
        ] = {}
        for wave, wave_lcu in lcu.items():
            yerr = (
                (wave_lcu[2] - wave_lcu[0]) / 2,
                (wave_lcu[2] + wave_lcu[0]) / 2,
                (wave_lcu[2] - wave_lcu[0]) / 2,
            )  # symmetric, prevents issues with wave_lcu[0] < 0
            error_bars[wave] = yerr
        return error_bars

    def get_lower_center_upper(
        self,
        *,
        bootstrap_mode: Literal['SE', 'CI', 'CI-BC'] | str = 'CI-BC',
        confidence_percent: int = 68,
    ) -> dict[
        int,
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    ]:
        if (
            self.lcu_cache is not None
            and (cache := self.lcu_cache.get(bootstrap_mode)) is not None
        ):
            return cache  # this will be fine for non-bootstrap uncertainty estimation, since SE is default
        if self.uncertainty == 'sqrt':
            histograms = self.fit_result.get_histograms()
            lcu = {
                waveset: (
                    np.array(
                        histogram.counts - np.sqrt(histogram.counts),
                        dtype=np.float64,
                    ),
                    np.array(histogram.counts, dtype=np.float64),
                    np.array(
                        histogram.counts + np.sqrt(histogram.counts),
                        dtype=np.float64,
                    ),
                )
                for waveset, histogram in histograms.items()
            }
            if self.lcu_cache is None:
                self.lcu_cache = {bootstrap_mode: lcu}
            else:
                self.lcu_cache[bootstrap_mode] = lcu
            return lcu
        data_datasets = self.fit_result.paths.get_data_datasets_binned(
            self.fit_result.binning
        )
        accmc_datasets = self.fit_result.paths.get_accmc_datasets_binned(
            self.fit_result.binning
        )
        wavesets = Wave.power_set(self.fit_result.waves)
        lower_quantile: dict[int, list[float]] = {
            Wave.encode_waves(waveset): [] for waveset in wavesets
        }
        center_quantile: dict[int, list[float]] = {
            Wave.encode_waves(waveset): [] for waveset in wavesets
        }
        upper_quantile: dict[int, list[float]] = {
            Wave.encode_waves(waveset): [] for waveset in wavesets
        }
        fit_histograms = self.fit_result.get_histograms()
        for ibin in range(self.fit_result.binning.bins):
            intensities_in_bin: dict[int, list[float]] = {
                Wave.encode_waves(waveset): [] for waveset in wavesets
            }
            for isample, sample in enumerate(self.samples[ibin]):
                nlls = (
                    [
                        ld.NLL(
                            self.fit_result.model,
                            ds_data[ibin].bootstrap(isample),
                            ds_accmc[ibin],
                        )
                        for ds_data, ds_accmc in zip(
                            data_datasets, accmc_datasets
                        )
                    ]
                    if self.uncertainty == 'bootstrap'
                    else [
                        ld.NLL(
                            self.fit_result.model,
                            ds_data[ibin],
                            ds_accmc[ibin],
                        )
                        for ds_data, ds_accmc in zip(
                            data_datasets, accmc_datasets
                        )
                    ]
                )
                for waveset in wavesets:
                    intensities_in_bin[Wave.encode_waves(waveset)].append(
                        np.sum(
                            np.concatenate(
                                [
                                    nll.project_with(
                                        sample,
                                        Wave.get_waveset_names(
                                            waveset,
                                            mass_dependent=False,
                                            phase_factor=self.fit_result.phase_factor,
                                        ),
                                    )
                                    for nll in nlls
                                ]
                            )
                        )
                    )
            for waveset in wavesets:
                if (
                    self.uncertainty == 'bootstrap' and bootstrap_mode == 'CI'
                ) or self.uncertainty == 'mcmc':
                    a_lo = (1 - confidence_percent / 100) / 2
                    a_hi = 1 - a_lo
                    quantiles = np.quantile(
                        intensities_in_bin[Wave.encode_waves(waveset)],
                        [a_lo, 0.5, a_hi],
                    )
                elif (
                    self.uncertainty == 'bootstrap'
                    and bootstrap_mode == 'CI-BC'
                ):
                    fit_value = fit_histograms[
                        Wave.encode_waves(waveset)
                    ].counts[ibin]
                    n_b = len(self.samples[ibin])
                    phi = norm().cdf  # pyright:ignore[reportUnknownVariableType]
                    phi_inv = norm().ppf  # pyright:ignore[reportUnknownVariableType]

                    def cdf_b(x: float) -> float:
                        return (
                            np.sum(
                                np.array(
                                    intensities_in_bin[
                                        Wave.encode_waves(waveset)
                                    ]
                                )
                                < x
                            )
                            / n_b
                        )

                    a = (1 - confidence_percent / 100) / 2
                    z_a_lo = phi_inv(a)  # pyright:ignore[reportUnknownVariableType]
                    z_a_hi = phi_inv(1 - a)  # pyright:ignore[reportUnknownVariableType]
                    z_0 = phi_inv(cdf_b(fit_value))  # pyright:ignore[reportUnknownVariableType]
                    a_lo = phi(2 * z_0 + z_a_lo)  # pyright:ignore[reportUnknownVariableType]
                    a_hi = phi(2 * z_0 + z_a_hi)  # pyright:ignore[reportUnknownVariableType]

                    quantiles = np.quantile(
                        intensities_in_bin[Wave.encode_waves(waveset)],
                        [a_lo, 0.5, a_hi],
                    )
                else:
                    # bootstrap-SE only
                    fit_value = fit_histograms[
                        Wave.encode_waves(waveset)
                    ].counts[ibin]
                    wave_intensities_in_bin = intensities_in_bin[
                        Wave.encode_waves(waveset)
                    ]
                    # mean = np.mean(wave_intensities_in_bin)
                    std_err = np.std(wave_intensities_in_bin, ddof=1)
                    quantiles = np.array(
                        [fit_value - std_err, fit_value, fit_value + std_err],
                        dtype=np.float64,
                    )

                lower_quantile[Wave.encode_waves(waveset)].append(quantiles[0])
                center_quantile[Wave.encode_waves(waveset)].append(quantiles[1])
                upper_quantile[Wave.encode_waves(waveset)].append(quantiles[2])
        lcu = {
            Wave.encode_waves(waveset): (
                np.array(lower_quantile[Wave.encode_waves(waveset)]),
                np.array(center_quantile[Wave.encode_waves(waveset)]),
                np.array(upper_quantile[Wave.encode_waves(waveset)]),
            )
            for waveset in wavesets
        }
        if self.lcu_cache is None:
            self.lcu_cache = {bootstrap_mode: lcu}
        else:
            self.lcu_cache[bootstrap_mode] = lcu
        return lcu


def calculate_bootstrap_uncertainty_binned(
    fit_result: BinnedFitResult,
    *,
    nboot: int = 30,
) -> BinnedFitResultUncertainty:
    data_datasets = fit_result.paths.get_data_datasets_binned(
        fit_result.binning
    )
    accmc_datasets = fit_result.paths.get_accmc_datasets_binned(
        fit_result.binning
    )
    samples: list[list[NDArray[np.float64]]] = []
    for ibin in range(fit_result.binning.bins):
        logger.info(f'Bootstrapping {ibin=}')
        bin_samples: list[NDArray[np.float64]] = []
        for iboot in range(nboot):
            manager = ld.LikelihoodManager()
            bin_model = ld.likelihood_sum(
                [
                    manager.register(
                        ld.NLL(
                            fit_result.model,
                            ds_data[ibin].bootstrap(iboot),
                            ds_accmc[ibin],
                        ).as_term()
                    )
                    for ds_data, ds_accmc in zip(data_datasets, accmc_datasets)
                ]
            )
            nll = manager.load(bin_model)
            status = nll.minimize(
                fit_result.statuses[ibin].x,
                # observers=[LoggingObserver()],
                threads=NUM_THREADS,
                skip_hessian=True,
            )
            if status.converged:
                bin_samples.append(status.x)
        samples.append(bin_samples)
    return BinnedFitResultUncertainty(
        samples,
        fit_result,
        uncertainty='bootstrap',
    )


class CustomAutocorrelationObserver(ld.MCMCObserver):
    def __init__(
        self,
        nlls: list[ld.NLL],
        waves: list[Wave],
        *,
        phase_factor: bool,
        min_steps: int = 100,
        ncheck: int = 20,
        dact: float = 0.05,
        nact: int = 20,
        discard: float = 0.5,
    ) -> None:
        self.nlls: list[ld.NLL] = nlls
        self.phase_factor: bool = phase_factor
        self.min_steps: int = min_steps
        self.ncheck: int = ncheck
        self.dact: float = dact
        self.nact: int = nact
        self.discard: float = discard
        self.latest_tau: float = np.inf
        self.waves: list[Wave] = waves
        self.wavesets: list[list[Wave]] = Wave.power_set(waves)
        self.waveset_results: dict[int, list[list[float]]] = {
            Wave.encode(wave): [] for wave in self.waves
        }

    @override
    def callback(
        self, step: int, ensemble: ld.Ensemble
    ) -> tuple[ld.Ensemble, bool]:
        latest_step = ensemble.get_chain()[:, -1, :]
        waveset_results_list: dict[int, list[float]] = {
            Wave.encode(wave): [] for wave in self.waves
        }
        waveset_results_list[Wave.encode_waves(self.waves)] = []
        for i_walker in range(ensemble.dimension[0]):
            for wave in self.waves:
                amplitude_names = Wave.get_amplitude_names(
                    wave,
                    mass_dependent=False,
                    phase_factor=self.phase_factor,
                )
                waveset_results_list[Wave.encode(wave)].append(
                    np.sum(
                        np.concatenate(
                            [
                                nll.project_with(
                                    latest_step[i_walker], amplitude_names
                                )
                                for nll in self.nlls
                            ]
                        )
                    )
                )
            waveset_results_list[Wave.encode_waves(self.waves)].append(
                np.sum(
                    np.concatenate(
                        [
                            nll.project(latest_step[i_walker])
                            for nll in self.nlls
                        ]
                    )
                )
            )
        for wave in self.waves:
            self.waveset_results[Wave.encode(wave)].append(
                waveset_results_list[Wave.encode(wave)]
            )
        if step % self.ncheck == 0:
            logger.info('Checking Autocorrelation (custom)')
            logger.info(
                f'Chain dimensions: {ensemble.dimension[0]} walkers, {ensemble.dimension[1]} steps, {ensemble.dimension[2]} parameters'
            )
            chain = np.array(
                [res for res in self.waveset_results.values()]
            ).transpose(2, 1, 0)  # (walkers, steps, parameters)
            chain = chain[
                :,
                min(
                    int(step * self.discard),
                    int(self.latest_tau * self.nact)
                    if np.isfinite(self.latest_tau)
                    else int(step * self.discard),
                ) :,
            ]
            taus = ld.integrated_autocorrelation_times(chain)
            logger.info(f'τ = [{", ".join(str(t) for t in taus)}]')
            tau = np.mean(taus)
            logger.info(f'mean τ = {tau}')
            logger.info(f'steps to converge = {int(tau * self.nact)}')
            logger.info(f'steps remaining = {int(tau * self.nact) - step}')
            logger.info(
                f'Δτ/τ = {abs(self.latest_tau - tau) / tau} (converges if < {self.dact})'
            )
            logger.info('Creating monitoring plot "corner.svg"')
            nw, ns, nx = chain.shape
            flat_chain = chain.reshape(nw * ns, nx)
            corner(flat_chain, show_titles=True, quantiles=[0.16, 0.5, 0.84])
            plt.savefig('corner.svg')
            plt.close()
            logger.info('End of custom Autocorrelation check')
            converged = (
                (tau * self.nact < step)
                and (abs(self.latest_tau - tau) / tau < self.dact)
                and step > self.min_steps
            )
            self.latest_tau = float(tau)
            return (ensemble, bool(converged))

        return (ensemble, False)


def calculate_mcmc_uncertainty_binned(
    fit_result: BinnedFitResult,
    *,
    nwalkers: int = 10,
    nsteps_min: int = 100,
) -> BinnedFitResultUncertainty:
    data_datasets = fit_result.paths.get_data_datasets_binned(
        fit_result.binning
    )
    accmc_datasets = fit_result.paths.get_accmc_datasets_binned(
        fit_result.binning
    )
    samples: list[list[NDArray[np.float64]]] = []
    for ibin in range(fit_result.binning.bins):
        manager = ld.LikelihoodManager()
        nlls = [
            ld.NLL(
                fit_result.model,
                ds_data[ibin],
                ds_accmc[ibin],
            )
            for ds_data, ds_accmc in zip(data_datasets, accmc_datasets)
        ]
        nlls_clone = [nll for nll in nlls]
        bin_model = ld.likelihood_sum(
            [manager.register(nll.as_term()) for nll in nlls]
        )
        nll = manager.load(bin_model)
        rng = np.random.default_rng(0)
        p0 = fit_result.statuses[ibin].x * rng.normal(
            0.0, scale=0.01, size=(nwalkers, len(nll.parameters))
        )
        caco = CustomAutocorrelationObserver(
            nlls_clone,
            fit_result.waves,
            phase_factor=fit_result.phase_factor,
            ncheck=10,
            nact=5,
        )
        ensemble = nll.mcmc(
            p0,
            2000,
            method='ESS',
            observers=caco,
            ess_moves=[('differential', 0.7), ('gaussian', 0.3)],
            verbose=True,
        )
        n_steps_burned = ensemble.dimension[1] - int(caco.latest_tau * 5)
        excess_steps = n_steps_burned - nsteps_min
        thin = 1 if excess_steps < 0 else n_steps_burned // nsteps_min
        samples.append(
            [
                sample
                for sample in ensemble.get_flat_chain(
                    burn=int(caco.latest_tau * 5), thin=thin
                )
            ]
        )
    return BinnedFitResultUncertainty(samples, fit_result, uncertainty='mcmc')


@dataclass
class GuidedFitResult:
    binned_fit_result: BinnedFitResultUncertainty
    fit_result: UnbinnedFitResult


def fit_guided(
    binned_fit_result_uncertainty: BinnedFitResultUncertainty,
    *,
    p0: NDArray[np.float64] | None = None,
    bootstrap_mode: Literal['SE', 'CI', 'CI-BC'] | str = 'CI-BC',
    iters: int,
) -> GuidedFitResult:
    logger.info('Starting Guided Fit')
    waves = binned_fit_result_uncertainty.fit_result.waves
    binning = binned_fit_result_uncertainty.fit_result.binning
    phase_factor = binned_fit_result_uncertainty.fit_result.phase_factor
    paths = binned_fit_result_uncertainty.fit_result.paths
    model = Wave.get_model(
        waves, mass_dependent=True, phase_factor=phase_factor
    )
    data_datasets = (
        binned_fit_result_uncertainty.fit_result.paths.get_data_datasets()
    )
    accmc_datasets = (
        binned_fit_result_uncertainty.fit_result.paths.get_accmc_datasets()
    )
    n_accmc_tot = sum(
        [accmc_dataset.n_events_weighted for accmc_dataset in accmc_datasets]
    )
    histograms = binned_fit_result_uncertainty.fit_result.get_histograms()
    res_mass = ld.Mass([2, 3])
    manager = ld.LikelihoodManager()
    wavesets = Wave.power_set(waves)
    error_sets = None
    quantiles = binned_fit_result_uncertainty.get_lower_center_upper(
        bootstrap_mode=bootstrap_mode
    )
    error_bars = binned_fit_result_uncertainty.get_error_bars(
        bootstrap_mode=bootstrap_mode
    )
    error_sets = [
        [
            (
                quantiles[Wave.encode_waves(waveset)][2]
                - quantiles[Wave.encode_waves(waveset)][0]
            )
            / 2
            * accmc_dataset.n_events_weighted
            / n_accmc_tot
            for waveset in wavesets
        ]
        for accmc_dataset in accmc_datasets
    ]
    nlls = [
        ld.NLL(model, ds_data, ds_accmc)
        for ds_data, ds_accmc in zip(data_datasets, accmc_datasets)
    ]
    nlls_clone = nlls[::]
    masses = [res_mass.value_on(ds_accmc) for ds_accmc in accmc_datasets]
    likelihood_model = ld.likelihood_sum(
        [
            manager.register(
                ld.experimental.BinnedGuideTerm(
                    nlls[i],
                    res_mass,
                    amplitude_sets=[
                        Wave.get_waveset_names(
                            waveset,
                            mass_dependent=True,
                            phase_factor=phase_factor,
                        )
                        for waveset in wavesets
                    ],
                    bins=binning.bins,
                    range=binning.range,
                    count_sets=[
                        histograms[Wave.encode_waves(waveset)].counts
                        * accmc_dataset.n_events_weighted
                        / n_accmc_tot
                        for waveset in wavesets
                    ],
                    error_sets=error_sets[i],
                )
            )
            for i, (_, accmc_dataset) in enumerate(
                zip(data_datasets, accmc_datasets)
            )
        ]
    )
    nll = manager.load(likelihood_model)
    ndof = binning.bins * len(wavesets) - len(nll.parameters)
    best_nll = np.inf
    best_status = None
    rng = np.random.default_rng(0)
    init_mag = (
        0.0001
        if binned_fit_result_uncertainty.fit_result.phase_factor
        else 1000.0
    )
    for _ in range(iters):
        p_init = (
            p0
            if p0 is not None
            else rng.uniform(-init_mag, init_mag, len(nll.parameters))
        )
        status = nll.minimize(
            p_init,
            observers=GuidedLoggingObserver(
                masses,
                [
                    accmc_dataset.n_events_weighted
                    for accmc_dataset in accmc_datasets
                ],
                nlls_clone,
                wavesets,
                histograms,
                error_bars,
                phase_factor=phase_factor,
                binning=binning,
                ndof=ndof,
            ),
            threads=NUM_THREADS,
            max_steps=GUIDED_MAX_STEPS,
            skip_hessian=True,
        )
        if status.converged:
            if status.fx < best_nll:
                best_nll = status.fx
                best_status = status
    if best_status is None:
        raise Exception('No fit converged')
    return GuidedFitResult(
        binned_fit_result_uncertainty,
        UnbinnedFitResult(
            best_status,
            waves,
            model,
            paths,
            phase_factor,
        ),
    )


def fit_binned_regularized(
    unregularized_fit_result: BinnedFitResult,
    lda: float,
    *,
    gamma: int = 2,
    iters: int,
) -> BinnedFitResult:
    waves = unregularized_fit_result.waves
    binning = unregularized_fit_result.binning
    phase_factor = unregularized_fit_result.phase_factor
    paths = unregularized_fit_result.paths
    data_datasets = paths.get_data_datasets_binned(binning)
    accmc_datasets = paths.get_accmc_datasets_binned(binning)
    model = Wave.get_model(
        waves, mass_dependent=False, phase_factor=phase_factor
    )
    statuses: list[ld.Status] = []
    for ibin in range(binning.bins):
        logger.info(f'Fitting bin {ibin} (regularized, λ={lda}')
        manager = ld.LikelihoodManager()
        bin_model = ld.likelihood_sum(
            [
                manager.register(
                    ld.NLL(model, ds_data[ibin], ds_accmc[ibin]).as_term()
                )
                for ds_data, ds_accmc in zip(data_datasets, accmc_datasets)
            ]
        )
        parameter_sets = [
            [f'{wave.coefficient_name} real', f'{wave.coefficient_name} imag']
            for wave in waves
            if wave.l != 0
        ]
        weight_sets = [
            1
            / np.power(
                np.abs(
                    np.array(
                        [
                            unregularized_fit_result.statuses[ibin].x[
                                unregularized_fit_result.model.parameters.index(
                                    p
                                )
                            ]
                            for p in parameter_set
                        ]
                    )
                ),
                gamma,
            )
            for parameter_set in parameter_sets
        ]
        ridge_term = ld.likelihood_sum(
            [
                manager.register(
                    ld.experimental.Regularizer(
                        parameter_set, lda, 2, weights=weight_set
                    )
                )
                for parameter_set, weight_set in zip(
                    parameter_sets, weight_sets
                )
            ]
        )
        nll = manager.load(bin_model + ridge_term)
        best_nll = np.inf
        best_status = None
        rng = np.random.default_rng(0)
        for iiter in range(iters):
            p_init = rng.uniform(-1000.0, 1000.0, len(nll.parameters))
            status = nll.minimize(
                p_init,
                observers=[LoggingObserver()],
                threads=NUM_THREADS,
                skip_hessian=True,
            )
            if status.converged:
                if status.fx < best_nll:
                    best_nll = status.fx
                    best_status = status
        if best_status is None:
            raise Exception('No fit converged')
        statuses.append(best_status)
    return BinnedFitResult(statuses, waves, model, paths, binning, phase_factor)
