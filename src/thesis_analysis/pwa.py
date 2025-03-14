from abc import ABC, abstractmethod
from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from typing import Literal

import laddu as ld
import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from thesis_analysis.constants import GUIDED_MAX_STEPS, NUM_THREADS
from thesis_analysis.logger import logger
from thesis_analysis.utils import Histogram
from thesis_analysis.wave import Wave


class LoggingObserver(ld.Observer):
    @override
    def callback(self, step: int, status: ld.Status) -> tuple[ld.Status, bool]:
        logger.info(f'Step {step}: {status.fx}')
        return status, False


class GuidedLoggingObserver(ld.Observer):
    def __init__(self, ndof: int):
        self.ndof: int = ndof

    @override
    def callback(self, step: int, status: ld.Status) -> tuple[ld.Status, bool]:
        logger.info(f'Step {step}: {status.fx} ({status.fx / self.ndof})')
        if status.fx / self.ndof <= 1.0:
            return status, True
        return status, False


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


@dataclass
class BinnedFitResult:
    statuses: list[ld.Status]
    waves: set[Wave]
    model: ld.Model
    paths: PathSet
    binning: Binning
    phase_factor: bool

    def get_data_histogram(self) -> Histogram:
        data_datasets = self.paths.get_data_datasets()
        res_mass = ld.Mass([2, 3])
        values = np.concatenate(
            [res_mass.value_on(dataset) for dataset in data_datasets]
        )
        weights = np.concatenate([dataset.weights for dataset in data_datasets])
        return Histogram(
            *np.histogram(
                values,
                bins=self.binning.bins,
                range=self.binning.range,
                weights=weights,
            )
        )

    def get_histograms(self) -> dict[set[Wave], Histogram]:
        data_datasets = self.paths.get_data_datasets_binned(self.binning)
        accmc_datasets = self.paths.get_accmc_datasets_binned(self.binning)
        wavesets = Wave.power_set(self.waves)
        counts: dict[set[Wave], list[float]] = {}
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
                counts[waveset].append(
                    np.sum(
                        [
                            nll.project_with(
                                status.x,
                                Wave.get_waveset_names(
                                    waveset, mass_dependent=False
                                ),
                            )
                            for nll in nlls
                        ]
                    )
                )
        return {
            waveset: Histogram(edges, np.array(counts[waveset]))
            for waveset in wavesets
        }


def fit_binned(
    waves: set[Wave],
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
        statuses.append(best_status)
    return BinnedFitResult(statuses, waves, model, paths, binning, phase_factor)


@dataclass
class UnbinnedFitResult:
    status: ld.Status
    waves: set[Wave]
    model: ld.Model
    paths: PathSet
    phase_factor: bool


def fit_unbinned(
    waves: set[Wave],
    paths: PathSet,
    *,
    p0: NDArray[np.float64] | None = None,
    iters: int,
    phase_factor: bool = False,
) -> UnbinnedFitResult:
    data_datasets = paths.get_data_datasets()
    accmc_datasets = paths.get_accmc_datasets()
    model = Wave.get_model(
        waves, mass_dependent=False, phase_factor=phase_factor
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
    for iiter in range(iters):
        p_init = (
            p0
            if p0 is not None
            else rng.uniform(-1000.0, 1000.0, len(nll.parameters))
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
    return UnbinnedFitResult(best_status, waves, model, paths, phase_factor)


@dataclass
class BinnedFitResultUncertainty:
    samples: list[list[NDArray[np.float64]]]
    fit_result: BinnedFitResult
    _: KW_ONLY
    uncertainty: Literal['sqrt', 'bootstrap', 'mcmc'] = 'sqrt'

    def get_lower_center_upper(
        self,
    ) -> dict[
        set[Wave],
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    ]:
        if self.uncertainty == 'sqrt':
            histograms = self.fit_result.get_histograms()
            return {
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
        data_datasets = self.fit_result.paths.get_data_datasets_binned(
            self.fit_result.binning
        )
        accmc_datasets = self.fit_result.paths.get_accmc_datasets_binned(
            self.fit_result.binning
        )
        wavesets = Wave.power_set(self.fit_result.waves)
        lower_quantile: dict[set[Wave], list[float]] = {
            waveset: [] for waveset in wavesets
        }
        center_quantile: dict[set[Wave], list[float]] = {
            waveset: [] for waveset in wavesets
        }
        upper_quantile: dict[set[Wave], list[float]] = {
            waveset: [] for waveset in wavesets
        }
        for ibin in range(self.fit_result.binning.bins):
            intensities_in_bin: dict[set[Wave], list[float]] = {
                waveset: [] for waveset in wavesets
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
                    intensities_in_bin[waveset].append(
                        np.sum(
                            [
                                nll.project_with(
                                    sample,
                                    Wave.get_waveset_names(
                                        waveset, mass_dependent=False
                                    ),
                                )
                                for nll in nlls
                            ]
                        )
                    )
            for waveset in wavesets:
                quantiles = np.quantile(
                    intensities_in_bin[waveset], [0.16, 0.5, 0.84]
                )
                lower_quantile[waveset].append(quantiles[0])
                center_quantile[waveset].append(quantiles[1])
                upper_quantile[waveset].append(quantiles[2])
        return {
            waveset: (
                np.array(lower_quantile[waveset]),
                np.array(center_quantile[waveset]),
                np.array(upper_quantile[waveset]),
            )
            for waveset in wavesets
        }


def calculate_bootstrap_uncertainty_binned(
    fit_result: BinnedFitResult,
    *,
    nboot: int = 20,
) -> BinnedFitResultUncertainty:
    data_datasets = fit_result.paths.get_data_datasets_binned(
        fit_result.binning
    )
    accmc_datasets = fit_result.paths.get_accmc_datasets_binned(
        fit_result.binning
    )
    samples: list[list[NDArray[np.float64]]] = []
    for ibin in range(fit_result.binning.bins):
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
                observers=[LoggingObserver()],
                threads=NUM_THREADS,
                skip_hessian=True,
            )
            if status.converged:
                bin_samples.append(status.x)
        samples.append(bin_samples)
    return BinnedFitResultUncertainty(
        samples, fit_result, uncertainty='bootstrap'
    )


class CustomAutocorrelationObserver(ld.MCMCObserver):
    def __init__(
        self,
        nlls: list[ld.NLL],
        waves: set[Wave],
        ncheck: int = 20,
        dact: float = 0.05,
        nact: int = 20,
        discard: float = 0.5,
    ) -> None:
        self.nlls: list[ld.NLL] = nlls
        self.ncheck: int = ncheck
        self.dact: float = dact
        self.nact: int = nact
        self.discard: float = discard
        self.latest_tau: float = np.inf
        self.waves: set[Wave] = waves
        self.wavesets: list[set[Wave]] = Wave.power_set(waves)
        self.waveset_results: dict[set[Wave], list[list[float]]] = {
            waveset: [] for waveset in self.wavesets
        }

    @override
    def callback(
        self, step: int, ensemble: ld.Ensemble
    ) -> tuple[ld.Ensemble, bool]:
        latest_step = ensemble.get_chain()[:, -1, :]
        waveset_results_list: dict[set[Wave], list[float]] = {
            waveset: [] for waveset in self.wavesets
        }
        for i_walker in range(ensemble.dimension[0]):
            for waveset in self.wavesets:
                amplitude_names = Wave.get_waveset_names(
                    waveset, mass_dependent=False
                )
                waveset_results_list[waveset].append(
                    np.sum(
                        [
                            nll.project_with(
                                latest_step[i_walker], amplitude_names
                            )
                            for nll in self.nlls
                        ]
                    )
                )
        for waveset in self.wavesets:
            self.waveset_results[waveset].append(waveset_results_list[waveset])
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
            logger.info('End of custom Autocorrelation check')
            converged = (tau * self.nact < step) and (
                abs(self.latest_tau - tau) / tau < self.dact
            )
            self.latest_tau = float(tau)
            return (ensemble, bool(converged))

        return (ensemble, False)


def calculate_mcmc_uncertainty_binned(
    fit_result: BinnedFitResult,
    *,
    nwalkers: int = 100,
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
        caco = CustomAutocorrelationObserver(nlls_clone, fit_result.waves)
        ensemble = nll.mcmc(p0, 30000, observers=caco)
        n_steps_burned = ensemble.dimension[1] - int(caco.latest_tau * 10)
        excess_steps = n_steps_burned - nsteps_min
        thin = 1 if excess_steps < 0 else n_steps_burned // nsteps_min
        samples.append(
            [
                sample
                for sample in ensemble.get_flat_chain(
                    burn=int(caco.latest_tau * 10), thin=thin
                )
            ]
        )
    return BinnedFitResultUncertainty(samples, fit_result, uncertainty='mcmc')


def fit_guided(
    binned_fit_result: BinnedFitResult,
    binned_fit_result_uncertainty: BinnedFitResultUncertainty | None,
    *,
    iters: int,
) -> UnbinnedFitResult:
    data_datasets = binned_fit_result.paths.get_data_datasets()
    accmc_datasets = binned_fit_result.paths.get_accmc_datasets()
    histograms = binned_fit_result.get_histograms()
    res_mass = ld.Mass([2, 3])
    manager = ld.LikelihoodManager()
    wavesets = Wave.power_set(binned_fit_result.waves)
    error_sets = None
    if binned_fit_result_uncertainty is not None:
        quantiles = binned_fit_result_uncertainty.get_lower_center_upper()
        error_sets = [
            (quantiles[waveset][2] - quantiles[waveset][1]) / 2
            for waveset in wavesets
        ]
    likelihood_model = ld.likelihood_sum(
        [
            manager.register(
                ld.experimental.BinnedGuideTerm(
                    ld.NLL(binned_fit_result.model, ds_data, ds_accmc),
                    res_mass,
                    amplitude_sets=[
                        Wave.get_waveset_names(waveset, mass_dependent=True)
                        for waveset in wavesets
                    ],
                    bins=binned_fit_result.binning.bins,
                    range=binned_fit_result.binning.range,
                    count_sets=[
                        histograms[waveset].counts for waveset in wavesets
                    ],
                    error_sets=error_sets,
                )
            )
            for ds_data, ds_accmc in zip(data_datasets, accmc_datasets)
        ]
    )
    nll = manager.load(likelihood_model)
    ndof = (
        binned_fit_result.binning.bins * len(wavesets) - len(nll.parameters)
    ) * len(data_datasets)
    best_nll = np.inf
    best_status = None
    rng = np.random.default_rng(0)
    for iiter in range(iters):
        p_init = rng.uniform(-1000.0, 1000.0, len(nll.parameters))
        status = nll.minimize(
            p_init,
            observers=[GuidedLoggingObserver(ndof)],
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
    return UnbinnedFitResult(
        best_status,
        binned_fit_result.waves,
        binned_fit_result.model,
        binned_fit_result.paths,
        binned_fit_result.phase_factor,
    )
