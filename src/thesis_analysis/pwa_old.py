from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import override

import laddu as ld
import numpy as np
from numpy.typing import NDArray

from thesis_analysis.constants import (
    GUIDED_MAX_STEPS,
    NBINS,
    NUM_THREADS,
    RANGE,
)
from thesis_analysis.logger import logger
from thesis_analysis.utils import Histogram


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


class Waveset(Enum):
    TOT = auto()
    S0 = auto()
    D2 = auto()
    P = auto()
    N = auto()
    S0P = auto()
    S0N = auto()
    D2P = auto()


WAVESETS = {
    Waveset.TOT: [
        'z00p',
        'f0p',
        'a0p',
        'z00n',
        'f0n',
        'a0n',
        'z22p',
        'f2',
        'a2',
    ],
    Waveset.S0: ['z00p', 'f0p', 'a0p', 'z00n', 'f0n', 'a0n'],
    Waveset.P: ['z00p', 'f0p', 'a0p', 'z22p', 'f2', 'a2'],
    Waveset.N: ['z00n', 'f0n', 'a0n'],
    Waveset.S0P: ['z00p', 'f0p', 'a0p'],
    Waveset.S0N: ['z00n', 'f0n', 'a0n'],
    Waveset.D2P: ['z22p', 'f2', 'a2'],
    Waveset.D2: ['z22p', 'f2', 'a2'],
}

BINNED_WAVESETS = {
    Waveset.TOT: ['z00p', 's0p', 'z00n', 's0n', 'z22p', 'd2p'],
    Waveset.S0: ['z00p', 's0p', 'z00n', 's0n'],
    Waveset.P: ['z00p', 's0p', 'z22p', 'd2p'],
    Waveset.N: ['z00n', 's0n'],
    Waveset.S0P: ['z00p', 's0p'],
    Waveset.S0N: ['z00n', 's0n'],
    Waveset.D2P: ['z22p', 'd2p'],
    Waveset.D2: ['z22p', 'd2p'],
}


class AnyAnalysisPaths(ABC):
    @abstractmethod
    def get_datasets(self) -> tuple[list[ld.Dataset], list[ld.Dataset]]: ...

    @abstractmethod
    def get_binned_datasets(
        self, *, bins: int, range: tuple[float, float]
    ) -> tuple[list[ld.BinnedDataset], list[ld.BinnedDataset]]: ...

    @property
    @abstractmethod
    def data_paths(self) -> list[Path]: ...

    @property
    @abstractmethod
    def accmc_paths(self) -> list[Path]: ...


@dataclass
class AnalysisPath(AnyAnalysisPaths):
    data: Path
    accmc: Path

    @override
    def get_datasets(self) -> tuple[list[ld.Dataset], list[ld.Dataset]]:
        return (
            [ld.open_amptools(str(self.data), pol_in_beam=True)],
            [ld.open_amptools(str(self.accmc), pol_in_beam=True)],
        )

    @override
    def get_binned_datasets(
        self, *, bins: int, range: tuple[float, float]
    ) -> tuple[list[ld.BinnedDataset], list[ld.BinnedDataset]]:
        res_mass = ld.Mass([2, 3])
        return (
            [
                ld.open_amptools(str(self.data), pol_in_beam=True).bin_by(
                    res_mass, bins, range
                )
            ],
            [
                ld.open_amptools(str(self.accmc), pol_in_beam=True).bin_by(
                    res_mass, bins, range
                )
            ],
        )

    @property
    @override
    def data_paths(self) -> list[Path]:
        return [self.data]

    @property
    @override
    def accmc_paths(self) -> list[Path]:
        return [self.accmc]


@dataclass
class AnalysisPathSet(AnyAnalysisPaths):
    data_s17: Path
    data_s18: Path
    data_f18: Path
    data_s20: Path
    accmc_s17: Path
    accmc_s18: Path
    accmc_f18: Path
    accmc_s20: Path

    @override
    def get_datasets(self) -> tuple[list[ld.Dataset], list[ld.Dataset]]:
        return (
            [
                ld.open_amptools(str(data_path), pol_in_beam=True)
                for data_path in self.data_paths
            ],
            [
                ld.open_amptools(str(accmc_path), pol_in_beam=True)
                for accmc_path in self.accmc_paths
            ],
        )

    @override
    def get_binned_datasets(
        self, *, bins: int, range: tuple[float, float]
    ) -> tuple[list[ld.BinnedDataset], list[ld.BinnedDataset]]:
        res_mass = ld.Mass([2, 3])
        return (
            [
                ld.open_amptools(str(data_path), pol_in_beam=True).bin_by(
                    res_mass, bins, range
                )
                for data_path in self.data_paths
            ],
            [
                ld.open_amptools(str(accmc_path), pol_in_beam=True).bin_by(
                    res_mass, bins, range
                )
                for accmc_path in self.accmc_paths
            ],
        )

    @property
    @override
    def data_paths(self) -> list[Path]:
        return [self.data_s17, self.data_s18, self.data_f18, self.data_s20]

    @property
    @override
    def accmc_paths(self) -> list[Path]:
        return [self.accmc_s17, self.accmc_s18, self.accmc_f18, self.accmc_s20]


def get_binned_model(*, phase_factor: bool = False) -> ld.Model:
    angles = ld.Angles(0, [1], [2], [2, 3])
    polarization = ld.Polarization(0, [1], 0)
    manager = ld.Manager()
    z00p = manager.register(ld.Zlm('z00p', 0, 0, '+', angles, polarization))
    s0p = manager.register(ld.Scalar('s0p', ld.parameter('S0+ re')))
    z00n = manager.register(ld.Zlm('z00n', 0, 0, '-', angles, polarization))
    s0n = manager.register(ld.Scalar('s0n', ld.parameter('S0- re')))
    z22p = manager.register(ld.Zlm('z22p', 2, 2, '+', angles, polarization))
    d2p = manager.register(
        ld.ComplexScalar('d2p', ld.parameter('D2+ re'), ld.parameter('D2+ im'))
    )
    if phase_factor:
        m_resonance = ld.Mass([2, 3])
        m_1 = ld.Mass([2])
        m_2 = ld.Mass([3])
        m_recoil = ld.Mass([1])
        s = ld.Mandelstam([0], [], [2, 3], [1], channel='s')
        kappa = manager.register(
            ld.PhaseSpaceFactor('kappa', m_recoil, m_1, m_2, m_resonance, s)
        )
        pos_re = (
            s0p * kappa * z00p.real() + d2p * kappa * z22p.real()
        ).norm_sqr()
        pos_im = (
            s0p * kappa * z00p.imag() + d2p * kappa * z22p.imag()
        ).norm_sqr()
        neg_re = (s0n * kappa * z00n.real()).norm_sqr()
        neg_im = (s0n * kappa * z00n.imag()).norm_sqr()
    else:
        pos_re = (s0p * z00p.real() + d2p * z22p.real()).norm_sqr()
        pos_im = (s0p * z00p.imag() + d2p * z22p.imag()).norm_sqr()
        neg_re = (s0n * z00n.real()).norm_sqr()
        neg_im = (s0n * z00n.imag()).norm_sqr()
    model = manager.model(pos_re + pos_im + neg_re + neg_im)
    return model


def get_unbinned_model(*, phase_factor: bool = False) -> ld.Model:
    res_mass = ld.Mass([2, 3])
    angles = ld.Angles(0, [1], [2], [2, 3])
    polarization = ld.Polarization(0, [1], 0)
    manager = ld.Manager()
    z00p = manager.register(ld.Zlm('z00p', 0, 0, '+', angles, polarization))
    z00n = manager.register(ld.Zlm('z00n', 0, 0, '-', angles, polarization))
    z22p = manager.register(ld.Zlm('z22p', 2, 2, '+', angles, polarization))
    f0p = manager.register(
        ld.amplitudes.kmatrix.KopfKMatrixF0(
            'f0p',
            (
                (ld.constant(0), ld.constant(0)),
                (ld.parameter('f0(980) re +'), ld.constant(0)),
                (
                    ld.parameter('f0(1370) re +'),
                    ld.parameter('f0(1370) im +'),
                ),
                (
                    ld.parameter('f0(1500) re +'),
                    ld.parameter('f0(1500) im +'),
                ),
                (
                    ld.parameter('f0(1710) re +'),
                    ld.parameter('f0(1710) im +'),
                ),
            ),
            2,
            res_mass,
        )
    )
    f0n = manager.register(
        ld.amplitudes.kmatrix.KopfKMatrixF0(
            'f0n',
            (
                (ld.constant(0), ld.constant(0)),
                (ld.parameter('f0(980) re -'), ld.constant(0)),
                (
                    ld.parameter('f0(1370) re -'),
                    ld.parameter('f0(1370) im -'),
                ),
                (
                    ld.parameter('f0(1500) re -'),
                    ld.parameter('f0(1500) im -'),
                ),
                (
                    ld.parameter('f0(1710) re -'),
                    ld.parameter('f0(1710) im -'),
                ),
            ),
            2,
            res_mass,
        )
    )
    f2 = manager.register(
        ld.amplitudes.kmatrix.KopfKMatrixF2(
            'f2',
            (
                (ld.parameter('f2(1270) re'), ld.parameter('f2(1270)')),
                (ld.parameter('f2(1525) re'), ld.parameter('f2(1525) im')),
                (ld.parameter('f2(1810) re'), ld.parameter('f2(1810) im')),
                (ld.parameter('f2(1950) re'), ld.parameter('f2(1950) im')),
            ),
            2,
            res_mass,
        )
    )
    a0p = manager.register(
        ld.amplitudes.kmatrix.KopfKMatrixA0(
            'a0p',
            (
                (ld.parameter('a0(980) re +'), ld.parameter('a0(980) +')),
                (
                    ld.parameter('a0(1450) re +'),
                    ld.parameter('a0(1450) im +'),
                ),
            ),
            1,
            res_mass,
        )
    )
    a0n = manager.register(
        ld.amplitudes.kmatrix.KopfKMatrixA0(
            'a0n',
            (
                (ld.parameter('a0(980) re -'), ld.parameter('a0(980) -')),
                (
                    ld.parameter('a0(1450) re -'),
                    ld.parameter('a0(1450) im -'),
                ),
            ),
            1,
            res_mass,
        )
    )
    a2 = manager.register(
        ld.amplitudes.kmatrix.KopfKMatrixA2(
            'a2',
            (
                (ld.parameter('a2(1320) re'), ld.parameter('a2(1320)')),
                (ld.parameter('a2(1700) re'), ld.parameter('a2(1700) im')),
            ),
            1,
            res_mass,
        )
    )
    if phase_factor:
        m_resonance = ld.Mass([2, 3])
        m_1 = ld.Mass([2])
        m_2 = ld.Mass([3])
        m_recoil = ld.Mass([1])
        s = ld.Mandelstam([0], [], [2, 3], [1], channel='s')
        kappa = manager.register(
            ld.PhaseSpaceFactor('kappa', m_recoil, m_1, m_2, m_resonance, s)
        )
        pos_re = (
            kappa * z00p.real() * (f0p + a0p) + kappa * z22p.real() * (f2 + a2)
        ).norm_sqr()
        pos_im = (
            kappa * z00p.imag() * (f0p + a0p) + kappa * z22p.imag() * (f2 + a2)
        ).norm_sqr()
        neg_re = (kappa * z00n.real() * (f0n + a0n)).norm_sqr()
        neg_im = (kappa * z00n.imag() * (f0n + a0n)).norm_sqr()
    else:
        pos_re = (
            z00p.real() * (f0p + a0p) + z22p.real() * (f2 + a2)
        ).norm_sqr()
        pos_im = (
            z00p.imag() * (f0p + a0p) + z22p.imag() * (f2 + a2)
        ).norm_sqr()
        neg_re = (z00n.real() * (f0n + a0n)).norm_sqr()
        neg_im = (z00n.imag() * (f0n + a0n)).norm_sqr()
    model = manager.model(pos_re + pos_im + neg_re + neg_im)
    return model


@dataclass
class UnbinnedFitResult:
    statuses: list[ld.Status]
    i_best: int
    paths: AnyAnalysisPaths
    model: ld.Model

    @property
    def best_status(self) -> ld.Status:
        return self.statuses[self.i_best]

    def get_datasets(self) -> tuple[list[ld.Dataset], list[ld.Dataset]]:
        return self.paths.get_datasets()

    def get_nlls(
        self, datasets: tuple[list[ld.Dataset], list[ld.Dataset]]
    ) -> list[ld.NLL]:
        return [
            ld.NLL(self.model, data_dataset, accmc_dataset)
            for data_dataset, accmc_dataset in zip(*datasets)
        ]

    def project(
        self,
        nlls: list[ld.NLL],
        *,
        threads: int | None = None,
    ) -> NDArray[np.float64]:
        return np.concatenate(
            [
                nll.project(self.statuses[self.i_best].x, threads=threads)
                for nll in nlls
            ]
        )

    def project_with(
        self,
        waveset: Waveset,
        nlls: list[ld.NLL],
        *,
        threads: int | None = None,
    ) -> NDArray[np.float64]:
        return np.concatenate(
            [
                nll.project_with(
                    self.statuses[self.i_best].x,
                    WAVESETS[waveset],
                    threads=threads,
                )
                for nll in nlls
            ]
        )

    def get_hist(
        self,
        datasets: list[ld.Dataset],
        *,
        bins: int,
        range: tuple[float, float],
        weights: NDArray[np.float64] | None = None,
    ) -> Histogram:
        res_mass = ld.Mass([2, 3])
        data = sum(datasets)
        assert data != 0
        counts, edges = np.histogram(
            res_mass.value_on(data),
            weights=data.weights if weights is None else weights,
            bins=bins,
            range=range,
        )
        return Histogram(counts, edges)


@dataclass
class BinnedFitResultBin:
    status: ld.Status
    count_data: float
    count_fit: float
    count_s0: float
    count_p: float
    count_s0p: float
    count_s0n: float
    count_d2p: float

    @property
    def count_n(self) -> float:
        return self.count_s0n

    @property
    def count_d2(self) -> float:
        return self.count_d2p


class BinnedFitResult:
    def __init__(
        self,
        results: list[BinnedFitResultBin],
        edges: NDArray[np.float64],
        paths: AnyAnalysisPaths,
        model: ld.Model,
    ):
        self.paths: AnyAnalysisPaths = paths
        self.model: ld.Model = model
        self.results: list[BinnedFitResultBin] = results
        self.edges: NDArray[np.float64] = edges
        self.waveset_hists: dict[Waveset, Histogram] = {
            Waveset.TOT: Histogram(
                np.array([res.count_fit for res in results]), edges
            ),
            Waveset.S0: Histogram(
                np.array([res.count_s0 for res in results]), edges
            ),
            Waveset.D2: Histogram(
                np.array([res.count_d2 for res in results]), edges
            ),
            Waveset.P: Histogram(
                np.array([res.count_p for res in results]), edges
            ),
            Waveset.N: Histogram(
                np.array([res.count_n for res in results]), edges
            ),
            Waveset.S0P: Histogram(
                np.array([res.count_s0p for res in results]), edges
            ),
            Waveset.S0N: Histogram(
                np.array([res.count_s0n for res in results]), edges
            ),
            Waveset.D2P: Histogram(
                np.array([res.count_d2p for res in results]), edges
            ),
        }
        self.data_hist: Histogram = Histogram(
            np.array([res.count_data for res in results]), edges
        )
        self.fit_hist: Histogram = Histogram(
            np.array([res.count_fit for res in results]), edges
        )

    @property
    def range(self) -> tuple[float, float]:
        return (self.edges[0], self.edges[-1])

    @property
    def bins(self) -> int:
        return len(self.results)

    def get_nlls(
        self, *, bootstrap_seed: int | None = None
    ) -> list[list[ld.NLL]]:
        binned_datasets = self.paths.get_binned_datasets(
            bins=self.bins, range=self.range
        )
        if bootstrap_seed is None:
            return [
                [
                    ld.NLL(self.model, ds_data[i], ds_accmc[i])
                    for ds_data, ds_accmc in zip(*binned_datasets)
                ]
                for i in range(self.bins)
            ]
        else:
            return [
                [
                    ld.NLL(
                        self.model,
                        ds_data[i].bootstrap(bootstrap_seed),
                        ds_accmc[i].bootstrap(bootstrap_seed),
                    )
                    for ds_data, ds_accmc in zip(*binned_datasets)
                ]
                for i in range(self.bins)
            ]


def fit_unbinned(
    paths: AnalysisPathSet,
    *,
    p0: NDArray[np.float64] | None = None,
    niters: int | None = None,
    phase_factor: bool = False,
) -> UnbinnedFitResult:
    if p0 is not None:
        niters = 1
    if niters is None:
        niters = 1
    datasets = paths.get_datasets()
    model = get_unbinned_model(phase_factor=phase_factor)
    nlls = [
        ld.NLL(model, ds_data, ds_accmc) for ds_data, ds_accmc in zip(*datasets)
    ]

    manager = ld.LikelihoodManager()
    s17 = manager.register(nlls[0].as_term())
    s18 = manager.register(nlls[1].as_term())
    f18 = manager.register(nlls[2].as_term())
    s20 = manager.register(nlls[3].as_term())
    likelihood_model = s17 + s18 + f18 + s20
    total_nll = manager.load(likelihood_model)
    logger.info('Model loaded')
    logger.info('Computing one NLL evaluation...')
    logger.info(
        f'First evaluation: {total_nll.evaluate([1.0] * len(total_nll.parameters))}'
    )
    all_statuses: list[ld.Status] = []
    status = None
    i_best = None
    best_nll = np.inf
    rng = np.random.default_rng(0)
    for iiter in range(niters):
        logger.info(f'Fitting iteration {iiter}')
        p_init = (
            p0
            if p0 is not None
            else rng.uniform(-1000.0, 1000.0, len(total_nll.parameters))
        )
        logger.debug(f'Initial p0: {p_init}')
        iter_status = total_nll.minimize(
            p_init,
            observers=[LoggingObserver()],
            threads=NUM_THREADS,
            skip_hessian=True,
        )
        all_statuses.append(iter_status)
        if iter_status.fx < best_nll:
            status = iter_status
            i_best = iiter
            best_nll = iter_status.fx
    if status is None or i_best is None:
        logger.error('All fits failed!')
        raise Exception('All fits failed!')
    logger.success(f'Done!\n{status}')
    return UnbinnedFitResult(all_statuses, i_best, paths, model)


def fit_binned(
    paths: AnyAnalysisPaths,
    *,
    nbins: int,
    niters: int,
    phase_factor: bool = False,
) -> BinnedFitResult:
    binned_datasets = paths.get_binned_datasets(bins=nbins, range=RANGE)
    results: list[BinnedFitResultBin] = []
    model = get_binned_model(phase_factor=phase_factor)
    for ibin in range(nbins):
        nlls = [
            ld.NLL(model, ds_data[ibin], ds_accmc[ibin])
            for ds_data, ds_accmc in zip(*binned_datasets)
        ]
        logger.info(f'Fitting bin {ibin}')
        manager = ld.LikelihoodManager()
        if isinstance(paths, AnalysisPath):
            likelihood_model = manager.register(nlls[0].as_term()) + 0
        else:
            s17 = manager.register(nlls[0].as_term())
            s18 = manager.register(nlls[1].as_term())
            f18 = manager.register(nlls[2].as_term())
            s20 = manager.register(nlls[3].as_term())
            likelihood_model = s17 + s18 + f18 + s20
        total_nll = manager.load(likelihood_model)
        logger.info('Model loaded')
        logger.info('Computing one NLL evaluation...')
        logger.info(
            f'First evaluation: {total_nll.evaluate([1.0] * len(total_nll.parameters))}'
        )
        status = None
        best_nll = np.inf
        rng = np.random.default_rng(0)
        for iiter in range(niters):
            logger.info(f'Fitting iteration {iiter}')
            p0 = rng.uniform(-100.0, 100.0, len(total_nll.parameters))
            logger.debug(f'Initial p0: {p0}')
            iter_status = total_nll.minimize(
                p0,
                observers=[LoggingObserver()],
                threads=NUM_THREADS,
                skip_hessian=True,
            )
            if iter_status.fx < best_nll:
                status = iter_status
                best_nll = iter_status.fx
        if status is None:
            logger.error(f'All fits failed for bin {ibin}!')
            raise Exception(f'All fits failed for bin {ibin}!')
        logger.success(f'Done!\n{status}')
        weights_fit: NDArray[np.float64] = np.array([])
        weights_s0: NDArray[np.float64] = np.array([])
        weights_p: NDArray[np.float64] = np.array([])
        weights_s0p: NDArray[np.float64] = np.array([])
        weights_s0n: NDArray[np.float64] = np.array([])
        weights_d2p: NDArray[np.float64] = np.array([])
        for nll in nlls:
            weights_fit = np.append(
                weights_fit, nll.project(status.x, threads=NUM_THREADS)
            )
            weights_s0 = np.append(
                weights_s0,
                nll.project_with(
                    status.x,
                    BINNED_WAVESETS[Waveset.S0],
                    threads=NUM_THREADS,
                ),
            )
            weights_p = np.append(
                weights_p,
                nll.project_with(
                    status.x,
                    BINNED_WAVESETS[Waveset.P],
                    threads=NUM_THREADS,
                ),
            )
            weights_s0p = np.append(
                weights_s0p,
                nll.project_with(
                    status.x, BINNED_WAVESETS[Waveset.S0P], threads=NUM_THREADS
                ),
            )
            weights_s0n = np.append(
                weights_s0n,
                nll.project_with(
                    status.x, BINNED_WAVESETS[Waveset.S0N], threads=NUM_THREADS
                ),
            )
            weights_d2p = np.append(
                weights_d2p,
                nll.project_with(
                    status.x, BINNED_WAVESETS[Waveset.D2P], threads=NUM_THREADS
                ),
            )

        weights_data = np.concatenate([nll.data.weights for nll in nlls])
        results.append(
            BinnedFitResultBin(
                status,
                float(np.sum(weights_data)),
                np.sum(weights_fit),
                np.sum(weights_s0),
                np.sum(weights_p),
                np.sum(weights_s0p),
                np.sum(weights_s0n),
                np.sum(weights_d2p),
            )
        )
    return BinnedFitResult(
        results, np.histogram_bin_edges([], nbins, range=RANGE), paths, model
    )


def fit_unbinned_guided(
    paths: AnalysisPathSet,
    binned_result: BinnedFitResult | list[BinnedFitResult],
    *,
    niters: int,
    phase_factor: bool = False,
) -> UnbinnedFitResult:
    averaged = isinstance(binned_result, BinnedFitResult)
    logger.info(
        f'Starting binned fit ({niters} iters, phase_factor={phase_factor}, averaged={averaged})'
    )
    datasets = paths.get_datasets()
    model = get_unbinned_model(phase_factor=phase_factor)
    nlls = [
        ld.NLL(model, ds_data, ds_accmc) for ds_data, ds_accmc in zip(*datasets)
    ]
    logger.info('Model loaded')
    res_mass = ld.Mass([2, 3])
    # We need every coherent combination and its constituents,
    # i.e. don't need the total here because it is |p|^2 + |n|^2
    # However, it seems to help with the absolute scale?
    # TODO: we might need to weight the TOT more than the others
    wavesets = [Waveset.TOT, Waveset.P, Waveset.N, Waveset.S0P, Waveset.D2P]
    guided_manager = ld.LikelihoodManager()
    guided_terms: list[ld.extensions.LikelihoodID] = []
    n_accmc_tot = sum([nll.accmc.n_events_weighted for nll in nlls])
    for i, nll in enumerate(nlls):
        n_accmc = nll.accmc.n_events_weighted
        if averaged:
            count_sets = [
                binned_result.waveset_hists[waveset].counts
                * n_accmc
                / n_accmc_tot
                for waveset in wavesets
            ]  # scale bin counts by fraction of total accmc
        else:
            count_sets = [
                binned_result[i].waveset_hists[waveset].counts
                for waveset in wavesets
            ]  # get a single binned result for each NLL (each run period)
        guided_terms.append(
            guided_manager.register(
                ld.experimental.BinnedGuideTerm(
                    nll,
                    res_mass,
                    [WAVESETS[waveset] for waveset in wavesets],
                    bins=NBINS,
                    range=RANGE,
                    count_sets=count_sets,
                    error_sets=None,
                )
            )
        )
    guided_model = (
        guided_terms[0] + guided_terms[1] + guided_terms[2] + guided_terms[3]
    )  # TODO: make sum() work here
    assert guided_model != 0
    guided_nll = guided_manager.load(guided_model)

    logger.info('Computing one NLL evaluation...')
    logger.info(
        f'First guided evaluation: {guided_nll.evaluate([1.0] * len(guided_nll.parameters))}'
    )
    ndof = NBINS * len(wavesets) - len(guided_nll.parameters)
    if not isinstance(binned_result, BinnedFitResult):
        ndof *= len(datasets[0])
    all_statuses: list[ld.Status] = []
    status = None
    i_best = None
    best_nll = np.inf
    rng = np.random.default_rng(0)
    for iiter in range(niters):
        logger.info(f'Fitting iteration {iiter}')
        logger.info('Starting guided fit')
        p0 = rng.uniform(-1000.0, 1000.0, len(guided_nll.parameters))
        logger.debug(f'Initial p0: {p0}')
        iter_status = guided_nll.minimize(
            p0,
            observers=[GuidedLoggingObserver(ndof)],
            threads=NUM_THREADS,
            max_steps=GUIDED_MAX_STEPS,
            skip_hessian=True,
        )
        all_statuses.append(iter_status)
        if iter_status.fx < best_nll:
            status = iter_status
            i_best = iiter
            best_nll = iter_status.fx
    if status is None or i_best is None:
        logger.error('All fits failed!')
        raise Exception('All fits failed!')
    logger.success(f'Done!\n{status}')
    return UnbinnedFitResult(all_statuses, i_best, paths, model)


class BinnedFitErrorsBin:
    def __init__(self, values: NDArray[np.floating]):
        quantiles = np.quantile(values, [0.16, 0.5, 0.84])
        self.lower_quantile: float = quantiles[0]
        self.center_quantile: float = quantiles[1]
        self.upper_quantile: float = quantiles[2]


@dataclass
class BinnedFitErrors:
    errors: list[BinnedFitErrorsBin]

    @property
    def lower(self) -> list[float]:
        return [e.center_quantile - e.lower_quantile for e in self.errors]

    @property
    def center(self) -> list[float]:
        return [e.center_quantile for e in self.errors]

    @property
    def upper(self) -> list[float]:
        return [e.upper_quantile - e.center_quantile for e in self.errors]


class CustomAutocorrelationObserver(ld.MCMCObserver):
    def __init__(
        self,
        nll: ld.NLL,
        wavesets: list[Waveset],
        ncheck: int = 20,
        dact: float = 0.05,
        nact: int = 20,
        discard: float = 0.5,
    ) -> None:
        self.nll: ld.NLL = nll
        self.ncheck: int = ncheck
        self.dact: float = dact
        self.nact: int = nact
        self.discard: float = discard
        self.latest_tau: float = np.inf
        self.wavesets: list[Waveset] = wavesets
        self.waveset_results: dict[Waveset, list[list[float]]] = {
            waveset: [] for waveset in wavesets
        }

    @override
    def callback(
        self, step: int, ensemble: ld.Ensemble
    ) -> tuple[ld.Ensemble, bool]:
        latest_step = ensemble.get_chain()[:, -1, :]
        waveset_results_list: dict[Waveset, list[float]] = {
            waveset: [] for waveset in self.wavesets
        }
        for i_walker in range(ensemble.dimension[0]):
            for waveset in self.wavesets:
                waveset_results_list[waveset].append(
                    np.sum(
                        self.nll.project_with(
                            latest_step[i_walker], BINNED_WAVESETS[waveset]
                        )
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


def compute_bootstrap_errors(
    binned_results: BinnedFitResult,
    *,
    nboot: int,
) -> dict[Waveset, BinnedFitErrors]:
    binned_errors: list[dict[Waveset, BinnedFitErrorsBin]] = []
    for bootstrap_seed in range(nboot):
        nlls = binned_results.get_nlls(bootstrap_seed=bootstrap_seed)
        binned_statuses: list[ld.Status] = []
        for ibin, binned_result in enumerate(binned_results.results):
            p_best = binned_result.status.x
            nlls_bin = nlls[ibin]
            manager = ld.LikelihoodManager()
            if isinstance(binned_results.paths, AnalysisPath):
                likelihood_model = manager.register(nlls_bin[0].as_term()) + 0
            else:
                s17 = manager.register(nlls_bin[0].as_term())
                s18 = manager.register(nlls_bin[1].as_term())
                f18 = manager.register(nlls_bin[2].as_term())
                s20 = manager.register(nlls_bin[3].as_term())
                likelihood_model = s17 + s18 + f18 + s20
            total_nll = manager.load(likelihood_model)
            binned_status = total_nll.minimize(
                p_best,
                observers=[LoggingObserver()],
                threads=NUM_THREADS,
                skip_hessian=True,
            )
            binned_statuses.append(binned_status)

        # waveset_results_list: dict[Waveset, list[float]] = {
        #     waveset: [] for waveset in self.wavesets
        # }
        #     for waveset in self.wavesets:
        #         waveset_results_list[waveset].append(
        #             np.sum(
        #                 self.nll.project_with(
        #                     binned_status, BINNED_WAVESETS[waveset]
        #                 )
        #             )
        #         )
        # for waveset in self.wavesets:
        #     self.waveset_results[waveset].append(waveset_results_list[waveset])
