import pickle
from dataclasses import dataclass
from pathlib import Path

import laddu as ld
import numpy as np
from iminuit import Minuit

from thesis_analysis.logger import logger


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
        self.accidental_scaling_factors = accidental_scaling_factors

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
    counts: np.ndarray
    bins: np.ndarray


class RCDBData:
    def __init__(
        self,
        pol_angles: dict[int, tuple[str, str, float | None]],
        pol_magnitudes: dict[str, dict[str, Histogram]],
    ):
        self.pol_angles = pol_angles
        self.pol_magnitudes = pol_magnitudes

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
        magnitude = pol_hist.counts[energy_index]
        return magnitude * np.cos(angle), magnitude * np.sin(angle), True


@dataclass
class FitResult:
    values: dict[str, float]
    errors: dict[str, float]

    @staticmethod
    def from_minuit(minuit: Minuit) -> 'FitResult':
        return FitResult(minuit.values.to_dict(), minuit.errors.to_dict())


@dataclass
class SPlotFitResult:
    lda_fits_sig: list[FitResult]
    lda_fits_bkg: list[FitResult]
    yield_fit: FitResult
    v: np.ndarray

    def save(self, path: Path | str):
        path = Path(path)
        pickle.dump(self, path.open('wb'))

    @staticmethod
    def load(path: Path | str) -> 'SPlotFitResult':
        path = Path(path)
        return pickle.load(path.open('wb'))

    @property
    def ldas(self) -> list[float]:
        return [fit.values['lda'] for fit in self.lda_fits_sig] + [
            fit.values['lda'] for fit in self.lda_fits_bkg
        ]

    @property
    def ldas_sig(self) -> list[float]:
        return [fit.values['lda'] for fit in self.lda_fits_sig]

    @property
    def ldas_bkg(self) -> list[float]:
        return [fit.values['lda'] for fit in self.lda_fits_bkg]

    @property
    def n_sig(self) -> int:
        return len(self.lda_fits_sig)

    @property
    def n_bkg(self) -> int:
        return len(self.lda_fits_bkg)

    @property
    def yields(self) -> list[float]:
        return [
            self.yield_fit.values[f'y{i}']
            for i in range(self.n_sig + self.n_bkg)
        ]

    @property
    def yields_sig(self) -> list[float]:
        return [self.yield_fit.values[f'y{i}'] for i in range(self.n_sig)]

    @property
    def yields_bkg(self) -> list[float]:
        return [
            self.yield_fit.values[f'y{i+self.n_sig}'] for i in range(self.n_bkg)
        ]

    def pdfs(self, rfl1: np.ndarray, rfl2: np.ndarray) -> list[np.ndarray]:
        return [
            exp_pdf(rfl1, rfl2, self.ldas[i])
            for i in range(self.n_sig + self.n_bkg)
        ]


def exp_pdf_single(rfl: np.ndarray, lda: float) -> np.ndarray:
    return np.exp(-rfl * lda) * lda


def exp_pdf(rfl1: np.ndarray, rfl2: np.ndarray, lda: float) -> np.ndarray:
    return exp_pdf_single(rfl1, lda) * exp_pdf_single(rfl2, lda)


def fit_lda(
    rfl1: np.ndarray, rfl2: np.ndarray, weight: np.ndarray, *, lda0: float
) -> Minuit:
    def nll(*args: float) -> float:
        return -2.0 * np.sum(
            np.sort(
                weight
                * np.log(exp_pdf(rfl1, rfl2, args[0]) + np.finfo(float).tiny)
            )
        )

    m = Minuit(nll, lda=lda0, name=('lda',))
    m.limits['lda'] = (0.0, 200.0)
    m.migrad(ncall=10_000)
    if not m.valid:
        logger.error('sPlot λ fit failed!')
        raise Exception('sPlot λ fit failed!')
    return m


def fit_components(
    rfl1: np.ndarray,
    rfl2: np.ndarray,
    mass: np.ndarray,
    weight: np.ndarray,
    *,
    n_spec: int,
) -> tuple[list[float], list[Minuit]]:
    tot_nevents = np.sum(weight)
    mass_bins = np.quantile(mass, np.linspace(0, 1, n_spec + 1))
    binned_nevents = []
    fits = []
    for m_lo, m_hi in zip(mass_bins[:-1], mass_bins[1:]):
        mask = (m_lo <= mass) & (mass < m_hi)
        nevents = np.sum(weight[mask])
        binned_nevents.append(nevents)
        fit = fit_lda(rfl1[mask], rfl2[mask], weight[mask], lda0=100.0)
        fits.append(fit)
    fit_fractions = [nevents / tot_nevents for nevents in binned_nevents]
    return fit_fractions, fits


def get_sweights(
    rfl1: np.ndarray,
    rfl2: np.ndarray,
    weight: np.ndarray,
    rfl1_sigmc: np.ndarray,
    rfl2_sigmc: np.ndarray,
    mass_sigmc: np.ndarray,
    weight_sigmc: np.ndarray,
    rfl1_bkgmc: np.ndarray,
    rfl2_bkgmc: np.ndarray,
    mass_bkgmc: np.ndarray,
    weight_bkgmc: np.ndarray,
    *,
    n_sig: int,
    n_bkg: int,
) -> tuple[SPlotFitResult, np.ndarray]:
    n_spec = n_sig + n_bkg
    fit_fractions_sig, fits_sig = fit_components(
        rfl1_sigmc, rfl2_sigmc, mass_sigmc, weight_sigmc, n_spec=n_sig
    )
    fit_fractions_bkg, fits_bkg = fit_components(
        rfl1_bkgmc, rfl2_bkgmc, mass_bkgmc, weight_bkgmc, n_spec=n_bkg
    )
    nevents = np.sum(weight)
    # assume half the data is signal-like to initialize the fit
    yields0 = [
        fit_fraction * 0.5 * nevents for fit_fraction in fit_fractions_sig
    ] + [fit_fraction * 0.5 * nevents for fit_fraction in fit_fractions_bkg]
    logger.debug(f'Yields init: {yields0}')
    # get λs from fits
    ldas0 = [fit.values['lda'] for fit in fits_sig] + [
        fit.values['lda'] for fit in fits_bkg
    ]
    logger.debug(f'λs init: {ldas0}')

    def nll(*args: float) -> float:
        likelihoods = weight * np.log(
            np.sum(
                [
                    args[i] * exp_pdf(rfl1, rfl2, ldas0[i])
                    for i in range(n_spec)
                ],
                axis=0,
            )
            + np.finfo(float).tiny
        )
        return -2 * (np.sum(np.sort(likelihoods)) - np.sum(args))

    m = Minuit(nll, *yields0, name=[f'y{i}' for i in range(n_spec)])
    for i in range(n_spec):
        m.limits[f'y{i}'] = (0.0, None)
    m.migrad(ncall=10_000)
    if not m.valid:
        logger.error('sPlot yield fit failed!')
        raise Exception('sPlot yield fit failed!')
    yields = [m.values[f'y{i}'] for i in range(n_spec)]
    logger.debug(f'Yields (fit): {yields}')
    pdfs = [exp_pdf(rfl1, rfl2, ldas0[i]) for i in range(n_spec)]
    denom = np.sum([yields[k] * pdfs[k] for k in range(n_spec)], axis=0)
    inds = np.argwhere(
        np.power(denom, 2) == 0.0
    )  # if a component is very small, this can happen
    denom[inds] += np.sqrt(
        np.finfo(float).eps
    )  # push these values just lightly away from zero
    v_inv = np.array(
        [
            [
                np.sum((weight * pdfs[i] * pdfs[j]) / np.power(denom, 2))
                for j in range(n_spec)
            ]
            for i in range(n_spec)
        ]
    )
    v = np.linalg.inv(v_inv)
    logger.debug(f'V = {v.tolist()}')
    fit_result = SPlotFitResult(
        [FitResult.from_minuit(fit_sig) for fit_sig in fits_sig],
        [FitResult.from_minuit(fit_bkg) for fit_bkg in fits_bkg],
        FitResult.from_minuit(m),
        v,
    )
    sweights = [
        np.sum([weight * v[i, j] * pdfs[j] for j in range(n_spec)], axis=0)
        / denom
        for i in range(n_spec)
    ]
    return fit_result, np.sum(sweights[:n_sig], axis=0)


@dataclass
class UnbinnedFitResult:
    status: ld.Status
    masses_data: np.ndarray
    weights_data: np.ndarray
    masses_accmc: np.ndarray
    weights_fit: np.ndarray
    weights_s0p: np.ndarray
    weights_s0n: np.ndarray
    weights_d2p: np.ndarray

    def get_data_hist(self, bins: int, range: tuple[float, float]) -> Histogram:
        counts, edges = np.histogram(
            self.masses_data, weights=self.weights_data, bins=bins, range=range
        )
        return Histogram(counts, edges)

    def get_fit_hist(self, bins: int, range: tuple[float, float]) -> Histogram:
        counts, edges = np.histogram(
            self.masses_accmc, weights=self.weights_fit, bins=bins, range=range
        )
        return Histogram(counts, edges)

    def get_s0p_hist(self, bins: int, range: tuple[float, float]) -> Histogram:
        counts, edges = np.histogram(
            self.masses_accmc, weights=self.weights_s0p, bins=bins, range=range
        )
        return Histogram(counts, edges)

    def get_s0n_hist(self, bins: int, range: tuple[float, float]) -> Histogram:
        counts, edges = np.histogram(
            self.masses_accmc, weights=self.weights_s0n, bins=bins, range=range
        )
        return Histogram(counts, edges)

    def get_d2p_hist(self, bins: int, range: tuple[float, float]) -> Histogram:
        counts, edges = np.histogram(
            self.masses_accmc, weights=self.weights_d2p, bins=bins, range=range
        )
        return Histogram(counts, edges)


def fit_unbinned(
    data_s17_path: str | Path,
    accmc_s17_path: str | Path,
    data_s18_path: str | Path,
    accmc_s18_path: str | Path,
    data_f18_path: str | Path,
    accmc_f18_path: str | Path,
    data_s20_path: str | Path,
    accmc_s20_path: str | Path,
) -> UnbinnedFitResult:
    data_ds_S17 = ld.open_amptools(str(data_s17_path), pol_in_beam=True)
    logger.info('S17 data loaded')
    accmc_ds_S17 = ld.open_amptools(str(accmc_s17_path), pol_in_beam=True)
    logger.info('S17 accmc loaded')
    data_ds_S18 = ld.open_amptools(str(data_s18_path), pol_in_beam=True)
    logger.info('S18 data loaded')
    accmc_ds_S18 = ld.open_amptools(str(accmc_s18_path), pol_in_beam=True)
    logger.info('S18 accmc loaded')
    data_ds_F18 = ld.open_amptools(str(data_f18_path), pol_in_beam=True)
    logger.info('F18 data loaded')
    accmc_ds_F18 = ld.open_amptools(str(accmc_f18_path), pol_in_beam=True)
    logger.info('F18 accmc loaded')
    data_ds_S20 = ld.open_amptools(str(data_s20_path), pol_in_beam=True)
    logger.info('S20 data loaded')
    accmc_ds_S20 = ld.open_amptools(str(accmc_s20_path), pol_in_beam=True)
    logger.info('S20 accmc loaded')

    def single_dataset_nll(ds_data: ld.Dataset, ds_mc: ld.Dataset) -> ld.NLL:
        res_mass = ld.Mass([2, 3])
        angles = ld.Angles(0, [1], [2], [2, 3])
        polarization = ld.Polarization(0, [1])
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
        pos_re = (
            z00p.real() * (f0p + a0p) + z22p.real() * (f2 + a2)
        ).norm_sqr()
        pos_im = (
            z00p.imag() * (f0p + a0p) + z22p.imag() * (f2 + a2)
        ).norm_sqr()
        neg_re = (z00n.real() * (f0n + a0n)).norm_sqr()
        neg_im = (z00n.imag() * (f0n + a0n)).norm_sqr()
        model = manager.model(pos_re + pos_im + neg_re + neg_im)

        return ld.NLL(model, ds_data, ds_mc)

    manager = ld.LikelihoodManager()
    nll_S17 = single_dataset_nll(data_ds_S17, accmc_ds_S17)
    logger.info('Evaluating S17 NLL...')
    logger.info(f'Result: {nll_S17.evaluate([1.0] * len(nll_S17.parameters))}')
    nll_S18 = single_dataset_nll(data_ds_S18, accmc_ds_S18)
    logger.info('Evaluating S18 NLL...')
    logger.info(f'Result: {nll_S18.evaluate([1.0] * len(nll_S18.parameters))}')
    nll_F18 = single_dataset_nll(data_ds_F18, accmc_ds_F18)
    logger.info('Evaluating F18 NLL...')
    logger.info(f'Result: {nll_F18.evaluate([1.0] * len(nll_F18.parameters))}')
    nll_S20 = single_dataset_nll(data_ds_S20, accmc_ds_S20)
    logger.info('Evaluating S20 NLL...')
    logger.info(f'Result: {nll_S20.evaluate([1.0] * len(nll_S20.parameters))}')
    s17 = manager.register(nll_S17.as_term())
    s18 = manager.register(nll_S18.as_term())
    f18 = manager.register(nll_F18.as_term())
    s20 = manager.register(nll_S20.as_term())
    model = s17 + s18 + f18 + s20
    nll = manager.load(model)
    logger.info('Model loaded')

    class LoggingObserver(ld.Observer):
        def callback(
            self, step: int, status: ld.Status
        ) -> tuple[ld.Status, bool]:
            logger.info(f'Step {step}:\n{status}')
            return status, False

    logger.info('Computing one NLL evaluation...')
    logger.info(
        f'First evaluation: {nll.evaluate([1.0] * len(nll.parameters))}'
    )
    status = nll.minimize(
        [1.0] * len(nll.parameters),
        observers=[LoggingObserver()],
    )
    weights_fit = np.array([])
    weights_s0p = np.array([])
    weights_s0n = np.array([])
    weights_d2p = np.array([])
    nlls = [nll_S17, nll_S18, nll_F18, nll_S20]
    for nll in nlls:
        nll.activate_all()
        weights_fit = np.append(weights_fit, nll.project(status.x))
        nll.isolate(['z00p', 'f0p', 'a0p'])
        weights_s0p = np.append(weights_s0p, nll.project(status.x))
        nll.isolate(['z00n', 'f0n', 'a0n'])
        weights_s0n = np.append(weights_s0n, nll.project(status.x))
        nll.isolate(['z22p', 'f2', 'a2'])
        weights_d2p = np.append(weights_d2p, nll.project(status.x))
        nll.activate_all()

    res_mass = ld.Mass([2, 3])
    masses_accmc = np.concatenate(
        [np.array(res_mass.value_on(nll.accmc)) for nll in nlls]
    )
    masses_data = np.concatenate(
        [np.array(res_mass.value_on(nll.data)) for nll in nlls]
    )
    weights_data = np.concatenate([nll.data.weights for nll in nlls])
    return UnbinnedFitResult(
        status,
        masses_data,
        weights_data,
        masses_accmc,
        weights_fit,
        weights_s0p,
        weights_s0n,
        weights_d2p,
    )
