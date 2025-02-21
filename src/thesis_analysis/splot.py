import itertools
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from iminuit import Minuit
from scipy.stats import chi2

from thesis_analysis.logger import logger
from thesis_analysis.utils import FitResult, Histogram
from thesis_analysis.constants import RFL_RANGE


def density_hist_to_pdf(
    bin_edges: np.ndarray, counts: np.ndarray
) -> Callable[[float], float]:
    def pdf(x: float) -> float:
        idx = np.searchsorted(bin_edges, x, side='right') - 1
        if idx < 0 or idx >= len(counts):
            return 0.0
        return counts[idx]

    return pdf


@dataclass
class SPlotFitResult:
    lda_fits_sig: list[FitResult]
    lda_fits_bkg: list[FitResult]
    total_fit: FitResult
    v: np.ndarray
    converged: bool

    def save(self, path: Path | str):
        path = Path(path)
        pickle.dump(self, path.open('wb'))

    @staticmethod
    def load(path: Path | str) -> 'SPlotFitResult':
        path = Path(path)
        return pickle.load(path.open('rb'))

    @property
    def aic(self) -> float:
        return self.total_fit.aic

    @property
    def bic(self) -> float:
        return self.total_fit.bic

    @property
    def ldas(self) -> list[float]:
        return self.ldas_sig + self.ldas_bkg

    @property
    def ldas_sig(self) -> list[float]:
        return [self.total_fit.values[f'lda{i}'] for i in range(self.nsig)]

    @property
    def ldas_bkg(self) -> list[float]:
        return [
            self.total_fit.values[f'lda{i + self.nsig}']
            for i in range(self.nbkg)
        ]

    @property
    def nsig(self) -> int:
        return len(self.lda_fits_sig)

    @property
    def nbkg(self) -> int:
        return len(self.lda_fits_bkg)

    @property
    def yields(self) -> list[float]:
        return self.yields_sig + self.yields_bkg

    @property
    def yields_sig(self) -> list[float]:
        return [self.total_fit.values[f'y{i}'] for i in range(self.nsig)]

    @property
    def yields_bkg(self) -> list[float]:
        return [
            self.total_fit.values[f'y{i + self.nsig}'] for i in range(self.nbkg)
        ]

    def pdfs(self, rfl1: np.ndarray, rfl2: np.ndarray) -> list[np.ndarray]:
        return [
            exp_pdf(rfl1, rfl2, self.ldas[i])
            for i in range(self.nsig + self.nbkg)
        ]


@dataclass
class SPlotFitResultD:
    hists_sig: list[tuple[Histogram, Histogram]]
    lda_fits_bkg: list[FitResult]
    total_fit: FitResult
    v: np.ndarray
    converged: bool

    def save(self, path: Path | str):
        path = Path(path)
        pickle.dump(self, path.open('wb'))

    @staticmethod
    def load(path: Path | str) -> 'SPlotFitResult':
        path = Path(path)
        return pickle.load(path.open('rb'))

    @property
    def aic(self) -> float:
        return self.total_fit.aic

    @property
    def bic(self) -> float:
        return self.total_fit.bic

    @property
    def ldas(self) -> list[float]:
        return self.ldas_bkg

    @property
    def ldas_bkg(self) -> list[float]:
        return [
            self.total_fit.values[f'lda{i + self.nsig}']
            for i in range(self.nbkg)
        ]

    @property
    def nsig(self) -> int:
        return len(self.hists_sig)

    @property
    def nbkg(self) -> int:
        return len(self.lda_fits_bkg)

    @property
    def yields(self) -> list[float]:
        return self.yields_sig + self.yields_bkg

    @property
    def yields_sig(self) -> list[float]:
        return [self.total_fit.values[f'y{i}'] for i in range(self.nsig)]

    @property
    def yields_bkg(self) -> list[float]:
        return [
            self.total_fit.values[f'y{i + self.nsig}'] for i in range(self.nbkg)
        ]

    def pdfs_sig(self, rfl1: np.ndarray, rfl2: np.ndarray) -> list[np.ndarray]:
        return [
            np.array(
                [
                    density_hist_to_pdf(
                        self.hists_sig[i][0].bins, self.hists_sig[i][0].counts
                    )(rfl1[j])
                    * density_hist_to_pdf(
                        self.hists_sig[i][1].bins, self.hists_sig[i][1].counts
                    )(rfl2[j])
                    for j in range(len(rfl1))
                ]
            )
            for i in range(self.nsig)
        ]

    def pdfs(self, rfl1: np.ndarray, rfl2: np.ndarray) -> list[np.ndarray]:
        return self.pdfs_sig(rfl1, rfl2) + [
            exp_pdf(rfl1, rfl2, self.ldas[i]) for i in range(self.nbkg)
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
        logger.debug(m)
        logger.error('sPlot λ fit failed!')
        raise Exception('sPlot λ fit failed!')
    return m


def fit_components(
    rfl1: np.ndarray,
    rfl2: np.ndarray,
    control: np.ndarray,
    weight: np.ndarray,
    *,
    n_spec: int,
) -> tuple[list[float], list[Minuit]]:
    tot_nevents = np.sum(weight)

    mass_bins = get_quantile_edges(control, bins=n_spec, weights=weight)
    binned_nevents = []
    fits = []
    for c_lo, c_hi in zip(mass_bins[:-1], mass_bins[1:]):
        mask = (c_lo <= control) & (control < c_hi)
        nevents = np.sum(weight[mask])
        binned_nevents.append(nevents)
        fit = fit_lda(rfl1[mask], rfl2[mask], weight[mask], lda0=100.0)
        fits.append(fit)
    fit_fractions = [nevents / tot_nevents for nevents in binned_nevents]
    return fit_fractions, fits


def fit_components_d(
    rfl1: np.ndarray,
    rfl2: np.ndarray,
    control: np.ndarray,
    weight: np.ndarray,
    *,
    n_spec: int,
    n_bins: int,
) -> tuple[
    list[float],
    list[Callable[[float, float], float]],
    list[tuple[Histogram, Histogram]],
]:
    tot_nevents = np.sum(weight)

    mass_bins = get_quantile_edges(control, bins=n_spec, weights=weight)
    binned_nevents = []
    pdfs = []
    histograms: list[tuple[Histogram, Histogram]] = []
    for c_lo, c_hi in zip(mass_bins[:-1], mass_bins[1:]):
        mask = (c_lo <= control) & (control < c_hi)
        nevents = np.sum(weight[mask])
        binned_nevents.append(nevents)
        hist1 = np.histogram(
            rfl1[mask],
            bins=n_bins,
            range=RFL_RANGE,
            weights=weight[mask],
            density=True,
        )
        hist2 = np.histogram(
            rfl2[mask],
            bins=n_bins,
            range=RFL_RANGE,
            weights=weight[mask],
            density=True,
        )

        def pdf(t1: float, t2: float) -> float:
            pdf1 = density_hist_to_pdf(*hist1)(t1)
            pdf2 = density_hist_to_pdf(*hist2)(t2)
            return pdf1 * pdf2

        pdfs.append(pdf)
        histograms.append((Histogram(*hist1), Histogram(*hist2)))
    fit_fractions = [nevents / tot_nevents for nevents in binned_nevents]
    return fit_fractions, pdfs, histograms


def run_splot_fit(
    rfl1: np.ndarray,
    rfl2: np.ndarray,
    weight: np.ndarray,
    rfl1_sigmc: np.ndarray,
    rfl2_sigmc: np.ndarray,
    control_sigmc: np.ndarray,
    weight_sigmc: np.ndarray,
    rfl1_bkgmc: np.ndarray,
    rfl2_bkgmc: np.ndarray,
    control_bkgmc: np.ndarray,
    weight_bkgmc: np.ndarray,
    *,
    nsig: int,
    nbkg: int,
    fixed_sig: bool,
    fixed_bkg: bool,
) -> SPlotFitResult:
    n_spec = nsig + nbkg
    fit_fractions_sig, fits_sig = fit_components(
        rfl1_sigmc, rfl2_sigmc, control_sigmc, weight_sigmc, n_spec=nsig
    )
    fit_fractions_bkg, fits_bkg = fit_components(
        rfl1_bkgmc, rfl2_bkgmc, control_bkgmc, weight_bkgmc, n_spec=nbkg
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
        yields = args[::2]
        ldas = args[1::2]
        likelihoods = weight * np.log(
            np.sum(
                [
                    yields[i] * exp_pdf(rfl1, rfl2, ldas[i])
                    for i in range(n_spec)
                ],
                axis=0,
            )
            + np.finfo(float).tiny
        )
        return -2 * (np.sum(np.sort(likelihoods)) - np.sum(yields))

    p0 = list(itertools.chain(*zip(yields0, ldas0)))
    name = list(
        itertools.chain(
            *zip(
                [f'y{i}' for i in range(n_spec)],
                [f'lda{i}' for i in range(n_spec)],
            )
        )
    )
    m = Minuit(nll, *p0, name=name)
    for i in range(n_spec):
        m.limits[f'y{i}'] = (0.0, None)
        m.limits[f'lda{i}'] = (max(ldas0[i] - 30, 0.01), ldas0[i] + 30)
        if fixed_sig and i < nsig:
            m.fixed[f'lda{i}'] = True
        if fixed_bkg and i >= nsig:
            m.fixed[f'lda{i}'] = True
    m.migrad(ncall=10_000)
    if not m.valid:
        logger.debug(m)
        logger.error('sPlot yield fit failed!')
        # raise Exception('sPlot yield fit failed!')
        # for now, just continue, we expect some of these to not perform as well
    yields = [m.values[f'y{i}'] for i in range(n_spec)]
    ldas = [m.values[f'lda{i}'] for i in range(n_spec)]
    logger.debug(f'Yields (fit): {yields}')
    logger.debug(f'λs (fit): {ldas}')
    pdfs = [exp_pdf(rfl1, rfl2, ldas[i]) for i in range(n_spec)]
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
        [
            FitResult.from_minuit(fit_sig, len(rfl1_sigmc))
            for fit_sig in fits_sig
        ],
        [
            FitResult.from_minuit(fit_bkg, len(rfl1_bkgmc))
            for fit_bkg in fits_bkg
        ],
        FitResult.from_minuit(m, len(rfl1)),
        v,
        m.valid,
    )
    return fit_result


def run_splot_fit_d(
    rfl1: np.ndarray,
    rfl2: np.ndarray,
    weight: np.ndarray,
    rfl1_sigmc: np.ndarray,
    rfl2_sigmc: np.ndarray,
    control_sigmc: np.ndarray,
    weight_sigmc: np.ndarray,
    rfl1_bkgmc: np.ndarray,
    rfl2_bkgmc: np.ndarray,
    control_bkgmc: np.ndarray,
    weight_bkgmc: np.ndarray,
    *,
    nsig: int,
    nsig_bins: int,
    nbkg: int,
    fixed_bkg: bool,
) -> SPlotFitResultD:
    n_spec = nsig + nbkg
    fit_fractions_sig, sig_pdfs, sig_hists = fit_components_d(
        rfl1_sigmc,
        rfl2_sigmc,
        control_sigmc,
        weight_sigmc,
        n_spec=nsig,
        n_bins=nsig_bins,
    )
    fit_fractions_bkg, fits_bkg = fit_components(
        rfl1_bkgmc, rfl2_bkgmc, control_bkgmc, weight_bkgmc, n_spec=nbkg
    )
    nevents = np.sum(weight)
    # assume half the data is signal-like to initialize the fit
    yields0 = [
        fit_fraction * 0.5 * nevents for fit_fraction in fit_fractions_sig
    ] + [fit_fraction * 0.5 * nevents for fit_fraction in fit_fractions_bkg]
    logger.debug(f'Yields init: {yields0}')
    # get λs from fits
    ldas0 = [fit.values['lda'] for fit in fits_bkg]
    logger.debug(f'λs init: {ldas0}')

    def nll(*args: float) -> float:
        yields_sig = args[:nsig]
        yields_bkg = args[nsig::2]
        ldas = args[nsig + 1 :: 2]
        likelihoods = weight * np.log(
            np.sum(
                [
                    [
                        yields_sig[i] * sig_pdfs[i](rfl1[j], rfl2[j])
                        for j in range(len(rfl1))
                    ]
                    for i in range(nsig)
                ],
                axis=0,
            )
            + np.sum(
                [
                    yields_bkg[i] * exp_pdf(rfl1, rfl2, ldas[i])
                    for i in range(nbkg)
                ],
                axis=0,
            )
            + np.finfo(float).tiny
        )
        return -2 * (np.sum(np.sort(likelihoods)) - np.sum(yields))

    p0 = list(itertools.chain(*zip(yields0, ldas0)))
    name = [f'y{i}' for i in range(nsig)] + list(
        itertools.chain(
            *zip(
                [f'y{nsig + i}' for i in range(nbkg)],
                [f'lda{nsig + i}' for i in range(nbkg)],
            )
        )
    )
    m = Minuit(nll, *p0, name=name)
    for i in range(n_spec):
        m.limits[f'y{i}'] = (0.0, None)
    for i in range(nbkg):
        m.limits[f'lda{nsig + i}'] = (max(ldas0[i] - 30, 0.01), ldas0[i] + 30)
        if fixed_bkg:
            m.fixed[f'lda{nsig + i}'] = True
    m.migrad(ncall=10_000)
    if not m.valid:
        logger.debug(m)
        logger.error('sPlot yield fit failed!')
        # raise Exception('sPlot yield fit failed!')
        # for now, just continue, we expect some of these to not perform as well
    yields = [m.values[f'y{i}'] for i in range(n_spec)]
    ldas = [m.values[f'lda{nsig + i}'] for i in range(nbkg)]
    logger.debug(f'Yields (fit): {yields}')
    logger.debug(f'Background λs (fit): {ldas}')
    pdfs = sig_pdfs + [exp_pdf(rfl1, rfl2, ldas[i]) for i in range(nbkg)]
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
    fit_result = SPlotFitResultD(
        sig_hists,
        [
            FitResult.from_minuit(fit_bkg, len(rfl1_bkgmc))
            for fit_bkg in fits_bkg
        ],
        FitResult.from_minuit(m, len(rfl1)),
        v,
        m.valid,
    )
    return fit_result


def get_sweights(
    fit_result: SPlotFitResult | SPlotFitResultD,
    rfl1: np.ndarray,
    rfl2: np.ndarray,
    weight: np.ndarray,
    *,
    nsig: int,
    nbkg: int,
) -> np.ndarray:
    n_spec = nsig + nbkg
    yields = fit_result.yields
    pdfs = fit_result.pdfs(rfl1, rfl2)
    denom = np.sum([yields[k] * pdfs[k] for k in range(n_spec)], axis=0)
    inds = np.argwhere(
        np.power(denom, 2) == 0.0
    )  # if a component is very small, this can happen
    denom[inds] += np.sqrt(
        np.finfo(float).eps
    )  # push these values just lightly away from zero
    v = fit_result.v
    logger.debug(f'V = {v.tolist()}')
    sweights = [
        np.sum([weight * v[i, j] * pdfs[j] for j in range(n_spec)], axis=0)
        / denom
        for i in range(n_spec)
    ]
    return np.sum(sweights[:nsig], axis=0)


@dataclass
class Significance:
    likelihood_ratio: float
    ndof: float

    @property
    def p(self) -> float:
        return float(chi2(self.ndof).sf(self.likelihood_ratio))


@dataclass
class FactorizationFitResult:
    h0: FitResult
    h1s: list[FitResult]
    ndof: int

    @property
    def significance(self):
        r = self.h0.n2ll - sum([h1.n2ll for h1 in self.h1s])
        return Significance(r, self.ndof)


def get_quantile_edges(
    variable: np.ndarray, *, bins: int, weights: np.ndarray
) -> np.ndarray:
    # This is a custom wrapper method around numpy.quantile that
    # first rescales the weights so that they are between 0 and 1
    # and then runs the quantile method with those weights
    scaled_weights = (weights - np.min(weights)) / (
        np.max(weights) - np.min(weights)
    )
    return np.quantile(
        variable,
        np.linspace(0, 1, bins + 1),
        weights=scaled_weights,
        method='inverted_cdf',
    )


def get_quantile_indices(
    variable: np.ndarray, *, bins: int, weights: np.ndarray
) -> list[np.ndarray]:
    quantiles = get_quantile_edges(variable, bins=bins, weights=weights)
    quantiles[-1] = (
        np.inf
    )  # ensure the maximum value gets placed in the last bin
    quantile_assignment = np.digitize(variable, quantiles)
    return [np.where(quantile_assignment == i)[0] for i in range(1, bins + 1)]


def run_factorization_fits(
    rfl1: np.ndarray,
    rfl2: np.ndarray,
    weight: np.ndarray,
    control: np.ndarray,
    *,
    bins: int,
) -> FactorizationFitResult:
    quantile_indices = get_quantile_indices(control, bins=bins, weights=weight)

    def generate_nll(rfl1s: np.ndarray, rfl2s: np.ndarray, weights: np.ndarray):
        def nll(z: float, lda_s: float, lda_b: float) -> float:
            likelihoods = weights * np.log(
                z * exp_pdf(rfl1s, rfl2s, lda_s)
                + (1 - z) * exp_pdf(rfl1s, rfl2s, lda_b)
                + np.finfo(float).tiny
            )
            return float(
                -2.0 * np.sum(np.sort(likelihoods))
            )  # the integral term doesn't matter here since we are using this in a ratio where it cancels

        return nll

    nlls = [
        generate_nll(
            rfl1[quantile_indices[i]],
            rfl2[quantile_indices[i]],
            weight[quantile_indices[i]],
        )
        for i in range(bins)
    ]

    # arguments are (z_0, z_1, ..., z_{n-1}, lda_s, lda_b)
    def nll0(*args: float) -> float:
        return np.sum(
            np.array(
                [nlls[i](args[i], args[-2], args[-1]) for i in range(bins)]
            )
        )

    h0 = Minuit(
        nll0,
        *[0.5] * bins,
        12.0,
        100.0,
        name=[f'z{i}' for i in range(bins)] + ['lda_s', 'lda_b'],
    )
    for i in range(bins):
        h0.limits[f'z{i}'] = (0.0, 1.0)
    h0.limits['lda_s'] = (5.0, 20.0)
    h0.limits['lda_b'] = (80.0, 120.0)
    h0.migrad(ncall=10_000)
    if not h0.valid:
        logger.debug(h0)
        logger.error('Null hypothesis fit failed!')
        raise Exception('Null hypothesis fit failed!')

    h1s = []
    for i in range(bins):
        h1 = Minuit(nlls[i], z=0.5, lda_s=12.0, lda_b=100.0)  # type: ignore
        h1.limits['z'] = (0.0, 1.0)
        h1.limits['lda_s'] = (5.0, 20.0)
        h1.limits['lda_b'] = (80.0, 120.0)
        h1.migrad(ncall=10_000)
        if not h1.valid:
            logger.debug(h1)
            logger.error(f'Alternative hypothesis (bin {i}) fit failed!')
            raise Exception(f'Null hypothesis (bin {i}) fit failed!')
        h1s.append(h1)

    return FactorizationFitResult(
        FitResult.from_minuit(h0, len(rfl1)),
        [
            FitResult.from_minuit(h1s[i], len(quantile_indices[i]))
            for i in range(bins)
        ],
        2 * bins - 2,
    )


def run_factorization_fits_mc(
    rfl1: np.ndarray,
    rfl2: np.ndarray,
    weight: np.ndarray,
    control: np.ndarray,
    *,
    bins: int,
) -> FactorizationFitResult:
    quantile_indices = get_quantile_indices(control, bins=bins, weights=weight)

    def generate_nll(rfl1s: np.ndarray, rfl2s: np.ndarray, weights: np.ndarray):
        def nll(lda: float) -> float:
            return float(
                -2.0
                * np.sum(
                    np.sort(
                        weights * np.log(exp_pdf(rfl1s, rfl2s, lda))
                        + np.finfo(float).tiny
                    )
                )
            )  # the integral term doesn't matter here since we are using this in a ratio where it cancels

        return nll

    nlls = [
        generate_nll(
            rfl1[quantile_indices[i]],
            rfl2[quantile_indices[i]],
            weight[quantile_indices[i]],
        )
        for i in range(bins)
    ]

    def nll0(lda: float) -> float:
        return np.sum(np.array([nlls[i](lda) for i in range(bins)]))

    h0 = Minuit(nll0, lda=50.0)  # type: ignore
    h0.limits['lda'] = (5.0, 120.0)
    h0.migrad(ncall=10_000)
    if not h0.valid:
        logger.debug(h0)
        logger.error('Null hypothesis fit failed!')
        raise Exception('Null hypothesis fit failed!')

    h1s = []
    for i in range(bins):
        h1 = Minuit(nlls[i], lda=50.0)  # type: ignore
        h1.limits['lda'] = (5.0, 120.0)
        h1.migrad(ncall=10_000)
        if not h1.valid:
            logger.debug(h1)
            logger.error(f'Alternative hypothesis (bin {i}) fit failed!')
            raise Exception(f'Null hypothesis (bin {i}) fit failed!')
        h1s.append(h1)

    return FactorizationFitResult(
        FitResult.from_minuit(h0, len(rfl1)),
        [
            FitResult.from_minuit(h1s[i], len(quantile_indices[i]))
            for i in range(bins)
        ],
        bins - 1,
    )
