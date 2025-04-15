import itertools
from dataclasses import dataclass
from typing import Callable

import numpy as np
from iminuit import Minuit
from numpy.typing import NDArray
from scipy.stats import chi2

from thesis_analysis.constants import RFL_RANGE
from thesis_analysis.logger import logger
from thesis_analysis.utils import FitResult, Histogram


def density_hist_to_pdf(
    counts: NDArray[np.float32], bin_edges: NDArray[np.float32]
) -> Callable[[NDArray[np.float32]], NDArray[np.float32]]:
    def pdf(x: float) -> float:
        idx = np.searchsorted(bin_edges, x, side='right') - 1
        if idx < 0 or idx >= len(counts):
            return 0.0
        return counts[idx]

    vpdf = np.vectorize(pdf)

    return vpdf


class SPlotFitFailure:
    pass


@dataclass
class SPlotFitResultExp:
    lda_fits_sig: list[FitResult]
    lda_fits_bkg: list[FitResult]
    total_fit: FitResult
    v: NDArray[np.float64]
    converged: bool

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

    def pdfs(
        self, rfl1: NDArray[np.float32], rfl2: NDArray[np.float32]
    ) -> list[NDArray[np.float64]]:
        return [
            exp_pdf(rfl1, rfl2, self.ldas[i])
            for i in range(self.nsig + self.nbkg)
        ]


@dataclass
class SPlotFitResult:
    hists_sig: list[tuple[Histogram, Histogram]]
    lda_fits_bkg: list[FitResult]
    total_fit: FitResult
    v: NDArray[np.float64]
    converged: bool

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

    def pdfs1(self, rfl1: NDArray[np.float32]) -> list[NDArray[np.float32]]:
        return [
            density_hist_to_pdf(
                self.hists_sig[i][0].counts, self.hists_sig[i][0].bins
            )(rfl1)
            for i in range(self.nsig)
        ]

    def pdfs2(self, rfl2: NDArray[np.float32]) -> list[NDArray[np.float32]]:
        return [
            density_hist_to_pdf(
                self.hists_sig[i][1].counts, self.hists_sig[i][1].bins
            )(rfl2)
            for i in range(self.nsig)
        ]

    def pdfs_sig(
        self, rfl1: NDArray[np.float32], rfl2: NDArray[np.float32]
    ) -> list[NDArray[np.floating]]:
        return [
            density_hist_to_pdf(
                self.hists_sig[i][0].counts, self.hists_sig[i][0].bins
            )(rfl1)
            * density_hist_to_pdf(
                self.hists_sig[i][1].counts, self.hists_sig[i][1].bins
            )(rfl2)
            for i in range(self.nsig)
        ]

    def pdfs(
        self, rfl1: NDArray[np.float32], rfl2: NDArray[np.float32]
    ) -> list[NDArray[np.floating]]:
        return self.pdfs_sig(rfl1, rfl2) + [
            exp_pdf(rfl1, rfl2, self.ldas[i]) for i in range(self.nbkg)
        ]


def exp_pdf_single(rfl: NDArray[np.float32], lda: float) -> NDArray[np.float64]:
    return np.exp(-rfl * lda) * lda


def exp_pdf(
    rfl1: NDArray[np.float32], rfl2: NDArray[np.float32], lda: float
) -> NDArray[np.float64]:
    return exp_pdf_single(rfl1, lda) * exp_pdf_single(rfl2, lda)


def fit_lda(
    rfl1: NDArray[np.float32],
    rfl2: NDArray[np.float32],
    weight: NDArray[np.float32],
    *,
    lda0: float,
) -> Minuit:
    def nll(*args: float) -> float:
        return float(
            -2.0
            * np.sum(
                np.sort(
                    weight
                    * np.log(
                        exp_pdf(rfl1, rfl2, args[0]) + np.finfo(float).tiny
                    )
                )
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


def fit_components_exp(
    rfl1: NDArray[np.float32],
    rfl2: NDArray[np.float32],
    control: NDArray[np.float32],
    weight: NDArray[np.float32],
    *,
    n_spec: int,
) -> tuple[list[np.float32], list[Minuit]]:
    tot_nevents = np.sum(weight)

    mass_bins = get_quantile_edges(control, bins=n_spec, weights=weight)
    binned_nevents: list[np.float32] = []
    fits: list[Minuit] = []
    for c_lo, c_hi in zip(mass_bins[:-1], mass_bins[1:]):
        mask: NDArray[np.bool] = (c_lo <= control) & (control < c_hi)
        nevents = np.sum(weight[mask])
        binned_nevents.append(nevents)
        fit = fit_lda(rfl1[mask], rfl2[mask], weight[mask], lda0=100.0)
        fits.append(fit)
    fit_fractions = [nevents / tot_nevents for nevents in binned_nevents]
    return fit_fractions, fits


def fit_components(
    rfl1: NDArray[np.float32],
    rfl2: NDArray[np.float32],
    control: NDArray[np.float32],
    weight: NDArray[np.float32],
    *,
    n_spec: int,
    n_bins: int,
) -> tuple[
    list[np.float32],
    list[
        Callable[
            [NDArray[np.float32], NDArray[np.float32]], NDArray[np.floating]
        ]
    ],
    list[tuple[Histogram, Histogram]],
]:
    tot_nevents = np.sum(weight)

    mass_bins = get_quantile_edges(control, bins=n_spec, weights=weight)
    binned_nevents: list[np.float32] = []
    pdfs: list[
        Callable[
            [NDArray[np.float32], NDArray[np.float32]], NDArray[np.floating]
        ]
    ] = []
    histograms: list[tuple[Histogram, Histogram]] = []
    for c_lo, c_hi in zip(mass_bins[:-1], mass_bins[1:]):
        mask: NDArray[np.bool] = (c_lo <= control) & (control < c_hi)
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

        def pdf(
            t1: NDArray[np.float32], t2: NDArray[np.float32]
        ) -> NDArray[np.floating]:
            pdf1 = density_hist_to_pdf(*hist1)(t1)
            pdf2 = density_hist_to_pdf(*hist2)(t2)
            return pdf1 * pdf2

        pdfs.append(pdf)
        histograms.append((Histogram(*hist1), Histogram(*hist2)))
    fit_fractions = [nevents / tot_nevents for nevents in binned_nevents]
    return fit_fractions, pdfs, histograms


def run_splot_fit_exp(
    rfl1: NDArray[np.float32],
    rfl2: NDArray[np.float32],
    weight: NDArray[np.float32],
    rfl1_sigmc: NDArray[np.float32],
    rfl2_sigmc: NDArray[np.float32],
    control_sigmc: NDArray[np.float32],
    weight_sigmc: NDArray[np.float32],
    rfl1_bkgmc: NDArray[np.float32],
    rfl2_bkgmc: NDArray[np.float32],
    control_bkgmc: NDArray[np.float32],
    weight_bkgmc: NDArray[np.float32],
    *,
    nsig: int,
    nbkg: int,
    fixed_sig: bool,
    fixed_bkg: bool,
) -> SPlotFitResultExp:
    n_spec = nsig + nbkg
    fit_fractions_sig, fits_sig = fit_components_exp(
        rfl1_sigmc, rfl2_sigmc, control_sigmc, weight_sigmc, n_spec=nsig
    )
    fit_fractions_bkg, fits_bkg = fit_components_exp(
        rfl1_bkgmc, rfl2_bkgmc, control_bkgmc, weight_bkgmc, n_spec=nbkg
    )
    nevents = np.sum(weight)
    # assume half the data is signal-like to initialize the fit
    yields0 = [
        fit_fraction * 0.5 * nevents for fit_fraction in fit_fractions_sig
    ] + [fit_fraction * 0.5 * nevents for fit_fraction in fit_fractions_bkg]
    logger.debug(f'Yields init: {yields0}')
    # get λs from fits
    ldas0: list[float] = [fit.values['lda'] for fit in fits_sig] + [
        fit.values['lda'] for fit in fits_bkg
    ]
    logger.debug(f'λs init: {ldas0}')

    def nll(*args: float) -> float:
        yields = args[::2]
        ldas = args[1::2]
        likelihoods: NDArray[np.float64] = weight * np.log(
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
        raise Exception('sPlot yield fit failed!')
    yields: list[float] = [m.values[f'y{i}'] for i in range(n_spec)]
    ldas: list[float] = [m.values[f'lda{i}'] for i in range(n_spec)]
    logger.debug(f'Yields (fit): {yields}')
    logger.debug(f'λs (fit): {ldas}')
    pdfs = [exp_pdf(rfl1, rfl2, ldas[i]) for i in range(n_spec)]
    denom: NDArray[np.float64] = np.sum(
        [yields[k] * pdfs[k] for k in range(n_spec)], axis=0
    )
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
    v: NDArray[np.float64] = np.linalg.inv(v_inv)  # pyright:ignore[reportAssignmentType]
    logger.debug(f'V = {v.tolist()}')
    fit_result = SPlotFitResultExp(
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


def run_splot_fit(
    rfl1: NDArray[np.float32],
    rfl2: NDArray[np.float32],
    weight: NDArray[np.float32],
    rfl1_sigmc: NDArray[np.float32],
    rfl2_sigmc: NDArray[np.float32],
    control_sigmc: NDArray[np.float32],
    weight_sigmc: NDArray[np.float32],
    rfl1_bkgmc: NDArray[np.float32],
    rfl2_bkgmc: NDArray[np.float32],
    control_bkgmc: NDArray[np.float32],
    weight_bkgmc: NDArray[np.float32],
    *,
    nsig: int,
    nsig_bins: int,
    nbkg: int,
    fixed_bkg: bool,
) -> SPlotFitResult:
    n_spec = nsig + nbkg
    fit_fractions_sig, sig_pdfs, sig_hists = fit_components(
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
    yields_sig = [
        fit_fraction * 0.5 * nevents for fit_fraction in fit_fractions_sig
    ]
    yields_bkg = [
        fit_fraction * 0.5 * nevents for fit_fraction in fit_fractions_bkg
    ]
    yields0 = yields_sig + yields_bkg
    logger.debug(f'Yields init: {yields0}')
    # get λs from fits
    ldas0 = [fit.values['lda'] for fit in fits_bkg]
    logger.debug(f'λs init: {ldas0}')

    sig_pdfs_evaluated = [sig_pdf(rfl1, rfl2) for sig_pdf in sig_pdfs]

    def nll(*args: float) -> float:
        yields_sig = args[:nsig]
        yields_bkg = args[nsig::2]
        ldas = args[nsig + 1 :: 2]
        yields = yields_sig + yields_bkg
        likelihoods: NDArray[np.float64] = weight * np.log(
            np.sum(
                [[yields_sig[i] * sig_pdfs_evaluated[i]] for i in range(nsig)],
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

    p0 = yields_sig + list(itertools.chain(*zip(yields_bkg, ldas0)))
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
        m.limits[f'lda{nsig + i}'] = (
            max(ldas0[i] - 30.0, 0.01),
            ldas0[i] + 30.0,
        )
        if fixed_bkg:
            m.fixed[f'lda{nsig + i}'] = True
    m.migrad(ncall=10_000)
    if not m.valid:
        logger.debug(m)
        logger.error('sPlot yield fit failed!')
        raise Exception('sPlot yield fit failed!')
    yields: list[float] = [m.values[f'y{i}'] for i in range(n_spec)]
    ldas: list[float] = [m.values[f'lda{nsig + i}'] for i in range(nbkg)]
    logger.debug(f'Yields (fit): {yields}')
    logger.debug(f'Background λs (fit): {ldas}')
    pdfs = sig_pdfs_evaluated + [
        exp_pdf(rfl1, rfl2, ldas[i]) for i in range(nbkg)
    ]
    denom: NDArray[np.float64] = np.sum(
        [yields[k] * pdfs[k] for k in range(n_spec)], axis=0
    )
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
    logger.debug(f'V⁻¹ = {v_inv.tolist()}')
    v: NDArray[np.float64] = np.linalg.inv(v_inv)  # pyright:ignore[reportAssignmentType]
    logger.debug(f'V = {v.tolist()}')
    fit_result = SPlotFitResult(
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
    fit_result: SPlotFitResult | SPlotFitResultExp,
    rfl1: NDArray[np.float32],
    rfl2: NDArray[np.float32],
    weight: NDArray[np.float32],
    *,
    nsig: int,
    nbkg: int,
) -> NDArray[np.float64]:
    n_spec = nsig + nbkg
    yields = fit_result.yields
    pdfs = fit_result.pdfs(rfl1, rfl2)
    denom: NDArray[np.float64] = np.sum(
        [yields[k] * pdfs[k] for k in range(n_spec)], axis=0
    )
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
    variable: NDArray[np.float32], *, bins: int, weights: NDArray[np.float32]
) -> NDArray[np.float32]:
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
    variable: NDArray[np.float32], *, bins: int, weights: NDArray[np.float32]
) -> list[NDArray[np.intp]]:
    quantiles = get_quantile_edges(variable, bins=bins, weights=weights)
    quantiles[-1] = (
        np.inf
    )  # ensure the maximum value gets placed in the last bin
    quantile_assignment = np.digitize(variable, quantiles)
    return [np.where(quantile_assignment == i)[0] for i in range(1, bins + 1)]


def run_factorization_fits_exp(
    rfl1: NDArray[np.float32],
    rfl2: NDArray[np.float32],
    weight: NDArray[np.float32],
    control: NDArray[np.float32],
    *,
    bins: int,
) -> FactorizationFitResult:
    quantile_indices = get_quantile_indices(control, bins=bins, weights=weight)

    def generate_nll(
        rfl1s: NDArray[np.float32],
        rfl2s: NDArray[np.float32],
        weights: NDArray[np.float32],
    ):
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

    h1s: list[Minuit] = []
    for i in range(bins):
        h1 = Minuit(nlls[i], z=0.5, lda_s=12.0, lda_b=100.0)  # pyright:ignore[reportArgumentType]
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
    rfl1: NDArray[np.float32],
    rfl2: NDArray[np.float32],
    weight: NDArray[np.float32],
    control: NDArray[np.float32],
    *,
    bins: int,
) -> FactorizationFitResult:
    quantile_indices = get_quantile_indices(control, bins=bins, weights=weight)

    def generate_nll(
        rfl1s: NDArray[np.float32],
        rfl2s: NDArray[np.float32],
        weights: NDArray[np.float32],
    ):
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

    h0 = Minuit(nll0, lda=50.0)  # pyright:ignore[reportArgumentType]
    h0.limits['lda'] = (5.0, 120.0)
    h0.migrad(ncall=10_000)
    if not h0.valid:
        logger.debug(h0)
        logger.error('Null hypothesis fit failed!')
        raise Exception('Null hypothesis fit failed!')

    h1s: list[Minuit] = []
    for i in range(bins):
        h1 = Minuit(nlls[i], lda=50.0)  # pyright:ignore[reportArgumentType]
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


def run_factorization_fits(
    rfl1: NDArray[np.float32],
    rfl2: NDArray[np.float32],
    weight: NDArray[np.float32],
    control: NDArray[np.float32],
    rfl1_sigmc: NDArray[np.float32],
    rfl2_sigmc: NDArray[np.float32],
    weight_sigmc: NDArray[np.float32],
    control_sigmc: NDArray[np.float32],
    *,
    bins: int,
    nsig_bins: int,
) -> FactorizationFitResult:
    quantile_indices = get_quantile_indices(control, bins=bins, weights=weight)

    fit_fractions_sig, sig_pdfs, sig_hists = fit_components(
        rfl1_sigmc,
        rfl2_sigmc,
        control_sigmc,
        weight_sigmc,
        n_spec=1,
        n_bins=nsig_bins,
    )
    sig_pdf_evaluated = sig_pdfs[0](rfl1, rfl2)

    def generate_nll(
        rfl1s: NDArray[np.float32],
        rfl2s: NDArray[np.float32],
        weights: NDArray[np.float32],
    ):
        def nll(z: float, lda_b: float) -> float:
            likelihoods: NDArray[np.float64] = weight * np.log(
                z * sig_pdf_evaluated
                + (1 - z) * exp_pdf(rfl1s, rfl2s, lda_b)
                + np.finfo(float).tiny
            )
            return (
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

    # arguments are (z_0, z_1, ..., z_{n-1}, lda_b)
    def nll0(*args: float) -> float:
        return np.sum(
            np.array([nlls[i](args[i], args[-1]) for i in range(bins)])
        )

    h0 = Minuit(
        nll0,
        *[0.5] * bins,
        100.0,
        name=[f'z{i}' for i in range(bins)] + ['lda_b'],
    )
    for i in range(bins):
        h0.limits[f'z{i}'] = (0.0, 1.0)
    h0.limits['lda_b'] = (80.0, 120.0)
    h0.migrad(ncall=10_000)
    if not h0.valid:
        logger.debug(h0)
        logger.error('Null hypothesis fit failed!')
        raise Exception('Null hypothesis fit failed!')

    h1s: list[Minuit] = []
    for i in range(bins):
        h1 = Minuit(nlls[i], z=0.5, lda_b=100.0)  # pyright:ignore[reportArgumentType]
        h1.limits['z'] = (0.0, 1.0)
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
        bins - 1,
    )
