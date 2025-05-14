import pickle
from pathlib import Path
from typing import final, override

import luigi
import numpy as np

from thesis_analysis.constants import NBOOT
from thesis_analysis.paths import Paths
from thesis_analysis.pwa import (
    BinnedFitResultUncertainty,
)
from thesis_analysis.tasks.binned_fit_uncertainty import BinnedFitUncertainty
from thesis_analysis.wave import Wave


@final
class BinnedFitReport(luigi.Task):
    waves = luigi.IntParameter()
    chisqdof = luigi.FloatParameter()
    ksb_costheta = luigi.FloatParameter()
    cut_baryons = luigi.OptionalBoolParameter(True)
    splot_method = luigi.Parameter()
    nsig = luigi.IntParameter()
    nbkg = luigi.IntParameter()
    niters = luigi.IntParameter(default=3, significant=False)
    phase_factor = luigi.BoolParameter(default=False)
    uncertainty = luigi.Parameter(default='bootstrap')

    @override
    def requires(self):
        return [
            BinnedFitUncertainty(
                self.waves,
                self.chisqdof,
                self.ksb_costheta,
                self.cut_baryons,
                self.splot_method,
                self.nsig,
                self.nbkg,
                self.niters,
                self.phase_factor,
                self.uncertainty,
            ),
        ]

    @override
    def output(self):
        return [
            luigi.LocalTarget(
                Paths.reports
                / f'binned_fit_chisqdof_{self.chisqdof:.1f}_ksb_costheta_{self.ksb_costheta:.2f}{"mesons" if self.cut_baryons else "baryons"}_splot_{self.splot_method}_{self.nsig}s_{self.nbkg}b{"_phase_factor" if self.phase_factor else ""}_waves{self.waves}_uncertainty_{self.uncertainty}.txt'
            ),
        ]

    @override
    def run(self):
        binned_fit_path = Path(str(self.input()[0][0]))

        output_report_path = Path(self.output()[0].path)
        output_report_path.parent.mkdir(parents=True, exist_ok=True)

        fit_result: BinnedFitResultUncertainty = pickle.load(
            binned_fit_path.open('rb')
        )
        waves = int(self.waves)  # pyright:ignore[reportArgumentType]

        output = r"""
\begin{center}
    \begin{longtable}{clrrr}\toprule
         Bin (GeV/$c^2$) & Wave & Real & Imaginary & Total ($\abs{F}^2$) \\\midrule
        \endhead
"""
        statuses = fit_result.fit_result.statuses
        edges = fit_result.fit_result.binning.edges
        for ibin in range(len(statuses)):
            bin_status = statuses[ibin]
            last_bin = (
                ibin == len(statuses) - 1 or ibin + 1 == len(statuses) - 1
            )
            bin_edges = rf'{edges[ibin]:.3f}\textendash {edges[ibin + 1]:.3f}'
            for iwave, wave in enumerate(Wave.decode_waves(waves)):
                last_wave = iwave == len(Wave.decode_waves(waves)) - 1
                coefficient_name = wave.coefficient_name
                i_re = fit_result.fit_result.model.parameters.index(
                    f'{coefficient_name} real'
                )
                c_re = bin_status.x[i_re]
                e_re = np.std(
                    [
                        fit_result.samples[ibin][j][i_re]
                        for j in range(len(fit_result.samples[ibin]))
                    ],
                    ddof=1,
                )
                if wave.l == 0:
                    c_im = 0.0
                    e_im = 0.0
                    c_tot = c_re**2
                    e_tot = np.std(
                        [
                            fit_result.samples[ibin][j][i_re] ** 2
                            for j in range(len(fit_result.samples[ibin]))
                        ],
                        ddof=1,
                    )
                else:
                    i_im = fit_result.fit_result.model.parameters.index(
                        f'{coefficient_name} imag'
                    )
                    c_im = bin_status.x[i_im]
                    e_im = np.std(
                        [
                            fit_result.samples[ibin][j][i_im]
                            for j in range(len(fit_result.samples[ibin]))
                        ],
                        ddof=1,
                    )
                    c_tot = c_re**2 + c_im**2
                    e_tot = np.std(
                        [
                            fit_result.samples[ibin][j][i_re] ** 2
                            + fit_result.samples[ibin][j][i_im] ** 2
                            for j in range(len(fit_result.samples[ibin]))
                        ],
                        ddof=1,
                    )
                output += rf'\n            {bin_edges if iwave == 0 else ""} & {wave.latex} & {latex(c_re, float(e_re))} & {latex(c_im, float(e_im))} & {latex(c_tot, float(e_tot))} \\*'
                if last_wave:
                    if last_bin:
                        output += r'\bottomrule'
                    else:
                        output += r'\midrule'
        output += rf"""
    \caption{{The parameter values and uncertainties for the binned fit of <?> waves to data with $\chi^2_\nu < {self.chisqdof:.1f}$. Uncertainties are calculated from the standard error over ${NBOOT}$ bootstrap iterations.}}\label{{tab:binned-fit-chisqdof-{self.chisqdof:.1f}-<?>}}
    \end{{longtable}}
\end{{center}}
"""
        output_report_path.write_text(output)


def latex(val: float, unc: float) -> str:
    if val == 0.0 and unc == 0.0:
        return r'$0.0$ (fixed)'
    unc_trunc = round(unc, -int(np.floor(np.log10(abs(unc)))) + 1)
    val_trunc = round(val, -int(np.floor(np.log10(abs(unc)))) + 1)
    ndigits = int(np.floor(np.log10(abs(unc)))) - 1
    expo = int(
        np.floor(np.log10(abs(val_trunc if val_trunc != 0.0 else unc_trunc)))
    )
    val_mantissa = val_trunc / 10**expo
    unc_mantissa = unc_trunc / 10**expo
    return rf'$({val_mantissa:.{expo - ndigits}f} \pm {unc_mantissa:.{expo - ndigits}f}) \times 10^{{{expo}}}$'
