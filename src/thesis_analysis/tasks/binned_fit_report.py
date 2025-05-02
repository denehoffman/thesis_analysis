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
                / f'binned_fit_chisqdof_{self.chisqdof:.1f}_splot_{self.splot_method}_{self.nsig}s_{self.nbkg}b{"_phase_factor" if self.phase_factor else ""}_waves{self.waves}_uncertainty_{self.uncertainty}.txt'
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
    \begin{longtable}{ccrcccr}\toprule
        Bin (GeV/$c^2$) & Parameter & Value & \hspace{1em} & Bin (GeV/$c^2$) & Parameter & Value \\\midrule
        \endhead
"""
        statuses = fit_result.fit_result.statuses
        edges = fit_result.fit_result.binning.edges
        for ibin in range(0, len(statuses), 2):
            bin_status = statuses[ibin]
            bin_status2 = statuses[ibin + 1]
            last_bin = (
                ibin == len(statuses) - 1 or ibin + 1 == len(statuses) - 1
            )
            bin_edges = rf'{edges[ibin]:.3f}\textendash {edges[ibin + 1]:.3f}'
            bin_edges2 = (
                rf'{edges[ibin + 1]:.3f}\textendash {edges[ibin + 2]:.3f}'
            )
            for iwave, wave in enumerate(Wave.decode_waves(waves)):
                last_wave = iwave == len(Wave.decode_waves(waves)) - 1
                coefficient_name = wave.coefficient_name
                l_re = rf'$\Re\left[{wave.latex.replace("$", "")}\right]$'
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
                c_re2 = bin_status2.x[i_re]
                e_re2 = np.std(
                    [
                        fit_result.samples[ibin + 1][j][i_re]
                        for j in range(len(fit_result.samples[ibin + 1]))
                    ],
                    ddof=1,
                )
                output += '\n'
                if iwave == 0:
                    output += rf'            {bin_edges} & {l_re} & {latex(c_re, float(e_re))} & & {bin_edges2} & {l_re} & {latex(c_re2, float(e_re2))} \\*'
                else:
                    output += rf'               & {l_re} & {latex(c_re, float(e_re))} & &    & {l_re} & {latex(c_re2, float(e_re2))} \\*'
                l_im = rf'$\Im\left[{wave.latex.replace("$", "")}\right]$'
                if wave.l != 0:
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
                    c_im2 = bin_status2.x[i_im]
                    e_im2 = np.std(
                        [
                            fit_result.samples[ibin + 1][j][i_im]
                            for j in range(len(fit_result.samples[ibin + 1]))
                        ],
                        ddof=1,
                    )
                    output += '\n'
                    output += rf'& {l_im} & {latex(c_im, float(e_im))} & &    & {l_im} & {latex(c_im2, float(e_im2))} \\*'
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
    unc_trunc = round(unc, -int(np.floor(np.log10(abs(unc)))) + 1)
    val_trunc = round(val, -int(np.floor(np.log10(abs(unc)))) + 1)
    ndigits = int(np.floor(np.log10(abs(unc)))) - 1
    expo = int(
        np.floor(np.log10(abs(val_trunc if val_trunc != 0.0 else unc_trunc)))
    )
    val_mantissa = val_trunc / 10**expo
    unc_mantissa = unc_trunc / 10**expo
    return rf'$({val_mantissa:.{expo - ndigits}f} \pm {unc_mantissa:.{expo - ndigits}f}) \times 10^{{{expo}}}$'
