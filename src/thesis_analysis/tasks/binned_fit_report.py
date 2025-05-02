import pickle
from pathlib import Path
from typing import final, override

import luigi
import numpy as np

from thesis_analysis.paths import Paths
from thesis_analysis.pwa import (
    BinnedFitResultUncertainty,
)
from thesis_analysis.tasks.binned_fit_uncertainty import BinnedFitUncertainty
from thesis_analysis.waves import Wave


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
                / f'binned_fit_chisqdof_{self.chisqdof:.1f}_splot_{self.splot_method}_{self.nsig}s_{self.nbkg}b{"_phase_factor" if self.phase_factor else ""}_waves{self.waves}_uncertainty_{self.uncertainty}{f"-{self.bootstrap_mode}" if str(self.uncertainty) == "bootstrap" else ""}.txt'
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
\begin{{table}}
    \begin{{center}}
        \begin{{tabular}}{{cccc}}\toprule
            Bin & Parameter & Value & $\abs{{F}}^2$ \\"""
        for ibin, bin_status in enumerate(fit_result.fit_result.statuses):
            for iwave, wave in enumerate(Wave.decode_waves(waves)):
                coefficient_name = wave.coefficient_name
                l_re = rf'$\Re{{{wave.latex}}}$'
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
                l_im = rf'$\Im{{{wave.latex}}}$'
                if wave.l == 0:
                    c_im = 0.0
                    e_im = 0.0
                    f_sq = c_re**2
                    e_f_sq = np.std(
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
                    f_sq = c_re**2 + c_im**2
                    e_f_sq = np.std(
                        [
                            fit_result.samples[ibin][j][i_re] ** 2
                            + fit_result.samples[ibin][j][i_im] ** 2
                            for j in range(len(fit_result.samples[ibin]))
                        ],
                        ddof=1,
                    )
                if iwave == 0:
                    output += f'\\midrule\n            {ibin:2}  '
                else:
                    output += '\n                '
                output += rf'& {l_re} & {latex(c_re, e_re)} & {latex(f_sq, e_f_sq)} \\\n'
                output += rf'& {l_im} & {latex(c_im, e_im)} & \\'
        output += r"""\bottomrule
            \end{tabular}
        \caption{<insert caption>}\label{tab:<insert table name>}
    \end{center}
\end{table}
"""
        output_report_path.write_text(output)


def latex(val: float, unc: float) -> str:
    unc_trunc = round(unc, -int(np.floor(np.log10(abs(unc)))) + 1)
    val_trunc = round(val, -int(np.floor(np.log10(abs(unc)))) - 1)
    expo = int(np.floor(np.log10(abs(val_trunc))))
    val_mantissa = val_trunc / 10**expo
    unc_mantissa = unc_trunc / 10**expo
    decimals = -int(np.floor(np.log10(abs(unc_mantissa)))) + 1
    value_string = f'{val_mantissa:.{decimals}f}'[:-2]
    uncertainty_string = f'{unc_mantissa:.{decimals}f}'[decimals:]
    return rf'${value_string}({uncertainty_string}) \times 10^{{{expo}}}$'
