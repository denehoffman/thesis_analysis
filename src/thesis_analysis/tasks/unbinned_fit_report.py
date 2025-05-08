import pickle
from pathlib import Path
from typing import final, override

import luigi
import numpy as np

from thesis_analysis.constants import NBOOT
from thesis_analysis.paths import Paths
from thesis_analysis.pwa import (
    UnbinnedFitResultUncertainty,
)
from thesis_analysis.tasks.unbinned_fit_uncertainty import (
    UnbinnedFitUncertainty,
)
from thesis_analysis.wave import Wave


@final
class UnbinnedFitReport(luigi.Task):
    waves = luigi.IntParameter()
    chisqdof = luigi.FloatParameter()
    splot_method = luigi.Parameter()
    nsig = luigi.IntParameter()
    nbkg = luigi.IntParameter()
    niters = luigi.IntParameter(default=3, significant=False)
    guided = luigi.BoolParameter(default=False)
    phase_factor = luigi.BoolParameter(default=False)
    uncertainty = luigi.Parameter(default='bootstrap')
    bootstrap_mode = luigi.Parameter(default='SE')

    @override
    def requires(self):
        return [
            UnbinnedFitUncertainty(
                self.waves,
                self.chisqdof,
                self.splot_method,
                self.nsig,
                self.nbkg,
                self.niters,
                self.guided,
                self.phase_factor,
                self.uncertainty,
                self.bootstrap_mode,
            )
        ]

    @override
    def output(self):
        return [
            luigi.LocalTarget(
                Paths.reports
                / f'unbinned_fit_chisqdof_{self.chisqdof:.1f}_splot_{self.splot_method}_{self.nsig}s_{self.nbkg}b{"_guided" if self.guided else ""}{"_phase_factor" if self.phase_factor else ""}_waves{self.waves}_uncertainty_{self.uncertainty}_unbinned_bootstrap.txt'
            ),
        ]

    @override
    def run(self):
        binned_fit_path = Path(str(self.input()[0][0]))

        output_report_path = Path(self.output()[0].path)
        output_report_path.parent.mkdir(parents=True, exist_ok=True)

        fit_result: UnbinnedFitResultUncertainty = pickle.load(
            binned_fit_path.open('rb')
        )

        output = r"""
\begin{table}
    \begin{center}
        \begin{tabular}{llrrr}\toprule
            Wave & Resonance & Real & Imaginary & Total ($\abs{F}^2$) \\\midrule"""

        model = fit_result.fit_result.model
        status = fit_result.fit_result.status
        latest_wave = None
        for i, parameter in enumerate(model.parameters):
            if parameter.endswith('real'):
                parameter_real = parameter
                parameter_imag = parameter_real.replace('real', 'imag')
                latex_res_name, latex_wave_name = (
                    Wave.kmatrix_parameter_name_to_latex_parts(parameter_real)
                )
                value_real = status.x[i]
                unc_real = np.std(
                    [
                        fit_result.samples[j][i]
                        for j in range(len(fit_result.samples))
                    ],
                    ddof=1,
                )
                if parameter_imag in model.parameters:
                    value_imag = status.x[i + 1]
                    unc_imag = np.std(
                        [
                            fit_result.samples[j][i + 1]
                            for j in range(len(fit_result.samples))
                        ],
                        ddof=1,
                    )
                    total_mag = value_real**2 + value_imag**2
                    total_mag_unc = np.std(
                        [
                            fit_result.samples[j][i] ** 2
                            + fit_result.samples[j][i + 1] ** 2
                            for j in range(len(fit_result.samples))
                        ],
                        ddof=1,
                    )
                else:
                    value_imag = 0.0
                    unc_imag = 0.0
                    total_mag = value_real**2
                    total_mag_unc = np.std(
                        [
                            fit_result.samples[j][i] ** 2
                            for j in range(len(fit_result.samples))
                        ],
                        ddof=1,
                    )
                if latex_wave_name == latest_wave:
                    wave = ''
                else:
                    wave = latex_wave_name
                    latest_wave = latex_wave_name
                output += f'\n{wave} & {latex_res_name} & {latex(value_real, unc_real)} & {latex(value_imag, unc_imag)} & {latex(total_mag, total_mag_unc)} \\\\'

        output += rf"""\bottomrule
        \end{{tabular}}
    \caption{{The parameter values and uncertainties for the unbinned {'(guided) ' if self.guided else ''}fit of <?> waves to data with $\chi^2_\nu < {self.chisqdof:.1f}$. Uncertainties are calculated from the standard error over ${NBOOT}$ bootstrap iterations.}}\label{{tab:unbinned-fit-chisqdof-{self.chisqdof:.1f}{'-guided' if self.guided else ''}-<?>}}
    \end{{center}}
\end{{table}}
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
