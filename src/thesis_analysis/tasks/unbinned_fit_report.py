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
        \begin{tabular}{cr}\toprule
            Parameter & Value \\\midrule"""

        model = fit_result.fit_result.model
        status = fit_result.fit_result.status
        for i, parameter in enumerate(model.parameters):
            latex_par_name = Wave.kmatrix_parameter_name_to_latex(parameter)
            value = status.x[i]
            unc = np.std(
                [
                    fit_result.samples[j][i]
                    for j in range(len(fit_result.samples))
                ],
                ddof=1,
            )
            output += f'\n{latex_par_name} & {latex(value, float(unc))}\\\\'

        output += rf"""\bottomrule
        \end{{tabular}}
    \caption{{The parameter values and uncertainties for the unbinned {'(guided) ' if self.guided else ''}fit of <?> waves to data with $\chi^2_\nu < {self.chisqdof:.1f}$. Uncertainties are calculated from the standard error over ${NBOOT}$ bootstrap iterations.}}\label{{tab:unbinned-fit-chisqdof-{self.chisqdof:.1f}{'-guided' if self.guided else ''}-<?>}}
    \end{{center}}
\end{{table}}
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
