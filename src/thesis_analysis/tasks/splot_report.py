import pickle
from pathlib import Path
from typing import final, override

import luigi
import numpy as np

from thesis_analysis.constants import SPLOT_METHODS
from thesis_analysis.paths import Paths
from thesis_analysis.splot import SPlotFitFailure
from thesis_analysis.tasks.splot_fit import SPlotFit
from thesis_analysis.tasks.splot_plot import SPlotPlot


@final
class SPlotReport(luigi.Task):
    chisqdof = luigi.FloatParameter()
    ksb_costheta = luigi.FloatParameter()
    cut_baryons = luigi.OptionalBoolParameter(True)
    nsig_max = luigi.IntParameter()
    nbkg_max = luigi.IntParameter()

    @override
    def requires(self):
        nsig_max = int(self.nsig_max)  # pyright:ignore[reportArgumentType]
        nbkg_max = int(self.nbkg_max)  # pyright:ignore[reportArgumentType]
        return [
            SPlotFit(
                'data',
                self.chisqdof,
                self.ksb_costheta,
                self.cut_baryons,
                splot_method,
                nsig,
                nbkg,
            )
            for splot_method in SPLOT_METHODS
            for nsig in range(1, nsig_max + 1)
            for nbkg in range(1, nbkg_max + 1)
        ] + [
            SPlotPlot(
                'data',
                self.chisqdof,
                self.ksb_costheta,
                self.cut_baryons,
                splot_method,
                nsig,
                nbkg,
            )
            for splot_method in SPLOT_METHODS
            for nsig in range(1, nsig_max + 1)  # type: ignore
            for nbkg in range(1, nbkg_max + 1)  # type: ignore
        ]

    @override
    def output(self):
        return [
            luigi.LocalTarget(
                Paths.reports
                / f'splot_report_chisqdof_{self.chisqdof:.1f}_ksb_costheta_{self.ksb_costheta:.2f}{"mesons" if self.cut_baryons else "baryons"}.txt'
            ),
        ]

    @override
    def run(self):
        nsig_max = int(self.nsig_max)  # pyright:ignore[reportArgumentType]
        nbkg_max = int(self.nbkg_max)  # pyright:ignore[reportArgumentType]
        fit_results = {
            method: {
                nsig: {
                    nbkg: pickle.load(
                        Path(
                            self.input()[
                                i * nsig_max * nbkg_max + j * nbkg_max + k
                            ][0].path
                        ).open('rb')
                    )
                    for k, nbkg in enumerate(range(1, nbkg_max + 1))
                }
                for j, nsig in enumerate(range(1, nsig_max + 1))
            }
            for i, method in enumerate(SPLOT_METHODS)
        }
        output_path = Path(self.output()[0].path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        out_text = r"""\begin{table}
  \begin{center}
    \begin{tabular}{ccccc}\toprule
    & \multicolumn{2}{c}{\# Components} & \\\cmidrule(lr){2-3}
      Method & Signal & Background & $r\text{AIC}$ & $r\text{BIC}$\\\midrule
"""
        min_aic = np.inf
        min_aic_non_factorizing = np.inf
        min_bic = np.inf
        min_bic_non_factorizing = np.inf
        for method in SPLOT_METHODS:
            for nsig in range(1, nsig_max + 1):
                for nbkg in range(1, nbkg_max + 1):
                    fit_result = fit_results[method][nsig][nbkg]
                    if isinstance(fit_result, SPlotFitFailure):
                        continue
                    if fit_result.aic < min_aic:
                        min_aic = fit_result.aic
                    if fit_result.bic < min_bic:
                        min_bic = fit_result.bic
                    if nsig > 1 and nbkg > 1:
                        if fit_result.aic < min_aic_non_factorizing:
                            min_aic_non_factorizing = fit_result.aic
                        if fit_result.bic < min_bic_non_factorizing:
                            min_bic_non_factorizing = fit_result.bic
        current_method = ''
        for method in SPLOT_METHODS:
            for nsig in range(1, nsig_max + 1):
                for nbkg in range(1, nbkg_max + 1):
                    if method == current_method:
                        method_string = ''
                    else:
                        current_method = method
                        method_string = f'${method}$'
                    out_text += (
                        f'      {method_string} & ${nsig}$ & ${nbkg}$ & '
                    )
                    fit_result = fit_results[method][nsig][nbkg]
                    print(fit_result)
                    if not isinstance(fit_result, SPlotFitFailure):
                        if (
                            fit_result.aic == min_aic
                            and fit_result.aic == min_aic_non_factorizing
                        ):
                            out_text += f'\\fcolorbox{{red}}{{white}}{{\\underline{{${fit_result.aic - min_aic:.3f}$}}}} & '
                        elif fit_result.aic == min_aic:
                            out_text += f'\\underline{{${fit_result.aic - min_aic:.3f}$}} & '
                        elif fit_result.aic == min_aic_non_factorizing:
                            out_text += f'\\fcolorbox{{red}}{{white}}{{${fit_result.aic - min_aic:.3f}$}} & '
                        else:
                            out_text += f'${fit_result.aic - min_aic:.3f}$ & '
                        if (
                            fit_result.bic == min_bic
                            and fit_result.bic == min_bic_non_factorizing
                        ):
                            out_text += f'\\fcolorbox{{red}}{{white}}{{\\underline{{${fit_result.bic - min_bic:.3f}$}}}}'
                        elif fit_result.bic == min_bic:
                            out_text += f'\\underline{{${fit_result.bic - min_bic:.3f}$}}'
                        elif fit_result.bic == min_bic_non_factorizing:
                            out_text += f'\\fcolorbox{{red}}{{white}}{{${fit_result.bic - min_bic:.3f}$}}'
                        else:
                            out_text += f'${fit_result.bic - min_bic:.3f}$'
                    else:
                        out_text += r'\textemdash & \textemdash'
                    if (
                        nsig == nsig_max
                        and nbkg == nbkg_max
                        and method == SPLOT_METHODS[-1]
                    ):
                        out_text += '\\\\\\bottomrule\n'
                    elif nsig == nsig_max and nbkg == nbkg_max:
                        out_text += '\\\\\\midrule\n'
                    else:
                        out_text += '\\\\\n'
        out_text += r"""    \end{tabular}
    \caption{Relative AIC and BIC values for each fitting method. The absolute minimum values in each column are underlined, and the minimums excluding models with only one signal or background component are boxed. Methods for which the fits did not converge or $V^{-1}$ was singular are omitted.}\label{tab:splot-model-results}
  \end{center}
\end{table}"""
        output_path.write_text(out_text)
