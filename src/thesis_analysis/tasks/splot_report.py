import pickle
from pathlib import Path

import luigi
import numpy as np

from thesis_analysis.constants import SPLOT_METHODS
from thesis_analysis.paths import Paths
from thesis_analysis.splot import SPlotFitFailure
from thesis_analysis.tasks.splot_fit import SPlotFit
from thesis_analysis.tasks.splot_plot import SPlotPlot


class SPlotReport(luigi.Task):
    chisqdof = luigi.FloatParameter()
    nsig_max = luigi.IntParameter()
    nbkg_max = luigi.IntParameter()

    def requires(self):
        return [
            SPlotFit(
                'data',
                self.chisqdof,
                splot_method,
                nsig,
                nbkg,
            )
            for splot_method in SPLOT_METHODS
            for nsig in range(1, int(self.nsig_max) + 1)  # type: ignore
            for nbkg in range(1, int(self.nbkg_max) + 1)  # type: ignore
        ] + [
            SPlotPlot(
                'data',
                self.chisqdof,
                splot_method,
                nsig,
                nbkg,
            )
            for splot_method in SPLOT_METHODS
            for nsig in range(1, int(self.nsig_max) + 1)  # type: ignore
            for nbkg in range(1, int(self.nbkg_max) + 1)  # type: ignore
        ]

    def output(self):
        return [
            luigi.LocalTarget(
                Paths.reports / f'splot_report_chisqdof_{self.chisqdof:.1f}.txt'
            ),
        ]

    def run(self):
        fit_results = {
            method: {
                nsig: {
                    nbkg: pickle.load(
                        Path(
                            self.input()[
                                i
                                * int(self.nsig_max)  # type: ignore
                                * int(self.nbkg_max)  # type: ignore
                                + j * int(self.nbkg_max)  # type: ignore
                                + k
                            ][0].path
                        ).open('rb')
                    )
                    for k, nbkg in enumerate(range(1, int(self.nbkg_max) + 1))  # type: ignore
                }
                for j, nsig in enumerate(range(1, int(self.nsig_max) + 1))  # type: ignore
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
            for nsig in range(1, int(self.nsig_max) + 1):  # type: ignore
                for nbkg in range(1, int(self.nbkg_max) + 1):  # type: ignore
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
            for nsig in range(1, int(self.nsig_max) + 1):  # type: ignore
                for nbkg in range(1, int(self.nbkg_max) + 1):  # type: ignore
                    if method == current_method:
                        method_string = ''
                    else:
                        current_method = method
                        method_string = f'${method}$'
                    out_text += (
                        f'      {method_string} & ${nsig}$ & ${nbkg}$ & '
                    )
                    fit_result = fit_results[method][nsig][nbkg]
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
                        nsig == int(self.nsig_max)  # type: ignore
                        and nbkg == int(self.nbkg_max)  # type: ignore
                        and method == SPLOT_METHODS[-1]
                    ):
                        out_text += '\\\\\\bottomrule\n'
                    elif (
                        nsig == int(self.nsig_max)  # type: ignore
                        and nbkg == int(self.nbkg_max)  # type: ignore
                    ):
                        out_text += '\\\\\\midrule\n'
                    else:
                        out_text += '\\\\\n'
        out_text += r"""    \end{tabular}
    \caption{Relative AIC and BIC values for each fitting method. The absolute minimum values in each column are underlined, and the minimums excluding models with only one signal or background component are boxed. Methods for which the fits did not converge or $V^{-1}$ was singular are omitted.}\label{tab:splot-model-results}
  \end{center}
\end{table}"""
        output_path.write_text(out_text)
