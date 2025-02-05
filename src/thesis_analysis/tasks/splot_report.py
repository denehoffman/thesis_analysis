from pathlib import Path

import luigi
import numpy as np

from thesis_analysis.constants import RUN_PERIODS, SPLOT_METHODS
from thesis_analysis.paths import Paths
from thesis_analysis.tasks.plot_splot import PlotSPlot
from thesis_analysis.tasks.splot_fit import SPlotFit
from thesis_analysis.utils import SPlotFitResult


class SPlotReport(luigi.Task):
    chisqdof = luigi.FloatParameter()
    nsig_max = luigi.IntParameter()
    nbkg_max = luigi.IntParameter()

    def requires(self):
        return [
            SPlotFit(
                'data',
                run_period,
                self.chisqdof,
                splot_method,
                nsig,
                nbkg,
            )
            for splot_method in SPLOT_METHODS
            for nsig in range(1, int(self.nsig_max) + 1)  # type: ignore
            for nbkg in range(1, int(self.nbkg_max) + 1)  # type: ignore
            for run_period in RUN_PERIODS
        ] + [
            PlotSPlot(
                'data',
                run_period,
                self.chisqdof,
                splot_method,
                nsig,
                nbkg,
            )
            for splot_method in SPLOT_METHODS
            for nsig in range(1, int(self.nsig_max) + 1)  # type: ignore
            for nbkg in range(1, int(self.nbkg_max) + 1)  # type: ignore
            for run_period in RUN_PERIODS
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
                    nbkg: {
                        run_period: SPlotFitResult.load(
                            Path(
                                self.input()[
                                    i
                                    * int(self.nsig_max)  # type: ignore
                                    * int(self.nbkg_max)  # type: ignore
                                    * len(RUN_PERIODS)
                                    + j * int(self.nbkg_max) * len(RUN_PERIODS)  # type: ignore
                                    + k * len(RUN_PERIODS)
                                    + m
                                ][0].path
                            )
                        )
                        for m, run_period in enumerate(RUN_PERIODS)
                    }
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
    \begin{tabular}{ccccccccccccc}\toprule
    & \multicolumn{2}{c}{\# Components} & \multicolumn{2}{c}{Spring 2017} & \multicolumn{2}{c}{Spring 2018} & \multicolumn{2}{c}{Fall 2018} & \multicolumn{2}{c}{Spring 2020}\\\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}
      Method & Signal & Background & $r\text{AIC}$ & $r\text{BIC}$ & $r\text{AIC}$ & $r\text{BIC}$ & $r\text{AIC}$ & $r\text{BIC}$ & $r\text{AIC}$ & $r\text{BIC}$\\\midrule
"""
        min_aic = {run_period: np.inf for run_period in RUN_PERIODS}
        min_aic_non_factorizing = {
            run_period: np.inf for run_period in RUN_PERIODS
        }
        min_bic = {run_period: np.inf for run_period in RUN_PERIODS}
        min_bic_non_factorizing = {
            run_period: np.inf for run_period in RUN_PERIODS
        }
        for method in SPLOT_METHODS:
            for nsig in range(1, int(self.nsig_max) + 1):  # type: ignore
                for nbkg in range(1, int(self.nbkg_max) + 1):  # type: ignore
                    for run_period in RUN_PERIODS:
                        fit_result = fit_results[method][nsig][nbkg][
                            run_period
                        ]
                        if fit_result.aic < min_aic[run_period]:
                            min_aic[run_period] = fit_result.aic
                        if fit_result.bic < min_bic[run_period]:
                            min_bic[run_period] = fit_result.bic
                        if nsig > 1 and nbkg > 1:
                            if (
                                fit_result.aic
                                < min_aic_non_factorizing[run_period]
                            ):
                                min_aic_non_factorizing[run_period] = (
                                    fit_result.aic
                                )
                            if (
                                fit_result.bic
                                < min_bic_non_factorizing[run_period]
                            ):
                                min_bic_non_factorizing[run_period] = (
                                    fit_result.bic
                                )
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
                    for run_period in RUN_PERIODS:
                        fit_result = fit_results[method][nsig][nbkg][
                            run_period
                        ]
                        if (
                            fit_result.aic == min_aic[run_period]
                            and fit_result.aic
                            == min_aic_non_factorizing[run_period]
                        ):
                            out_text += f'\\fcolorbox{{red}}{{white}}{{\\underline{{${fit_result.aic - min_aic[run_period]:.3f}$}}}} & '
                        elif fit_result.aic == min_aic[run_period]:
                            out_text += f'\\underline{{${fit_result.aic - min_aic[run_period]:.3f}$}} & '
                        elif (
                            fit_result.aic
                            == min_aic_non_factorizing[run_period]
                        ):
                            out_text += f'\\fcolorbox{{red}}{{white}}{{${fit_result.aic - min_aic[run_period]:.3f}$}} & '
                        else:
                            out_text += f'${fit_result.aic - min_aic[run_period]:.3f}$ & '
                        if (
                            fit_result.bic == min_bic[run_period]
                            and fit_result.bic
                            == min_bic_non_factorizing[run_period]
                        ):
                            out_text += f'\\fcolorbox{{red}}{{white}}{{\\underline{{${fit_result.bic - min_bic[run_period]:.3f}$}}}}'
                        elif fit_result.bic == min_bic[run_period]:
                            out_text += f'\\underline{{${fit_result.bic - min_bic[run_period]:.3f}$}}'
                        elif (
                            fit_result.bic
                            == min_bic_non_factorizing[run_period]
                        ):
                            out_text += f'\\fcolorbox{{red}}{{white}}{{${fit_result.bic - min_bic[run_period]:.3f}$}}'
                        else:
                            out_text += (
                                f'${fit_result.bic - min_bic[run_period]:.3f}$'
                            )
                        if (
                            run_period == RUN_PERIODS[-1]
                            and nsig == int(self.nsig_max)  # type: ignore
                            and nbkg == int(self.nbkg_max)  # type: ignore
                            and method == SPLOT_METHODS[-1]
                        ):
                            out_text += '\\\\\\bottomrule\n'
                        elif (
                            run_period == RUN_PERIODS[-1]
                            and nsig == int(self.nsig_max)  # type: ignore
                            and nbkg == int(self.nbkg_max)  # type: ignore
                        ):
                            out_text += '\\\\\\midrule\n'
                        elif run_period == RUN_PERIODS[-1]:
                            out_text += '\\\\\n'
                        else:
                            out_text += ' & '
        out_text += r"""    \end{tabular}
    \caption{Relative AIC and BIC values for each fitting method (relative within each run period). The absolute minimum values in each column are underlined, and the minimums excluding models with only one signal or background component are boxed.}\label{tab:splot-model-results}
  \end{center}
\end{table}"""
        output_path.write_text(out_text)
