import pickle
from pathlib import Path

import luigi

from thesis_analysis.constants import (
    RUN_PERIOD_LABELS,
    RUN_PERIODS,
)
from thesis_analysis.paths import Paths
from thesis_analysis.tasks.factorization_fit import FactorizationFit
from thesis_analysis.tasks.factorization_plot import FactorizationPlot
from thesis_analysis.utils import (
    FactorizationFitResult,
)


class FactorizationReport(luigi.Task):
    chisqdof = luigi.FloatParameter()
    max_quantiles = luigi.IntParameter()

    def requires(self):
        return [
            FactorizationFit(data_type, run_period, self.chisqdof, n_quantiles)
            for run_period in RUN_PERIODS
            for n_quantiles in range(2, int(self.max_quantiles) + 1)  # type: ignore
            for data_type in ['data', 'accmc', 'bkgmc']
        ] + [
            FactorizationPlot(
                data_type=data_type,
                run_period=run_period,
                chisqdof=self.chisqdof,
                n_quantiles=n_quantiles,
            )
            for run_period in RUN_PERIODS
            for n_quantiles in range(2, int(self.max_quantiles) + 1)  # type: ignore
            for data_type in ['data', 'accmc', 'bkgmc']
        ]

    def output(self):
        return [
            luigi.LocalTarget(
                Paths.reports
                / f'factorization_report_chisqdof_{self.chisqdof:.1f}_{self.max_quantiles}_quantiles.txt'
            ),
        ]

    def run(self):
        n_quantiles = int(self.max_quantiles) - 1  # type: ignore
        max_quantiles = int(self.max_quantiles)  # type: ignore
        input_fits = {
            run_period: {
                j: {
                    data_type: pickle.load(
                        Path(
                            self.input()[n_quantiles * 3 * i + 3 * (j - 2) + k][
                                0
                            ].path
                        ).open('rb')
                    )
                    for k, data_type in enumerate(['data', 'accmc', 'bkgmc'])
                }
                for j in range(2, max_quantiles + 1)
            }
            for i, run_period in enumerate(RUN_PERIODS)
        }
        output_report_path = Path(self.output()[0].path)
        output_report_path.parent.mkdir(parents=True, exist_ok=True)

        output = rf"""\begin{{table}}
  \begin{{center}}
    \begin{{tabular}}{{cc{'c' * n_quantiles}}}\toprule
       & & \multicolumn{{{n_quantiles}}}{{c}}{{\# quantiles}} \\\cmidrule(lr){{3-{3 + n_quantiles - 1}}}
       & & {' & '.join([str(i) for i in range(2, max_quantiles + 1)])} \\
       Data Type & Run Period & {' & '.join(['$p$'] * n_quantiles)} \\\midrule
"""
        for data_type, label in [
            ('data', 'Data'),
            ('accmc', '$K_S^0K_S^0$ MC'),
            ('bkgmc', r'$4\pi$ MC'),
        ]:
            first = True
            for run_period, run_period_latex in zip(
                RUN_PERIODS, RUN_PERIOD_LABELS
            ):
                if first:
                    output += f'      {label} & '
                    first = False
                else:
                    output += '       & '
                output += f'{run_period_latex}'
                for n_quantiles in range(2, max_quantiles + 1):
                    fit: FactorizationFitResult = input_fits[run_period][
                        n_quantiles
                    ][data_type]
                    sig_p = fit.significance.p
                    if sig_p == 0.0:
                        output += r' & $<2.23\times 10^{-308}$'
                    else:
                        output += f' & {latex(sig_p)}'
                if data_type == 'bkgmc' and run_period == RUN_PERIODS[-1]:
                    output += '\\\\\\bottomrule\n'
                elif run_period == RUN_PERIODS[-1]:
                    output += '\\\\\\midrule\n'
                else:
                    output += '\\\\\n'
        output += r"""    \end{tabular}
    \caption{The probability of accepting the null hypothesis (that the rest-frame lifetime is statistically independent of the invariant mass of $K_S^0K_S^0$) for the tests described in \Cref{eq:independence-test} for data and \Cref{eq:independence-test-mc} for Monte Carlo with the given number of quantiles. All values are calculated with a $\chi^2_\nu < 3.0$ selection on each type of data over each run period. Values listed as $<2.23 \times 10^{-308}$ are nonzero but smaller than the smallest representable 64-bit floating point number}\label{tab:factorization-results}
  \end{center}
\end{table}"""
        output_report_path.write_text(output)


def latex(value: float) -> str:
    mantissa, exponent = f'{value:.2E}'.split('E')
    return f'${mantissa} \\times 10^{{{exponent}}}$'
