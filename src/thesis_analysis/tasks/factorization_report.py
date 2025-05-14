import pickle
from pathlib import Path
from typing import final, override

import luigi

from thesis_analysis.paths import Paths
from thesis_analysis.splot import (
    FactorizationFitResult,
)
from thesis_analysis.tasks.factorization_fit import FactorizationFit
from thesis_analysis.tasks.factorization_plot import FactorizationPlot


@final
class FactorizationReport(luigi.Task):
    chisqdof = luigi.FloatParameter()
    ksb_costheta = luigi.FloatParameter()
    cut_baryons = luigi.OptionalBoolParameter(True)
    max_quantiles = luigi.IntParameter()

    @override
    def requires(self):
        max_quantiles = int(self.max_quantiles)  # pyright:ignore[reportArgumentType]
        return [
            FactorizationFit(
                data_type,
                self.chisqdof,
                self.ksb_costheta,
                self.cut_baryons,
                n_quantiles,
            )
            for n_quantiles in range(2, max_quantiles + 1)  # type: ignore
            for data_type in ['data', 'accmc', 'bkgmc']
        ] + [
            FactorizationPlot(
                data_type=data_type,
                chisqdof=self.chisqdof,
                ksb_costheta=self.ksb_costheta,
                cut_baryons=self.cut_baryons,
                n_quantiles=n_quantiles,
            )
            for n_quantiles in range(2, max_quantiles + 1)  # type: ignore
            for data_type in ['data', 'accmc', 'bkgmc']
        ]

    @override
    def output(self):
        return [
            luigi.LocalTarget(
                Paths.reports
                / f'factorization_report_chisqdof_{self.chisqdof:.1f}_ksb_costheta_{self.ksb_costheta:.2f}{"mesons" if self.cut_baryons else "baryons"}_{self.max_quantiles}_quantiles.txt'
            ),
        ]

    @override
    def run(self):
        max_quantiles = int(self.max_quantiles)  # pyright:ignore[reportArgumentType]
        n_quantiles = max_quantiles - 1
        input_fits = {
            j: {
                data_type: pickle.load(
                    Path(self.input()[3 * (j - 2) + k][0].path).open('rb')
                )
                for k, data_type in enumerate(['data', 'accmc', 'bkgmc'])
            }
            for j in range(2, max_quantiles + 1)
        }
        output_report_path = Path(self.output()[0].path)
        output_report_path.parent.mkdir(parents=True, exist_ok=True)

        output = rf"""\begin{{table}}
  \begin{{center}}
    \begin{{tabular}}{{c{'c' * n_quantiles}}}\toprule
       & \multicolumn{{{n_quantiles}}}{{c}}{{\# quantiles}} \\\cmidrule(lr){{2-{2 + n_quantiles - 1}}}
       & {' & '.join([str(i) for i in range(2, max_quantiles + 1)])} \\
       Data Type & {' & '.join(['$p$'] * n_quantiles)} \\\midrule
"""
        for data_type, label in [
            ('data', 'Data'),
            ('accmc', '$K_S^0K_S^0$ MC'),
            ('bkgmc', r'$4\pi$ MC'),
        ]:
            output += f'      {label} '
            for n_quantiles in range(2, max_quantiles + 1):
                fit: FactorizationFitResult = input_fits[n_quantiles][data_type]
                sig_p = fit.significance.p
                if sig_p == 0.0:
                    output += r' & $<2.23\times 10^{-308}$'
                else:
                    output += f' & {latex(sig_p)}'
            if data_type == 'bkgmc':
                output += '\\\\\\bottomrule\n'
            else:
                output += '\\\\\n'
        output += r"""    \end{tabular}
    \caption{The probability of accepting the null hypothesis (that the rest-frame lifetime is statistically independent of the invariant mass of $K_S^0K_S^0$) for the tests described in \Cref{eq:independence-test} for data and \Cref{eq:independence-test-mc} for Monte Carlo with the given number of quantiles. All values are calculated with a $\chi^2_\nu < 3.0$ selection on each type of data over all run period combined. Values listed as $<2.23 \times 10^{-308}$ are nonzero but smaller than the smallest representable 64-bit floating point number}\label{tab:factorization-results}
  \end{center}
\end{table}"""
        output_report_path.write_text(output)


def latex(value: float) -> str:
    mantissa, exponent = f'{value:.2E}'.split('E')
    return f'${mantissa} \\times 10^{{{exponent}}}$'
