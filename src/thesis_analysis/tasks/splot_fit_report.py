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


from thesis_analysis import root_io
from thesis_analysis.constants import get_branch
from thesis_analysis.splot import (
    SPlotFitResult,
    SPlotFitResultExp,
    get_sweights,
    splot_method_rename,
)
from thesis_analysis.tasks.chisqdof import ChiSqDOF
from thesis_analysis.tasks.splot_fit import SPlotFit
from thesis_analysis.tasks.splot_plot import SPlotPlot


@final
class SPlotFitReport(luigi.Task):
    data_type = luigi.Parameter()
    chisqdof = luigi.FloatParameter()
    splot_method = luigi.Parameter()
    nsig = luigi.IntParameter()
    nbkg = luigi.IntParameter()

    @override
    def requires(self):
        return [
            SPlotFit(
                self.data_type,
                self.chisqdof,
                self.splot_method,
                self.nsig,
                self.nbkg,
            ),
        ]

    @override
    def output(self):
        return [
            luigi.LocalTarget(
                Paths.reports
                / f'splot_fit_chisqdof_{self.chisqdof:.1f}_splot_{self.splot_method}_{self.nsig}s_{self.nbkg}b.txt'
            ),
        ]

    @override
    def run(self):
        fit_path = Path(str(self.input()[0][0]))

        output_report_path = Path(self.output()[0].path)
        output_report_path.parent.mkdir(parents=True, exist_ok=True)

        fit_result: SPlotFitResult | SPlotFitResultExp = pickle.load(
            fit_path.open('rb')
        )

        nsig = int(self.nsig)  # pyright:ignore[reportArgumentType]
        nbkg = int(self.nbkg)  # pyright:ignore[reportArgumentType]

        output = r"""
\begin{table}
    \begin{center}
        \begin{tabular}{cr}\toprule
            Parameter & Value \\\midrule"""
        for (par_name, value), (_, error) in zip(
            fit_result.total_fit.values.items(),
            fit_result.total_fit.errors.items(),
        ):
            if par_name.startswith('y'):
                component = int(par_name[1:])
                if component < nsig:
                    normalized_name = rf'Signal Yield $\#{component + 1}$'
                else:
                    normalized_name = (
                        rf'Background Yield $\#{component - nsig + 1}$'
                    )
            else:
                component = int(par_name[3:])  # 'lda' + #
                if component < nsig:
                    normalized_name = rf'Signal $\lambda$ $\#{component + 1}$'
                else:
                    normalized_name = (
                        rf'Background $\lambda$ $\#{component - nsig + 1}$'
                    )
            output += f'\n{normalized_name} & {latex(value, error)} \\\\'
        if (method := splot_method_rename(str(self.splot_method))) in {
            'A',
            'B',
        } and nsig == 1:
            method_name = f'${method}{nbkg}$'
        else:
            method_name = (
                f'{method} for {nsig} signal and {nbkg} background components'
            )

        output += rf"""\bottomrule
        \end{{tabular}}
        \caption{{The parameter values and uncertainties for the sPlot fit of data with $\chi^2_\nu < {self.chisqdof:.1f}$ using method {method_name}. Uncertainties are calculated using the covariance matrix of the fit. All $\lambda$ parameters have units of $\si{{\nano\second}}^{{-1}}$.}}\label{{tab:splot-fit-chisqdof-{self.chisqdof:.1f}-splot-{self.splot_method}-{self.nsig}s-{self.nbkg}b}}
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
