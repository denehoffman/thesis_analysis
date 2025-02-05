import pickle
from pathlib import Path

import luigi
import matplotlib.pyplot as plt
import matplotlib.style as mpl_style

from thesis_analysis import colors, root_io
from thesis_analysis.constants import (
    BRANCH_NAME_TO_LATEX,
    BRANCH_NAME_TO_LATEX_UNITS,
    MC_TYPES,
    SPLOT_CONTROL,
    get_branch,
)
from thesis_analysis.paths import Paths
from thesis_analysis.tasks.chisqdof import ChiSqDOF
from thesis_analysis.tasks.factorization_fits import FactorizationFits
from thesis_analysis.utils import (
    FactorizationFitResult,
    get_quantile_edges,
)


class FactorizationPlot(luigi.Task):
    data_type = luigi.Parameter()
    run_period = luigi.Parameter()
    chisqdof = luigi.FloatParameter()
    n_quantiles = luigi.IntParameter()

    def requires(self):
        return [
            ChiSqDOF(self.data_type, self.run_period, self.chisqdof),
            FactorizationFits(
                self.data_type, self.run_period, self.chisqdof, self.n_quantiles
            ),
        ]

    def output(self):
        return [
            luigi.LocalTarget(
                Paths.plots
                / f'factorization_plot_{self.data_type}_{self.run_period}_chisqdof_{self.chisqdof:.1f}_{self.n_quantiles}_quantiles.png'
            ),
        ]

    def run(self):
        input_data_path = Path(self.input()[0][0].path)
        output_plot_path = Path(self.output()[0].path)
        output_plot_path.parent.mkdir(parents=True, exist_ok=True)

        data_df = root_io.get_branches(
            input_data_path,
            [
                get_branch('RFL1'),
                get_branch('Weight'),
                get_branch(SPLOT_CONTROL),
            ],
        )

        input_fit: FactorizationFitResult = pickle.load(
            Path(self.input()[1][0].path).open('rb')
        )

        quantile_edges = get_quantile_edges(
            data_df[SPLOT_CONTROL],
            bins=int(self.n_quantiles),  # type: ignore
        )
        quantile_centers = (quantile_edges[1:] + quantile_edges[:-1]) / 2
        mpl_style.use('thesis_analysis.thesis')
        fig, ax = plt.subplots()

        if self.data_type in MC_TYPES:
            for quantile_center, quantile_fit in zip(
                quantile_centers, input_fit.h1s
            ):
                ax.errorbar(
                    quantile_center,
                    quantile_fit.values['lda'],
                    yerr=quantile_fit.errors['lda'],
                    color=colors.black,
                    fmt='.',
                )
                ax.set_ylabel(r'$\lambda$ (ns${}^{-1}$)')
        else:
            ax_bkg = ax.twinx()
            ax.yaxis.label.set_color(colors.blue)
            ax.tick_params(axis='y', colors=colors.blue)
            ax_bkg.yaxis.label.set_color(colors.red)
            ax_bkg.spines.left.set_edgecolor(colors.blue)
            ax_bkg.spines.right.set_edgecolor(colors.red)
            ax_bkg.tick_params(axis='y', colors=colors.red)
            for quantile_center, quantile_fit in zip(
                quantile_centers, input_fit.h1s
            ):
                ax.errorbar(
                    quantile_center,
                    quantile_fit.values['lda_s'],
                    yerr=quantile_fit.errors['lda_s'],
                    color=colors.blue,
                    fmt='.',
                )
                ax_bkg.errorbar(
                    quantile_center,
                    quantile_fit.values['lda_b'],
                    yerr=quantile_fit.errors['lda_b'],
                    color=colors.red,
                    fmt='.',
                )
                ax.set_ylabel(r'$\lambda_S$ (ns${}^{-1}$)')
                ax_bkg.set_ylabel(r'$\lambda_B$ (ns${}^{-1}$)')
        ax.set_xlabel(
            f'{BRANCH_NAME_TO_LATEX[SPLOT_CONTROL]} ({BRANCH_NAME_TO_LATEX_UNITS[SPLOT_CONTROL]})'
        )

        fig.savefig(output_plot_path)
        plt.close()
