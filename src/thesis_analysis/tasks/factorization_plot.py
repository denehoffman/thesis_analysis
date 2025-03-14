import pickle
from pathlib import Path
from typing import final, override

import luigi
import matplotlib.pyplot as plt
import matplotlib.style as mpl_style
import numpy as np
from thesis_analysis import colors, root_io
from thesis_analysis.constants import (
    BRANCH_NAME_TO_LATEX,
    BRANCH_NAME_TO_LATEX_UNITS,
    MC_TYPES,
    RUN_PERIODS,
    SPLOT_CONTROL,
    get_branch,
)
from thesis_analysis.paths import Paths
from thesis_analysis.splot import (
    FactorizationFitResult,
    get_quantile_edges,
)
from thesis_analysis.tasks.chisqdof import ChiSqDOF
from thesis_analysis.tasks.factorization_fit import FactorizationFit


@final
class FactorizationPlot(luigi.Task):
    data_type = luigi.Parameter()
    chisqdof = luigi.FloatParameter()
    n_quantiles = luigi.IntParameter()

    @override
    def requires(self):
        return [
            ChiSqDOF(self.data_type, run_period, self.chisqdof)
            for run_period in RUN_PERIODS
        ] + [
            FactorizationFit(self.data_type, self.chisqdof, self.n_quantiles),
        ]

    @override
    def output(self):
        return [
            luigi.LocalTarget(
                Paths.plots
                / f'factorization_plot_{self.data_type}_chisqdof_{self.chisqdof:.1f}_{self.n_quantiles}_quantiles.png'
            ),
        ]

    @override
    def run(self):
        input_data_paths = [
            Path(self.input()[i][0].path) for i in range(len(RUN_PERIODS))
        ]
        input_fit: FactorizationFitResult = pickle.load(
            Path(self.input()[-1][0].path).open('rb')
        )
        output_plot_path = Path(self.output()[0].path)
        output_plot_path.parent.mkdir(parents=True, exist_ok=True)

        n_quantiles = int(self.n_quantiles)  # pyright:ignore[reportArgumentType]

        data_dfs = {
            i: root_io.get_branches(
                input_data_paths[i],
                [
                    get_branch('RFL1'),
                    get_branch('Weight'),
                    get_branch(SPLOT_CONTROL),
                ],
            )
            for i in range(len(RUN_PERIODS))
        }
        data_df = {
            'RFL1': np.concatenate(
                [data_dfs[i]['RFL1'] for i in range(len(RUN_PERIODS))]
            ),
            'Weight': np.concatenate(
                [data_dfs[i]['Weight'] for i in range(len(RUN_PERIODS))]
            ),
            SPLOT_CONTROL: np.concatenate(
                [data_dfs[i][SPLOT_CONTROL] for i in range(len(RUN_PERIODS))]
            ),
        }

        quantile_edges = get_quantile_edges(
            data_df[SPLOT_CONTROL],
            bins=n_quantiles,
            weights=data_df['Weight'],
        )
        quantile_centers = (quantile_edges[1:] + quantile_edges[:-1]) / 2
        mpl_style.use('thesis_analysis.thesis')
        fig, ax = plt.subplots()
        ax.vlines(
            quantile_edges,
            0,
            1,
            transform=ax.get_xaxis_transform(),
            colors=colors.black,
        )

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
