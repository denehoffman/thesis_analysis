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
        input_paths = [
            Path(self.input()[i][0].path) for i in range(len(RUN_PERIODS))
        ]
        input_fit: FactorizationFitResult = pickle.load(
            Path(self.input()[-1][0].path).open('rb')
        )
        output_plot_path = Path(self.output()[0].path)
        output_plot_path.parent.mkdir(parents=True, exist_ok=True)

        n_quantiles = int(self.n_quantiles)  # pyright:ignore[reportArgumentType]

        branches = [
            get_branch('RFL1'),
            get_branch('Weight'),
            get_branch(SPLOT_CONTROL),
        ]
        flat_data = root_io.concatenate_branches(
            input_paths, branches, root=False
        )
        quantile_edges = get_quantile_edges(
            flat_data[SPLOT_CONTROL],
            bins=n_quantiles,
            weights=flat_data['Weight'],
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
            for quantile_center, quantile_fit in zip(
                quantile_centers, input_fit.h1s
            ):
                ax.errorbar(
                    quantile_center,
                    quantile_fit.values['lda_b'],
                    yerr=quantile_fit.errors['lda_b'],
                    color=colors.blue,
                    fmt='.',
                )
                ax.set_ylabel(r'$\lambda_B$ (ns${}^{-1}$)')
        ax.set_xlabel(
            f'{BRANCH_NAME_TO_LATEX[SPLOT_CONTROL]} ({BRANCH_NAME_TO_LATEX_UNITS[SPLOT_CONTROL]})'
        )

        fig.savefig(output_plot_path)
        plt.close()
