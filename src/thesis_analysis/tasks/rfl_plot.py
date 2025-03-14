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
    DATA_TYPE_TO_LATEX,
    RUN_PERIODS,
    get_branch,
)
from thesis_analysis.paths import Paths
from thesis_analysis.tasks.chisqdof import ChiSqDOF


@final
class RFLPlot(luigi.Task):
    data_type = luigi.Parameter()
    chisqdof = luigi.FloatParameter()

    @override
    def requires(self):
        return [
            ChiSqDOF(self.data_type, run_period, self.chisqdof)
            for run_period in RUN_PERIODS
        ]

    @override
    def output(self):
        return [
            luigi.LocalTarget(
                Paths.plots
                / f'rfl_{self.data_type}_chisqdof_{self.chisqdof:.1f}.png'
            ),
        ]

    @override
    def run(self):
        input_paths = [
            Path(self.input()[i][0].path) for i in range(len(RUN_PERIODS))
        ]
        output_plot_path = Path(self.output()[0].path)
        output_plot_path.parent.mkdir(parents=True, exist_ok=True)

        data_dfs = {
            i: root_io.get_branches(
                input_paths[i],
                [
                    get_branch('RFL1'),
                    get_branch('RFL2'),
                    get_branch('Weight'),
                ],
            )
            for i in range(len(RUN_PERIODS))
        }
        data_df = {
            'RFL1': np.concatenate(
                [data_dfs[i]['RFL1'] for i in range(len(RUN_PERIODS))]
            ),
            'RFL2': np.concatenate(
                [data_dfs[i]['RFL2'] for i in range(len(RUN_PERIODS))]
            ),
            'Weight': np.concatenate(
                [data_dfs[i]['Weight'] for i in range(len(RUN_PERIODS))]
            ),
        }

        nbins = 100
        mpl_style.use('thesis_analysis.thesis')
        fig, ax = plt.subplots()
        ax.hist(
            data_df['RFL1'],
            weights=data_df['Weight'],
            bins=nbins,
            range=(0.0, 0.2),
            color=colors.blue,
            label=DATA_TYPE_TO_LATEX[str(self.data_type)],
        )
        bin_width = 0.2 / nbins
        ax.set_xlabel(
            f'{BRANCH_NAME_TO_LATEX["RFL1"]} ({BRANCH_NAME_TO_LATEX_UNITS["RFL1"]})'
        )
        ax.set_ylabel(
            f'Counts / {bin_width:.3f} {BRANCH_NAME_TO_LATEX_UNITS["RFL1"]}'
        )
        ax.legend()
        fig.savefig(output_plot_path)
        plt.close()
