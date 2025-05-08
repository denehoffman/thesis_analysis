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
from thesis_analysis.tasks.data import GetData


@final
class RFPlot(luigi.Task):
    data_type = luigi.Parameter()
    chisqdof = luigi.OptionalFloatParameter(None)
    protonz = luigi.OptionalBoolParameter(False)

    @override
    def requires(self):
        return [
            GetData(self.data_type, run_period) for run_period in RUN_PERIODS
        ]

    @override
    def output(self):
        return [
            luigi.LocalTarget(
                Paths.plots
                / f'{self.data_type}_RF{f"_chisqdof_{self.chisqdof:.1f}" if self.chisqdof is not None else ""}{"_protonz" if self.protonz else ""}.png'
            ),
        ]

    @override
    def run(self):
        input_paths = [
            Path(self.input()[i][0].path) for i in range(len(RUN_PERIODS))
        ]
        output_path_rf = self.output()[0].path

        branches = [
            get_branch('ChiSqDOF'),
            get_branch('RF'),
            get_branch('Proton_Z'),
        ]
        flat_data = root_io.concatenate_branches(
            input_paths, branches, root=False
        )
        if self.chisqdof is not None:
            mask = flat_data['ChiSqDOF'] <= float(self.chisqdof)
        else:
            mask = np.ones(len(flat_data['ChiSqDOF']), dtype=bool)

        if self.protonz:
            mask = np.logical_and(
                mask,
                np.logical_and(
                    flat_data['Proton_Z'] >= 50, flat_data['Proton_Z'] <= 80
                ),
            )

        mpl_style.use('thesis_analysis.thesis')
        fig, ax = plt.subplots()
        bins = 200
        ax.hist(
            flat_data['RF'][mask],
            bins=bins,
            range=(-20.0, 20.0),
            color=colors.blue,
            label=DATA_TYPE_TO_LATEX[self.data_type],
        )
        ax.set_xlabel(
            f'{BRANCH_NAME_TO_LATEX["RF"]} ({BRANCH_NAME_TO_LATEX_UNITS["RF"]})'
        )
        bin_width = 1.0 / bins
        ax.set_ylabel(
            f'Counts / {bin_width:.3f} {BRANCH_NAME_TO_LATEX_UNITS["RF"]}'
        )
        ax.legend()
        fig.savefig(output_path_rf)
        plt.close()
