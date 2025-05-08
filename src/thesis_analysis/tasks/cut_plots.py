from pathlib import Path
from typing import final, override

import luigi
import matplotlib.pyplot as plt
import matplotlib.style as mpl_style
import numpy as np

from thesis_analysis import colors, root_io
from thesis_analysis.constants import (
    BRANCH_NAME_TO_LATEX,
    DATA_TYPE_TO_LATEX,
    RUN_PERIODS,
    get_branch,
)
from thesis_analysis.logger import logger
from thesis_analysis.paths import Paths
from thesis_analysis.tasks.accid_and_pol import AccidentalsAndPolarization


@final
class CutPlots(luigi.Task):
    data_type = luigi.Parameter()
    chisqdof = luigi.OptionalFloatParameter(None)
    protonz = luigi.OptionalBoolParameter(False)

    @override
    def requires(self):
        return [
            AccidentalsAndPolarization(self.data_type, run_period)
            for run_period in RUN_PERIODS
        ]

    @override
    def output(self):
        return [
            luigi.LocalTarget(
                Paths.plots
                / f'{self.data_type}_chisqdof{f"_chisqdof_{self.chisqdof:.1f}" if self.chisqdof is not None else ""}{"_protonz" if self.protonz else ""}.png'
            ),
            luigi.LocalTarget(
                Paths.plots
                / f'{self.data_type}_protonz{f"_chisqdof_{self.chisqdof:.1f}" if self.chisqdof is not None else ""}{"_protonz" if self.protonz else ""}.png'
            ),
            luigi.LocalTarget(
                Paths.plots
                / f'{self.data_type}_rfl{f"_chisqdof_{self.chisqdof:.1f}" if self.chisqdof is not None else ""}{"_protonz" if self.protonz else ""}.png'
            ),
            luigi.LocalTarget(
                Paths.plots
                / f'{self.data_type}_mm2{f"_chisqdof_{self.chisqdof:.1f}" if self.chisqdof is not None else ""}{"_protonz" if self.protonz else ""}.png'
            ),
        ]

    @override
    def run(self):
        logger.debug(f'Running cut plots for {self.data_type}')
        input_paths = [
            Path(self.input()[i][0].path) for i in range(len(RUN_PERIODS))
        ]
        output_path_chisqdof = self.output()[0].path
        output_path_protonz = self.output()[1].path
        output_path_rfl = self.output()[2].path
        output_path_mm2 = self.output()[3].path

        branches = [
            # get_branch('M_Resonance'),
            get_branch('ChiSqDOF'),
            get_branch('Proton_Z'),
            get_branch('RFL1'),
            get_branch('MM2'),
        ]
        data = [
            root_io.get_branches(input_path, branches, root=False)
            for input_path in input_paths
        ]
        flat_data = {
            branch.name: np.concatenate(
                [data[i][branch.name] for i in range(len(RUN_PERIODS))]
            )
            for branch in branches
        }

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
            flat_data['ChiSqDOF'][mask],
            bins=bins,
            range=(0.0, 200.0 if 'original' in str(self.data_type) else 10.0),
            color=colors.blue,
            label=DATA_TYPE_TO_LATEX[self.data_type],
        )
        ax.set_xlabel(BRANCH_NAME_TO_LATEX['ChiSqDOF'])
        bin_width = 1.0 / bins
        ax.set_ylabel(f'Counts / {bin_width:.2f}')
        ax.legend()
        fig.savefig(output_path_chisqdof)
        plt.close()

        # Proton Z
        fig, ax = plt.subplots()
        bins = 100
        ax.hist(
            flat_data['Proton_Z'][mask],
            bins=bins,
            range=(20.0, 120.0),
            color=colors.blue,
            label=DATA_TYPE_TO_LATEX[self.data_type],
        )
        ax.set_xlabel(BRANCH_NAME_TO_LATEX['Proton_Z'])
        bin_width = 1.0 / bins
        ax.set_ylabel(f'Counts / {bin_width:.2f}')
        ax.legend()
        fig.savefig(output_path_protonz)
        plt.close()

        # RFL
        fig, ax = plt.subplots()
        bins = 100
        ax.hist(
            flat_data['RFL1'][mask],
            bins=bins,
            range=(0.0, 0.2),
            color=colors.blue,
            label=DATA_TYPE_TO_LATEX[self.data_type],
        )
        ax.set_xlabel(BRANCH_NAME_TO_LATEX['RFL1'])
        bin_width = 1.0 / bins
        ax.set_ylabel(f'Counts / {bin_width:.2f}')
        ax.set_ylabel(f'Counts / {bin_width:.2f}')
        ax.legend()
        fig.savefig(output_path_rfl)
        plt.close()

        # MM2
        fig, ax = plt.subplots()
        bins = 100
        ax.hist(
            flat_data['MM2'][mask],
            bins=bins,
            color=colors.blue,
            label=DATA_TYPE_TO_LATEX[self.data_type],
        )
        ax.set_xlabel('Missing Mass Squared')
        bin_width = 1.0 / bins
        ax.set_ylabel(f'Counts / {bin_width:.2f}')
        ax.set_yscale('log')
        ax.legend()
        fig.savefig(output_path_mm2)
        plt.close()
