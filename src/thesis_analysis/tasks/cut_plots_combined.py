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
from thesis_analysis.tasks.cut_plots import CutPlots
from thesis_analysis.tasks.data import GetData


@final
class CutPlotsCombined(luigi.Task):
    data_type = luigi.Parameter()
    chisqdof = luigi.OptionalFloatParameter(None)
    protonz = luigi.OptionalBoolParameter(False)

    @override
    def requires(self):
        data_type_sigmc = str(self.data_type).replace('data', 'accmc')
        data_type_bkgmc = str(self.data_type).replace('data', 'bkgmc')
        return (
            [GetData(self.data_type, run_period) for run_period in RUN_PERIODS]
            + [
                GetData(data_type_sigmc, run_period)
                for run_period in RUN_PERIODS
            ]
            + [
                GetData(data_type_bkgmc, run_period)
                for run_period in RUN_PERIODS
            ]
            + [
                CutPlots(
                    data_type=self.data_type,
                    chisqdof=self.chisqdof,
                    protonz=self.protonz,
                ),
                CutPlots(
                    data_type=data_type_sigmc,
                    chisqdof=self.chisqdof,
                    protonz=self.protonz,
                ),
                CutPlots(
                    data_type=data_type_bkgmc,
                    chisqdof=self.chisqdof,
                    protonz=self.protonz,
                ),
            ]
        )

    @override
    def output(self):
        return [
            luigi.LocalTarget(
                Paths.plots
                / f'{self.data_type}_combined_chisqdof{f"_chisqdof_{self.chisqdof:.1f}" if self.chisqdof is not None else ""}{"_protonz" if self.protonz else ""}.png'
            ),
            luigi.LocalTarget(
                Paths.plots
                / f'{self.data_type}_combined_chisqdof_youden_j{f"_chisqdof_{self.chisqdof:.1f}" if self.chisqdof is not None else ""}{"_protonz" if self.protonz else ""}.png'
            ),
            luigi.LocalTarget(
                Paths.plots
                / f'{self.data_type}_combined_chisqdof_roc{f"_chisqdof_{self.chisqdof:.1f}" if self.chisqdof is not None else ""}{"_protonz" if self.protonz else ""}.png'
            ),
            luigi.LocalTarget(
                Paths.plots
                / f'{self.data_type}_combined_protonz{f"_chisqdof_{self.chisqdof:.1f}" if self.chisqdof is not None else ""}{"_protonz" if self.protonz else ""}.png'
            ),
            luigi.LocalTarget(
                Paths.plots
                / f'{self.data_type}_combined_rfl{f"_chisqdof_{self.chisqdof:.1f}" if self.chisqdof is not None else ""}{"_protonz" if self.protonz else ""}.png'
            ),
            luigi.LocalTarget(
                Paths.plots
                / f'{self.data_type}_combined_mm2{f"_chisqdof_{self.chisqdof:.1f}" if self.chisqdof is not None else ""}{"_protonz" if self.protonz else ""}.png'
            ),
        ]

    @override
    def run(self):
        data_type_sigmc = str(self.data_type).replace('data', 'accmc')
        data_type_bkgmc = str(self.data_type).replace('data', 'bkgmc')
        logger.debug(f'Running cut plots for {self.data_type}')
        input_paths_data = [
            Path(self.input()[i][0].path) for i in range(len(RUN_PERIODS))
        ]
        input_paths_sigmc = [
            Path(self.input()[i + len(RUN_PERIODS)][0].path)
            for i in range(len(RUN_PERIODS))
        ]
        input_paths_bkgmc = [
            Path(self.input()[i + 2 * len(RUN_PERIODS)][0].path)
            for i in range(len(RUN_PERIODS))
        ]
        output_path_chisqdof = self.output()[0].path
        output_path_chisqdof_youden_j = self.output()[1].path
        output_path_chisqdof_roc = self.output()[2].path
        output_path_protonz = self.output()[3].path
        output_path_rfl = self.output()[4].path
        output_path_mm2 = self.output()[5].path

        branches = [
            get_branch('ChiSqDOF'),
            get_branch('Proton_Z'),
            get_branch('RFL1'),
            get_branch('MM2'),
        ]
        df_data = [
            root_io.get_branches(input_path, branches)
            for input_path in input_paths_data
        ]
        flat_data = {
            branch.name: np.concatenate(
                [df_data[i][branch.name] for i in range(len(RUN_PERIODS))]
            )
            for branch in branches
        }
        df_sigmc = [
            root_io.get_branches(input_path, branches)
            for input_path in input_paths_sigmc
        ]
        flat_sigmc = {
            branch.name: np.concatenate(
                [df_sigmc[i][branch.name] for i in range(len(RUN_PERIODS))]
            )
            for branch in branches
        }
        df_bkgmc = [
            root_io.get_branches(input_path, branches)
            for input_path in input_paths_bkgmc
        ]
        flat_bkgmc = {
            branch.name: np.concatenate(
                [df_bkgmc[i][branch.name] for i in range(len(RUN_PERIODS))]
            )
            for branch in branches
        }

        if self.chisqdof is not None:
            mask_data = flat_data['ChiSqDOF'] <= float(self.chisqdof)
            mask_sigmc = flat_sigmc['ChiSqDOF'] <= float(self.chisqdof)
            mask_bkgmc = flat_bkgmc['ChiSqDOF'] <= float(self.chisqdof)
        else:
            mask_data = np.ones(len(flat_data['ChiSqDOF']), dtype=bool)
            mask_sigmc = np.ones(len(flat_sigmc['ChiSqDOF']), dtype=bool)
            mask_bkgmc = np.ones(len(flat_bkgmc['ChiSqDOF']), dtype=bool)

        if self.protonz:
            mask_data = np.logical_and(
                mask_data,
                np.logical_and(
                    flat_data['Proton_Z'] >= 50, flat_data['Proton_Z'] <= 80
                ),
            )
            mask_sigmc = np.logical_and(
                mask_sigmc,
                np.logical_and(
                    flat_sigmc['Proton_Z'] >= 50, flat_sigmc['Proton_Z'] <= 80
                ),
            )
            mask_bkgmc = np.logical_and(
                mask_bkgmc,
                np.logical_and(
                    flat_bkgmc['Proton_Z'] >= 50, flat_bkgmc['Proton_Z'] <= 80
                ),
            )

        mpl_style.use('thesis_analysis.thesis')
        fig, ax = plt.subplots()
        bins = 200
        ax.hist(
            flat_data['ChiSqDOF'][mask_data],
            bins=bins,
            range=(0.0, 200.0 if 'original' in str(self.data_type) else 10.0),
            color=colors.blue,
            histtype='step',
            density=True,
            label=DATA_TYPE_TO_LATEX[self.data_type],
        )
        ax.hist(
            flat_sigmc['ChiSqDOF'][mask_sigmc],
            bins=bins,
            range=(0.0, 200.0 if 'original' in str(self.data_type) else 10.0),
            color=colors.green,
            histtype='step',
            density=True,
            label=DATA_TYPE_TO_LATEX[data_type_sigmc],
        )
        ax.hist(
            flat_bkgmc['ChiSqDOF'][mask_bkgmc],
            bins=bins,
            range=(0.0, 200.0 if 'original' in str(self.data_type) else 10.0),
            color=colors.red,
            histtype='step',
            density=True,
            label=DATA_TYPE_TO_LATEX[data_type_bkgmc],
        )
        ax.set_xlabel(BRANCH_NAME_TO_LATEX['ChiSqDOF'])
        bin_width = 1.0 / bins
        ax.set_ylabel(f'Normalized Counts / {bin_width:.2f}')
        ax.legend()
        fig.savefig(output_path_chisqdof)
        plt.close()

        # ChiSqDOF Youden J
        fig, ax = plt.subplots()
        bins = 200
        h_sig, edges = np.histogram(
            flat_sigmc['ChiSqDOF'][mask_sigmc],
            bins=bins,
            range=(0.0, 200.0 if 'original' in str(self.data_type) else 10.0),
            density=True,
        )
        h_bkg, _ = np.histogram(
            flat_bkgmc['ChiSqDOF'][mask_bkgmc],
            bins=bins,
            range=(0.0, 200.0 if 'original' in str(self.data_type) else 10.0),
            density=True,
        )
        sig_cumsum = np.cumsum(h_sig)
        bkg_cumsum = np.cumsum(h_bkg)
        sig_eff = sig_cumsum / sig_cumsum[-1]
        bkg_eff = bkg_cumsum / bkg_cumsum[-1]
        youden_j = sig_eff - bkg_eff
        max_j_index = np.argmax(youden_j)
        cut_values = edges[:-1]
        max_j = cut_values[max_j_index]
        ax.plot(
            cut_values,
            youden_j,
            color=colors.blue,
            label=r'J = $\epsilon_S - \epsilon_B$',
        )
        ax.axvline(max_j, color=colors.red, ls=':', label='Cut Value')
        ax.text(
            max_j,
            0.3,
            f'{max_j:0.2f}',
            color=colors.red,
            ha='right',
            va='top',
            rotation=90,
            transform=ax.get_xaxis_transform(),
        )
        ax.set_xlabel(BRANCH_NAME_TO_LATEX['ChiSqDOF'])
        bin_width = 1.0 / bins
        ax.set_ylabel('Youden J-Statistic')
        ax.legend()
        fig.savefig(output_path_chisqdof_youden_j)
        plt.close()

        # ChiSqDOF ROC
        fig, ax = plt.subplots()
        bins = 200
        h_sig, edges = np.histogram(
            flat_sigmc['ChiSqDOF'][mask_sigmc],
            bins=bins,
            range=(0.0, 200.0 if 'original' in str(self.data_type) else 10.0),
            density=True,
        )
        h_bkg, _ = np.histogram(
            flat_bkgmc['ChiSqDOF'][mask_bkgmc],
            bins=bins,
            range=(0.0, 200.0 if 'original' in str(self.data_type) else 10.0),
            density=True,
        )
        sig_cumsum = np.cumsum(h_sig)
        bkg_cumsum = np.cumsum(h_bkg)
        sig_eff = sig_cumsum / sig_cumsum[-1]
        bkg_eff = bkg_cumsum / bkg_cumsum[-1]
        cut_values = edges[:-1]
        ax.plot(
            bkg_eff,
            sig_eff,
            lw=1.5,
            color=colors.purple,
            label='ROC Curve',
        )
        ax.plot([0, 1], [0, 1], color=colors.black, lw=1.5, ls=':')
        ax.set_xlabel(r'False Positive Rate ($\epsilon_B$)')
        ax.set_ylabel(r'True Positive Rate ($\epsilon_S$)')
        ax.grid(True)
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('square')
        fig.savefig(output_path_chisqdof_roc)
        plt.close()

        # Proton Z
        fig, ax = plt.subplots()
        bins = 100
        ax.hist(
            flat_data['Proton_Z'][mask_data],
            bins=bins,
            range=(20.0, 120.0),
            color=colors.blue,
            histtype='step',
            density=True,
            label=DATA_TYPE_TO_LATEX[self.data_type],
        )
        ax.hist(
            flat_sigmc['Proton_Z'][mask_sigmc],
            bins=bins,
            range=(20.0, 120.0),
            color=colors.green,
            histtype='step',
            density=True,
            label=DATA_TYPE_TO_LATEX[data_type_sigmc],
        )
        ax.hist(
            flat_bkgmc['Proton_Z'][mask_bkgmc],
            bins=bins,
            range=(20.0, 120.0),
            color=colors.red,
            histtype='step',
            density=True,
            label=DATA_TYPE_TO_LATEX[data_type_bkgmc],
        )
        ax.set_xlabel(BRANCH_NAME_TO_LATEX['Proton_Z'])
        bin_width = 1.0 / bins
        ax.set_ylabel(f'Normalized Counts / {bin_width:.2f}')
        ax.legend()
        fig.savefig(output_path_protonz)
        plt.close()

        # RFL
        fig, ax = plt.subplots()
        bins = 100
        ax.hist(
            flat_data['RFL1'][mask_data],
            bins=bins,
            range=(0.0, 0.2),
            color=colors.blue,
            histtype='step',
            density=True,
            label=DATA_TYPE_TO_LATEX[self.data_type],
        )
        ax.hist(
            flat_sigmc['RFL1'][mask_sigmc],
            bins=bins,
            range=(0.0, 0.2),
            color=colors.green,
            histtype='step',
            density=True,
            label=DATA_TYPE_TO_LATEX[data_type_sigmc],
        )
        ax.hist(
            flat_bkgmc['RFL1'][mask_bkgmc],
            bins=bins,
            range=(0.0, 0.2),
            color=colors.red,
            histtype='step',
            density=True,
            label=DATA_TYPE_TO_LATEX[data_type_bkgmc],
        )
        ax.set_xlabel(BRANCH_NAME_TO_LATEX['RFL1'])
        bin_width = 1.0 / bins
        ax.set_ylabel(f'Normalized Counts / {bin_width:.2f}')
        ax.legend()
        fig.savefig(output_path_rfl)
        plt.close()

        # MM2
        fig, ax = plt.subplots()
        bins = 100
        ax.hist(
            flat_data['MM2'][mask_data],
            bins=bins,
            range=(-0.1, 0.1),
            color=colors.blue,
            histtype='step',
            density=True,
            label=DATA_TYPE_TO_LATEX[self.data_type],
        )
        ax.hist(
            flat_sigmc['MM2'][mask_sigmc],
            bins=bins,
            range=(-0.1, 0.1),
            color=colors.green,
            histtype='step',
            density=True,
            label=DATA_TYPE_TO_LATEX[data_type_sigmc],
        )
        ax.hist(
            flat_bkgmc['MM2'][mask_bkgmc],
            bins=bins,
            range=(-0.1, 0.1),
            color=colors.red,
            histtype='step',
            density=True,
            label=DATA_TYPE_TO_LATEX[data_type_bkgmc],
        )
        ax.set_xlabel('Missing Mass Squared')
        bin_width = 1.0 / bins
        ax.set_ylabel(f'Normalized Counts / {bin_width:.2f}')
        ax.set_yscale('log')
        ax.legend()
        fig.savefig(output_path_mm2)
        plt.close()
