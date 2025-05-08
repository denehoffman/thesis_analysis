from pathlib import Path
from typing import Any, final, override

import laddu as ld
import luigi
import matplotlib.pyplot as plt
import matplotlib.style as mpl_style
import numpy as np

from thesis_analysis import colors, root_io
from thesis_analysis.constants import (
    BRANCH_NAME_TO_LATEX,
    DATA_TYPE_TO_LATEX,
    RUN_PERIODS,
    RootBranchDict,
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
    ksb_costheta = luigi.OptionalFloatParameter(None)

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
                / f'{self.data_type}_combined_chisqdof{f"_chisqdof_{self.chisqdof:.1f}" if self.chisqdof is not None else ""}{"_protonz" if self.protonz else ""}{"_ksb_costheta_{self.ksb_costheta:.2f}" if self.ksb_costheta is not None else ""}.png'
            ),
            luigi.LocalTarget(
                Paths.plots
                / f'{self.data_type}_combined_chisqdof_youden_j{f"_chisqdof_{self.chisqdof:.1f}" if self.chisqdof is not None else ""}{"_protonz" if self.protonz else ""}{"_ksb_costheta_{self.ksb_costheta:.2f}" if self.ksb_costheta is not None else ""}.png'
            ),
            luigi.LocalTarget(
                Paths.plots
                / f'{self.data_type}_combined_chisqdof_roc{f"_chisqdof_{self.chisqdof:.1f}" if self.chisqdof is not None else ""}{"_protonz" if self.protonz else ""}{"_ksb_costheta_{self.ksb_costheta:.2f}" if self.ksb_costheta is not None else ""}.png'
            ),
            luigi.LocalTarget(
                Paths.plots
                / f'{self.data_type}_combined_protonz{f"_chisqdof_{self.chisqdof:.1f}" if self.chisqdof is not None else ""}{"_protonz" if self.protonz else ""}{"_ksb_costheta_{self.ksb_costheta:.2f}" if self.ksb_costheta is not None else ""}.png'
            ),
            luigi.LocalTarget(
                Paths.plots
                / f'{self.data_type}_combined_rfl{f"_chisqdof_{self.chisqdof:.1f}" if self.chisqdof is not None else ""}{"_protonz" if self.protonz else ""}{"_ksb_costheta_{self.ksb_costheta:.2f}" if self.ksb_costheta is not None else ""}.png'
            ),
            luigi.LocalTarget(
                Paths.plots
                / f'{self.data_type}_combined_mm2{f"_chisqdof_{self.chisqdof:.1f}" if self.chisqdof is not None else ""}{"_protonz" if self.protonz else ""}{"_ksb_costheta_{self.ksb_costheta:.2f}" if self.ksb_costheta is not None else ""}.png'
            ),
            luigi.LocalTarget(
                Paths.plots
                / f'{self.data_type}_combined_ksbp{f"_chisqdof_{self.chisqdof:.1f}" if self.chisqdof is not None else ""}{"_protonz" if self.protonz else ""}{"_ksb_costheta_{self.ksb_costheta:.2f}" if self.ksb_costheta is not None else ""}.png'
            ),
            luigi.LocalTarget(
                Paths.plots
                / f'{self.data_type}_combined_ksks{f"_chisqdof_{self.chisqdof:.1f}" if self.chisqdof is not None else ""}{"_protonz" if self.protonz else ""}{"_ksb_costheta_{self.ksb_costheta:.2f}" if self.ksb_costheta is not None else ""}.png'
            ),
            luigi.LocalTarget(
                Paths.plots
                / f'{self.data_type}_combined_ksb_costheta{f"_chisqdof_{self.chisqdof:.1f}" if self.chisqdof is not None else ""}{"_protonz" if self.protonz else ""}{"_ksb_costheta_{self.ksb_costheta:.2f}" if self.ksb_costheta is not None else ""}.png'
            ),
            luigi.LocalTarget(
                Paths.plots
                / f'{self.data_type}_combined_ksb_costheta_v_ksbp{f"_chisqdof_{self.chisqdof:.1f}" if self.chisqdof is not None else ""}{"_protonz" if self.protonz else ""}{"_ksb_costheta_{self.ksb_costheta:.2f}" if self.ksb_costheta is not None else ""}.png'
            ),
            luigi.LocalTarget(
                Paths.plots
                / f'{self.data_type}_combined_ksb_costheta_v_ksks{f"_chisqdof_{self.chisqdof:.1f}" if self.chisqdof is not None else ""}{"_protonz" if self.protonz else ""}{"_ksb_costheta_{self.ksb_costheta:.2f}" if self.ksb_costheta is not None else ""}.png'
            ),
            luigi.LocalTarget(
                Paths.plots
                / f'{self.data_type}_combined_me{f"_chisqdof_{self.chisqdof:.1f}" if self.chisqdof is not None else ""}{"_protonz" if self.protonz else ""}{"_ksb_costheta_{self.ksb_costheta:.2f}" if self.ksb_costheta is not None else ""}.png'
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
        output_path_ksbp = self.output()[6].path
        output_path_ksks = self.output()[7].path
        output_path_ksb_costheta = self.output()[8].path
        output_path_ksbp_costheta_v_ksbp = self.output()[9].path
        output_path_ksbp_costheta_v_ksks = self.output()[10].path
        output_path_me = self.output()[11].path

        branches = [
            get_branch('E_FinalState', dim=3),
            get_branch('Px_FinalState', dim=3),
            get_branch('Py_FinalState', dim=3),
            get_branch('Pz_FinalState', dim=3),
            get_branch('ChiSqDOF'),
            get_branch('Proton_Z'),
            get_branch('RFL1'),
            get_branch('MM2'),
            get_branch('ME'),
        ]
        flat_data = root_io.concatenate_branches(
            input_paths_data, branches, root=False
        )
        flat_data = get_fs_branches(flat_data)

        flat_sigmc = root_io.concatenate_branches(
            input_paths_sigmc, branches, root=False
        )
        flat_sigmc = get_fs_branches(flat_sigmc)
        flat_bkgmc = root_io.concatenate_branches(
            input_paths_bkgmc, branches, root=False
        )
        flat_bkgmc = get_fs_branches(flat_bkgmc)

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
        if self.ksb_costheta is not None:
            logger.debug(f'KsB Cut is not None! ({self.ksb_costheta})')
            mask_data = flat_data['KShortB_CosTheta'] <= float(
                self.ksb_costheta
            )
            mask_sigmc = flat_sigmc['KShortB_CosTheta'] <= float(
                self.ksb_costheta
            )
            mask_bkgmc = flat_bkgmc['KShortB_CosTheta'] <= float(
                self.ksb_costheta
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

        # KShort_B + Proton
        fig, ax = plt.subplots()
        bins = 100
        ax.hist(
            flat_data['M_KShortB_Proton'][mask_data],
            bins=bins,
            range=(1.4, 3.7),
            color=colors.blue,
            histtype='step',
            density=True,
            label=DATA_TYPE_TO_LATEX[self.data_type],
        )
        ax.hist(
            flat_sigmc['M_KShortB_Proton'][mask_sigmc],
            bins=bins,
            range=(1.4, 3.7),
            color=colors.green,
            histtype='step',
            density=True,
            label=DATA_TYPE_TO_LATEX[data_type_sigmc],
        )
        ax.hist(
            flat_bkgmc['M_KShortB_Proton'][mask_bkgmc],
            bins=bins,
            range=(1.4, 3.7),
            color=colors.red,
            histtype='step',
            density=True,
            label=DATA_TYPE_TO_LATEX[data_type_bkgmc],
        )
        ax.set_xlabel('Invariant Mass of $K_{S,B}^0 p$')
        bin_width = 1.0 / bins
        ax.set_ylabel(f'Normalized Counts / {bin_width:.2f}')
        # ax.set_yscale('log')
        ax.legend()
        fig.savefig(output_path_ksbp)
        plt.close()

        # KShort1 + KShort2
        fig, ax = plt.subplots()
        bins = 100
        ax.hist(
            flat_data['M_KShort1_KShort2'][mask_data],
            bins=bins,
            range=(0.9, 3.0),
            color=colors.blue,
            histtype='step',
            density=True,
            label=DATA_TYPE_TO_LATEX[self.data_type],
        )
        ax.hist(
            flat_sigmc['M_KShort1_KShort2'][mask_sigmc],
            bins=bins,
            range=(0.9, 3.0),
            color=colors.green,
            histtype='step',
            density=True,
            label=DATA_TYPE_TO_LATEX[data_type_sigmc],
        )
        ax.hist(
            flat_bkgmc['M_KShort1_KShort2'][mask_bkgmc],
            bins=bins,
            range=(0.9, 3.0),
            color=colors.red,
            histtype='step',
            density=True,
            label=DATA_TYPE_TO_LATEX[data_type_bkgmc],
        )
        ax.set_xlabel('Invariant Mass of $K_{S,1}^0 K_{S,2}^0$')
        bin_width = 1.0 / bins
        ax.set_ylabel(f'Normalized Counts / {bin_width:.2f}')
        # ax.set_yscale('log')
        ax.legend()
        fig.savefig(output_path_ksks)
        plt.close()

        # KShort_B CosTheta
        fig, ax = plt.subplots()
        bins = 100
        ax.hist(
            flat_data['KShortB_CosTheta'][mask_data],
            bins=bins,
            range=(-1, 1),
            color=colors.blue,
            histtype='step',
            density=True,
            label=DATA_TYPE_TO_LATEX[self.data_type],
        )
        ax.hist(
            flat_sigmc['KShortB_CosTheta'][mask_sigmc],
            bins=bins,
            range=(-1, 1),
            color=colors.green,
            histtype='step',
            density=True,
            label=DATA_TYPE_TO_LATEX[data_type_sigmc],
        )
        ax.hist(
            flat_bkgmc['KShortB_CosTheta'][mask_bkgmc],
            bins=bins,
            range=(-1, 1),
            color=colors.red,
            histtype='step',
            density=True,
            label=DATA_TYPE_TO_LATEX[data_type_bkgmc],
        )
        ax.set_xlabel(r'$\cos\theta$ of $K_{S,B}^0$')
        bin_width = 1.0 / bins
        ax.set_ylabel(f'Normalized Counts / {bin_width:.2f}')
        ax.legend()
        fig.savefig(output_path_ksb_costheta)
        plt.close()

        # KShort_B + Proton v KShort_B CosTheta
        fig, ax = plt.subplots()
        bins = 100
        ax.hist2d(
            flat_data['M_KShortB_Proton'][mask_data],
            flat_data['KShortB_CosTheta'][mask_data],
            bins=bins,
            range=[(1.4, 3.7), (-1, 1)],
        )
        # ax.hist(
        #     flat_sigmc['M_KShortB_Proton'][mask_sigmc],
        #     bins=bins,
        #     range=(1.4, 3.7),
        #     color=colors.green,
        #     histtype='step',
        #     density=True,
        #     label=DATA_TYPE_TO_LATEX[data_type_sigmc],
        # )
        # ax.hist(
        #     flat_bkgmc['M_KShortB_Proton'][mask_bkgmc],
        #     bins=bins,
        #     range=(1.4, 3.7),
        #     color=colors.red,
        #     histtype='step',
        #     density=True,
        #     label=DATA_TYPE_TO_LATEX[data_type_bkgmc],
        # )
        ax.set_xlabel('Invariant Mass of $K_{S,B}^0 p$')
        ax.set_ylabel(r'$\cos\theta$ of $K_{S,B}^0$')
        fig.savefig(output_path_ksbp_costheta_v_ksbp)
        plt.close()

        # KShort1 + KShort2 v KShort_B CosTheta
        fig, ax = plt.subplots()
        bins = 100
        ax.hist2d(
            flat_data['M_KShort1_KShort2'][mask_data],
            flat_data['KShortB_CosTheta'][mask_data],
            bins=bins,
            range=[(0.9, 3.0), (-1, 1)],
        )
        # ax.hist(
        #     flat_sigmc['M_KShortB_Proton'][mask_sigmc],
        #     bins=bins,
        #     range=(1.4, 3.7),
        #     color=colors.green,
        #     histtype='step',
        #     density=True,
        #     label=DATA_TYPE_TO_LATEX[data_type_sigmc],
        # )
        # ax.hist(
        #     flat_bkgmc['M_KShortB_Proton'][mask_bkgmc],
        #     bins=bins,
        #     range=(1.4, 3.7),
        #     color=colors.red,
        #     histtype='step',
        #     density=True,
        #     label=DATA_TYPE_TO_LATEX[data_type_bkgmc],
        # )
        ax.set_xlabel('Invariant Mass of $K_{S,B}^0 p$')
        ax.set_ylabel(r'$\cos\theta$ of $K_{S,B}^0$')
        fig.savefig(output_path_ksbp_costheta_v_ksks)
        plt.close()

        # ME (missing energy)
        fig, ax = plt.subplots()
        bins = 100
        ax.hist(
            flat_data['ME'][mask_data],
            bins=bins,
            range=(-0.1, 0.1),
            color=colors.blue,
            histtype='step',
            density=True,
            label=DATA_TYPE_TO_LATEX[self.data_type],
        )
        ax.hist(
            flat_sigmc['ME'][mask_sigmc],
            bins=bins,
            range=(-0.1, 0.1),
            color=colors.green,
            histtype='step',
            density=True,
            label=DATA_TYPE_TO_LATEX[data_type_sigmc],
        )
        ax.hist(
            flat_bkgmc['ME'][mask_bkgmc],
            bins=bins,
            range=(-0.1, 0.1),
            color=colors.red,
            histtype='step',
            density=True,
            label=DATA_TYPE_TO_LATEX[data_type_bkgmc],
        )
        ax.set_xlabel('Missing Energy')
        bin_width = 1.0 / bins
        ax.set_ylabel(f'Normalized Counts / {bin_width:.2f}')
        ax.set_yscale('log')
        ax.legend()
        fig.savefig(output_path_me)
        plt.close()


def get_pion_branches(
    flat_data: dict[str, np.typing.NDArray[Any]],
) -> dict[str, np.typing.NDArray[Any]]:
    flat_data['PiPlus1_P4'] = np.array(
        [
            ld.Vector4(px, py, pz, e)
            for px, py, pz, e in zip(
                flat_data['Px_PiPlus1'],
                flat_data['Py_PiPlus1'],
                flat_data['Pz_PiPlus1'],
                flat_data['E_PiPlus1'],
            )
        ]
    )
    flat_data['PiMinus1_P4'] = np.array(
        [
            ld.Vector4(px, py, pz, e)
            for px, py, pz, e in zip(
                flat_data['Px_PiMinus1'],
                flat_data['Py_PiMinus1'],
                flat_data['Pz_PiMinus1'],
                flat_data['E_PiMinus1'],
            )
        ]
    )
    flat_data['PiPlus2_P4'] = np.array(
        [
            ld.Vector4(px, py, pz, e)
            for px, py, pz, e in zip(
                flat_data['Px_PiPlus2'],
                flat_data['Py_PiPlus2'],
                flat_data['Pz_PiPlus2'],
                flat_data['E_PiPlus2'],
            )
        ]
    )
    flat_data['PiMinus2_P4'] = np.array(
        [
            ld.Vector4(px, py, pz, e)
            for px, py, pz, e in zip(
                flat_data['Px_PiMinus2'],
                flat_data['Py_PiMinus2'],
                flat_data['Pz_PiMinus2'],
                flat_data['E_PiMinus2'],
            )
        ]
    )
    return flat_data


def get_fs_branches(
    flat_data: RootBranchDict,
) -> dict[str, np.typing.NDArray[Any]]:
    flat_data: dict[str, np.typing.NDArray[Any]] = dict(flat_data)  # pyright:ignore[reportAssignmentType]
    flat_data['Proton_P4'] = np.array(
        [
            ld.Vector4(px[0], py[0], pz[0], e[0])
            for px, py, pz, e in zip(
                flat_data['Px_FinalState'],
                flat_data['Py_FinalState'],
                flat_data['Pz_FinalState'],
                flat_data['E_FinalState'],
            )
        ]
    )
    flat_data['KShort1_P4'] = np.array(
        [
            ld.Vector4(px[1], py[1], pz[1], e[1])
            for px, py, pz, e in zip(
                flat_data['Px_FinalState'],
                flat_data['Py_FinalState'],
                flat_data['Pz_FinalState'],
                flat_data['E_FinalState'],
            )
        ]
    )
    flat_data['KShort2_P4'] = np.array(
        [
            ld.Vector4(px[2], py[2], pz[2], e[2])
            for px, py, pz, e in zip(
                flat_data['Px_FinalState'],
                flat_data['Py_FinalState'],
                flat_data['Pz_FinalState'],
                flat_data['E_FinalState'],
            )
        ]
    )
    flat_data['KShortF_P4'] = np.array(
        [
            ks1 if ks1.vec3.costheta > ks2.vec3.costheta else ks2
            for ks1, ks2 in zip(
                flat_data['KShort1_P4'], flat_data['KShort2_P4']
            )
        ]
    )
    flat_data['KShortB_P4'] = np.array(
        [
            ks2 if ks1.vec3.costheta > ks2.vec3.costheta else ks1
            for ks1, ks2 in zip(
                flat_data['KShort1_P4'], flat_data['KShort2_P4']
            )
        ]
    )
    com_frame = [
        ks1 + ks2 + p
        for ks1, ks2, p in zip(
            flat_data['KShort1_P4'],
            flat_data['KShort2_P4'],
            flat_data['Proton_P4'],
        )
    ]
    flat_data['KShortB_P4_COM'] = np.array(
        [
            ksb.boost(-com.beta)
            for ksb, com in zip(flat_data['KShortB_P4'], com_frame)
        ]
    )
    flat_data['KShortB_CosTheta'] = np.array(
        [ksb.vec3.costheta for ksb in flat_data['KShortB_P4_COM']]
    )
    flat_data['M_KShortB_Proton'] = np.array(
        [
            (ksb + proton).m
            for ksb, proton in zip(
                flat_data['KShortB_P4'], flat_data['Proton_P4']
            )
        ]
    )
    flat_data['M_KShort1_KShort2'] = np.array(
        [
            (ks1 + ks2).m
            for ks1, ks2 in zip(
                flat_data['KShort1_P4'], flat_data['KShort2_P4']
            )
        ]
    )
    return flat_data
