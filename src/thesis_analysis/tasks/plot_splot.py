from pathlib import Path

import luigi
import matplotlib.pyplot as plt
import matplotlib.style as mpl_style
import numpy as np

from thesis_analysis import colors, root_io
from thesis_analysis.constants import (
    BRANCH_NAME_TO_LATEX,
    BRANCH_NAME_TO_LATEX_UNITS,
    get_branch,
)
from thesis_analysis.paths import Paths
from thesis_analysis.tasks.chisqdof import ChiSqDOF
from thesis_analysis.tasks.splot_fit import SPlotFit
from thesis_analysis.utils import SPlotFitResult, exp_pdf_single


class PlotSPlot(luigi.Task):
    data_type = luigi.Parameter()
    run_period = luigi.Parameter()
    chisqdof = luigi.FloatParameter()
    splot_method = luigi.Parameter()
    nsig = luigi.IntParameter()
    nbkg = luigi.IntParameter()

    def requires(self):
        return [
            ChiSqDOF(self.data_type, self.run_period, self.chisqdof),
            SPlotFit(
                self.data_type,
                self.run_period,
                self.chisqdof,
                self.splot_method,
                self.nsig,
                self.nbkg,
            ),
        ]

    def output(self):
        return [
            luigi.LocalTarget(
                Paths.plots
                / f'splot_fit_{self.data_type}_{self.run_period}_chisqdof_{self.chisqdof:.1f}_splot_{self.splot_method}_{self.nsig}s_{self.nbkg}b.png'
            ),
        ]

    def run(self):
        input_path = Path(self.input()[0][0].path)
        input_fit_path = Path(self.input()[1][0].path)
        output_plot_path = Path(self.output()[0].path)
        output_plot_path.parent.mkdir(parents=True, exist_ok=True)

        data_df = root_io.get_branches(
            input_path,
            [get_branch('RFL1'), get_branch('RFL2'), get_branch('Weight')],
        )

        fit_result = SPlotFitResult.load(input_fit_path)

        nbins = 100
        mpl_style.use('thesis_analysis.thesis')
        fig, ax = plt.subplots()
        counts_data, _, _ = ax.hist(
            data_df['RFL1'],
            weights=data_df['Weight'],
            bins=nbins,
            range=(0.0, 0.2),
            color=colors.blue,
            histtype='step',
            label='Data',
        )
        rfls = np.linspace(0.0, 0.2, 1000)
        ndata = np.sum(data_df['Weight'])
        z_total = np.sum(fit_result.yields)
        sig_lines = [
            exp_pdf_single(rfls, fit_result.ldas_sig[i])
            * fit_result.yields_sig[i]
            / z_total
            for i in range(fit_result.nsig)
        ]
        sig_tot = np.sum(np.array(sig_lines), axis=0)
        bkg_lines = [
            exp_pdf_single(rfls, fit_result.ldas_bkg[i])
            * fit_result.yields_bkg[i]
            / z_total
            for i in range(fit_result.nbkg)
        ]
        bkg_tot = np.sum(np.array(bkg_lines), axis=0)
        tot_line = sig_tot + bkg_tot
        bin_width = 0.2 / nbins
        for i, sig_line in enumerate(sig_lines):
            ax.plot(
                rfls,
                sig_line * ndata * bin_width,
                ls=':',
                color=colors.green,
                label='Signal Components' if i == 0 else None,
            )
        ax.plot(
            rfls,
            sig_tot * ndata * bin_width,
            color=colors.green,
            label='Total Signal',
        )
        for i, bkg_line in enumerate(bkg_lines):
            ax.plot(
                rfls,
                bkg_line * ndata * bin_width,
                ls=':',
                color=colors.red,
                label='Background Components' if i == 0 else None,
            )
        ax.plot(
            rfls,
            bkg_tot * ndata * bin_width,
            color=colors.red,
            label='Total Background',
        )
        ax.plot(
            rfls,
            tot_line * ndata * bin_width,
            color=colors.purple,
            label='Total Fit',
        )
        ax.set_xlabel(
            f'{BRANCH_NAME_TO_LATEX['RFL1']} ({BRANCH_NAME_TO_LATEX_UNITS['RFL1']})'
        )
        ax.set_ylabel(
            f'Counts / {bin_width:.4f} {BRANCH_NAME_TO_LATEX_UNITS["RFL1"]}'
        )
        ax.legend()
        ax.set_xlim(0.0, 0.2)
        ax.set_ylim(0.0, 1.1 * np.max(counts_data))
        fig.savefig(output_plot_path)
        plt.close()

        # fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))
        # counts_data, _, _ = ax[0, 0].hist(
        #     data_df['RFL1'],
        #     weights=data_df['Weight'],
        #     bins=100,
        #     range=(0.0, 0.2),
        #     color=colors.blue,
        #     histtype='step',
        # )
        # ax[0, 1].hist(
        #     data_df['RFL2'],
        #     weights=data_df['Weight'],
        #     bins=100,
        #     range=(0.0, 0.2),
        #     color=colors.blue,
        #     histtype='step',
        # )
        # counts_sigmc, _, _ = ax[1, 0].hist(
        #     sigmc_df['RFL1'],
        #     weights=sigmc_df['Weight'],
        #     bins=100,
        #     range=(0.0, 0.2),
        #     color=colors.green,
        #     histtype='step',
        # )
        # ax[1, 1].hist(
        #     sigmc_df['RFL2'],
        #     weights=sigmc_df['Weight'],
        #     bins=100,
        #     range=(0.0, 0.2),
        #     color=colors.green,
        #     histtype='step',
        # )
        # counts_bkgmc, _, _ = ax[2, 0].hist(
        #     bkgmc_df['RFL1'],
        #     weights=bkgmc_df['Weight'],
        #     bins=100,
        #     range=(0.0, 0.2),
        #     color=colors.red,
        #     histtype='step',
        # )
        # ax[2, 1].hist(
        #     bkgmc_df['RFL2'],
        #     weights=bkgmc_df['Weight'],
        #     bins=100,
        #     range=(0.0, 0.2),
        #     color=colors.red,
        #     histtype='step',
        # )
        # rfls = np.linspace(0.0, 0.2, 1000)
        # ndata = np.sum(data_df['Weight'])
        # nsigmc = np.sum(sigmc_df['Weight'])
        # nbkgmc = np.sum(bkgmc_df['Weight'])
        # z_total = np.sum(fit_result.yields)
        # sig_lines = [
        #     exp_pdf_single(rfls, fit_result.ldas_sig[i])
        #     * fit_result.yields_sig[i]
        #     / z_total
        #     for i in range(fit_result.nsig)
        # ]
        # sig_tot = np.sum(np.array(sig_lines), axis=0)
        # bkg_lines = [
        #     exp_pdf_single(rfls, fit_result.ldas_bkg[i])
        #     * fit_result.yields_bkg[i]
        #     / z_total
        #     for i in range(fit_result.nbkg)
        # ]
        # bkg_tot = np.sum(np.array(bkg_lines), axis=0)
        # tot_line = sig_tot + bkg_tot
        # bin_width = 0.2 / 100
        # for sig_line in sig_lines:
        #     ax[0, 0].plot(
        #         rfls,
        #         sig_line * ndata * bin_width,
        #         ls=':',
        #         color=colors.green,
        #     )
        # ax[0, 0].plot(rfls, sig_tot * ndata * bin_width, color=colors.green)
        # for bkg_line in bkg_lines:
        #     ax[0, 0].plot(
        #         rfls,
        #         bkg_line * ndata * bin_width,
        #         ls=':',
        #         color=colors.red,
        #     )
        # ax[0, 0].plot(rfls, bkg_tot * ndata * bin_width, color=colors.red)
        # ax[0, 0].plot(
        #     rfls,
        #     tot_line * ndata * bin_width,
        #     color=colors.purple,
        # )
        # for sig_line in sig_lines:
        #     ax[0, 1].plot(
        #         rfls,
        #         sig_line * ndata * bin_width,
        #         ls=':',
        #         color=colors.green,
        #     )
        # ax[0, 1].plot(
        #     rfls,
        #     sig_tot * ndata * bin_width,
        #     color=colors.green,
        # )
        # for bkg_line in bkg_lines:
        #     ax[0, 1].plot(
        #         rfls,
        #         bkg_line * ndata * bin_width,
        #         ls=':',
        #         color=colors.red,
        #     )
        # ax[0, 1].plot(rfls, bkg_tot * ndata * bin_width, color=colors.red)
        # ax[0, 1].plot(rfls, tot_line * ndata * bin_width, color=colors.purple)
        #
        # sigmc_lines = [
        #     exp_pdf_single(rfls, fit_result.ldas_sig[i])
        #     for i in range(fit_result.nsig)
        # ]
        # for sigmc_line in sigmc_lines:
        #     ax[1, 0].plot(
        #         rfls, sigmc_line * nsigmc * bin_width, color=colors.purple
        #     )
        #     ax[1, 1].plot(
        #         rfls, sigmc_line * nsigmc * bin_width, color=colors.purple
        #     )
        # bkgmc_lines = [
        #     exp_pdf_single(rfls, fit_result.ldas_bkg[i])
        #     for i in range(fit_result.nbkg)
        # ]
        # for bkgmc_line in bkgmc_lines:
        #     ax[2, 0].plot(
        #         rfls, bkgmc_line * nbkgmc * bin_width, color=colors.purple
        #     )
        #     ax[2, 1].plot(
        #         rfls, bkgmc_line * nbkgmc * bin_width, color=colors.purple
        #     )
        #
        # ax[0, 0].set_ylim(0, np.amax(counts_data) + 100)
        # ax[0, 1].set_ylim(0, np.amax(counts_data) + 100)
        # ax[1, 0].set_ylim(0, np.amax(counts_sigmc) + 100)
        # ax[1, 1].set_ylim(0, np.amax(counts_sigmc) + 100)
        # ax[2, 0].set_ylim(0, np.amax(counts_bkgmc) + 100)
        # ax[2, 1].set_ylim(0, np.amax(counts_bkgmc) + 100)
        #
        # fig.savefig(output_plot_path)
        # plt.close()
