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
    DATA_TYPE_TO_LATEX,
    RUN_PERIODS,
    get_branch,
)
from thesis_analysis.paths import Paths
from thesis_analysis.splot import (
    SPlotFitFailure,
    SPlotFitResultExp,
    exp_pdf_single,
)
from thesis_analysis.tasks.chisqdof import ChiSqDOF
from thesis_analysis.tasks.splot_fit import SPlotFit


@final
class SPlotPlot(luigi.Task):
    data_type = luigi.Parameter()
    chisqdof = luigi.FloatParameter()
    splot_method = luigi.Parameter()
    nsig = luigi.IntParameter()
    nbkg = luigi.IntParameter()

    @override
    def requires(self):
        return [
            ChiSqDOF(self.data_type, run_period, self.chisqdof)
            for run_period in RUN_PERIODS
        ] + [
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
                Paths.plots
                / f'splot_fit_{self.data_type}_chisqdof_{self.chisqdof:.1f}_splot_{self.splot_method}_{self.nsig}s_{self.nbkg}b.png'
            ),
        ]

    @override
    def run(self):
        input_paths = [
            Path(self.input()[i][0].path) for i in range(len(RUN_PERIODS))
        ]
        input_fit_path = Path(self.input()[-1][0].path)
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

        fit_result = pickle.load(input_fit_path.open('rb'))

        nbins = 100
        mpl_style.use('thesis_analysis.thesis')
        fig, ax = plt.subplots()
        if isinstance(fit_result, SPlotFitFailure):
            plt.text(0.35, 0.5, 'Fit Failed!', dict(size=30))
            fig.savefig(output_plot_path)
            plt.close()
            return

        ax.hist(
            data_df['RFL1'],
            weights=data_df['Weight'],
            bins=nbins,
            range=(0.0, 0.2),
            color=colors.blue,
            histtype='step',
            label=DATA_TYPE_TO_LATEX[str(self.data_type)],
        )
        rfls = np.linspace(0.0, 0.2, 1000)
        ndata = np.sum(data_df['Weight'])
        z_total = np.sum(fit_result.yields)
        if isinstance(fit_result, SPlotFitResultExp):
            sig_lines = [
                exp_pdf_single(rfls, fit_result.ldas_sig[i])
                * fit_result.yields_sig[i]
                / z_total
                for i in range(fit_result.nsig)
            ]
        else:
            sig_lines = [
                fit_result.pdfs1(rfls)[i] * fit_result.yields_sig[i] / z_total
                for i in range(fit_result.nsig)
            ]
        bkg_lines = [
            exp_pdf_single(rfls, fit_result.ldas_bkg[i])
            * fit_result.yields_bkg[i]
            / z_total
            for i in range(fit_result.nbkg)
        ]
        sig_tot = np.sum(np.array(sig_lines), axis=0)
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
        # ax.plot(
        #     rfls,
        #     sig_tot * ndata * bin_width,
        #     color=colors.green,
        #     label='Total Signal',
        # )
        for i, bkg_line in enumerate(bkg_lines):
            ax.plot(
                rfls,
                bkg_line * ndata * bin_width,
                ls=':',
                color=colors.red,
                label='Background Components' if i == 0 else None,
            )
        # ax.plot(
        #     rfls,
        #     bkg_tot * ndata * bin_width,
        #     color=colors.red,
        #     label='Total Background',
        # )
        ax.plot(
            rfls,
            tot_line * ndata * bin_width,
            color=colors.purple,
            lw=1,
            label='Total Fit',
        )
        ax.set_xlabel(
            f'{BRANCH_NAME_TO_LATEX["RFL1"]} ({BRANCH_NAME_TO_LATEX_UNITS["RFL1"]})'
        )
        ax.set_ylabel(
            f'Counts / {bin_width:.3f} {BRANCH_NAME_TO_LATEX_UNITS["RFL1"]}'
        )
        ax.set_yscale('log')
        ax.legend()
        fig.savefig(output_plot_path)
        plt.close()
