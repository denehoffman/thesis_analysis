import pickle
from pathlib import Path

import luigi
import matplotlib.pyplot as plt
import matplotlib.style as mpl_style
from thesis_analysis import colors
from thesis_analysis.constants import RUN_PERIODS
from thesis_analysis.paths import Paths
from thesis_analysis.tasks.splot import SPlot
from thesis_analysis.utils import fit_unbinned


class FitUnbinned(luigi.Task):
    chisqdof = luigi.FloatParameter()
    n_sig = luigi.IntParameter()
    n_bkg = luigi.IntParameter()

    resources = {'exclusive_task': 1}

    def requires(self):
        return [
            SPlot('data', rp, self.chisqdof, self.n_sig, self.n_bkg)
            for rp in RUN_PERIODS
        ] + [
            SPlot('accmc', rp, self.chisqdof, self.n_sig, self.n_bkg)
            for rp in RUN_PERIODS
        ]

    def output(self):
        return [
            luigi.LocalTarget(
                Paths.fits
                / Path(f'chisqdof_{self.chisqdof:.1f}')
                / Path(f'splot_{self.n_sig}s_{self.n_bkg}b')
                / 'unbinned_fit.pkl'
            ),
            luigi.LocalTarget(
                Paths.plots
                / f'unbinned_fit_chisqdof_{self.chisqdof:.1f}_splot_{self.n_sig}s_{self.n_bkg}b.png'
            ),
        ]

    def run(self):
        data_s17_path = self.input()[0][0]
        data_s18_path = self.input()[1][0]
        data_f18_path = self.input()[2][0]
        data_s20_path = self.input()[3][0]
        accmc_s17_path = self.input()[4][0]
        accmc_s18_path = self.input()[5][0]
        accmc_f18_path = self.input()[6][0]
        accmc_s20_path = self.input()[7][0]

        output_fit_path = Path(self.output()[0].path)
        output_fit_path.parent.mkdir(parents=True, exist_ok=True)
        output_plot_path = Path(self.output()[1].path)
        output_plot_path.parent.mkdir(parents=True, exist_ok=True)

        fit_result = fit_unbinned(
            data_s17_path,
            accmc_s17_path,
            data_s18_path,
            accmc_s18_path,
            data_f18_path,
            accmc_f18_path,
            data_s20_path,
            accmc_s20_path,
        )
        pickle.dump(fit_result, output_fit_path.open('wb'))

        mpl_style.use('thesis_analysis.thesis')
        data_hist = fit_result.get_data_hist(50, (1.0, 2.0))
        fit_hist = fit_result.get_fit_hist(50, (1.0, 2.0))
        s0p_hist = fit_result.get_s0p_hist(50, (1.0, 2.0))
        s0n_hist = fit_result.get_s0n_hist(50, (1.0, 2.0))
        d2p_hist = fit_result.get_d2p_hist(50, (1.0, 2.0))
        fig, ax = plt.subplots(ncols=2, sharey=True)
        ax[0].stairs(
            data_hist.counts,
            data_hist.bins,
            color=colors.black,
            label='Data',
        )
        ax[0].stairs(
            fit_hist.counts,
            fit_hist.bins,
            color=colors.black,
            label='Fit',
            fill=True,
            alpha=0.2,
        )
        ax[1].stairs(
            data_hist.counts,
            data_hist.bins,
            histtype='step',
            color=colors.black,
            label='Data',
        )
        ax[1].stairs(
            fit_hist.counts,
            fit_hist.bins,
            color=colors.black,
            label='Fit',
            fill=True,
            alpha=0.2,
        )
        ax[0].stairs(
            s0p_hist.counts,
            s0p_hist.bins,
            color=colors.red,
            label='$S_0^+$',
            fill=True,
            alpha=0.2,
        )
        ax[0].stairs(
            s0n_hist.counts,
            s0n_hist.bins,
            color=colors.blue,
            label='$S_0^-$',
            fill=True,
            alpha=0.2,
        )
        ax[1].stairs(
            d2p_hist.counts,
            d2p_hist.bins,
            color=colors.red,
            label='$D_2^+$',
            fill=True,
            alpha=0.2,
        )
        ax[0].legend()
        ax[1].legend()
        ax[0].set_xlabel('Invariant Mass of $K_S^0K_S^0$ (GeV/$c^2$)')
        ax[1].set_xlabel('Invariant Mass of $K_S^0K_S^0$ (GeV/$c^2$)')
        bin_width_mev = int(1000 / 50)
        ax[0].set_ylabel(f'Counts / {bin_width_mev} MeV/$c^2$')
        fig.savefig(output_plot_path)
        plt.close()
