import pickle
from pathlib import Path

import luigi
import matplotlib.pyplot as plt
import matplotlib.style as mpl_style
from thesis_analysis import colors
from thesis_analysis.constants import NBINS
from thesis_analysis.logger import logger
from thesis_analysis.paths import Paths
from thesis_analysis.pwa import BinnedFitResult, Waveset
from thesis_analysis.tasks.single_binned_fit import SingleBinnedFit


class PlotSingleBinned(luigi.Task):
    run_period = luigi.Parameter()
    chisqdof = luigi.FloatParameter()
    splot_method = luigi.Parameter()
    nsig = luigi.IntParameter()
    nbkg = luigi.IntParameter()

    def requires(self):
        return [
            SingleBinnedFit(
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
                / f'binned_fit_chisqdof_{self.chisqdof:.1f}_splot_{self.splot_method}_{self.nsig}s_{self.nbkg}b_{self.run_period}.png'
            ),
        ]

    def run(self):
        binned_fit_path = Path(str(self.input()[0][0]))

        output_plot_path = Path(str(self.output()[0].path))
        output_plot_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(output_plot_path)

        fit_result: BinnedFitResult = pickle.load(binned_fit_path.open('rb'))

        mpl_style.use('thesis_analysis.thesis')
        data_hist = fit_result.data_hist
        fit_hist = fit_result.fit_hist
        s0p_hist = fit_result.waveset_hists[Waveset.S0P]
        s0n_hist = fit_result.waveset_hists[Waveset.S0N]
        d2p_hist = fit_result.waveset_hists[Waveset.D2P]
        fig, ax = plt.subplots(ncols=2, sharey=True)
        ax[0].stairs(
            data_hist.counts,
            data_hist.bins,
            color=colors.black,
            label=f'Data ({str(self.run_period).upper()})',
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
        bin_width_mev = int(1000 / NBINS)
        ax[0].set_ylabel(f'Counts / {bin_width_mev} MeV/$c^2$')
        fig.savefig(output_plot_path)
        plt.close()
