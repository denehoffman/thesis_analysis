import pickle
from pathlib import Path

import luigi
import matplotlib.pyplot as plt
import matplotlib.style as mpl_style
from thesis_analysis import colors
from thesis_analysis.constants import NBINS, NUM_THREADS, RANGE
from thesis_analysis.paths import Paths
from thesis_analysis.pwa import UnbinnedFitResult, Waveset
from thesis_analysis.tasks.unbinned_fit import UnbinnedFit


class UnbinnedPlot(luigi.Task):
    chisqdof = luigi.FloatParameter()
    splot_method = luigi.Parameter()
    nsig = luigi.IntParameter()
    nbkg = luigi.IntParameter()
    niters = luigi.IntParameter(default=3, significant=False)
    guided = luigi.BoolParameter(default=False)
    averaged = luigi.BoolParameter(default=False)

    def requires(self):
        return [
            UnbinnedFit(
                self.chisqdof,
                self.splot_method,
                self.nsig,
                self.nbkg,
                self.niters,
                self.guided,
                self.averaged,
            ),
        ]

    def output(self):
        return [
            luigi.LocalTarget(
                Paths.plots
                / f'unbinned_fit_chisqdof_{self.chisqdof:.1f}_splot_{self.splot_method}_{self.nsig}s_{self.nbkg}b{"_guided" if self.guided else ""}{"_averaged" if self.averaged else ""}.png'
            ),
        ]

    def run(self):
        unbinned_fit_path = Path(str(self.input()[0][0]))

        output_plot_path = Path(str(self.output()[0].path))
        output_plot_path.parent.mkdir(parents=True, exist_ok=True)

        fit_result: UnbinnedFitResult = pickle.load(
            unbinned_fit_path.open('rb')
        )

        datasets = fit_result.get_datasets()
        nlls = fit_result.get_nlls(datasets)
        weights_fit = fit_result.project(nlls, threads=NUM_THREADS)
        weights_s0p = fit_result.project_with(
            Waveset.S0P, nlls, threads=NUM_THREADS
        )
        weights_s0n = fit_result.project_with(
            Waveset.S0N, nlls, threads=NUM_THREADS
        )
        weights_d2p = fit_result.project_with(
            Waveset.D2P, nlls, threads=NUM_THREADS
        )

        data_hist = fit_result.get_hist(
            datasets[0], bins=NBINS, range=RANGE, weights=None
        )
        fit_hist = fit_result.get_hist(
            datasets[1], bins=NBINS, range=RANGE, weights=weights_fit
        )
        s0p_hist = fit_result.get_hist(
            datasets[1], bins=NBINS, range=RANGE, weights=weights_s0p
        )
        s0n_hist = fit_result.get_hist(
            datasets[1], bins=NBINS, range=RANGE, weights=weights_s0n
        )
        d2p_hist = fit_result.get_hist(
            datasets[1], bins=NBINS, range=RANGE, weights=weights_d2p
        )

        mpl_style.use('thesis_analysis.thesis')
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
