import pickle
from pathlib import Path

import luigi
import matplotlib.pyplot as plt
import matplotlib.style as mpl_style

from thesis_analysis import colors
from thesis_analysis.constants import NBINS, NUM_THREADS, RANGE
from thesis_analysis.logger import logger
from thesis_analysis.paths import Paths
from thesis_analysis.pwa import (
    BinnedFitResult,
    UnbinnedFitResult,
    Waveset,
)
from thesis_analysis.tasks.fit_binned import FitBinned
from thesis_analysis.tasks.fit_unbinned_guided import FitUnbinnedGuided


class PlotUnbinnedGuided(luigi.Task):
    chisqdof = luigi.FloatParameter()
    splot_method = luigi.Parameter()
    nsig = luigi.IntParameter()
    nbkg = luigi.IntParameter()

    resources = {'fit': 1}

    def requires(self):
        return [
            FitUnbinnedGuided(
                self.chisqdof, self.splot_method, self.nsig, self.nbkg
            ),
            FitBinned(self.chisqdof, self.splot_method, self.nsig, self.nbkg),
        ]

    def output(self):
        return [
            luigi.LocalTarget(
                Paths.plots
                / f'unbinned_fit_guided_chisqdof_{self.chisqdof:.1f}_splot_{self.splot_method}_{self.nsig}s_{self.nbkg}b.png'
            ),
        ]

    def run(self):
        guided_unbinned_fit_path = Path(str(self.input()[0][0].path))
        guided_unbinned_result: UnbinnedFitResult = pickle.load(
            guided_unbinned_fit_path.open('rb')
        )

        binned_fit_path = Path(str(self.input()[1][0].path))
        logger.debug(f'Binned fit: {binned_fit_path}')
        binned_result: BinnedFitResult = pickle.load(binned_fit_path.open('rb'))

        datasets = guided_unbinned_result.get_datasets()
        nlls = guided_unbinned_result.get_nlls(datasets)
        weights_fit = guided_unbinned_result.project(nlls, threads=NUM_THREADS)
        weights_p = guided_unbinned_result.project_with(
            Waveset.P, nlls, threads=NUM_THREADS
        )
        weights_n = guided_unbinned_result.project_with(
            Waveset.N, nlls, threads=NUM_THREADS
        )
        weights_s0p = guided_unbinned_result.project_with(
            Waveset.S0P, nlls, threads=NUM_THREADS
        )
        weights_d2p = guided_unbinned_result.project_with(
            Waveset.D2P, nlls, threads=NUM_THREADS
        )

        fit_hist = guided_unbinned_result.get_hist(
            datasets[1], bins=NBINS, range=RANGE, weights=weights_fit
        )
        p_hist = guided_unbinned_result.get_hist(
            datasets[1], bins=NBINS, range=RANGE, weights=weights_p
        )
        n_hist = guided_unbinned_result.get_hist(
            datasets[1], bins=NBINS, range=RANGE, weights=weights_n
        )
        s0p_hist = guided_unbinned_result.get_hist(
            datasets[1], bins=NBINS, range=RANGE, weights=weights_s0p
        )
        d2p_hist = guided_unbinned_result.get_hist(
            datasets[1], bins=NBINS, range=RANGE, weights=weights_d2p
        )

        data_hist = binned_result.data_hist
        p_hist_binned = binned_result.waveset_hists[Waveset.P]
        n_hist_binned = binned_result.waveset_hists[Waveset.N]
        s0p_hist_binned = binned_result.waveset_hists[Waveset.S0P]
        d2p_hist_binned = binned_result.waveset_hists[Waveset.D2P]

        output_plot_path = Path(str(self.output()[0].path))
        output_plot_path.parent.mkdir(parents=True, exist_ok=True)

        mpl_style.use('thesis_analysis.thesis')
        fig, ax = plt.subplots(
            ncols=2, nrows=2, sharex=True, sharey=True, figsize=(5.2, 6.4)
        )
        for axis in ax.flatten():
            axis.stairs(
                data_hist.counts,
                data_hist.bins,
                color=colors.black,
                label='Data',
            )
            axis.stairs(
                fit_hist.counts,
                fit_hist.bins,
                color=colors.black,
                label='Fit',
                fill=True,
                alpha=0.2,
            )
        ax[0][0].stairs(
            p_hist.counts,
            p_hist.bins,
            color=colors.red,
            label=r'$\varepsilon = +$ (guided)',
            fill=True,
            alpha=0.2,
        )
        ax[0][0].stairs(
            p_hist_binned.counts,
            p_hist_binned.bins,
            color=colors.red,
            label=r'$\varepsilon = +$ (fit)',
        )
        ax[0][1].stairs(
            n_hist.counts,
            n_hist.bins,
            color=colors.blue,
            label=r'$\varepsilon = -$ (guided)',
            fill=True,
            alpha=0.2,
        )
        ax[0][1].stairs(
            n_hist_binned.counts,
            n_hist_binned.bins,
            color=colors.blue,
            label=r'$\varepsilon = -$ (fit)',
        )
        ax[1][0].stairs(
            s0p_hist.counts,
            s0p_hist.bins,
            color=colors.red,
            label='$S_0^+$ (guided)',
            fill=True,
            alpha=0.2,
        )
        ax[1][0].stairs(
            s0p_hist_binned.counts,
            s0p_hist_binned.bins,
            color=colors.red,
            label='$S_0^+$ (fit)',
        )
        ax[1][0].stairs(
            n_hist.counts,
            n_hist.bins,
            color=colors.blue,
            label='$S_0^-$ (guided)',
            fill=True,
            alpha=0.2,
        )
        ax[1][0].stairs(
            n_hist_binned.counts,
            n_hist_binned.bins,
            color=colors.blue,
            label='$S_0^-$ (fit)',
        )
        ax[1][1].stairs(
            d2p_hist.counts,
            d2p_hist.bins,
            color=colors.red,
            label='$D_2^+$ (guided)',
            fill=True,
            alpha=0.2,
        )
        ax[1][1].stairs(
            d2p_hist_binned.counts,
            d2p_hist_binned.bins,
            color=colors.red,
            label='$D_2^+$ (fit)',
        )
        ax[0][0].legend()
        ax[0][1].legend()
        ax[1][0].legend()
        ax[1][1].legend()
        ax[1][0].set_xlabel('Invariant Mass of $K_S^0K_S^0$ (GeV/$c^2$)')
        ax[1][1].set_xlabel('Invariant Mass of $K_S^0K_S^0$ (GeV/$c^2$)')
        bin_width_mev = int(1000 / NBINS)
        ax[0][0].set_ylabel(f'Counts / {bin_width_mev} MeV/$c^2$')
        ax[1][0].set_ylabel(f'Counts / {bin_width_mev} MeV/$c^2$')
        fig.savefig(output_plot_path)
        plt.close()
