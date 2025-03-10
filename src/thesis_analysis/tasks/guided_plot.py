import pickle
from pathlib import Path
from typing import final

import luigi
import matplotlib.pyplot as plt
import matplotlib.style as mpl_style
from typing_extensions import override

from thesis_analysis import colors
from thesis_analysis.constants import NBINS, NUM_THREADS, RANGE
from thesis_analysis.paths import Paths
from thesis_analysis.pwa import BinnedFitResult, UnbinnedFitResult, Waveset
from thesis_analysis.tasks.binned_fit import BinnedFit
from thesis_analysis.tasks.guided_fit import GuidedFit


@final
class GuidedPlot(luigi.Task):
    chisqdof = luigi.FloatParameter()
    splot_method = luigi.Parameter()
    nsig = luigi.IntParameter()
    nbkg = luigi.IntParameter()
    niters = luigi.IntParameter(default=3, significant=False)
    averaged = luigi.BoolParameter(default=False)
    phase_factor = luigi.BoolParameter(default=False)

    @override
    def requires(self):
        return [
            BinnedFit(
                self.chisqdof,
                self.splot_method,
                self.nsig,
                self.nbkg,
                self.niters,
                self.phase_factor,
            ),
            GuidedFit(
                self.chisqdof,
                self.splot_method,
                self.nsig,
                self.nbkg,
                self.niters,
                self.averaged,
                self.phase_factor,
            ),
        ]

    @override
    def output(self):
        return [
            luigi.LocalTarget(
                Paths.plots
                / f'guided_fit_chisqdof_{self.chisqdof:.1f}_splot_{self.splot_method}_{self.nsig}s_{self.nbkg}b{"_averaged" if self.averaged else ""}{"_phase_factor" if self.phase_factor else ""}.png'
            ),
        ]

    @override
    def run(self):
        binned_fit_path = Path(str(self.input()[0][0]))
        unbinned_fit_path = Path(str(self.input()[1][0]))

        output_plot_path = Path(str(self.output()[0].path))
        output_plot_path.parent.mkdir(parents=True, exist_ok=True)

        binned_fit_result: BinnedFitResult = pickle.load(
            binned_fit_path.open('rb')
        )
        unbinned_fit_result: UnbinnedFitResult = pickle.load(
            unbinned_fit_path.open('rb')
        )

        datasets = unbinned_fit_result.get_datasets()
        nlls = unbinned_fit_result.get_nlls(datasets)
        weights_fit = unbinned_fit_result.project(nlls, threads=NUM_THREADS)
        weights_s0p = unbinned_fit_result.project_with(
            Waveset.S0P, nlls, threads=NUM_THREADS
        )
        weights_s0n = unbinned_fit_result.project_with(
            Waveset.S0N, nlls, threads=NUM_THREADS
        )
        weights_d2p = unbinned_fit_result.project_with(
            Waveset.D2P, nlls, threads=NUM_THREADS
        )

        unbinned_data_hist = unbinned_fit_result.get_hist(
            datasets[0], bins=NBINS, range=RANGE, weights=None
        )
        unbinned_fit_hist = unbinned_fit_result.get_hist(
            datasets[1], bins=NBINS, range=RANGE, weights=weights_fit
        )
        unbinned_s0p_hist = unbinned_fit_result.get_hist(
            datasets[1], bins=NBINS, range=RANGE, weights=weights_s0p
        )
        unbinned_s0n_hist = unbinned_fit_result.get_hist(
            datasets[1], bins=NBINS, range=RANGE, weights=weights_s0n
        )
        unbinned_d2p_hist = unbinned_fit_result.get_hist(
            datasets[1], bins=NBINS, range=RANGE, weights=weights_d2p
        )
        binned_fit_hist = binned_fit_result.fit_hist
        binned_s0p_hist = binned_fit_result.waveset_hists[Waveset.S0P]
        binned_s0n_hist = binned_fit_result.waveset_hists[Waveset.S0N]
        binned_d2p_hist = binned_fit_result.waveset_hists[Waveset.D2P]

        mpl_style.use('thesis_analysis.thesis')
        fig, ax = plt.subplots(ncols=2, sharey=True)
        ax[0].stairs(
            binned_fit_hist.counts,
            binned_fit_hist.bins,
            color=colors.black,
            label='Fit (binned)',
        )
        ax[1].stairs(
            binned_fit_hist.counts,
            binned_fit_hist.bins,
            color=colors.black,
            label='Fit (binned)',
        )
        ax[0].stairs(
            binned_s0p_hist.counts,
            binned_s0p_hist.bins,
            color=colors.red,
            label='$S_0^+$ (binned)',
        )
        ax[0].stairs(
            binned_s0n_hist.counts,
            binned_s0n_hist.bins,
            color=colors.blue,
            label='$S_0^-$ (binned)',
        )
        ax[1].stairs(
            binned_d2p_hist.counts,
            binned_d2p_hist.bins,
            color=colors.red,
            label='$D_2^+$ (binned)',
        )
        ax[0].stairs(
            unbinned_data_hist.counts,
            unbinned_data_hist.bins,
            color=colors.black,
            label='Data',
        )
        ax[0].stairs(
            unbinned_fit_hist.counts,
            unbinned_fit_hist.bins,
            color=colors.black,
            label='Fit (unbinned)',
            fill=True,
            alpha=0.2,
        )
        ax[1].stairs(
            unbinned_data_hist.counts,
            unbinned_data_hist.bins,
            color=colors.black,
            label='Data',
        )
        ax[1].stairs(
            unbinned_fit_hist.counts,
            unbinned_fit_hist.bins,
            color=colors.black,
            label='Fit (guided)',
            fill=True,
            alpha=0.2,
        )
        ax[0].stairs(
            unbinned_s0p_hist.counts,
            unbinned_s0p_hist.bins,
            color=colors.red,
            label='$S_0^+$ (guided)',
            fill=True,
            alpha=0.2,
        )
        ax[0].stairs(
            unbinned_s0n_hist.counts,
            unbinned_s0n_hist.bins,
            color=colors.blue,
            label='$S_0^-$ (guided)',
            fill=True,
            alpha=0.2,
        )
        ax[1].stairs(
            unbinned_d2p_hist.counts,
            unbinned_d2p_hist.bins,
            color=colors.red,
            label='$D_2^+$ (guided)',
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
