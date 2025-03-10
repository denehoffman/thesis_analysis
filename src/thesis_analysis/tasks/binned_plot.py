import pickle
from pathlib import Path
from typing import final

import luigi
import matplotlib.pyplot as plt
import matplotlib.style as mpl_style
from typing_extensions import override

from thesis_analysis import colors
from thesis_analysis.constants import NBINS
from thesis_analysis.paths import Paths
from thesis_analysis.pwa import BinnedFitResult, Waveset
from thesis_analysis.tasks.binned_fit import BinnedFit


@final
class BinnedPlot(luigi.Task):
    chisqdof = luigi.FloatParameter()
    splot_method = luigi.Parameter()
    nsig = luigi.IntParameter()
    nbkg = luigi.IntParameter()
    niters = luigi.IntParameter(default=3, significant=False)
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
        ]

    @override
    def output(self):
        return [
            luigi.LocalTarget(
                Paths.plots
                / f'binned_fit_chisqdof_{self.chisqdof:.1f}_splot_{self.splot_method}_{self.nsig}s_{self.nbkg}b{"_phase_factor" if self.phase_factor else ""}.png'
            ),
        ]

    @override
    def run(self):
        binned_fit_path = Path(str(self.input()[0][0]))

        output_plot_path = Path(str(self.output()[0].path))
        output_plot_path.parent.mkdir(parents=True, exist_ok=True)

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
