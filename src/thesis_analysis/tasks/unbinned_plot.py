import pickle
from pathlib import Path
from typing import final, override

import luigi
import matplotlib.pyplot as plt
import matplotlib.style as mpl_style

from thesis_analysis import colors
from thesis_analysis.constants import NBINS, RANGE
from thesis_analysis.paths import Paths
from thesis_analysis.pwa import Binning, UnbinnedFitResult
from thesis_analysis.tasks.unbinned_fit import UnbinnedFit
from thesis_analysis.wave import Wave


@final
class UnbinnedPlot(luigi.Task):
    waves = luigi.IntParameter()
    chisqdof = luigi.FloatParameter()
    splot_method = luigi.Parameter()
    nsig = luigi.IntParameter()
    nbkg = luigi.IntParameter()
    niters = luigi.IntParameter(default=3, significant=False)
    guided = luigi.BoolParameter(default=False)
    phase_factor = luigi.BoolParameter(default=False)
    uncertainty = luigi.Parameter(default='bootstrap')

    @override
    def requires(self):
        return [
            UnbinnedFit(
                self.waves,
                self.chisqdof,
                self.splot_method,
                self.nsig,
                self.nbkg,
                self.niters,
                self.guided,
                self.phase_factor,
                self.uncertainty,
            ),
        ]

    @override
    def output(self):
        return [
            luigi.LocalTarget(
                Paths.plots
                / f'unbinned_fit_chisqdof_{self.chisqdof:.1f}_splot_{self.splot_method}_{self.nsig}s_{self.nbkg}b{"_guided" if self.guided else ""}{"_phase_factor" if self.phase_factor else ""}_waves{self.waves}_uncertainty_{self.uncertainty}.png'
            ),
        ]

    @override
    def run(self):
        unbinned_fit_path = Path(str(self.input()[0][0]))

        output_plot_path = Path(str(self.output()[0].path))
        output_plot_path.parent.mkdir(parents=True, exist_ok=True)

        fit_result: UnbinnedFitResult = pickle.load(
            unbinned_fit_path.open('rb')
        )
        waves = int(self.waves)  # pyright:ignore[reportArgumentType]

        mpl_style.use('thesis_analysis.thesis')
        data_hist = fit_result.get_data_histogram(Binning(NBINS, RANGE))
        fit_hists = fit_result.get_histograms(Binning(NBINS, RANGE))
        fig, ax = plt.subplots(ncols=2, sharey=True)
        for i in {0, 1}:
            ax[i].stairs(
                data_hist.counts,
                data_hist.bins,
                color=colors.black,
                label='Data',
            )
            ax[i].stairs(
                fit_hists[waves].counts,
                fit_hists[waves].bins,
                color=colors.black,
                label='Fit',
                fill=True,
                alpha=0.2,
            )
        for wave in Wave.decode_waves(waves):
            wave_hist = fit_hists[Wave.encode(wave)]
            ax[wave.plot_index(double=True)[0]].stairs(
                wave_hist.counts,
                wave_hist.bins,
                color=wave.plot_color,
                label=wave.latex,
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
