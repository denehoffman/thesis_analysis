import pickle
from pathlib import Path
from typing import final, override

import luigi
import matplotlib.pyplot as plt
import matplotlib.style as mpl_style

from thesis_analysis import colors
from thesis_analysis.constants import NBINS
from thesis_analysis.paths import Paths
from thesis_analysis.pwa import (
    BinnedFitResultUncertainty,
)
from thesis_analysis.tasks.binned_fit_uncertainty import BinnedFitUncertainty
from thesis_analysis.wave import Wave


@final
class BinnedPlot(luigi.Task):
    waves = luigi.IntParameter()
    chisqdof = luigi.FloatParameter()
    splot_method = luigi.Parameter()
    nsig = luigi.IntParameter()
    nbkg = luigi.IntParameter()
    niters = luigi.IntParameter(default=3, significant=False)
    phase_factor = luigi.BoolParameter(default=False)
    uncertainty = luigi.Parameter(default='bootstrap')
    bootstrap_mode = luigi.Parameter(default='CI-BC')

    @override
    def requires(self):
        return [
            BinnedFitUncertainty(
                self.waves,
                self.chisqdof,
                self.splot_method,
                self.nsig,
                self.nbkg,
                self.niters,
                self.phase_factor,
                self.uncertainty,
            ),
        ]

    @override
    def output(self):
        return [
            luigi.LocalTarget(
                Paths.plots
                / f'binned_fit_chisqdof_{self.chisqdof:.1f}_splot_{self.splot_method}_{self.nsig}s_{self.nbkg}b{"_phase_factor" if self.phase_factor else ""}_waves{self.waves}_uncertainty_{self.uncertainty}{f"-{self.bootstrap_mode}" if str(self.uncertainty) == "bootstrap" else ""}.png'
            ),
        ]

    @override
    def run(self):
        binned_fit_path = Path(str(self.input()[0][0]))

        output_plot_path = Path(str(self.output()[0].path))
        output_plot_path.parent.mkdir(parents=True, exist_ok=True)

        fit_result: BinnedFitResultUncertainty = pickle.load(
            binned_fit_path.open('rb')
        )
        waves = int(self.waves)  # pyright:ignore[reportArgumentType]
        bootstrap_mode = str(self.bootstrap_mode)

        mpl_style.use('thesis_analysis.thesis')
        data_hist = fit_result.fit_result.get_data_histogram()
        fit_hists = fit_result.fit_result.get_histograms()
        print('available wavesets:')
        for wave in fit_hists.keys():
            print(Wave.decode_waves(wave))
        fit_error_bars = fit_result.get_error_bars(
            bootstrap_mode=bootstrap_mode
        )
        fig, ax = plt.subplots(ncols=2, sharey=True)
        for i in {0, 1}:
            ax[i].stairs(
                data_hist.counts,
                data_hist.bins,
                color=colors.black,
                label='Data',
            )
            fit_hist = fit_hists[waves]
            err = fit_error_bars[waves]
            centers = (fit_hist.bins[1:] + fit_hist.bins[:-1]) / 2
            ax[i].errorbar(
                centers,
                fit_hist.counts,
                yerr=0,
                fmt='.',
                markersize=3,
                color=colors.black,
                label='Fit Total',
            )
            ax[i].errorbar(
                centers,
                err[1],
                yerr=(err[0], err[2]),
                fmt='none',
                color=colors.black,
            )
        for wave in Wave.decode_waves(waves):
            wave_hist = fit_hists[Wave.encode(wave)]
            err = fit_error_bars[Wave.encode(wave)]
            centers = (wave_hist.bins[1:] + wave_hist.bins[:-1]) / 2
            ax[wave.plot_index(double=True)[0]].errorbar(
                centers,
                wave_hist.counts,
                yerr=0,
                fmt='.',
                markersize=3,
                color=wave.plot_color,
                label=wave.latex,
            )
            ax[wave.plot_index(double=True)[0]].errorbar(
                centers,
                err[1],
                yerr=(err[0], err[2]),
                fmt='none',
                color=wave.plot_color,
            )
        ax[0].legend()
        ax[1].legend()
        ax[0].set_xlabel('Invariant Mass of $K_S^0K_S^0$ (GeV/$c^2$)')
        ax[1].set_xlabel('Invariant Mass of $K_S^0K_S^0$ (GeV/$c^2$)')
        bin_width_mev = int(1000 / NBINS)
        ax[0].set_ylabel(f'Counts / {bin_width_mev} MeV/$c^2$')
        ax[0].set_ylim(0)
        ax[1].set_ylim(0)
        fig.savefig(output_plot_path)
        plt.close()
