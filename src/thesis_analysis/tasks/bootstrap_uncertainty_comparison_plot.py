import pickle
from pathlib import Path
from typing import final, override

import luigi
import matplotlib.pyplot as plt
import matplotlib.style as mpl_style
import numpy as np

from thesis_analysis import colors
from thesis_analysis.constants import NBINS
from thesis_analysis.paths import Paths
from thesis_analysis.pwa import (
    BinnedFitResultUncertainty,
)
from thesis_analysis.tasks.binned_fit_uncertainty import BinnedFitUncertainty
from thesis_analysis.wave import Wave


@final
class BootstrapUncertaintyComparisonPlot(luigi.Task):
    waves = luigi.IntParameter()
    chisqdof = luigi.FloatParameter()
    ksb_costheta = luigi.FloatParameter()
    cut_baryons = luigi.OptionalBoolParameter(True)
    splot_method = luigi.Parameter()
    nsig = luigi.IntParameter()
    nbkg = luigi.IntParameter()
    niters = luigi.IntParameter(default=3, significant=False)
    phase_factor = luigi.BoolParameter(default=False)

    @override
    def requires(self):
        return [
            BinnedFitUncertainty(
                self.waves,
                self.chisqdof,
                self.ksb_costheta,
                self.cut_baryons,
                self.splot_method,
                self.nsig,
                self.nbkg,
                self.niters,
                self.phase_factor,
                uncertainty='bootstrap',
            ),
        ]

    @override
    def output(self):
        return [
            luigi.LocalTarget(
                Paths.plots
                / f'binned_fit_chisqdof_{self.chisqdof:.1f}_ksb_costheta_{self.ksb_costheta:.2f}{"mesons" if self.cut_baryons else "baryons"}_splot_{self.splot_method}_{self.nsig}s_{self.nbkg}b{"_phase_factor" if self.phase_factor else ""}_waves{self.waves}_uncertainty_bootstrap-comparison.png'
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

        mpl_style.use('thesis_analysis.thesis')
        data_hist = fit_result.fit_result.get_data_histogram()
        fit_hists = fit_result.fit_result.get_histograms()
        print('available wavesets:')
        for wave in fit_hists.keys():
            print(Wave.decode_waves(wave))
        modes = ['SE', 'CI', 'CI-BC']
        ecolors = [colors.orange, colors.purple, colors.brown]
        fit_error_bars = {
            mode: fit_result.get_error_bars(bootstrap_mode=mode)
            for mode in modes
        }
        fig, ax = plt.subplots(ncols=2, sharey=True)
        for i in {0, 1}:
            ax[i].stairs(
                data_hist.counts,
                data_hist.bins,
                color=colors.black,
                label='Data',
            )
            fit_hist = fit_hists[waves]
            for j, (mode, ecolor) in enumerate(zip(modes, ecolors)):
                err = fit_error_bars[mode][waves]
                centers = (fit_hist.bins[1:] + fit_hist.bins[:-1]) / 2
                offset = (np.diff(centers)[0] / 3) * (j - 1)
                ax[i].scatter(
                    centers + offset,
                    fit_hist.counts,
                    marker='.',
                    s=1,
                    color=colors.black,
                )
                ax[i].errorbar(
                    centers + offset,
                    err[1],
                    yerr=(err[0], err[2]),
                    fmt='none',
                    capsize=0.0,
                    ecolor=ecolor,
                    color=colors.black,
                    label=f'Fit Total ({mode})',
                )
        for wave in Wave.decode_waves(waves):
            wave_hist = fit_hists[Wave.encode(wave)]
            for j, (mode, ecolor) in enumerate(zip(modes, ecolors)):
                err = fit_error_bars[mode][Wave.encode(wave)]
                centers = (wave_hist.bins[1:] + wave_hist.bins[:-1]) / 2
                offset = (np.diff(centers)[0] / 3) * (j - 1)
                ax[wave.plot_index(double=True)[0]].scatter(
                    centers + offset,
                    wave_hist.counts,
                    marker='.',
                    s=1,
                    color=wave.plot_color,
                )
                ax[wave.plot_index(double=True)[0]].errorbar(
                    centers + offset,
                    err[1],
                    yerr=(err[0], err[2]),
                    fmt='none',
                    capsize=0.0,
                    ecolor=ecolor,
                    color=wave.plot_color,
                    label=f'{wave.latex} ({mode})',
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
