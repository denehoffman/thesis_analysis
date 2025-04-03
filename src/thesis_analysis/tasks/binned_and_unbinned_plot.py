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
    UnbinnedFitResult,
)
from thesis_analysis.tasks.binned_fit_uncertainty import BinnedFitUncertainty
from thesis_analysis.tasks.binned_plot import BinnedPlot
from thesis_analysis.tasks.guided_plot import GuidedPlot
from thesis_analysis.tasks.unbinned_fit import UnbinnedFit
from thesis_analysis.tasks.unbinned_plot import UnbinnedPlot
from thesis_analysis.wave import Wave


@final
class BinnedAndUnbinnedPlot(luigi.Task):
    waves = luigi.IntParameter()
    chisqdof = luigi.FloatParameter()
    splot_method = luigi.Parameter()
    nsig = luigi.IntParameter()
    nbkg = luigi.IntParameter()
    niters = luigi.IntParameter(default=3, significant=False)
    guided = luigi.BoolParameter(default=False)
    phase_factor = luigi.BoolParameter(default=False)
    uncertainty = luigi.Parameter(default='bootstrap')
    bootstrap_mode = luigi.Parameter(default='CI-BC')

    @override
    def requires(self):
        reqs = [
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
            BinnedPlot(
                self.waves,
                self.chisqdof,
                self.splot_method,
                self.nsig,
                self.nbkg,
                self.niters,
                self.phase_factor,
                self.uncertainty,
                self.bootstrap_mode,
            ),
            UnbinnedPlot(
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
        if self.guided:
            reqs += [
                GuidedPlot(
                    self.waves,
                    self.chisqdof,
                    self.splot_method,
                    self.nsig,
                    self.nbkg,
                    self.niters,
                    self.phase_factor,
                    self.uncertainty,
                    self.bootstrap_mode,
                )
            ]
        return reqs

    @override
    def output(self):
        return [
            luigi.LocalTarget(
                Paths.plots
                / f'binned_and_unbinned_fit_chisqdof_{self.chisqdof:.1f}_splot_{self.splot_method}_{self.nsig}s_{self.nbkg}b{"_guided" if self.guided else ""}{"_phase_factor" if self.phase_factor else ""}_waves{self.waves}_uncertainty_{self.uncertainty}{f"-{self.bootstrap_mode}" if str(self.uncertainty) == "bootstrap" else ""}.png'
            ),
        ]

    @override
    def run(self):
        binned_fit_path = Path(str(self.input()[0][0]))
        unbinned_fit_path = Path(str(self.input()[1][0]))

        output_plot_path = Path(str(self.output()[0].path))
        output_plot_path.parent.mkdir(parents=True, exist_ok=True)

        binned_fit_result: BinnedFitResultUncertainty = pickle.load(
            binned_fit_path.open('rb')
        )
        unbinned_fit_result: UnbinnedFitResult = pickle.load(
            unbinned_fit_path.open('rb')
        )
        waves = int(self.waves)  # pyright:ignore[reportArgumentType]
        bootstrap_mode = str(self.bootstrap_mode)

        mpl_style.use('thesis_analysis.thesis')
        fig, ax = plt.subplots(ncols=2, sharey=True)

        # Binned Plot
        data_hist = binned_fit_result.fit_result.get_data_histogram()
        fit_hists = binned_fit_result.fit_result.get_histograms()
        unbinned_fit_hists = unbinned_fit_result.get_histograms(
            binned_fit_result.fit_result.binning
        )
        fit_error_bars = binned_fit_result.get_error_bars(
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
        # Unbinned Plot
        for i in {0, 1}:
            ax[i].stairs(
                unbinned_fit_hists[waves].counts,
                unbinned_fit_hists[waves].bins,
                color=colors.black,
                label='Fit (Unbinned)',
                fill=True,
                alpha=0.2,
            )
        for wave in Wave.decode_waves(waves):
            wave_hist = unbinned_fit_hists[Wave.encode(wave)]
            ax[wave.plot_index(double=True)[0]].stairs(
                wave_hist.counts,
                wave_hist.bins,
                color=wave.plot_color,
                label=f'{wave.latex} (Unbinned)',
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
