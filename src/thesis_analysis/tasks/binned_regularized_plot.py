import pickle
from pathlib import Path
from typing import final, override

import luigi
import matplotlib.pyplot as plt
import matplotlib.style as mpl_style

from thesis_analysis import colors
from thesis_analysis.constants import NBINS
from thesis_analysis.logger import logger
from thesis_analysis.paths import Paths
from thesis_analysis.pwa import BinnedFitResult
from thesis_analysis.tasks.binned_regularized_fit import BinnedRegularizedFit
from thesis_analysis.wave import Wave


@final
class BinnedRegularizedPlot(luigi.Task):
    waves = luigi.IntParameter()
    chisqdof = luigi.FloatParameter()
    splot_method = luigi.Parameter()
    nsig = luigi.IntParameter()
    nbkg = luigi.IntParameter()
    niters = luigi.IntParameter(default=3, significant=False)
    phase_factor = luigi.BoolParameter(default=False)
    lda = luigi.FloatParameter()
    gamma = luigi.FloatParameter()

    @override
    def requires(self):
        logger.debug('checking dependencies for binned regularized plot')
        return [
            BinnedRegularizedFit(
                self.waves,
                self.chisqdof,
                self.splot_method,
                self.nsig,
                self.nbkg,
                self.niters,
                self.phase_factor,
                self.lda,
                self.gamma,
            ),
        ]

    @override
    def output(self):
        return [
            luigi.LocalTarget(
                Paths.plots
                / f'binned_regularized_fit_chisqdof_{self.chisqdof:.1f}_splot_{self.splot_method}_{self.nsig}s_{self.nbkg}b{"_phase_factor" if self.phase_factor else ""}_waves{self.waves}_lda_{self.lda}_gamma_{self.gamma}.png'
            ),
        ]

    @override
    def run(self):
        logger.info(f'Beginning Binned Regularized Plot (λ={self.lda})')
        binned_fit_path = Path(str(self.input()[0][0]))

        output_plot_path = Path(str(self.output()[0].path))
        output_plot_path.parent.mkdir(parents=True, exist_ok=True)

        fit_result: BinnedFitResult = pickle.load(binned_fit_path.open('rb'))
        waves = int(self.waves)  # pyright:ignore[reportArgumentType]

        mpl_style.use('thesis_analysis.thesis')
        data_hist = fit_result.get_data_histogram()
        fit_hists = fit_result.get_histograms()
        if Wave.needs_full_plot(Wave.decode_waves(waves)):
            fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
            for i in {0, 1}:
                for j in {0, 1, 2}:
                    if not Wave.has_wave_at_index(
                        Wave.decode_waves(waves), (i, j)
                    ):
                        continue
                    ax[i][j].stairs(
                        data_hist.counts,
                        data_hist.bins,
                        color=colors.black,
                        label='Data',
                    )
                    fit_hist = fit_hists[waves]
                    centers = (fit_hist.bins[1:] + fit_hist.bins[:-1]) / 2
                    ax[i][j].errorbar(
                        centers,
                        fit_hist.counts,
                        yerr=0,
                        fmt='.',
                        markersize=3,
                        color=colors.black,
                        label='Fit Total',
                    )
            for wave in Wave.decode_waves(waves):
                wave_hist = fit_hists[Wave.encode(wave)]
                centers = (wave_hist.bins[1:] + wave_hist.bins[:-1]) / 2
                plot_index = wave.plot_index(double=False)
                ax[plot_index[0]][plot_index[1]].errorbar(
                    centers,
                    wave_hist.counts,
                    yerr=0,
                    fmt='.',
                    markersize=3,
                    color=wave.plot_color,
                    label=wave.latex,
                )
            for i in {0, 1}:
                for j in {0, 1, 2}:
                    if not Wave.has_wave_at_index(
                        Wave.decode_waves(waves), (i, j)
                    ):
                        latex_group = Wave.get_latex_group_at_index((i, j))
                        ax[i][j].text(
                            0.5,
                            0.5,
                            f'No {latex_group}',
                            ha='center',
                            va='center',
                            transform=ax[i][j].transAxes,
                        )
                    else:
                        ax[i][j].legend()
                        ax[i][j].set_ylim(0)
            fig.supxlabel('Invariant Mass of $K_S^0K_S^0$ (GeV/$c^2$)')
            bin_width_mev = int(1000 / NBINS)
            fig.supylabel(f'Counts / {bin_width_mev} MeV/$c^2$')
            fig.savefig(output_plot_path)
            plt.close()
        else:
            fig, ax = plt.subplots(ncols=2, sharey=True)
            for i in {0, 1}:
                ax[i].stairs(
                    data_hist.counts,
                    data_hist.bins,
                    color=colors.black,
                    label='Data',
                )
                fit_hist = fit_hists[waves]
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
            for wave in Wave.decode_waves(waves):
                wave_hist = fit_hists[Wave.encode(wave)]
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
            ax[0].legend()
            ax[1].legend()
            ax[0].set_ylim(0)
            ax[1].set_ylim(0)
            fig.supxlabel('Invariant Mass of $K_S^0K_S^0$ (GeV/$c^2$)')
            bin_width_mev = int(1000 / NBINS)
            fig.supylabel(f'Counts / {bin_width_mev} MeV/$c^2$')
            fig.savefig(output_plot_path)
            plt.close()
