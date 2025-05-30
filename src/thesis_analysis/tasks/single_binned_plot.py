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
from thesis_analysis.tasks.pol_gen import PolarizeGenerated
from thesis_analysis.tasks.single_binned_fit_uncertainty import (
    SingleBinnedFitUncertainty,
)
from thesis_analysis.wave import Wave


@final
class SingleBinnedPlot(luigi.Task):
    waves = luigi.IntParameter()
    run_period = luigi.Parameter()
    chisqdof = luigi.FloatParameter()
    ksb_costheta = luigi.FloatParameter()
    cut_baryons = luigi.OptionalBoolParameter(True)
    splot_method = luigi.Parameter()
    nsig = luigi.IntParameter()
    nbkg = luigi.IntParameter()
    niters = luigi.IntParameter(default=3, significant=False)
    phase_factor = luigi.BoolParameter(default=False)
    uncertainty = luigi.Parameter(default='bootstrap')
    bootstrap_mode = luigi.Parameter(default='CI-BC')
    acceptance_corrected = luigi.BoolParameter(default=False)

    @override
    def requires(self):
        reqs = [
            SingleBinnedFitUncertainty(
                self.waves,
                self.run_period,
                self.chisqdof,
                self.ksb_costheta,
                self.cut_baryons,
                self.splot_method,
                self.nsig,
                self.nbkg,
                self.niters,
                self.phase_factor,
                self.uncertainty,
            ),
        ]
        if self.acceptance_corrected:
            reqs += [PolarizeGenerated(run_period=self.run_period)]
        return reqs

    @override
    def output(self):
        return [
            luigi.LocalTarget(
                Paths.plots
                / f'binned_fit_{self.run_period}_chisqdof_{self.chisqdof:.1f}_ksb_costheta_{self.ksb_costheta:.2f}{"mesons" if self.cut_baryons else "baryons"}_splot_{self.splot_method}_{self.nsig}s_{self.nbkg}b{"_phase_factor" if self.phase_factor else ""}_waves{self.waves}_uncertainty_{self.uncertainty}{"_acc" if self.acceptance_corrected else ""}.png'
            ),
        ]

    @override
    def run(self):
        binned_fit_path = Path(str(self.input()[0][0]))
        if self.acceptance_corrected:
            genmc_paths = [Path(str(self.input()[1][0]))]
        else:
            genmc_paths = None

        output_plot_path = Path(str(self.output()[0].path))
        output_plot_path.parent.mkdir(parents=True, exist_ok=True)

        fit_result: BinnedFitResultUncertainty = pickle.load(
            binned_fit_path.open('rb')
        )
        waves = int(self.waves)  # pyright:ignore[reportArgumentType]
        bootstrap_mode = str(self.bootstrap_mode)

        mpl_style.use('thesis_analysis.thesis')
        data_hist = fit_result.fit_result.get_data_histogram()
        fit_hists = fit_result.fit_result.get_histograms(mc_paths=genmc_paths)
        fit_error_bars = fit_result.get_error_bars(
            bootstrap_mode=bootstrap_mode, mc_paths=genmc_paths
        )
        if Wave.needs_full_plot(Wave.decode_waves(waves)):
            fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
            for i in {0, 1}:
                for j in {0, 1, 2}:
                    if not Wave.has_wave_at_index(
                        Wave.decode_waves(waves), (i, j)
                    ):
                        continue
                    if not self.acceptance_corrected:
                        ax[i][j].stairs(
                            data_hist.counts,
                            data_hist.bins,
                            color=colors.black,
                            label='Data',
                        )
                    fit_hist = fit_hists[waves]
                    err = fit_error_bars[waves]
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
                    ax[i][j].errorbar(
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
                ax[plot_index[0]][plot_index[1]].errorbar(
                    centers,
                    err[1],
                    yerr=(err[0], err[2]),
                    fmt='none',
                    color=wave.plot_color,
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
                if not self.acceptance_corrected:
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
            ax[0].set_ylim(0)
            ax[1].set_ylim(0)
            fig.supxlabel('Invariant Mass of $K_S^0K_S^0$ (GeV/$c^2$)')
            bin_width_mev = int(1000 / NBINS)
            fig.supylabel(f'Counts / {bin_width_mev} MeV/$c^2$')
            fig.savefig(output_plot_path)
            plt.close()
