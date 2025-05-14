import pickle
from pathlib import Path
from typing import final, override

import luigi
import matplotlib.pyplot as plt
import matplotlib.style as mpl_style

from thesis_analysis import colors
from thesis_analysis.constants import NBINS, RANGE, RUN_PERIODS
from thesis_analysis.paths import Paths
from thesis_analysis.pwa import Binning, UnbinnedFitResult
from thesis_analysis.tasks.pol_gen import PolarizeGenerated
from thesis_analysis.tasks.unbinned_fit import UnbinnedFit
from thesis_analysis.wave import Wave


@final
class UnbinnedPlot(luigi.Task):
    waves = luigi.IntParameter()
    chisqdof = luigi.FloatParameter()
    ksb_costheta = luigi.FloatParameter()
    cut_baryons = luigi.OptionalBoolParameter(True)
    splot_method = luigi.Parameter()
    nsig = luigi.IntParameter()
    nbkg = luigi.IntParameter()
    niters = luigi.IntParameter(default=3, significant=False)
    guided = luigi.BoolParameter(default=False)
    phase_factor = luigi.BoolParameter(default=False)
    uncertainty = luigi.Parameter(default='bootstrap')
    bootstrap_mode = luigi.Parameter(default='SE')
    acceptance_corrected = luigi.BoolParameter(default=False)

    @override
    def requires(self):
        reqs = [
            UnbinnedFit(
                self.waves,
                self.chisqdof,
                self.ksb_costheta,
                self.cut_baryons,
                self.splot_method,
                self.nsig,
                self.nbkg,
                self.niters,
                self.guided,
                self.phase_factor,
                self.uncertainty,
                self.bootstrap_mode,
            ),
        ]
        if self.acceptance_corrected:
            reqs += [
                PolarizeGenerated(run_period=run_period)
                for run_period in RUN_PERIODS
            ]
        return reqs

    @override
    def output(self):
        return [
            luigi.LocalTarget(
                Paths.plots
                / f'unbinned_fit_chisqdof_{self.chisqdof:.1f}_ksb_costheta_{self.ksb_costheta:.2f}{"mesons" if self.cut_baryons else "baryons"}_splot_{self.splot_method}_{self.nsig}s_{self.nbkg}b{"_guided" if self.guided else ""}{"_phase_factor" if self.phase_factor else ""}_waves{self.waves}_uncertainty_{self.uncertainty}{"_acc" if self.acceptance_corrected else ""}.png'
            ),
        ]

    @override
    def run(self):
        unbinned_fit_path = Path(str(self.input()[0][0]))
        if self.acceptance_corrected:
            genmc_paths = [
                Path(str(self.input()[i + 1][0]))
                for i in range(len(RUN_PERIODS))
            ]
        else:
            genmc_paths = None

        output_plot_path = Path(str(self.output()[0].path))
        output_plot_path.parent.mkdir(parents=True, exist_ok=True)

        fit_result: UnbinnedFitResult = pickle.load(
            unbinned_fit_path.open('rb')
        )
        waves = int(self.waves)  # pyright:ignore[reportArgumentType]

        mpl_style.use('thesis_analysis.thesis')
        data_hist = fit_result.get_data_histogram(Binning(NBINS, RANGE))
        fit_hists = fit_result.get_histograms(
            Binning(NBINS, RANGE), mc_paths=genmc_paths
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
                    ax[i][j].stairs(
                        fit_hists[waves].counts,
                        fit_hists[waves].bins,
                        color=colors.black,
                        label='Fit (Unbinned)',
                        fill=True,
                        alpha=0.2,
                    )
            for wave in Wave.decode_waves(waves):
                wave_hist = fit_hists[Wave.encode(wave)]
                plot_index = wave.plot_index(double=False)
                ax[plot_index[0]][plot_index[1]].stairs(
                    wave_hist.counts,
                    wave_hist.bins,
                    color=wave.plot_color,
                    label=f'{wave.latex} (Unbinned)',
                    fill=True,
                    alpha=0.2,
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
                ax[i].stairs(
                    fit_hists[waves].counts,
                    fit_hists[waves].bins,
                    color=colors.black,
                    label='Fit (Unbinned)',
                    fill=True,
                    alpha=0.2,
                )
            for wave in Wave.decode_waves(waves):
                wave_hist = fit_hists[Wave.encode(wave)]
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
            fig.supxlabel('Invariant Mass of $K_S^0K_S^0$ (GeV/$c^2$)')
            bin_width_mev = int(1000 / NBINS)
            fig.supylabel(f'Counts / {bin_width_mev} MeV/$c^2$')
            fig.savefig(output_plot_path)
            plt.close()
