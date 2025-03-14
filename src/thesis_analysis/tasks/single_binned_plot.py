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
from thesis_analysis.pwa import (
    BinnedFitResultUncertainty,
)
from thesis_analysis.tasks.single_binned_fit_uncertainty import (
    SingleBinnedFitUncertainty,
)
from thesis_analysis.wave import Wave


@final
class SingleBinnedPlot(luigi.Task):
    waves = luigi.IntParameter()
    run_period = luigi.Parameter()
    chisqdof = luigi.FloatParameter()
    splot_method = luigi.Parameter()
    nsig = luigi.IntParameter()
    nbkg = luigi.IntParameter()
    niters = luigi.IntParameter(default=3, significant=False)
    phase_factor = luigi.BoolParameter(default=False)
    uncertainty = luigi.Parameter(default='sqrt')

    @override
    def requires(self):
        return [
            SingleBinnedFitUncertainty(
                self.waves,
                self.run_period,
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
                / f'binned_fit_{self.run_period}_chisqdof_{self.chisqdof:.1f}_splot_{self.splot_method}_{self.nsig}s_{self.nbkg}b{"_phase_factor" if self.phase_factor else ""}_waves{self.waves}_uncertainty_{self.uncertainty}.png'
            ),
        ]

    @override
    def run(self):
        binned_fit_path = Path(str(self.input()[0][0]))

        output_plot_path = Path(str(self.output()[0].path))
        output_plot_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(output_plot_path)

        fit_result: BinnedFitResultUncertainty = pickle.load(
            binned_fit_path.open('rb')
        )
        waves = int(self.waves)  # pyright:ignore[reportArgumentType]

        mpl_style.use('thesis_analysis.thesis')
        data_hist = fit_result.fit_result.get_data_histogram()
        fit_hists = fit_result.fit_result.get_histograms()
        fit_lower_center_upper = fit_result.get_lower_center_upper()
        fig, ax = plt.subplots(ncols=2, sharey=True)
        ax[0].stairs(
            data_hist.counts,
            data_hist.bins,
            color=colors.black,
            label=f'Data ({str(self.run_period).upper()})',
        )
        ax[1].stairs(
            data_hist.counts,
            data_hist.bins,
            color=colors.black,
            label='Data',
        )
        for wave in Wave.decode_waves(waves):
            wave_hist = fit_hists[set([wave])]
            wave_lcu = fit_lower_center_upper[set([wave])]
            centers = (wave_hist.bins[1:] + wave_hist.bins[:-1]) / 2
            ax[wave.plot_index(double=True)[0]].plot(
                centers,
                wave_hist.counts,
                color=wave.plot_color,
                marker='_',  # TODO: maybe make this with the xerr arg of errorbar instead?
                linestyle='none',
            )
            ax[wave.plot_index(double=True)[0]].errorbar(
                centers,
                wave_lcu[1],
                yerr=(wave_lcu[1] - wave_lcu[0], wave_lcu[2] - wave_lcu[1]),
                fmt='none',
                color=wave.plot_color,
            )
        ax[0].legend()
        ax[1].legend()
        ax[0].set_xlabel('Invariant Mass of $K_S^0K_S^0$ (GeV/$c^2$)')
        ax[1].set_xlabel('Invariant Mass of $K_S^0K_S^0$ (GeV/$c^2$)')
        bin_width_mev = int(1000 / NBINS)
        ax[0].set_ylabel(f'Counts / {bin_width_mev} MeV/$c^2$')
        fig.savefig(output_plot_path)
        plt.close()
