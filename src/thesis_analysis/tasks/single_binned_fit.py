import pickle
from pathlib import Path
from typing import final

import luigi
from typing_extensions import override

from thesis_analysis.constants import NBINS, RANGE
from thesis_analysis.paths import Paths
from thesis_analysis.pwa import (
    Binning,
    SinglePathSet,
    fit_binned,
)
from thesis_analysis.tasks.chisqdof import ChiSqDOF
from thesis_analysis.tasks.splot_weights import SPlotWeights
from thesis_analysis.wave import Wave


@final
class SingleBinnedFit(luigi.Task):
    waves = luigi.IntParameter()
    run_period = luigi.Parameter()
    chisqdof = luigi.FloatParameter()
    splot_method = luigi.Parameter()
    nsig = luigi.IntParameter()
    nbkg = luigi.IntParameter()
    niters = luigi.IntParameter(default=3, significant=False)
    phase_factor = luigi.BoolParameter(default=False)

    resources = {'fit': 1}

    @override
    def requires(self):
        return [
            SPlotWeights(
                'data',
                self.run_period,
                self.chisqdof,
                self.splot_method,
                self.nsig,
                self.nbkg,
            ),
            ChiSqDOF(
                'accmc',
                self.run_period,
                self.chisqdof,
            ),
        ]

    @override
    def output(self):
        return [
            luigi.LocalTarget(
                Paths.fits
                / f'binned_fit_{self.run_period}_chisqdof_{self.chisqdof:.1f}_splot_{self.splot_method}_{self.nsig}s_{self.nbkg}b{"_phase_factor" if self.phase_factor else ""}_waves{self.waves}.pkl'
            ),
        ]

    @override
    def run(self):
        analysis_path_set = SinglePathSet(
            Path(self.input()[0][0].path), Path(self.input()[1][0].path)
        )

        output_fit_path = Path(str(self.output()[0].path))
        output_fit_path.parent.mkdir(parents=True, exist_ok=True)

        waves = int(self.waves)  # pyright:ignore[reportArgumentType]
        niters = int(self.niters)  # pyright:ignore[reportArgumentType]
        phase_factor = bool(self.phase_factor)

        binning = Binning(NBINS, RANGE)

        fit_result = fit_binned(
            Wave.decode_waves(waves),
            analysis_path_set,
            binning,
            iters=niters,
            phase_factor=phase_factor,
        )

        pickle.dump(fit_result, output_fit_path.open('wb'))
