import pickle
from pathlib import Path

import luigi

from thesis_analysis.constants import NBINS
from thesis_analysis.paths import Paths
from thesis_analysis.pwa import (
    AnalysisPath,
    fit_binned,
)
from thesis_analysis.tasks.chisqdof import ChiSqDOF
from thesis_analysis.tasks.splot_weights import SPlotWeights


class SingleBinnedFit(luigi.Task):
    run_period = luigi.Parameter()
    chisqdof = luigi.FloatParameter()
    splot_method = luigi.Parameter()
    nsig = luigi.IntParameter()
    nbkg = luigi.IntParameter()
    niters = luigi.IntParameter(default=3, significant=False)
    phase_factor = luigi.BoolParameter(default=False)

    resources = {'fit': 1}

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

    def output(self):
        return [
            luigi.LocalTarget(
                Paths.fits
                / f'binned_fit_{self.run_period}_chisqdof_{self.chisqdof:.1f}_splot_{self.splot_method}_{self.nsig}s_{self.nbkg}b{"_phase_factor" if self.phase_factor else ""}.pkl'
            ),
        ]

    def run(self):
        analysis_path_set = AnalysisPath(
            Path(self.input()[0][0].path), Path(self.input()[1][0].path)
        )

        output_fit_path = Path(str(self.output()[0].path))
        output_fit_path.parent.mkdir(parents=True, exist_ok=True)

        niters = int(self.niters)  # type: ignore
        phase_factor = bool(self.phase_factor)  # type: ignore

        fit_result = fit_binned(
            analysis_path_set,
            nbins=NBINS,
            niters=niters,
            phase_factor=phase_factor,
        )

        pickle.dump(fit_result, output_fit_path.open('wb'))
