import pickle
from pathlib import Path

import luigi

from thesis_analysis.constants import NBINS, RUN_PERIODS
from thesis_analysis.paths import Paths
from thesis_analysis.pwa import (
    AnalysisPathSet,
    fit_binned,
)
from thesis_analysis.tasks.chisqdof import ChiSqDOF
from thesis_analysis.tasks.splot_weights import SPlotWeights


class BinnedFit(luigi.Task):
    chisqdof = luigi.FloatParameter()
    splot_method = luigi.Parameter()
    nsig = luigi.IntParameter()
    nbkg = luigi.IntParameter()
    niters = luigi.IntParameter(default=3, significant=False)

    resources = {'fit': 1}

    def requires(self):
        return [
            SPlotWeights(
                'data',
                run_period,
                self.chisqdof,
                self.splot_method,
                self.nsig,
                self.nbkg,
            )
            for run_period in RUN_PERIODS
        ] + [
            ChiSqDOF(
                'accmc',
                run_period,
                self.chisqdof,
            )
            for run_period in RUN_PERIODS
        ]

    def output(self):
        return [
            luigi.LocalTarget(
                Paths.fits
                / f'binned_fit_chisqdof_{self.chisqdof:.1f}_splot_{self.splot_method}_{self.nsig}s_{self.nbkg}b.pkl'
            ),
        ]

    def run(self):
        analysis_path_set = AnalysisPathSet(
            *[Path(self.input()[i][0].path) for i in range(len(RUN_PERIODS))],
            *[
                Path(self.input()[i + len(RUN_PERIODS)][0].path)
                for i in range(len(RUN_PERIODS))
            ],
        )

        output_fit_path = Path(str(self.output()[0].path))
        output_fit_path.parent.mkdir(parents=True, exist_ok=True)

        niters = int(self.niters)  # type: ignore

        fit_result = fit_binned(analysis_path_set, nbins=NBINS, niters=niters)

        pickle.dump(fit_result, output_fit_path.open('wb'))
