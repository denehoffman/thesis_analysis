import pickle
from pathlib import Path

import luigi

from thesis_analysis.constants import RUN_PERIODS
from thesis_analysis.paths import Paths
from thesis_analysis.pwa import (
    AnalysisPathSet,
    UnbinnedFitResult,
    fit_unbinned,
)
from thesis_analysis.tasks.chisqdof import ChiSqDOF
from thesis_analysis.tasks.guided_fit import GuidedFit
from thesis_analysis.tasks.splot_weights import SPlotWeights


class UnbinnedFit(luigi.Task):
    chisqdof = luigi.FloatParameter()
    splot_method = luigi.Parameter()
    nsig = luigi.IntParameter()
    nbkg = luigi.IntParameter()
    niters = luigi.IntParameter(default=3, significant=False)
    guided = luigi.BoolParameter(default=False)
    averaged = luigi.BoolParameter(default=False)

    resources = {'fit': 1}

    def requires(self):
        reqs = [
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
        if self.guided:
            reqs += [
                GuidedFit(
                    self.chisqdof,
                    self.splot_method,
                    self.nsig,
                    self.nbkg,
                    self.niters,
                    self.averaged,
                )
            ]
        return reqs

    def output(self):
        return [
            luigi.LocalTarget(
                Paths.fits
                / f'unbinned_fit_chisqdof_{self.chisqdof:.1f}_splot_{self.splot_method}_{self.nsig}s_{self.nbkg}b{'_guided' if self.guided else ''}.pkl'
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
        p0 = None
        if self.guided:
            guided_result: UnbinnedFitResult = pickle.load(
                Path(self.input()[-1][0].path).open('rb')
            )
            p0 = guided_result.best_status.x

        fit_result = fit_unbinned(analysis_path_set, p0=p0, niters=niters)

        pickle.dump(fit_result, output_fit_path.open('wb'))
