import pickle
from pathlib import Path

import luigi

from thesis_analysis.constants import RUN_PERIODS
from thesis_analysis.paths import Paths
from thesis_analysis.pwa import (
    AnalysisPathSet,
    BinnedFitResult,
    fit_unbinned_guided,
)
from thesis_analysis.tasks.binned_fit import BinnedFit
from thesis_analysis.tasks.chisqdof import ChiSqDOF
from thesis_analysis.tasks.single_binned_fit import SingleBinnedFit
from thesis_analysis.tasks.splot_weights import SPlotWeights


class GuidedFit(luigi.Task):
    chisqdof = luigi.FloatParameter()
    splot_method = luigi.Parameter()
    nsig = luigi.IntParameter()
    nbkg = luigi.IntParameter()
    niters = luigi.IntParameter(default=3, significant=False)
    averaged = luigi.BoolParameter(default=False)

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
        if self.averaged:
            reqs += [
                BinnedFit(
                    self.chisqdof, self.splot_method, self.nsig, self.nbkg
                )
            ]
        else:
            reqs += [
                SingleBinnedFit(
                    run_period,
                    self.chisqdof,
                    self.splot_method,
                    self.nsig,
                    self.nbkg,
                )
                for run_period in RUN_PERIODS
            ]
        return reqs

    def output(self):
        return [
            luigi.LocalTarget(
                Paths.fits
                / f'guided_fit_chisqdof_{self.chisqdof:.1f}_splot_{self.splot_method}_{self.nsig}s_{self.nbkg}b{'_averaged' if self.averaged else ''}.pkl'
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

        binned_fit_result: BinnedFitResult | list[BinnedFitResult] = (
            pickle.load(
                Path(self.input()[2 * len(RUN_PERIODS)][0].path).open('rb')
            )
            if self.averaged
            else [
                pickle.load(
                    Path(self.input()[2 * len(RUN_PERIODS) + i][0].path).open(
                        'rb'
                    )
                )
                for i in range(len(RUN_PERIODS))
            ]
        )

        output_fit_path = Path(str(self.output()[0].path))
        output_fit_path.parent.mkdir(parents=True, exist_ok=True)

        niters = int(self.niters)  # type: ignore

        fit_result = fit_unbinned_guided(
            analysis_path_set, binned_fit_result, niters=niters
        )

        pickle.dump(fit_result, output_fit_path.open('wb'))
