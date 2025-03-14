import pickle
from pathlib import Path
from typing import final, override

import luigi
from thesis_analysis.constants import RUN_PERIODS
from thesis_analysis.paths import Paths
from thesis_analysis.pwa import (
    FullPathSet,
    SinglePathSet,
    UnbinnedFitResult,
    fit_unbinned,
)
from thesis_analysis.tasks.chisqdof import ChiSqDOF
from thesis_analysis.tasks.guided_fit import GuidedFit
from thesis_analysis.tasks.splot_weights import SPlotWeights
from thesis_analysis.wave import Wave


@final
class UnbinnedFit(luigi.Task):
    waves = luigi.IntParameter()
    chisqdof = luigi.FloatParameter()
    splot_method = luigi.Parameter()
    nsig = luigi.IntParameter()
    nbkg = luigi.IntParameter()
    niters = luigi.IntParameter(default=3, significant=False)
    guided = luigi.BoolParameter(default=False)
    averaged = luigi.BoolParameter(default=False)
    phase_factor = luigi.BoolParameter(default=False)
    uncertainty = luigi.Parameter(default='sqrt')

    resources = {'fit': 1}

    @override
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
                    self.waves,
                    self.chisqdof,
                    self.splot_method,
                    self.nsig,
                    self.nbkg,
                    self.niters,
                    self.averaged,
                    self.phase_factor,
                    self.uncertainty,
                )
            ]
        return reqs

    @override
    def output(self):
        return [
            luigi.LocalTarget(
                Paths.fits
                / f'unbinned_fit_chisqdof_{self.chisqdof:.1f}_splot_{self.splot_method}_{self.nsig}s_{self.nbkg}b{"_guided" if self.guided else ""}{"_averaged" if self.averaged else ""}{"_phase_factor" if self.phase_factor else ""}_waves{self.waves}_uncertainty_{self.uncertainty}.pkl'
            ),
        ]

    @override
    def run(self):
        analysis_path_set = FullPathSet(
            *[
                SinglePathSet(data_path, accmc_path)
                for data_path, accmc_path in zip(
                    [
                        Path(self.input()[i][0].path)
                        for i in range(len(RUN_PERIODS))
                    ],
                    [
                        Path(self.input()[i + len(RUN_PERIODS)][0].path)
                        for i in range(len(RUN_PERIODS))
                    ],
                )
            ]
        )

        output_fit_path = Path(str(self.output()[0].path))
        output_fit_path.parent.mkdir(parents=True, exist_ok=True)

        waves = int(self.waves)  # pyright:ignore[reportArgumentType]
        niters = int(self.niters)  # pyright:ignore[reportArgumentType]
        phase_factor = bool(self.phase_factor)
        p0 = None
        if self.guided:
            guided_result: UnbinnedFitResult = pickle.load(
                Path(self.input()[-1][0].path).open('rb')
            )
            p0 = guided_result.status.x

        fit_result = fit_unbinned(
            Wave.decode_waves(waves),
            analysis_path_set,
            p0=p0,
            iters=niters,
            phase_factor=phase_factor,
        )

        pickle.dump(fit_result, output_fit_path.open('wb'))
