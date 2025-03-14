import pickle
from pathlib import Path
from typing import final

import luigi
from typing_extensions import override

from thesis_analysis.constants import RUN_PERIODS
from thesis_analysis.paths import Paths
from thesis_analysis.pwa import (
    BinnedFitResult,
    BinnedFitResultUncertainty,
    fit_guided,
)
from thesis_analysis.tasks.binned_fit import BinnedFit
from thesis_analysis.tasks.binned_fit_uncertainty import BinnedFitUncertainty
from thesis_analysis.tasks.single_binned_fit import SingleBinnedFit
from thesis_analysis.tasks.single_binned_fit_uncertainty import (
    SingleBinnedFitUncertainty,
)


@final
class GuidedFit(luigi.Task):
    waves = luigi.IntParameter()
    chisqdof = luigi.FloatParameter()
    splot_method = luigi.Parameter()
    nsig = luigi.IntParameter()
    nbkg = luigi.IntParameter()
    niters = luigi.IntParameter(default=3, significant=False)
    averaged = luigi.BoolParameter(default=False)
    phase_factor = luigi.BoolParameter(default=False)
    uncertainty = luigi.Parameter(default='sqrt')

    resources = {'fit': 1}

    @override
    def requires(self):
        if self.averaged:
            return [
                BinnedFit(
                    self.waves,
                    self.chisqdof,
                    self.splot_method,
                    self.nsig,
                    self.nbkg,
                    self.niters,
                    self.phase_factor,
                ),
                BinnedFitUncertainty(
                    self.waves,
                    self.chisqdof,
                    self.splot_method,
                    self.nsig,
                    self.nbkg,
                    self.niters,
                    self.phase_factor,
                    self.uncertainty,
                ),
            ]
        else:
            return [
                SingleBinnedFit(
                    self.waves,
                    run_period,
                    self.chisqdof,
                    self.splot_method,
                    self.nsig,
                    self.nbkg,
                )
                for run_period in RUN_PERIODS
            ] + [
                SingleBinnedFitUncertainty(
                    self.waves,
                    run_period,
                    self.chisqdof,
                    self.splot_method,
                    self.nsig,
                    self.nbkg,
                    self.uncertainty,
                )
                for run_period in RUN_PERIODS
            ]

    @override
    def output(self):
        return [
            luigi.LocalTarget(
                Paths.fits
                / f'guided_fit_chisqdof_{self.chisqdof:.1f}_splot_{self.splot_method}_{self.nsig}s_{self.nbkg}b{"_averaged" if self.averaged else ""}{"_phase_factor" if self.phase_factor else ""}_waves{self.waves}_uncertainty_{self.uncertainty}.pkl'
            ),
        ]

    @override
    def run(self):
        binned_fit_result: BinnedFitResult | list[BinnedFitResult] = (
            pickle.load(Path(self.input()[0][0].path).open('rb'))
            if self.averaged
            else [
                pickle.load(Path(self.input()[i][0].path).open('rb'))
                for i in range(len(RUN_PERIODS))
            ]
        )

        binned_fit_result_uncertainty: (
            BinnedFitResultUncertainty | list[BinnedFitResultUncertainty]
        ) = (
            pickle.load(Path(self.input()[1][0].path).open('rb'))
            if self.averaged
            else [
                pickle.load(
                    Path(self.input()[len(RUN_PERIODS) + i][0].path).open('rb')
                )
                for i in range(len(RUN_PERIODS))
            ]
        )

        output_fit_path = Path(str(self.output()[0].path))
        output_fit_path.parent.mkdir(parents=True, exist_ok=True)

        niters = int(self.niters)  # pyright:ignore[reportArgumentType]

        if isinstance(binned_fit_result, list) and isinstance(
            binned_fit_result_uncertainty, list
        ):
            pass  # TODO: averaged guided fit

        if not isinstance(binned_fit_result, list) and not isinstance(
            binned_fit_result_uncertainty, list
        ):
            fit_result = fit_guided(
                binned_fit_result,
                binned_fit_result_uncertainty,
                iters=niters,
            )

            pickle.dump(fit_result, output_fit_path.open('wb'))
