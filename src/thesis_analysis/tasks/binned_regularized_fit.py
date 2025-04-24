import pickle
from pathlib import Path
from typing import final, override

import luigi

from thesis_analysis.paths import Paths
from thesis_analysis.pwa import (
    BinnedFitResult,
    fit_guided_regularized,
)
from thesis_analysis.tasks.binned_fit import BinnedFit


@final
class BinnedRegulariedFit(luigi.Task):
    waves = luigi.IntParameter()
    chisqdof = luigi.FloatParameter()
    splot_method = luigi.Parameter()
    nsig = luigi.IntParameter()
    nbkg = luigi.IntParameter()
    niters = luigi.IntParameter(default=3, significant=False)
    phase_factor = luigi.BoolParameter(default=False)
    lda = luigi.FloatParameter()

    resources = {'fit': 1}

    @override
    def requires(self):
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
        ]

    @override
    def output(self):
        return [
            luigi.LocalTarget(
                Paths.fits
                / f'binned_regularized_fit_chisqdof_{self.chisqdof:.1f}_splot_{self.splot_method}_{self.nsig}s_{self.nbkg}b{"_phase_factor" if self.phase_factor else ""}_waves{self.waves}_lda_{self.lda}.pkl'
            ),
        ]

    @override
    def run(self):
        binned_fit_result: BinnedFitResult = pickle.load(
            Path(self.input()[0][0].path).open('rb')
        )

        output_fit_path = Path(str(self.output()[0].path))
        output_fit_path.parent.mkdir(parents=True, exist_ok=True)

        niters = int(self.niters)  # pyright:ignore[reportArgumentType]
        lda = float(self.lda)  # pyright:ignore[reportArgumentType]

        fit_result = fit_guided_regularized(
            binned_fit_result,
            lda=lda,
            iters=niters,
        )

        pickle.dump(fit_result, output_fit_path.open('wb'))
