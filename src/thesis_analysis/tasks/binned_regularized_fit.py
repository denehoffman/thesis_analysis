import pickle
from pathlib import Path
from typing import final, override

import luigi

from thesis_analysis.logger import logger
from thesis_analysis.paths import Paths
from thesis_analysis.pwa import (
    BinnedFitResult,
    fit_binned_regularized,
)
from thesis_analysis.tasks.binned_fit import BinnedFit


@final
class BinnedRegularizedFit(luigi.Task):
    waves = luigi.IntParameter()
    chisqdof = luigi.FloatParameter()
    ksb_costheta = luigi.FloatParameter()
    cut_baryons = luigi.OptionalBoolParameter(True)
    splot_method = luigi.Parameter()
    nsig = luigi.IntParameter()
    nbkg = luigi.IntParameter()
    niters = luigi.IntParameter(default=3, significant=False)
    phase_factor = luigi.BoolParameter(default=False)
    lda = luigi.FloatParameter()
    gamma = luigi.FloatParameter()

    resources = {'fit': 1}

    @override
    def requires(self):
        logger.debug('checking dependencies for binned regularized fit')
        return [
            BinnedFit(
                self.waves,
                self.chisqdof,
                self.ksb_costheta,
                self.cut_baryons,
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
                / f'binned_regularized_fit_chisqdof_{self.chisqdof:.1f}_ksb_costheta_{self.ksb_costheta:.2f}{"mesons" if self.cut_baryons else "baryons"}_splot_{self.splot_method}_{self.nsig}s_{self.nbkg}b{"_phase_factor" if self.phase_factor else ""}_waves{self.waves}_lda_{self.lda}_gamma_{self.gamma}.pkl'
            ),
        ]

    @override
    def run(self):
        logger.info(f'Beginning Binned Regularized Fit (λ={self.lda})')
        binned_fit_result: BinnedFitResult = pickle.load(
            Path(self.input()[0][0].path).open('rb')
        )

        output_fit_path = Path(str(self.output()[0].path))
        output_fit_path.parent.mkdir(parents=True, exist_ok=True)

        niters = int(self.niters)  # pyright:ignore[reportArgumentType]
        lda = float(self.lda)  # pyright:ignore[reportArgumentType]
        gamma = float(self.gamma)  # pyright:ignore[reportArgumentType]

        fit_result = fit_binned_regularized(
            binned_fit_result,
            lda=lda,
            gamma=gamma,
            iters=niters,
        )

        pickle.dump(fit_result, output_fit_path.open('wb'))
