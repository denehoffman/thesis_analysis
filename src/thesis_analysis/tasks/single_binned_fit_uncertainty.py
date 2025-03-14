import pickle
from pathlib import Path
from typing import final

import luigi
from typing_extensions import override

from thesis_analysis.paths import Paths
from thesis_analysis.pwa import (
    BinnedFitResult,
    BinnedFitResultUncertainty,
    calculate_bootstrap_uncertainty_binned,
    calculate_mcmc_uncertainty_binned,
)
from thesis_analysis.tasks.single_binned_fit import SingleBinnedFit


@final
class SingleBinnedFitUncertainty(luigi.Task):
    waves = luigi.IntParameter()
    run_period = luigi.Parameter()
    chisqdof = luigi.FloatParameter()
    splot_method = luigi.Parameter()
    nsig = luigi.IntParameter()
    nbkg = luigi.IntParameter()
    niters = luigi.IntParameter(default=3, significant=False)
    phase_factor = luigi.BoolParameter(default=False)
    uncertainty = luigi.Parameter(default='sqrt')

    resources = {'fit': 1}

    @override
    def requires(self):
        return [
            SingleBinnedFit(
                self.waves,
                self.run_period,
                self.chisqdof,
                self.splot_method,
                self.nsig,
                self.nbkg,
                self.niters,
                self.phase_factor,
            )
        ]

    @override
    def output(self):
        return [
            luigi.LocalTarget(
                Paths.fits
                / f'binned_fit_{self.run_period}_chisqdof_{self.chisqdof:.1f}_splot_{self.splot_method}_{self.nsig}s_{self.nbkg}b{"_phase_factor" if self.phase_factor else ""}_waves{self.waves}_uncertainty_{self.uncertainty}.pkl'
            ),
        ]

    @override
    def run(self):
        binned_fit_result: BinnedFitResult = pickle.load(
            Path(self.input()[0][0].path).open('rb')
        )

        output_unc_path = Path(str(self.output()[0].path))
        output_unc_path.parent.mkdir(parents=True, exist_ok=True)

        uncertainty = str(self.uncertainty)

        if uncertainty == 'bootstrap':
            output = calculate_bootstrap_uncertainty_binned(binned_fit_result)
        elif uncertainty == 'mcmc':
            output = calculate_mcmc_uncertainty_binned(binned_fit_result)
        else:
            output = BinnedFitResultUncertainty(
                [], binned_fit_result, uncertainty='sqrt'
            )

        pickle.dump(output, output_unc_path.open('wb'))
