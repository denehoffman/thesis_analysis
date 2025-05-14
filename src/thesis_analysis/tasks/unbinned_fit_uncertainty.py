import pickle
from pathlib import Path
from typing import final, override

import luigi

from thesis_analysis.paths import Paths
from thesis_analysis.pwa import (
    UnbinnedFitResult,
    calculate_bootstrap_uncertainty_unbinned,
)
from thesis_analysis.tasks.unbinned_fit import UnbinnedFit


@final
class UnbinnedFitUncertainty(luigi.Task):
    waves = luigi.IntParameter()
    chisqdof = luigi.FloatParameter()
    ksb_costheta = luigi.FloatParameter()
    cut_baryons = luigi.OptionalBoolParameter(True)
    splot_method = luigi.Parameter()
    nsig = luigi.IntParameter()
    nbkg = luigi.IntParameter()
    niters = luigi.IntParameter(default=3, significant=False)
    guided = luigi.BoolParameter(default=False)
    phase_factor = luigi.BoolParameter(default=False)
    uncertainty = luigi.Parameter(default='bootstrap')
    bootstrap_mode = luigi.Parameter(default='SE')

    resources = {'fit': 1}

    @override
    def requires(self):
        return [
            UnbinnedFit(
                self.waves,
                self.chisqdof,
                self.ksb_costheta,
                self.cut_baryons,
                self.splot_method,
                self.nsig,
                self.nbkg,
                self.niters,
                self.guided,
                self.phase_factor,
                self.uncertainty,
                self.bootstrap_mode,
            )
        ]

    @override
    def output(self):
        return [
            luigi.LocalTarget(
                Paths.fits
                / f'unbinned_fit_chisqdof_{self.chisqdof:.1f}_ksb_costheta_{self.ksb_costheta:.2f}{"mesons" if self.cut_baryons else "baryons"}_splot_{self.splot_method}_{self.nsig}s_{self.nbkg}b{"_guided" if self.guided else ""}{"_phase_factor" if self.phase_factor else ""}_waves{self.waves}_uncertainty_{self.uncertainty}_unbinned_bootstrap.pkl'
            ),
        ]

    @override
    def run(self):
        unbinned_fit_result: UnbinnedFitResult = pickle.load(
            Path(self.input()[0][0].path).open('rb')
        )

        output_unc_path = Path(str(self.output()[0].path))
        output_unc_path.parent.mkdir(parents=True, exist_ok=True)

        output = calculate_bootstrap_uncertainty_unbinned(unbinned_fit_result)

        pickle.dump(output, output_unc_path.open('wb'))
