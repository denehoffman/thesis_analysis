import pickle
from pathlib import Path
from typing import final, override

import luigi

from thesis_analysis.paths import Paths
from thesis_analysis.pwa import (
    BinnedFitResultUncertainty,
    fit_guided,
)
from thesis_analysis.tasks.binned_fit_uncertainty import BinnedFitUncertainty


@final
class GuidedFit(luigi.Task):
    waves = luigi.IntParameter()
    chisqdof = luigi.FloatParameter()
    ksb_costheta = luigi.FloatParameter()
    cut_baryons = luigi.OptionalBoolParameter(True)
    splot_method = luigi.Parameter()
    nsig = luigi.IntParameter()
    nbkg = luigi.IntParameter()
    niters = luigi.IntParameter(default=3, significant=False)
    phase_factor = luigi.BoolParameter(default=False)
    uncertainty = luigi.Parameter(default='sqrt')
    bootstrap_mode = luigi.Parameter(default='CI-BC')

    resources = {'fit': 1}

    @override
    def requires(self):
        return [
            BinnedFitUncertainty(
                self.waves,
                self.chisqdof,
                self.ksb_costheta,
                self.cut_baryons,
                self.splot_method,
                self.nsig,
                self.nbkg,
                self.niters,
                self.phase_factor,
                self.uncertainty,
            ),
        ]

    @override
    def output(self):
        return [
            luigi.LocalTarget(
                Paths.fits
                / f'guided_fit_chisqdof_{self.chisqdof:.1f}_ksb_costheta_{self.ksb_costheta:.2f}{"mesons" if self.cut_baryons else "baryons"}_splot_{self.splot_method}_{self.nsig}s_{self.nbkg}b{"_phase_factor" if self.phase_factor else ""}_waves{self.waves}_uncertainty_{self.uncertainty}{f"-{self.bootstrap_mode}" if str(self.uncertainty) == "bootstrap" else ""}.pkl'
            ),
        ]

    @override
    def run(self):
        binned_fit_result_uncertainty: BinnedFitResultUncertainty = pickle.load(
            Path(self.input()[0][0].path).open('rb')
        )

        output_fit_path = Path(str(self.output()[0].path))
        output_fit_path.parent.mkdir(parents=True, exist_ok=True)

        niters = int(self.niters)  # pyright:ignore[reportArgumentType]
        bootstrap_mode = str(self.bootstrap_mode)

        fit_result = fit_guided(
            binned_fit_result_uncertainty,
            bootstrap_mode=bootstrap_mode,
            iters=niters,
        )

        pickle.dump(fit_result, output_fit_path.open('wb'))
