import pickle
from pathlib import Path
from typing import final, override

import luigi

from thesis_analysis.constants import NBINS, RANGE, RUN_PERIODS
from thesis_analysis.logger import logger
from thesis_analysis.paths import Paths
from thesis_analysis.pwa import (
    Binning,
    FullPathSet,
    SinglePathSet,
    fit_binned,
)
from thesis_analysis.tasks.chisqdof import ChiSqDOF
from thesis_analysis.tasks.splot_weights import SPlotWeights
from thesis_analysis.wave import Wave


@final
class BinnedFit(luigi.Task):
    waves = luigi.IntParameter()
    chisqdof = luigi.FloatParameter()
    ksb_costheta = luigi.FloatParameter()
    cut_baryons = luigi.OptionalBoolParameter(True)
    splot_method = luigi.Parameter()
    nsig = luigi.IntParameter()
    nbkg = luigi.IntParameter()
    niters = luigi.IntParameter(default=3, significant=False)
    phase_factor = luigi.BoolParameter(default=False)

    resources = {'fit': 1}

    @override
    def requires(self):
        return [
            SPlotWeights(
                'data',
                run_period,
                self.chisqdof,
                self.ksb_costheta,
                self.cut_baryons,
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
                self.ksb_costheta,
                self.cut_baryons,
            )
            for run_period in RUN_PERIODS
        ]

    @override
    def output(self):
        return [
            luigi.LocalTarget(
                Paths.fits
                / f'binned_fit_chisqdof_{self.chisqdof:.1f}_ksb_costheta_{self.ksb_costheta:.2f}{"mesons" if self.cut_baryons else "baryons"}_splot_{self.splot_method}_{self.nsig}s_{self.nbkg}b{"_phase_factor" if self.phase_factor else ""}_waves{self.waves}.pkl'
            ),
        ]

    @override
    def run(self):
        logger.info('Beginning Binned Fit')
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

        binning = Binning(NBINS, RANGE)

        fit_result = fit_binned(
            Wave.decode_waves(waves),
            analysis_path_set,
            binning,
            iters=niters,
            phase_factor=phase_factor,
        )

        pickle.dump(fit_result, output_fit_path.open('wb'))
