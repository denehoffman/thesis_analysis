import pickle
from pathlib import Path
from typing import final, override

import luigi
import numpy as np

from thesis_analysis import root_io
from thesis_analysis.constants import (
    NSIG_BINS,
    RUN_PERIODS,
    SPLOT_CONTROL,
    get_branch,
)
from thesis_analysis.logger import logger
from thesis_analysis.paths import Paths
from thesis_analysis.splot import (
    SPlotFitFailure,
    run_splot_fit,
    run_splot_fit_exp,
)
from thesis_analysis.tasks.chisqdof import ChiSqDOF


@final
class SPlotFit(luigi.Task):
    data_type = luigi.Parameter()
    chisqdof = luigi.FloatParameter()
    splot_method = luigi.Parameter()
    nsig = luigi.IntParameter()
    nbkg = luigi.IntParameter()

    @override
    def requires(self):
        return [
            ChiSqDOF(data_type, run_period, self.chisqdof)
            for run_period in RUN_PERIODS
            for data_type in [self.data_type, 'accmc', 'bkgmc']
        ]

    @override
    def output(self):
        return [
            luigi.LocalTarget(
                Paths.fits
                / f'splot_fit_{self.data_type}_chisqdof_{self.chisqdof:.1f}_{self.splot_method}_{self.nsig}s_{self.nbkg}b.pkl'
            )
        ]

    @override
    def run(self):
        input_data_paths = [
            Path(self.input()[3 * i + 0][0].path)
            for i in range(len(RUN_PERIODS))
        ]
        input_accmc_paths = [
            Path(self.input()[3 * i + 1][0].path)
            for i in range(len(RUN_PERIODS))
        ]
        input_bkgmc_paths = [
            Path(self.input()[3 * i + 2][0].path)
            for i in range(len(RUN_PERIODS))
        ]
        output_path = Path(self.output()[0].path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        nsig = int(self.nsig)  # pyright:ignore[reportArgumentType]
        nbkg = int(self.nbkg)  # pyright:ignore[reportArgumentType]

        branches = [
            get_branch('RFL1'),
            get_branch('RFL2'),
            get_branch('Weight'),
        ]
        flat_data = root_io.concatenate_branches(
            input_data_paths, branches, root=False
        )
        flat_accmc = root_io.concatenate_branches(
            input_accmc_paths, branches, root=False
        )
        flat_bkgmc = root_io.concatenate_branches(
            input_bkgmc_paths, branches, root=False
        )
        try:
            if not str(self.splot_method).startswith('D') and not str(
                self.splot_method
            ).startswith('E'):
                fit_result = run_splot_fit_exp(
                    flat_data['RFL1'],
                    flat_data['RFL2'],
                    flat_data['Weight'],
                    flat_accmc['RFL1'],
                    flat_accmc['RFL2'],
                    flat_accmc[SPLOT_CONTROL],
                    flat_accmc['Weight'],
                    flat_bkgmc['RFL1'],
                    flat_bkgmc['RFL2'],
                    flat_bkgmc[SPLOT_CONTROL],
                    flat_bkgmc['Weight'],
                    nsig=nsig,
                    nbkg=nbkg,
                    fixed_sig=str(self.splot_method) != 'A',
                    fixed_bkg=str(self.splot_method) == 'C',
                )
            else:
                fit_result = run_splot_fit(
                    flat_data['RFL1'],
                    flat_data['RFL2'],
                    flat_data['Weight'],
                    flat_accmc['RFL1'],
                    flat_accmc['RFL2'],
                    flat_accmc[SPLOT_CONTROL],
                    flat_accmc['Weight'],
                    flat_bkgmc['RFL1'],
                    flat_bkgmc['RFL2'],
                    flat_bkgmc[SPLOT_CONTROL],
                    flat_bkgmc['Weight'],
                    nsig=nsig,
                    nsig_bins=NSIG_BINS,
                    nbkg=nbkg,
                    fixed_bkg=str(self.splot_method) == 'E',
                )
        except Exception as e:
            logger.error(e)
            fit_result = SPlotFitFailure()

        pickle.dump(fit_result, output_path.open('wb'))
