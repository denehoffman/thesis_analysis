import pickle
from pathlib import Path
from typing import final

import luigi
import numpy as np
from typing_extensions import override

from thesis_analysis import root_io
from thesis_analysis.constants import (
    NSIG_BINS_DE,
    RUN_PERIODS,
    SPLOT_CONTROL,
    get_branch,
)
from thesis_analysis.paths import Paths
from thesis_analysis.splot import (
    SPlotFitFailure,
    run_splot_fit,
    run_splot_fit_d,
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

        data_dfs = {
            i: root_io.get_branches(
                input_data_paths[i],
                [
                    get_branch('RFL1'),
                    get_branch('RFL2'),
                    get_branch('Weight'),
                ],
            )
            for i in range(len(RUN_PERIODS))
        }
        data_df = {
            'RFL1': np.concatenate(
                [data_dfs[i]['RFL1'] for i in range(len(RUN_PERIODS))]
            ),
            'RFL2': np.concatenate(
                [data_dfs[i]['RFL2'] for i in range(len(RUN_PERIODS))]
            ),
            'Weight': np.concatenate(
                [data_dfs[i]['Weight'] for i in range(len(RUN_PERIODS))]
            ),
        }

        accmc_dfs = {
            i: root_io.get_branches(
                input_accmc_paths[i],
                [
                    get_branch('RFL1'),
                    get_branch('RFL2'),
                    get_branch('Weight'),
                    get_branch(SPLOT_CONTROL),
                ],
            )
            for i in range(len(RUN_PERIODS))
        }
        accmc_df = {
            'RFL1': np.concatenate(
                [accmc_dfs[i]['RFL1'] for i in range(len(RUN_PERIODS))]
            ),
            'RFL2': np.concatenate(
                [accmc_dfs[i]['RFL2'] for i in range(len(RUN_PERIODS))]
            ),
            'Weight': np.concatenate(
                [accmc_dfs[i]['Weight'] for i in range(len(RUN_PERIODS))]
            ),
            SPLOT_CONTROL: np.concatenate(
                [accmc_dfs[i][SPLOT_CONTROL] for i in range(len(RUN_PERIODS))]
            ),
        }

        bkgmc_dfs = {
            i: root_io.get_branches(
                input_bkgmc_paths[i],
                [
                    get_branch('RFL1'),
                    get_branch('RFL2'),
                    get_branch('Weight'),
                    get_branch(SPLOT_CONTROL),
                ],
            )
            for i in range(len(RUN_PERIODS))
        }
        bkgmc_df = {
            'RFL1': np.concatenate(
                [bkgmc_dfs[i]['RFL1'] for i in range(len(RUN_PERIODS))]
            ),
            'RFL2': np.concatenate(
                [bkgmc_dfs[i]['RFL2'] for i in range(len(RUN_PERIODS))]
            ),
            'Weight': np.concatenate(
                [bkgmc_dfs[i]['Weight'] for i in range(len(RUN_PERIODS))]
            ),
            SPLOT_CONTROL: np.concatenate(
                [bkgmc_dfs[i][SPLOT_CONTROL] for i in range(len(RUN_PERIODS))]
            ),
        }

        try:
            if not str(self.splot_method).startswith('D') and not str(
                self.splot_method
            ).startswith('E'):
                fit_result = run_splot_fit(
                    data_df['RFL1'],
                    data_df['RFL2'],
                    data_df['Weight'],
                    accmc_df['RFL1'],
                    accmc_df['RFL2'],
                    accmc_df[SPLOT_CONTROL],
                    accmc_df['Weight'],
                    bkgmc_df['RFL1'],
                    bkgmc_df['RFL2'],
                    bkgmc_df[SPLOT_CONTROL],
                    bkgmc_df['Weight'],
                    nsig=nsig,
                    nbkg=nbkg,
                    fixed_sig=str(self.splot_method) != 'A',
                    fixed_bkg=str(self.splot_method) == 'C',
                )
            else:
                fit_result = run_splot_fit_d(
                    data_df['RFL1'],
                    data_df['RFL2'],
                    data_df['Weight'],
                    accmc_df['RFL1'],
                    accmc_df['RFL2'],
                    accmc_df[SPLOT_CONTROL],
                    accmc_df['Weight'],
                    bkgmc_df['RFL1'],
                    bkgmc_df['RFL2'],
                    bkgmc_df[SPLOT_CONTROL],
                    bkgmc_df['Weight'],
                    nsig=nsig,
                    nsig_bins=NSIG_BINS_DE,
                    nbkg=nbkg,
                    fixed_bkg=str(self.splot_method) == 'E',
                )
        except Exception:
            fit_result = SPlotFitFailure()

        pickle.dump(fit_result, output_path.open('wb'))
