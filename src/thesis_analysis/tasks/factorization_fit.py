import pickle
from pathlib import Path

import luigi
import numpy as np
from thesis_analysis import root_io
from thesis_analysis.constants import (
    RUN_PERIODS,
    SPLOT_CONTROL,
    get_branch,
)
from thesis_analysis.paths import Paths
from thesis_analysis.splot import (
    run_factorization_fits,
    run_factorization_fits_mc,
)
from thesis_analysis.tasks.chisqdof import ChiSqDOF


class FactorizationFit(luigi.Task):
    data_type = luigi.Parameter()
    chisqdof = luigi.FloatParameter()
    n_quantiles = luigi.IntParameter()

    def requires(self):
        return [
            ChiSqDOF(self.data_type, run_period, self.chisqdof)
            for run_period in RUN_PERIODS
        ]

    def output(self):
        return [
            luigi.LocalTarget(
                Paths.fits
                / f'factorization_{self.data_type}_chisqdof_{self.chisqdof:.1f}_{self.n_quantiles}_quantiles.pkl'
            ),
        ]

    def run(self):
        input_data_paths = [
            Path(self.input()[i][0].path) for i in range(len(RUN_PERIODS))
        ]
        output_fit_path = Path(self.output()[0].path)
        output_fit_path.parent.mkdir(parents=True, exist_ok=True)

        n_quantiles = int(self.n_quantiles)  # type: ignore
        data_dfs = {
            i: root_io.get_branches(
                input_data_paths[i],
                [
                    get_branch('RFL1'),
                    get_branch('RFL2'),
                    get_branch('Weight'),
                    get_branch(SPLOT_CONTROL),
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
            SPLOT_CONTROL: np.concatenate(
                [data_dfs[i][SPLOT_CONTROL] for i in range(len(RUN_PERIODS))]
            ),
        }

        if self.data_type in ['accmc', 'bkgmc']:
            fit_result = run_factorization_fits_mc(
                data_df['RFL1'],
                data_df['RFL2'],
                data_df['Weight'],
                data_df[SPLOT_CONTROL],
                bins=n_quantiles,
            )
        else:
            fit_result = run_factorization_fits(
                data_df['RFL1'],
                data_df['RFL2'],
                data_df['Weight'],
                data_df[SPLOT_CONTROL],
                bins=n_quantiles,
            )
        pickle.dump(fit_result, output_fit_path.open('wb'))
