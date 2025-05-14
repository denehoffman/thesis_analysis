import pickle
from pathlib import Path
from typing import final, override

import luigi
import numpy as np

from thesis_analysis import root_io
from thesis_analysis.constants import (
    MC_TYPES,
    NSIG_BINS,
    RUN_PERIODS,
    SPLOT_CONTROL,
    get_branch,
)
from thesis_analysis.paths import Paths
from thesis_analysis.splot import (
    run_factorization_fits,
    run_factorization_fits_mc,
)
from thesis_analysis.tasks.baryon_cut import BaryonCut


@final
class FactorizationFit(luigi.Task):
    data_type = luigi.Parameter()
    chisqdof = luigi.FloatParameter()
    ksb_costheta = luigi.FloatParameter()
    cut_baryons = luigi.OptionalBoolParameter(True)
    n_quantiles = luigi.IntParameter()

    @override
    def requires(self):
        reqs = [
            BaryonCut(
                self.data_type,
                run_period,
                self.chisqdof,
                self.ksb_costheta,
                self.cut_baryons,
            )
            for run_period in RUN_PERIODS
        ]
        if self.data_type == 'data':
            reqs += [
                BaryonCut(
                    'accmc',
                    run_period,
                    self.chisqdof,
                    self.ksb_costheta,
                    self.cut_baryons,
                )
                for run_period in RUN_PERIODS
            ]
        return reqs

    @override
    def output(self):
        return [
            luigi.LocalTarget(
                Paths.fits
                / f'factorization_{self.data_type}_chisqdof_{self.chisqdof:.1f}_ksb_costheta_{self.ksb_costheta:.2f}{"mesons" if self.cut_baryons else "baryons"}_{self.n_quantiles}_quantiles.pkl'
            ),
        ]

    @override
    def run(self):
        input_data_paths = [
            Path(self.input()[i][0].path) for i in range(len(RUN_PERIODS))
        ]
        output_fit_path = Path(self.output()[0].path)
        output_fit_path.parent.mkdir(parents=True, exist_ok=True)

        n_quantiles = int(self.n_quantiles)  # pyright:ignore[reportArgumentType]
        data_dfs = {
            i: root_io.get_branches(
                input_data_paths[i],
                [
                    get_branch('RFL1'),
                    get_branch('RFL2'),
                    get_branch('Weight'),
                    get_branch(SPLOT_CONTROL),
                ],
                root=False,
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

        if self.data_type in MC_TYPES:
            fit_result = run_factorization_fits_mc(
                data_df['RFL1'],
                data_df['RFL2'],
                data_df['Weight'],
                data_df[SPLOT_CONTROL],
                bins=n_quantiles,
            )
        else:
            input_sigmc_paths = [
                Path(self.input()[len(RUN_PERIODS) + i][0].path)
                for i in range(len(RUN_PERIODS))
            ]
            branches = [
                get_branch('RFL1'),
                get_branch('RFL2'),
                get_branch('Weight'),
                get_branch(SPLOT_CONTROL),
            ]
            sigmc_df = root_io.concatenate_branches(
                input_sigmc_paths,
                branches,
                root=False,
            )
            fit_result = run_factorization_fits(
                data_df['RFL1'],
                data_df['RFL2'],
                data_df['Weight'],
                data_df[SPLOT_CONTROL],
                sigmc_df['RFL1'],
                sigmc_df['RFL2'],
                sigmc_df['Weight'],
                sigmc_df[SPLOT_CONTROL],
                bins=n_quantiles,
                nsig_bins=NSIG_BINS,
            )

        pickle.dump(fit_result, output_fit_path.open('wb'))
