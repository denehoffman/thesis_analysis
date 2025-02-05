import pickle
from pathlib import Path

import luigi

from thesis_analysis import root_io
from thesis_analysis.constants import (
    SPLOT_CONTROL,
    get_branch,
)
from thesis_analysis.paths import Paths
from thesis_analysis.tasks.chisqdof import ChiSqDOF
from thesis_analysis.utils import (
    run_factorization_fits,
    run_factorization_fits_mc,
)


class FactorizationFit(luigi.Task):
    data_type = luigi.Parameter()
    run_period = luigi.Parameter()
    chisqdof = luigi.FloatParameter()
    n_quantiles = luigi.IntParameter()

    def requires(self):
        return [
            ChiSqDOF(self.data_type, self.run_period, self.chisqdof),
        ]

    def output(self):
        return [
            luigi.LocalTarget(
                Paths.fits
                / f'factorization_{self.data_type}_{self.run_period}_chisqdof_{self.chisqdof:.1f}_{self.n_quantiles}_quantiles.pkl'
            ),
        ]

    def run(self):
        input_data_path = Path(self.input()[0][0].path)
        output_fit_path = Path(self.output()[0].path)
        output_fit_path.parent.mkdir(parents=True, exist_ok=True)

        n_quantiles = int(self.n_quantiles)  # type: ignore

        data_df = root_io.get_branches(
            input_data_path,
            [
                get_branch('RFL1'),
                get_branch('RFL2'),
                get_branch('Weight'),
                get_branch(SPLOT_CONTROL),
            ],
        )

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
