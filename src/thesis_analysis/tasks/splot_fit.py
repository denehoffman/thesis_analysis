import pickle
from pathlib import Path

import luigi

from thesis_analysis import root_io
from thesis_analysis.constants import SPLOT_CONTROL, get_branch
from thesis_analysis.paths import Paths
from thesis_analysis.tasks.chisqdof import ChiSqDOF
from thesis_analysis.utils import run_splot_fit


class SPlotFit(luigi.Task):
    data_type = luigi.Parameter()
    run_period = luigi.Parameter()
    chisqdof = luigi.FloatParameter()
    splot_method = luigi.Parameter()
    nsig = luigi.IntParameter()
    nbkg = luigi.IntParameter()

    def requires(self):
        return [
            ChiSqDOF(self.data_type, self.run_period, self.chisqdof),
            ChiSqDOF('accmc', self.run_period, self.chisqdof),
            ChiSqDOF('bkgmc', self.run_period, self.chisqdof),
        ]

    def output(self):
        return [
            luigi.LocalTarget(
                Paths.fits
                / f'splot_fit_{self.data_type}_{self.run_period}_chisqdof_{self.chisqdof:.1f}_{self.splot_method}_{self.nsig}s_{self.nbkg}b.root'
            )
        ]

    def run(self):
        input_data_path = Path(self.input()[0][0].path)
        input_accmc_path = Path(self.input()[1][0].path)
        input_bkgmc_path = Path(self.input()[2][0].path)
        output_path = Path(self.output()[0].path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        nsig = int(self.nsig)  # type: ignore
        nbkg = int(self.nbkg)  # type: ignore

        data_df = root_io.get_branches(
            input_data_path,
            [get_branch('RFL1'), get_branch('RFL2'), get_branch('Weight')],
        )
        accmc_df = root_io.get_branches(
            input_accmc_path,
            [
                get_branch('RFL1'),
                get_branch('RFL2'),
                get_branch('Weight'),
                get_branch(SPLOT_CONTROL),
            ],
        )
        bkgmc_df = root_io.get_branches(
            input_bkgmc_path,
            [
                get_branch('RFL1'),
                get_branch('RFL2'),
                get_branch('Weight'),
                get_branch(SPLOT_CONTROL),
            ],
        )

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
            fixed_sig=self.splot_method != 'A',
            fixed_bkg=self.splot_method == 'C',
        )

        pickle.dump(fit_result, output_path.open('wb'))
