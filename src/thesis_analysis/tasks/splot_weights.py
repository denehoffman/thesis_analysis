from pathlib import Path

import luigi
from numpy.typing import NDArray

from thesis_analysis import root_io
from thesis_analysis.constants import get_branch
from thesis_analysis.tasks.chisqdof import ChiSqDOF
from thesis_analysis.tasks.splot_fit import SPlotFit
from thesis_analysis.tasks.splot_plot import SPlotPlot
from thesis_analysis.utils import SPlotFitResult, get_sweights


class SPlotWeights(luigi.Task):
    data_type = luigi.Parameter()
    run_period = luigi.Parameter()
    chisqdof = luigi.FloatParameter()
    splot_method = luigi.Parameter()
    nsig = luigi.IntParameter()
    nbkg = luigi.IntParameter()

    def requires(self):
        return [
            ChiSqDOF(self.data_type, self.run_period, self.chisqdof),
            SPlotFit(
                self.data_type,
                self.chisqdof,
                self.splot_method,
                self.nsig,
                self.nbkg,
            ),
            SPlotPlot(
                self.data_type,
                self.chisqdof,
                self.splot_method,
                self.nsig,
                self.nbkg,
            ),
        ]

    def output(self):
        input_path = Path(self.input()[0][0].path)
        return [
            luigi.LocalTarget(
                input_path.parent
                / f'{input_path.stem}_splot_{self.splot_method}_{self.nsig}s_{self.nbkg}b.root'
            )
        ]

    def run(self):
        input_path = Path(self.input()[0][0].path)
        input_fit_path = Path(self.input()[1][0].path)
        output_path = Path(self.output()[0].path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data_df = root_io.get_branches(
            input_path,
            [get_branch('RFL1'), get_branch('RFL2'), get_branch('Weight')],
        )
        fit_result = SPlotFitResult.load(input_fit_path)
        weights = get_sweights(
            fit_result,
            data_df['RFL1'],
            data_df['RFL2'],
            data_df['Weight'],
            nsig=self.nsig,  # type: ignore
            nbkg=self.nbkg,  # type: ignore
        )

        branches = [
            get_branch('Weight'),
        ]

        def process(
            i: int,
            weight: NDArray,
        ) -> bool:
            weight[0] = weights[i]
            return weight[0] != 0.0

        root_io.process_root_tree(
            input_path,
            output_path,
            branches,
            process,
        )
