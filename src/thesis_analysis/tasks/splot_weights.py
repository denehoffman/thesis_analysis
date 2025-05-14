import pickle
from pathlib import Path
from typing import final, override

import luigi
import numpy as np
from numpy.typing import NDArray

from thesis_analysis import root_io
from thesis_analysis.constants import get_branch
from thesis_analysis.splot import get_sweights
from thesis_analysis.tasks.baryon_cut import BaryonCut
from thesis_analysis.tasks.splot_fit import SPlotFit
from thesis_analysis.tasks.splot_plot import SPlotPlot


@final
class SPlotWeights(luigi.Task):
    data_type = luigi.Parameter()
    run_period = luigi.Parameter()
    chisqdof = luigi.FloatParameter()
    ksb_costheta = luigi.FloatParameter()
    cut_baryons = luigi.OptionalBoolParameter(True)
    splot_method = luigi.Parameter()
    nsig = luigi.IntParameter()
    nbkg = luigi.IntParameter()

    @override
    def requires(self):
        return [
            BaryonCut(
                self.data_type,
                self.run_period,
                self.chisqdof,
                self.ksb_costheta,
                self.cut_baryons,
            ),
            SPlotFit(
                self.data_type,
                self.chisqdof,
                self.ksb_costheta,
                self.cut_baryons,
                self.splot_method,
                self.nsig,
                self.nbkg,
            ),
            SPlotPlot(
                self.data_type,
                self.chisqdof,
                self.ksb_costheta,
                self.cut_baryons,
                self.splot_method,
                self.nsig,
                self.nbkg,
            ),
        ]

    @override
    def output(self):
        input_path = Path(self.input()[0][0].path)
        return [
            luigi.LocalTarget(
                input_path.parent
                / f'{input_path.stem}_splot_{self.splot_method}_ksb_costheta_{self.ksb_costheta:.2f}{"mesons" if self.cut_baryons else "baryons"}_{self.nsig}s_{self.nbkg}b.root'
            )
        ]

    @override
    def run(self):
        input_path = Path(self.input()[0][0].path)
        input_fit_path = Path(self.input()[1][0].path)
        output_path = Path(self.output()[0].path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        nsig = int(self.nsig)  # pyright:ignore[reportArgumentType]
        nbkg = int(self.nbkg)  # pyright:ignore[reportArgumentType]

        data_df = root_io.get_branches(
            input_path,
            [get_branch('RFL1'), get_branch('RFL2'), get_branch('Weight')],
        )
        fit_result = pickle.load(input_fit_path.open('rb'))
        weights = get_sweights(
            fit_result,
            data_df['RFL1'],
            data_df['RFL2'],
            data_df['Weight'],
            nsig=nsig,
            nbkg=nbkg,
        )

        branches = [
            get_branch('Weight'),
        ]

        def process(
            i: int,
            weight: NDArray[np.float32],
        ) -> bool:
            weight[0] = weights[i]
            return weight[0] != 0.0

        root_io.process_root_tree(
            input_path,
            output_path,
            branches,
            process,
        )
