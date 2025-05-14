from pathlib import Path
from typing import final, override

import luigi
import matplotlib.pyplot as plt
import matplotlib.style as mpl_style

from thesis_analysis import colors, root_io
from thesis_analysis.constants import (
    BRANCH_NAME_TO_LATEX,
    BRANCH_NAME_TO_LATEX_UNITS,
    DATA_TYPE_TO_LATEX,
    RUN_PERIODS,
    get_branch,
)
from thesis_analysis.utils import get_plot_paths, get_plot_requirements


@final
class RFLPlot(luigi.Task):
    data_type = luigi.Parameter()
    original = luigi.BoolParameter(False)
    chisqdof = luigi.OptionalFloatParameter(None)
    ksb_costheta = luigi.OptionalFloatParameter(None)
    cut_baryons = luigi.OptionalBoolParameter(True)
    splot_method = luigi.OptionalParameter(None)
    nsig = luigi.OptionalIntParameter(None)
    nbkg = luigi.OptionalIntParameter(None)

    @override
    def requires(self):
        return get_plot_requirements(
            data_type=self.data_type,
            original=self.original,
            chisqdof=self.chisqdof,
            ksb_costheta=self.ksb_costheta,
            cut_baryons=self.cut_baryons,
            splot_method=self.splot_method,
            nsig=self.nsig,
            nbkg=self.nbkg,
        )

    @override
    def output(self):
        return get_plot_paths(
            [
                'rfl',
            ],
            self.data_type,
            self.original,
            self.chisqdof,
            self.ksb_costheta,
            self.cut_baryons,
            self.splot_method,
            self.nsig,
            self.nbkg,
        )

    @override
    def run(self):
        input_paths = [
            Path(self.input()[i][0].path) for i in range(len(RUN_PERIODS))
        ]
        output_plot_path = Path(self.output()[0].path)
        output_plot_path.parent.mkdir(parents=True, exist_ok=True)

        branches = [
            get_branch('RFL1'),
            get_branch('RFL2'),
            get_branch('Weight'),
        ]
        flat_data = root_io.concatenate_branches(
            input_paths, branches, root=False
        )
        nbins = 100
        mpl_style.use('thesis_analysis.thesis')
        fig, ax = plt.subplots()
        ax.hist(
            flat_data['RFL1'],
            weights=flat_data['Weight'],
            bins=nbins,
            range=(0.0, 0.2),
            color=colors.blue,
            label=DATA_TYPE_TO_LATEX[str(self.data_type)],
        )
        bin_width = 0.2 / nbins
        ax.set_xlabel(
            f'{BRANCH_NAME_TO_LATEX["RFL1"]} ({BRANCH_NAME_TO_LATEX_UNITS["RFL1"]})'
        )
        ax.set_ylabel(
            f'Counts / {bin_width:.3f} {BRANCH_NAME_TO_LATEX_UNITS["RFL1"]}'
        )
        ax.legend()
        fig.savefig(output_plot_path)
        plt.close()
