# pyright: reportUnnecessaryComparison=false
from pathlib import Path
from typing import final, override

import luigi
import matplotlib.pyplot as plt
import matplotlib.style as mpl_style

import thesis_analysis.colors as colors
from thesis_analysis import root_io
from thesis_analysis.constants import (
    BRANCH_NAME_TO_LATEX,
    DATA_TYPE_TO_LATEX,
    RUN_PERIODS,
    get_branch,
)
from thesis_analysis.utils import get_plot_paths, get_plot_requirements


@final
class ChiSqDOFPlot(luigi.Task):
    data_type = luigi.Parameter()
    bins = luigi.IntParameter()
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
                'chisqdof',
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
        output_path = self.output()[0].path

        bins = int(self.bins)  # pyright:ignore[reportArgumentType]

        branches = [
            get_branch('ChiSqDOF'),
            get_branch('Weight'),
        ]
        flat_data = root_io.concatenate_branches(
            input_paths, branches, root=False
        )
        mpl_style.use('thesis_analysis.thesis')
        fig, ax = plt.subplots()
        max_range = (
            float(self.chisqdof)  # pyright:ignore[reportArgumentType]
            if (
                self.splot_method is not None
                and self.nsig is not None
                and self.nbkg is not None
            )
            else 10.0
        )

        ax.hist(
            flat_data['ChiSqDOF'],
            weights=flat_data['Weight'],
            bins=bins,
            range=(0.0, max_range),
            color=colors.blue,
            label=DATA_TYPE_TO_LATEX[str(self.data_type)],
        )

        if self.chisqdof is not None and (
            self.splot_method is None
            and self.nsig is None
            and self.nbkg is None
        ):
            chisqdof = float(self.chisqdof)  # pyright:ignore[reportArgumentType]
            ax.axvline(chisqdof, color=colors.red, ls=':')  # type: ignore

        ax.set_xlabel(BRANCH_NAME_TO_LATEX['ChiSqDOF'])
        bin_width = 1.0 / bins
        ax.set_ylabel(f'Counts / {bin_width:.2f}')
        ax.legend()
        fig.savefig(output_path)
        plt.close()
