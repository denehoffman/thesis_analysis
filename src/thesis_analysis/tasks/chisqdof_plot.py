# pyright: reportUnnecessaryComparison=false
from pathlib import Path
from typing import final, override

import luigi
import matplotlib.pyplot as plt
import matplotlib.style as mpl_style
import numpy as np

import thesis_analysis.colors as colors
from thesis_analysis import root_io
from thesis_analysis.constants import (
    BRANCH_NAME_TO_LATEX,
    DATA_TYPE_TO_LATEX,
    RUN_PERIODS,
    get_branch,
)
from thesis_analysis.paths import Paths
from thesis_analysis.tasks.accid_and_pol import AccidentalsAndPolarization
from thesis_analysis.tasks.data import GetData
from thesis_analysis.tasks.splot_weights import SPlotWeights


@final
class ChiSqDOFPlot(luigi.Task):
    data_type = luigi.Parameter()
    bins = luigi.IntParameter()
    original = luigi.BoolParameter(False)
    chisqdof = luigi.OptionalFloatParameter(None)
    splot_method = luigi.OptionalParameter(None)
    nsig = luigi.OptionalIntParameter(None)
    nbkg = luigi.OptionalIntParameter(None)

    @override
    def requires(self):
        if self.original:
            return [
                GetData(self.data_type, run_period)
                for run_period in RUN_PERIODS
            ]
        elif self.nsig is None and self.nbkg is None:
            return [
                AccidentalsAndPolarization(self.data_type, run_period)
                for run_period in RUN_PERIODS
            ]
        elif (
            self.splot_method is not None
            and self.nsig is not None
            and self.nbkg is not None
        ):
            return [
                SPlotWeights(
                    self.data_type,
                    run_period,
                    self.chisqdof,
                    self.splot_method,
                    self.nsig,
                    self.nbkg,
                )
                for run_period in RUN_PERIODS
            ]
        else:
            raise Exception('Invalid requirements for mass plotting!')

    @override
    def output(self):
        path = Paths.plots
        if self.original:
            return [luigi.LocalTarget(path / f'chisqdof_{self.data_type}.png')]
        elif self.chisqdof is None:
            return [
                luigi.LocalTarget(
                    path / f'chisqdof_{self.data_type}_accpol.png'
                )
            ]
        elif self.nsig is None and self.nbkg is None:
            return [
                luigi.LocalTarget(
                    path
                    / f'chisqdof_{self.data_type}_accpol_chisqdof_{self.chisqdof:.1f}.png'
                )
            ]
        elif (
            self.splot_method is not None
            and self.nsig is not None
            and self.nbkg is not None
        ):
            return [
                luigi.LocalTarget(
                    path
                    / f'chisqdof_{self.data_type}_accpol_chisqdof_{self.chisqdof:.1f}_splot_{self.splot_method}_{self.nsig}s_{self.nbkg}b.png'
                )
            ]
        else:
            raise Exception('Invalid requirements for chisqdof plotting!')

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
