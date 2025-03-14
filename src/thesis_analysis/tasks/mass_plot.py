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
    BRANCH_NAME_TO_LATEX_UNITS,
    DATA_TYPE_TO_LATEX,
    RUN_PERIODS,
    get_branch,
)
from thesis_analysis.paths import Paths
from thesis_analysis.tasks.accid_and_pol import AccidentalsAndPolarization
from thesis_analysis.tasks.chisqdof import ChiSqDOF
from thesis_analysis.tasks.data import GetData
from thesis_analysis.tasks.splot_weights import SPlotWeights


@final
class MassPlot(luigi.Task):
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
        elif self.chisqdof is None:
            return [
                AccidentalsAndPolarization(self.data_type, run_period)
                for run_period in RUN_PERIODS
            ]
        elif self.nsig is None and self.nbkg is None:
            return [
                ChiSqDOF(self.data_type, run_period, self.chisqdof)
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
            return [luigi.LocalTarget(path / f'mass_{self.data_type}.png')]
        elif self.chisqdof is None:
            return [
                luigi.LocalTarget(path / f'mass_{self.data_type}_accpol.png')
            ]
        elif self.nsig is None and self.nbkg is None:
            return [
                luigi.LocalTarget(
                    path
                    / f'mass_{self.data_type}_accpol_chisqdof_{self.chisqdof:.1f}.png'
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
                    / f'mass_{self.data_type}_accpol_chisqdof_{self.chisqdof:.1f}_splot_{self.splot_method}_{self.nsig}s_{self.nbkg}b.png'
                )
            ]
        else:
            raise Exception('Invalid requirements for mass plotting!')

    @override
    def run(self):
        input_path = Path(self.input()[0][0].path)
        output_path = self.output()[0].path

        bins = int(self.bins)  # pyright:ignore[reportArgumentType]

        branches = [
            get_branch('M_Resonance'),
            get_branch('Weight'),
        ]

        data = root_io.get_branches(input_path, branches)
        mpl_style.use('thesis_analysis.thesis')
        fig, ax = plt.subplots()
        ax.hist(
            data['M_Resonance'],
            weights=data['Weight'],
            bins=bins,
            range=(1.0, 2.0),
            color=colors.blue,
            label=DATA_TYPE_TO_LATEX[str(self.data_type)],
        )
        ax.set_xlabel(
            f'{BRANCH_NAME_TO_LATEX["M_Resonance"]} ({BRANCH_NAME_TO_LATEX_UNITS["M_Resonance"]})'
        )
        bin_width_mev = int(1000 / bins)
        ax.set_ylabel(f'Counts / {bin_width_mev} MeV/$c^2$')
        ax.legend()
        fig.savefig(output_path)
        plt.close()
