# pyright: reportUnnecessaryComparison=false
from pathlib import Path
from typing import final, override

import luigi
import matplotlib.pyplot as plt
import matplotlib.style as mpl_style
import numpy as np

from thesis_analysis import root_io
from thesis_analysis.constants import (
    BRANCH_NAME_TO_LATEX,
    BRANCH_NAME_TO_LATEX_UNITS,
    RUN_PERIODS,
    get_branch,
)
from thesis_analysis.utils import get_plot_paths, get_plot_requirements


@final
class PhiPlot(luigi.Task):
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
                'phi',
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
            get_branch('M_Resonance'),
            get_branch('HX_Phi'),
            get_branch('Weight'),
        ]

        flat_data = root_io.concatenate_branches(
            input_paths, branches, root=False
        )
        mpl_style.use('thesis_analysis.thesis')
        fig, ax = plt.subplots()
        ax.hist2d(
            np.concatenate(
                [flat_data['M_Resonance'], flat_data['M_Resonance']]
            ),
            np.concatenate(
                [
                    flat_data['HX_Phi'],
                    np.mod(flat_data['HX_Phi'] + 2 * np.pi, 2 * np.pi) - np.pi,
                ]
            ),
            weights=np.concatenate([flat_data['Weight'], flat_data['Weight']]),
            bins=(bins, 100),
            range=[(1.0, 2.0), (-np.pi, np.pi)],
        )
        ax.set_xlabel(
            f'{BRANCH_NAME_TO_LATEX["M_Resonance"]} ({BRANCH_NAME_TO_LATEX_UNITS["M_Resonance"]})'
        )
        ax.set_ylabel(f'{BRANCH_NAME_TO_LATEX["HX_Phi"]}')
        fig.savefig(output_path)
        plt.close()
