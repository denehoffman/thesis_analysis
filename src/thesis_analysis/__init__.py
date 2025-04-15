# pyright: reportImportCycles=false
from typing import override

import luigi

# from thesis_analysis.tasks.binned_and_unbinned_plot import BinnedAndUnbinnedPlot
from thesis_analysis.tasks.aux_plots import MakeAuxiliaryPlots

# from thesis_analysis.tasks.binned_and_unbinned_plot import BinnedAndUnbinnedPlot
from thesis_analysis.tasks.bggen import BGGENPlot
from thesis_analysis.tasks.binned_and_unbinned_plot import BinnedAndUnbinnedPlot
from thesis_analysis.tasks.binned_plot import BinnedPlot
from thesis_analysis.tasks.bootstrap_uncertainty_comparison_plot import (
    BootstrapUncertaintyComparisonPlot,
)
from thesis_analysis.tasks.chisqdof_plot import ChiSqDOFPlot
from thesis_analysis.tasks.costheta_plot import CosThetaPlot
from thesis_analysis.tasks.factorization_report import FactorizationReport

# from thesis_analysis.tasks.guided_plot import GuidedPlot
from thesis_analysis.tasks.mass_plot import MassPlot
from thesis_analysis.tasks.rfl_plot import RFLPlot
from thesis_analysis.tasks.single_binned_plot import SingleBinnedPlot
from thesis_analysis.tasks.splot_report import SPlotReport
from thesis_analysis.wave import Wave


def run_all(chisqdof: float) -> list[luigi.Task]:
    return [
        RFLPlot('data', chisqdof=chisqdof),
        RFLPlot('accmc', chisqdof=chisqdof),
        RFLPlot('bkgmc', chisqdof=chisqdof),
        ChiSqDOFPlot('data', bins=50),
        MassPlot('data', bins=50),
        ChiSqDOFPlot('accmc', bins=50),
        ChiSqDOFPlot('data', bins=50, chisqdof=chisqdof),
        MassPlot('data', bins=50, chisqdof=chisqdof),
        ChiSqDOFPlot('accmc', bins=50, chisqdof=chisqdof),
        ChiSqDOFPlot(
            'data',
            bins=50,
            chisqdof=chisqdof,
            splot_method='D',
            nsig=1,
            nbkg=2,
        ),
        MassPlot(
            'data',
            bins=50,
            chisqdof=chisqdof,
            splot_method='D',
            nsig=1,
            nbkg=2,
        ),
        CosThetaPlot(
            'data',
            bins=50,
            chisqdof=chisqdof,
            splot_method='D',
            nsig=1,
            nbkg=2,
        ),
        # FITS
        *[
            BinnedAndUnbinnedPlot(
                waves=waves,
                chisqdof=chisqdof,
                splot_method='D',
                nsig=1,
                nbkg=2,
                niters=1,
                guided=guided,
                phase_factor=phase_factor,
                uncertainty='bootstrap',
                bootstrap_mode='SE',
            )
            for waves in [
                Wave.encode_waves([Wave(0, 0, '+'), Wave(2, 2, '+')]),
                Wave.encode_waves(
                    [Wave(0, 0, '+'), Wave(0, 0, '-'), Wave(2, 2, '+')]
                ),
            ]
            for guided in [True, False]
            for phase_factor in [True, False]
        ],
        # BootstrapUncertaintyComparisonPlot(
        #     waves=Wave.encode_waves([Wave(0, 0, '+'), Wave(2, 2, '+')]),
        #     chisqdof=chisqdof,
        #     splot_method='D',
        #     nsig=1,
        #     nbkg=2,
        #     niters=1,
        #     phase_factor=False,
        # ),
        # BootstrapUncertaintyComparisonPlot(
        #     waves=Wave.encode_waves([Wave(0, 0, '+'), Wave(2, 2, '+')]),
        #     chisqdof=chisqdof,
        #     splot_method='D',
        #     nsig=1,
        #     nbkg=2,
        #     niters=1,
        #     phase_factor=True,
        # ),
    ]


class RunAll(luigi.WrapperTask):
    @override
    def requires(self):
        return [
            *run_all(3.0),
            *run_all(4.0),
            *run_all(5.0),
            *run_all(6.0),
            MakeAuxiliaryPlots(),
            BGGENPlot(run_period='s18'),
        ]
