from typing import override

import luigi

from thesis_analysis.tasks.aux_plots import MakeAuxiliaryPlots
from thesis_analysis.tasks.bggen import BGGENPlots
from thesis_analysis.tasks.binned_and_unbinned_plot import BinnedAndUnbinnedPlot
from thesis_analysis.tasks.binned_regularized_plot import BinnedRegularizedPlot
from thesis_analysis.tasks.chisqdof_plot import ChiSqDOFPlot
from thesis_analysis.tasks.costheta_plot import CosThetaPlot
from thesis_analysis.tasks.cut_plots_combined import CutPlotsCombined
from thesis_analysis.tasks.factorization_report import FactorizationReport
from thesis_analysis.tasks.mass_plot import MassPlot
from thesis_analysis.tasks.rf_plot import RFPlot
from thesis_analysis.tasks.rfl_plot import RFLPlot
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
        # Regularized Fits
        *[
            BinnedRegularizedPlot(
                waves=Wave.encode_waves(
                    [
                        Wave(0, 0, '+'),
                        Wave(2, 0, '+'),
                        Wave(2, 1, '+'),
                        Wave(2, 2, '+'),
                    ]
                ),
                chisqdof=chisqdof,
                splot_method='D',
                nsig=1,
                nbkg=2,
                niters=10,
                phase_factor=True,
                lda=lda,
            )
            for lda in [10.0, 1.0, 0.1, 0.01, 0.0]
        ],
        # Fits
        *[
            BinnedAndUnbinnedPlot(
                waves=waves,
                chisqdof=chisqdof,
                splot_method='D',
                nsig=1,
                nbkg=2,
                niters=1,
                guided=guided,
                phase_factor=True,
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
        ],
    ]


class RunAll(luigi.WrapperTask):
    @override
    def requires(self):
        return [
            # *run_all(3.0),
            *run_all(3.4),
            # *run_all(4.0),
            # *run_all(5.0),
            # *run_all(6.0),
            MakeAuxiliaryPlots(),
            BGGENPlots(run_period='s18'),
            CutPlotsCombined(data_type='data_original'),
            CutPlotsCombined(data_type='data_original', chisqdof=3.4),
            CutPlotsCombined(data_type='data_original', protonz=True),
            CutPlotsCombined(
                data_type='data_original', chisqdof=3.4, protonz=True
            ),
            RFPlot(data_type='data_original'),
            RFPlot(data_type='data_original', chisqdof=3.4),
            RFPlot(data_type='data_original', protonz=True),
            RFPlot(data_type='data_original', chisqdof=3.4, protonz=True),
            CutPlotsCombined(data_type='data'),
            FactorizationReport(3.4, max_quantiles=4),
            SPlotReport(3.4, nsig_max=1, nbkg_max=4),
        ]
