from typing import override

import luigi

from thesis_analysis.constants import NBINS
from thesis_analysis.tasks.aux_plots import MakeAuxiliaryPlots
from thesis_analysis.tasks.bggen import BGGENPlots
from thesis_analysis.tasks.binned_and_unbinned_plot import BinnedAndUnbinnedPlot
from thesis_analysis.tasks.binned_fit_report import BinnedFitReport
from thesis_analysis.tasks.binned_plot import BinnedPlot
from thesis_analysis.tasks.chisqdof_plot import ChiSqDOFPlot
from thesis_analysis.tasks.costheta_plot import CosThetaPlot
from thesis_analysis.tasks.cut_plots_combined import CutPlotsCombined
from thesis_analysis.tasks.factorization_report import FactorizationReport
from thesis_analysis.tasks.mass_plot import MassPlot
from thesis_analysis.tasks.pdg_plot import PDGPlot
from thesis_analysis.tasks.rf_plot import RFPlot
from thesis_analysis.tasks.rfl_plot import RFLPlot
from thesis_analysis.tasks.splot_fit_report import SPlotFitReport
from thesis_analysis.tasks.splot_report import SPlotReport
from thesis_analysis.tasks.unbinned_fit_report import UnbinnedFitReport
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
        *[
            UnbinnedFitReport(
                waves=waves,
                chisqdof=chisqdof,
                splot_method='D',
                nsig=1,
                nbkg=2,
                niters=5,
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
        *[
            BinnedFitReport(
                waves=waves,
                chisqdof=chisqdof,
                splot_method='D',
                nsig=1,
                nbkg=2,
                niters=1,
                phase_factor=True,
                uncertainty='bootstrap',
            )
            for waves in [
                Wave.encode_waves([Wave(0, 0, '+'), Wave(2, 2, '+')]),
                Wave.encode_waves(
                    [Wave(0, 0, '+'), Wave(0, 0, '-'), Wave(2, 2, '+')]
                ),
                Wave.encode_waves([Wave(0, 0, '+'), Wave(2, 1, '+')]),
                Wave.encode_waves([Wave(0, 0, '+'), Wave(2, 0, '+')]),
            ]
        ],
        SPlotFitReport(
            data_type='data',
            chisqdof=chisqdof,
            splot_method='D',
            nsig=1,
            nbkg=2,
        ),
        # Fits
        *[
            BinnedAndUnbinnedPlot(
                waves=waves,
                chisqdof=chisqdof,
                splot_method='D',
                nsig=1,
                nbkg=2,
                niters=5,
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
        *[
            BinnedPlot(
                waves=waves,
                chisqdof=chisqdof,
                splot_method='D',
                nsig=1,
                nbkg=2,
                niters=1,
                phase_factor=True,
                uncertainty='bootstrap',
                bootstrap_mode='CI-BC',
            )
            for waves in [
                Wave.encode_waves([Wave(0, 0, '+'), Wave(2, 2, '+')]),
                Wave.encode_waves([Wave(0, 0, '+'), Wave(2, 1, '+')]),
                Wave.encode_waves([Wave(0, 0, '+'), Wave(2, 0, '+')]),
                Wave.encode_waves(
                    [
                        Wave(0, 0, '+'),
                        Wave(0, 0, '-'),
                        Wave(2, 2, '+'),
                        Wave(2, 2, '-'),
                    ]
                ),
            ]
        ],
        *[
            BinnedAndUnbinnedPlot(
                waves=waves,
                chisqdof=chisqdof,
                splot_method='D',
                nsig=1,
                nbkg=2,
                niters=5,
                guided=guided,
                phase_factor=True,
                uncertainty='bootstrap',
                bootstrap_mode='SE',
                acceptance_corrected=True,
            )
            for waves in [
                Wave.encode_waves([Wave(0, 0, '+'), Wave(2, 2, '+')]),
                Wave.encode_waves(
                    [Wave(0, 0, '+'), Wave(0, 0, '-'), Wave(2, 2, '+')]
                ),
            ]
            for guided in [True, False]
        ],
        *[
            BinnedPlot(
                waves=waves,
                chisqdof=chisqdof,
                splot_method='D',
                nsig=1,
                nbkg=2,
                niters=1,
                phase_factor=True,
                uncertainty='bootstrap',
                bootstrap_mode='CI-BC',
                acceptance_corrected=True,
            )
            for waves in [
                Wave.encode_waves([Wave(0, 0, '+'), Wave(2, 2, '+')]),
                Wave.encode_waves([Wave(0, 0, '+'), Wave(2, 1, '+')]),
                Wave.encode_waves([Wave(0, 0, '+'), Wave(2, 0, '+')]),
                Wave.encode_waves(
                    [
                        Wave(0, 0, '+'),
                        Wave(0, 0, '-'),
                        Wave(2, 2, '+'),
                        Wave(2, 2, '-'),
                    ]
                ),
            ]
        ],
    ]


class RunAll(luigi.WrapperTask):
    @override
    def requires(self):
        return [
            # *run_all(1.4),
            # *run_all(2.4),
            *run_all(3.4),
            # *run_all(4.4),
            # *run_all(5.4),
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
            PDGPlot(
                data_type='data',
                bins=NBINS,
                original=False,
                chisqdof=3.4,
                splot_method='D',
                nsig=1,
                nbkg=2,
            ),
        ]
