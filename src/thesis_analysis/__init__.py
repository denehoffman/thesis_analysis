# pyright: reportImportCycles=false
import luigi
from typing_extensions import override

# from thesis_analysis.tasks.binned_and_unbinned_plot import BinnedAndUnbinnedPlot
from thesis_analysis.tasks.aux_plots import MakeAuxiliaryPlots

# from thesis_analysis.tasks.binned_and_unbinned_plot import BinnedAndUnbinnedPlot
from thesis_analysis.tasks.chisqdof_plot import ChiSqDOFPlot
from thesis_analysis.tasks.factorization_report import FactorizationReport

# from thesis_analysis.tasks.guided_plot import GuidedPlot
from thesis_analysis.tasks.mass_plot import MassPlot
from thesis_analysis.tasks.rfl_plot import RFLPlot
from thesis_analysis.tasks.single_binned_plot import SingleBinnedPlot
from thesis_analysis.tasks.splot_report import SPlotReport
from thesis_analysis.wave import Wave


class RunAll(luigi.WrapperTask):
    @override
    def requires(self):
        return [
            SPlotReport(
                chisqdof=3.0,
                nsig_max=3,
                nbkg_max=3,
            ),
            FactorizationReport(chisqdof=3.0, max_quantiles=4),
            RFLPlot('data', chisqdof=3.0),
            RFLPlot('accmc', chisqdof=3.0),
            RFLPlot('bkgmc', chisqdof=3.0),
            ChiSqDOFPlot('data', bins=50),
            MassPlot('data', bins=50),
            ChiSqDOFPlot('accmc', bins=50),
            ChiSqDOFPlot('data', bins=50, chisqdof=3.0),
            MassPlot('data', bins=50, chisqdof=3.0),
            ChiSqDOFPlot('accmc', bins=50, chisqdof=3.0),
            ChiSqDOFPlot(
                'data',
                bins=50,
                chisqdof=3.0,
                splot_method='B',
                nsig=2,
                nbkg=2,
            ),
            MassPlot(
                'data',
                bins=50,
                chisqdof=3.0,
                splot_method='B',
                nsig=2,
                nbkg=2,
            ),
            ChiSqDOFPlot(
                'data',
                bins=50,
                chisqdof=3.0,
                splot_method='D',
                nsig=1,
                nbkg=2,
            ),
            MassPlot(
                'data',
                bins=50,
                chisqdof=3.0,
                splot_method='D',
                nsig=1,
                nbkg=2,
            ),
            MakeAuxiliaryPlots(),
            # FITS
            SingleBinnedPlot(
                waves=Wave.encode_waves(
                    set([Wave(0, 0, '+'), Wave(0, 0, '-'), Wave(2, 2, '+')])
                ),
                run_period='s20',
                chisqdof=3.0,
                splot_method='D',
                nsig=1,
                nbkg=2,
                niters=1,
                phase_factor=False,
                uncertainty='sqrt',
            ),
            # BinnedAndUnbinnedPlot(
            #     chisqdof=3.0,
            #     splot_method='B',
            #     nsig=2,
            #     nbkg=2,
            #     guided=True,
            #     averaged=True,
            #     phase_factor=True,
            # ),
            # BinnedAndUnbinnedPlot(
            #     chisqdof=3.0,
            #     splot_method='B',
            #     nsig=2,
            #     nbkg=2,
            #     guided=True,
            #     averaged=True,
            # ),
            # BinnedAndUnbinnedPlot(
            #     chisqdof=3.0, splot_method='B', nsig=2, nbkg=2
            # ),
            # BinnedAndUnbinnedPlot(
            #     chisqdof=3.0,
            #     splot_method='D',
            #     nsig=1,
            #     nbkg=2,
            #     guided=True,
            #     averaged=True,
            #     phase_factor=True,
            # ),
            # BinnedAndUnbinnedPlot(
            #     chisqdof=3.0,
            #     splot_method='D',
            #     nsig=1,
            #     nbkg=2,
            #     guided=True,
            #     averaged=True,
            # ),
            # BinnedAndUnbinnedPlot(
            #     chisqdof=3.0, splot_method='D', nsig=1, nbkg=2
            # ),
        ]
