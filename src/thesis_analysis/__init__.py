import luigi

# from thesis_analysis.tasks.binned_and_unbinned_plot import BinnedAndUnbinnedPlot
from thesis_analysis.tasks.aux_plots import MakeAuxiliaryPlots
from thesis_analysis.tasks.binned_and_unbinned_plot import BinnedAndUnbinnedPlot
from thesis_analysis.tasks.chisqdof_plot import ChiSqDOFPlot
from thesis_analysis.tasks.factorization_report import FactorizationReport
from thesis_analysis.tasks.guided_plot import GuidedPlot
from thesis_analysis.tasks.mass_plot import MassPlot
from thesis_analysis.tasks.rfl_plot import RFLPlot
from thesis_analysis.tasks.splot_report import SPlotReport


class RunAll(luigi.WrapperTask):
    def requires(self):
        return (
            [
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
                GuidedPlot(
                    chisqdof=3.0,
                    splot_method='B',
                    nsig=2,
                    nbkg=2,
                    averaged=True,
                ),
                BinnedAndUnbinnedPlot(
                    chisqdof=3.0, splot_method='B', nsig=2, nbkg=2
                ),
                GuidedPlot(
                    chisqdof=3.0,
                    splot_method='B',
                    nsig=2,
                    nbkg=2,
                ),
                BinnedAndUnbinnedPlot(
                    chisqdof=3.0,
                    splot_method='B',
                    nsig=2,
                    nbkg=2,
                    guided=True,
                ),
                BinnedAndUnbinnedPlot(
                    chisqdof=3.0,
                    splot_method='B',
                    nsig=2,
                    nbkg=2,
                    guided=True,
                    averaged=True,
                ),
                BinnedAndUnbinnedPlot(
                    chisqdof=3.0, splot_method='D', nsig=1, nbkg=2
                ),
                GuidedPlot(
                    chisqdof=3.0,
                    splot_method='D',
                    nsig=1,
                    nbkg=2,
                ),
                BinnedAndUnbinnedPlot(
                    chisqdof=3.0,
                    splot_method='D',
                    nsig=1,
                    nbkg=2,
                    guided=True,
                ),
                GuidedPlot(
                    chisqdof=3.0,
                    splot_method='D',
                    nsig=1,
                    nbkg=2,
                    averaged=True,
                ),
                BinnedAndUnbinnedPlot(
                    chisqdof=3.0,
                    splot_method='D',
                    nsig=1,
                    nbkg=2,
                    guided=True,
                    averaged=True,
                ),
            ]
            # [
            #     PlotUnbinnedInitialized(chisqdof=3.0, nsig=1, nbkg=1),
            #     PlotUnbinnedGuided(chisqdof=3.0, nsig=1, nbkg=1),
            #     PlotUnbinned(chisqdof=3.0, nsig=1, nbkg=1),
            #     PlotBinned(chisqdof=3.0, nsig=1, nbkg=1),
            # ]
            # + [
            #     PlotSingleBinned(
            #         run_period=run_period, chisqdof=3.0, nsig=1, nbkg=1
            #     )
            #     for run_period in RUN_PERIODS
            # ]
            # + [
            #     AccidentalsAndPolarization('genmc', run_period)
            #     for run_period in RUN_PERIODS
            # ]
            # + [
            #     FactorizationTest(run_period, chisqdof=3.0, n_quantiles=n)
            #     for run_period, n in itertools.product(
            #         RUN_PERIODS, SIG_QUANTILES
            #     )
            # ]
            # + [
            #     PlotMass(data_type, run_period, NBINS, original=True)
            #     for data_type, run_period in itertools.product(
            #         DATA_TYPES, RUN_PERIODS
            #     )
            # ]
            # + [
            #     PlotMass(data_type, run_period, NBINS, original=False)
            #     for data_type, run_period in itertools.product(
            #         DATA_TYPES, RUN_PERIODS
            #     )
            # ]
            # + [
            #     PlotMass(
            #         data_type,
            #         run_period,
            #         NBINS,
            #         original=False,
            #         chisqdof=chisqdof,
            #     )
            #     for data_type, run_period, chisqdof in itertools.product(
            #         ['data', 'accmc', 'bkgmc'], RUN_PERIODS, CHISQDOF
            #     )
            # ]
            # + [
            #     PlotMass(
            #         data_type,
            #         run_period,
            #         NBINS,
            #         original=False,
            #         chisqdof=chisqdof,
            #     )
            #     for data_type, run_period, chisqdof in itertools.product(
            #         ['data', 'accmc'], RUN_PERIODS, CHISQDOF
            #     )
            # ]
            # + [
            #     PlotMass(
            #         data_type,
            #         run_period,
            #         NBINS,
            #         original=False,
            #         chisqdof=chisqdof,
            #         nsig=nsig,
            #         nbkg=nbkg,
            #     )
            #     for data_type, run_period, chisqdof, (
            #         nsig,
            #         nbkg,
            #     ) in itertools.product(
            #         ['data', 'accmc'], RUN_PERIODS, CHISQDOF, SPLOT_SB
            #     )
            # ]
        )
