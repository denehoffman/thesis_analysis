import itertools

import luigi

from thesis_analysis.constants import (
    CHISQDOF,
    DATA_TYPES,
    RUN_PERIODS,
    SPLOT_SB,
)
from thesis_analysis.tasks.accid_and_pol import AccidentalsAndPolarization
from thesis_analysis.tasks.fit_unbinned import FitUnbinned
from thesis_analysis.tasks.plot_mass import PlotMass


class RunAll(luigi.WrapperTask):
    def requires(self):
        return (
            [
                FitUnbinned(chisqdof, n_sig, n_bkg)
                for chisqdof, (
                    n_sig,
                    n_bkg,
                ) in itertools.product(CHISQDOF, SPLOT_SB)
            ]
            + [
                AccidentalsAndPolarization('genmc', run_period)
                for run_period in RUN_PERIODS
            ]
            + [
                PlotMass(data_type, run_period, 50, original=True)
                for data_type, run_period in itertools.product(
                    DATA_TYPES, RUN_PERIODS
                )
            ]
            + [
                PlotMass(data_type, run_period, 50, original=False)
                for data_type, run_period in itertools.product(
                    DATA_TYPES, RUN_PERIODS
                )
            ]
            + [
                PlotMass(
                    data_type, run_period, 50, original=False, chisqdof=chisqdof
                )
                for data_type, run_period, chisqdof in itertools.product(
                    ['data', 'accmc', 'bkgmc'], RUN_PERIODS, CHISQDOF
                )
            ]
            + [
                PlotMass(
                    data_type, run_period, 50, original=False, chisqdof=chisqdof
                )
                for data_type, run_period, chisqdof in itertools.product(
                    ['data', 'accmc'], RUN_PERIODS, CHISQDOF
                )
            ]
            + [
                PlotMass(
                    data_type,
                    run_period,
                    50,
                    original=False,
                    chisqdof=chisqdof,
                    n_sig=n_sig,
                    n_bkg=n_bkg,
                )
                for data_type, run_period, chisqdof, (
                    n_sig,
                    n_bkg,
                ) in itertools.product(
                    ['data', 'accmc'], RUN_PERIODS, CHISQDOF, SPLOT_SB
                )
            ]
        )
