from typing import Literal, override

import luigi

from thesis_analysis.constants import NBINS
from thesis_analysis.tasks.aux_plots import MakeAuxiliaryPlots
from thesis_analysis.tasks.baryon_plots import BaryonPlot
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
from thesis_analysis.tasks.phi_plot import PhiPlot
from thesis_analysis.tasks.rf_plot import RFPlot
from thesis_analysis.tasks.rfl_plot import RFLPlot
from thesis_analysis.tasks.splot_fit_report import SPlotFitReport
from thesis_analysis.tasks.splot_report import SPlotReport
from thesis_analysis.tasks.unbinned_fit_report import UnbinnedFitReport
from thesis_analysis.wave import Wave


def run_all(
    *,
    original: bool,
    chisqdof: float,
    ksb_costheta: float,
    cut_baryons: bool,
    max_fit: Literal['binned', 'unbinned', 'guided'] | None,
) -> list[luigi.Task]:
    binned_wavesets = [
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
    unbinned_wavesets = [
        Wave.encode_waves([Wave(0, 0, '+'), Wave(2, 2, '+')]),
        Wave.encode_waves([Wave(0, 0, '+'), Wave(0, 0, '-'), Wave(2, 2, '+')]),
    ]
    res: list[luigi.Task] = [
        *[
            RFLPlot(
                data_type,
                original=original,
                chisqdof=chisqdof,
                ksb_costheta=ksb_costheta,
                cut_baryons=cut_baryons,
            )
            for data_type in ['data', 'accmc', 'bkgmc']
        ],
        RFLPlot(
            'data',
            original=original,
            chisqdof=chisqdof,
            ksb_costheta=ksb_costheta,
            cut_baryons=cut_baryons,
            splot_method='D',
            nsig=1,
            nbkg=2,
        ),
        *[
            ChiSqDOFPlot(
                data_type,
                bins=50,
                original=original,
                chisqdof=chisqdof,
                ksb_costheta=ksb_costheta,
                cut_baryons=cut_baryons,
            )
            for data_type in ['data', 'accmc', 'bkgmc']
        ],
        ChiSqDOFPlot(
            'data',
            bins=50,
            original=original,
            chisqdof=chisqdof,
            ksb_costheta=ksb_costheta,
            cut_baryons=cut_baryons,
            splot_method='D',
            nsig=1,
            nbkg=2,
        ),
        *[
            MassPlot(
                data_type,
                bins=50,
                original=original,
                chisqdof=chisqdof,
                ksb_costheta=ksb_costheta,
                cut_baryons=cut_baryons,
            )
            for data_type in ['data', 'accmc', 'bkgmc']
        ],
        MassPlot(
            'data',
            bins=50,
            original=original,
            chisqdof=chisqdof,
            ksb_costheta=ksb_costheta,
            cut_baryons=cut_baryons,
            splot_method='D',
            nsig=1,
            nbkg=2,
        ),
        *[
            PDGPlot(
                data_type,
                bins=50,
                original=original,
                chisqdof=chisqdof,
                ksb_costheta=ksb_costheta,
                cut_baryons=cut_baryons,
            )
            for data_type in ['data', 'accmc', 'bkgmc']
        ],
        PDGPlot(
            'data',
            bins=50,
            original=original,
            chisqdof=chisqdof,
            ksb_costheta=ksb_costheta,
            cut_baryons=cut_baryons,
            splot_method='D',
            nsig=1,
            nbkg=2,
        ),
        *[
            CosThetaPlot(
                data_type,
                bins=100,
                original=original,
                chisqdof=chisqdof,
                ksb_costheta=ksb_costheta,
                cut_baryons=cut_baryons,
            )
            for data_type in ['data', 'accmc', 'bkgmc']
        ],
        CosThetaPlot(
            'data',
            bins=100,
            original=original,
            chisqdof=chisqdof,
            ksb_costheta=ksb_costheta,
            cut_baryons=cut_baryons,
            splot_method='D',
            nsig=1,
            nbkg=2,
        ),
        *[
            PhiPlot(
                data_type,
                bins=100,
                original=original,
                chisqdof=chisqdof,
                ksb_costheta=ksb_costheta,
                cut_baryons=cut_baryons,
            )
            for data_type in ['data', 'accmc', 'bkgmc']
        ],
        PhiPlot(
            'data',
            bins=100,
            original=original,
            chisqdof=chisqdof,
            ksb_costheta=ksb_costheta,
            cut_baryons=cut_baryons,
            splot_method='D',
            nsig=1,
            nbkg=2,
        ),
        *[
            BaryonPlot(
                data_type,
                bins=100,
                original=original,
                chisqdof=chisqdof,
                ksb_costheta=ksb_costheta,
                cut_baryons=cut_baryons,
            )
            for data_type in ['data', 'accmc', 'bkgmc']
        ],
        BaryonPlot(
            'data',
            bins=100,
            original=original,
            chisqdof=chisqdof,
            ksb_costheta=ksb_costheta,
            cut_baryons=cut_baryons,
            splot_method='D',
            nsig=1,
            nbkg=2,
        ),
        SPlotFitReport(
            data_type='data',
            chisqdof=chisqdof,
            ksb_costheta=ksb_costheta,
            cut_baryons=cut_baryons,
            splot_method='D',
            nsig=1,
            nbkg=2,
        ),
    ]
    if original or max_fit is None:
        return res

    if max_fit == 'binned':
        res.extend(
            [
                BinnedPlot(
                    waves=waves,
                    chisqdof=chisqdof,
                    ksb_costheta=ksb_costheta,
                    cut_baryons=cut_baryons,
                    splot_method='D',
                    nsig=1,
                    nbkg=2,
                    niters=1,
                    phase_factor=True,
                    uncertainty='bootstrap',
                    bootstrap_mode='CI-BC',
                )
                for waves in binned_wavesets
            ],
        )
        res.extend(
            [
                BinnedFitReport(
                    waves=waves,
                    chisqdof=chisqdof,
                    ksb_costheta=ksb_costheta,
                    cut_baryons=cut_baryons,
                    splot_method='D',
                    nsig=1,
                    nbkg=2,
                    niters=1,
                    phase_factor=True,
                    uncertainty='bootstrap',
                )
                for waves in binned_wavesets
            ],
        )
    if max_fit == 'unbinned':
        res.extend(
            [
                UnbinnedFitReport(
                    waves=waves,
                    chisqdof=chisqdof,
                    ksb_costheta=ksb_costheta,
                    cut_baryons=cut_baryons,
                    splot_method='D',
                    nsig=1,
                    nbkg=2,
                    niters=1,
                    guided=False,
                    phase_factor=True,
                    uncertainty='bootstrap',
                    bootstrap_mode='SE',
                )
                for waves in unbinned_wavesets
            ]
        )
        res.extend(
            [
                BinnedAndUnbinnedPlot(
                    waves=waves,
                    chisqdof=chisqdof,
                    ksb_costheta=ksb_costheta,
                    cut_baryons=cut_baryons,
                    splot_method='D',
                    nsig=1,
                    nbkg=2,
                    niters=5,
                    guided=False,
                    phase_factor=True,
                    uncertainty='bootstrap',
                    bootstrap_mode='SE',
                )
                for waves in unbinned_wavesets
            ]
        )
    if max_fit == 'guided':
        res.extend(
            [
                UnbinnedFitReport(
                    waves=waves,
                    chisqdof=chisqdof,
                    ksb_costheta=ksb_costheta,
                    cut_baryons=cut_baryons,
                    splot_method='D',
                    nsig=1,
                    nbkg=2,
                    niters=1,
                    guided=True,
                    phase_factor=True,
                    uncertainty='bootstrap',
                    bootstrap_mode='SE',
                )
                for waves in unbinned_wavesets
            ]
        )
        res.extend(
            [
                BinnedAndUnbinnedPlot(
                    waves=waves,
                    chisqdof=chisqdof,
                    ksb_costheta=ksb_costheta,
                    cut_baryons=cut_baryons,
                    splot_method='D',
                    nsig=1,
                    nbkg=2,
                    niters=5,
                    guided=True,
                    phase_factor=True,
                    uncertainty='bootstrap',
                    bootstrap_mode='SE',
                )
                for waves in unbinned_wavesets
            ]
        )
    return res


class RunAll(luigi.WrapperTask):
    @override
    def requires(self):
        chisqdof = 3.4
        return [
            *run_all(
                original=False,
                chisqdof=chisqdof,
                ksb_costheta=-1.00,
                cut_baryons=True,
                max_fit='guided',
            ),
            *run_all(
                original=False,
                chisqdof=chisqdof,
                ksb_costheta=0.00,
                cut_baryons=True,
                max_fit='guided',
            ),
            *run_all(
                original=False,
                chisqdof=chisqdof,
                ksb_costheta=0.00,
                cut_baryons=False,
                max_fit=None,
            ),
            MakeAuxiliaryPlots(),
            BGGENPlots(run_period='s18'),
            RFPlot(data_type='data_original'),
            RFPlot(data_type='data_original', chisqdof=chisqdof),
            RFPlot(data_type='data_original', protonz=True),
            RFPlot(data_type='data_original', chisqdof=chisqdof, protonz=True),
            FactorizationReport(chisqdof, max_quantiles=4),
            SPlotReport(chisqdof, nsig_max=1, nbkg_max=4),
            # CutPlotsCombined(data_type='data'),
            # CutPlotsCombined(data_type='data', chisqdof=3.4, protonz=True),
        ]
