from pathlib import Path

import luigi
import matplotlib.pyplot as plt
import matplotlib.style as mpl_style
import thesis_analysis.colors as colors
from thesis_analysis import root_io
from thesis_analysis.constants import get_branch
from thesis_analysis.paths import Paths
from thesis_analysis.tasks.accid_and_pol import AccidentalsAndPolarization
from thesis_analysis.tasks.chisqdof import ChiSqDOF
from thesis_analysis.tasks.data import GetData
from thesis_analysis.tasks.splot import SPlot


class PlotMass(luigi.Task):
    data_type = luigi.Parameter()
    run_period = luigi.Parameter()
    bins = luigi.IntParameter()
    original = luigi.BoolParameter(False)
    chisqdof = luigi.OptionalFloatParameter(None)
    n_sig = luigi.OptionalIntParameter(None)
    n_bkg = luigi.OptionalIntParameter(None)

    def requires(self):
        if self.original:
            return [GetData(self.data_type, self.run_period)]
        elif self.chisqdof is None:
            return [AccidentalsAndPolarization(self.data_type, self.run_period)]
        elif self.n_sig is None and self.n_bkg is None:
            return [ChiSqDOF(self.data_type, self.run_period, self.chisqdof)]
        elif self.n_sig is not None and self.n_bkg is not None:
            return [
                SPlot(
                    self.data_type,
                    self.run_period,
                    self.chisqdof,
                    self.n_sig,
                    self.n_bkg,
                )
            ]
        else:
            raise Exception('Invalid requirements for mass plotting!')

    def output(self):
        path = Paths.plots
        if self.original:
            return [
                luigi.LocalTarget(
                    path / f'mass_{self.data_type}_{self.run_period}.png'
                )
            ]
        elif self.chisqdof is None:
            return [
                luigi.LocalTarget(
                    path / f'mass_{self.data_type}_{self.run_period}_accpol.png'
                )
            ]
        elif self.n_sig is None and self.n_bkg is None:
            return [
                luigi.LocalTarget(
                    path
                    / f'mass_{self.data_type}_{self.run_period}_accpol_chisqdof_{self.chisqdof:.1f}.png'
                )
            ]
        elif self.n_sig is not None and self.n_bkg is not None:
            return [
                luigi.LocalTarget(
                    path
                    / f'mass_{self.data_type}_{self.run_period}_accpol_chisqdof_{self.chisqdof:.1f}_splot_{self.n_sig}s_{self.n_bkg}b.png'
                )
            ]
        else:
            raise Exception('Invalid requirements for mass plotting!')

    def run(self):
        input_path = Path(self.input()[0][0].path)
        output_path = self.output()[0].path

        branches = [
            get_branch('M_Resonance'),
            get_branch('Weight'),
            get_branch('RFL1'),
        ]

        data = root_io.get_branches(input_path, branches)
        mpl_style.use('thesis_analysis.thesis')
        fig, ax = plt.subplots()
        ax.hist(
            data['M_Resonance'],
            weights=data['Weight'],
            bins=self.bins,
            range=(1.0, 2.0),
            color=colors.blue,
        )
        ax.set_xlabel('Invariant Mass of $K_S^0K_S^0$ (GeV/$c^2$)')
        bin_width_mev = int(1000 / self.bins)
        ax.set_ylabel(f'Counts / {bin_width_mev} MeV/$c^2$')
        fig.savefig(output_path)
        plt.close()
