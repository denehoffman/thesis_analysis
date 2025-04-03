# pyright: reportUnnecessaryComparison=false
from pathlib import Path
from typing import final, override

import luigi
import matplotlib.pyplot as plt
import matplotlib.style as mpl_style
import numpy as np

from thesis_analysis import colors, root_io
from thesis_analysis.constants import BRANCH_NAME_TO_LATEX, get_branch
from thesis_analysis.paths import Paths
from thesis_analysis.tasks.data import GetData


def topo_to_latex(topo: str) -> str:
    topo = topo.replace('#plus', '+')
    topo = topo.replace('#minus', '-')
    topo = topo.replace('#', '\\')
    return f'${topo}$'


@final
class BGGENPlot(luigi.Task):
    run_period = luigi.Parameter()
    chisqdof = luigi.OptionalFloatParameter(None)
    protonz = luigi.OptionalBoolParameter(False)

    @override
    def requires(self):
        return [
            GetData('bggen', self.run_period),
        ]

    @override
    def output(self):
        return [
            luigi.LocalTarget(
                Paths.plots
                / f'bggen_chisqdof{f"_chisqdof_{self.chisqdof:.1f}" if self.chisqdof is not None else ""}{"_protonz" if self.protonz else ""}.png'
            ),
            luigi.LocalTarget(
                Paths.plots
                / f'bggen_protonz{f"_chisqdof_{self.chisqdof:.1f}" if self.chisqdof is not None else ""}{"_protonz" if self.protonz else ""}.png'
            ),
            luigi.LocalTarget(
                Paths.plots
                / f'bggen_rfl{f"_chisqdof_{self.chisqdof:.1f}" if self.chisqdof is not None else ""}{"_protonz" if self.protonz else ""}.png'
            ),
            luigi.LocalTarget(
                Paths.plots
                / f'bggen_mm2{f"_chisqdof_{self.chisqdof:.1f}" if self.chisqdof is not None else ""}{"_protonz" if self.protonz else ""}.png'
            ),
        ]

    @override
    def run(self):
        input_path = Path(self.input()[0][0].path)
        output_path_chisqdof = self.output()[0].path
        output_path_protonz = self.output()[1].path
        output_path_rfl = self.output()[2].path
        output_path_mm2 = self.output()[3].path

        if self.chisqdof is not None:
            chisqdof = float(self.chisqdof)  # pyright:ignore[reportArgumentType]
        else:
            chisqdof = None

        branches = [
            # get_branch('M_Resonance'),
            get_branch('ChiSqDOF'),
            get_branch('Proton_Z'),
            get_branch('RFL1'),
            get_branch('Topology'),
            get_branch('MM2'),
        ]
        data = root_io.get_branches(input_path, branches)

        unique_topos, counts = np.unique(data['Topology'], return_counts=True)
        sorted_count_indices = np.argsort(-counts)
        sorted_topos = unique_topos[sorted_count_indices]

        target_topo = '2#pi^{#plus}2#pi^{#minus}p[K^{0}_{S}]'
        target_mask = data['Topology'] == target_topo

        top_five_topos = []
        top_five_masks = []
        remaining_mask = np.ones_like(data['Topology'], dtype=bool)

        count = 0
        for topo in sorted_topos:
            if topo == target_topo:
                continue
            if count >= 5:
                break
            top_five_topos.append(topo)
            mask = data['Topology'] == topo
            top_five_masks.append(mask)
            remaining_mask &= ~mask
            count += 1
        remaining_mask &= ~target_mask

        color_sequence = [
            colors.blue,
            colors.red,
            colors.green,
            colors.purple,
            colors.orange,
            colors.pink,
            colors.gray,
        ]

        masks = [target_mask] + top_five_masks + [remaining_mask]

        mpl_style.use('thesis_analysis.thesis')
        fig, ax = plt.subplots()
        bins = 200
        values = [data['ChiSqDOF'][mask] for mask in masks]
        ax.hist(
            values,
            bins=bins,
            range=(0.0, 200.0),
            color=color_sequence,
            stacked=True,
            label=[
                'Signal',
                *[topo_to_latex(topo) for topo in top_five_topos],
                'Other',
            ],
        )
        ax.set_xlabel(BRANCH_NAME_TO_LATEX['ChiSqDOF'])
        bin_width = 1.0 / bins
        ax.set_ylabel(f'Counts / {bin_width:.2f}')
        ax.legend()
        fig.savefig(output_path_chisqdof)
        plt.close()

        # Proton Z
        fig, ax = plt.subplots()
        bins = 100
        values = [data['Proton_Z'][mask] for mask in masks]
        ax.hist(
            values,
            bins=bins,
            range=(20.0, 120.0),
            color=color_sequence,
            stacked=True,
            label=[
                'Signal',
                *[topo_to_latex(topo) for topo in top_five_topos],
                'Other',
            ],
        )
        ax.set_xlabel(BRANCH_NAME_TO_LATEX['Proton_Z'])
        bin_width = 1.0 / bins
        ax.set_ylabel(f'Counts / {bin_width:.2f}')
        ax.legend()
        fig.savefig(output_path_protonz)
        plt.close()

        # RFL
        fig, ax = plt.subplots()
        bins = 100
        values = [data['RFL1'][mask] for mask in masks]
        ax.hist(
            values,
            bins=bins,
            range=(0.0, 0.2),
            color=color_sequence,
            stacked=True,
            label=[
                'Signal',
                *[topo_to_latex(topo) for topo in top_five_topos],
                'Other',
            ],
        )
        ax.set_xlabel(BRANCH_NAME_TO_LATEX['RFL1'])
        bin_width = 1.0 / bins
        ax.set_ylabel(f'Counts / {bin_width:.2f}')
        ax.set_yscale('log')
        ax.legend()
        fig.savefig(output_path_rfl)
        plt.close()

        # MM2
        fig, ax = plt.subplots()
        bins = 100

        values = [data['MM2'][mask] for mask in masks]

        ax.hist(
            values,
            bins=bins,
            color=color_sequence,
            stacked=True,
            label=[
                'Signal',
                *[topo_to_latex(topo) for topo in top_five_topos],
                'Other',
            ],
        )
        ax.set_xlabel('Missing Mass Squared')
        bin_width = 1.0 / bins
        ax.set_ylabel(f'Counts / {bin_width:.2f}')
        ax.legend()
        fig.savefig(output_path_mm2)
        plt.close()
