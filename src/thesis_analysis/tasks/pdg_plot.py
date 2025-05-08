# pyright: reportUnnecessaryComparison=false
from dataclasses import dataclass
from pathlib import Path
from typing import final, override

import luigi
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.style as mpl_style
import numpy as np

import thesis_analysis.colors as colors
from thesis_analysis import root_io
from thesis_analysis.constants import (
    BRANCH_NAME_TO_LATEX,
    BRANCH_NAME_TO_LATEX_UNITS,
    DATA_TYPE_TO_LATEX,
    RUN_PERIODS,
    get_branch,
)
from thesis_analysis.paths import Paths
from thesis_analysis.tasks.accid_and_pol import AccidentalsAndPolarization
from thesis_analysis.tasks.chisqdof import ChiSqDOF
from thesis_analysis.tasks.data import GetData
from thesis_analysis.tasks.splot_weights import SPlotWeights


@final
class PDGPlot(luigi.Task):
    data_type = luigi.Parameter()
    bins = luigi.IntParameter()
    original = luigi.BoolParameter(False)
    chisqdof = luigi.OptionalFloatParameter(None)
    splot_method = luigi.OptionalParameter(None)
    nsig = luigi.OptionalIntParameter(None)
    nbkg = luigi.OptionalIntParameter(None)

    @override
    def requires(self):
        if self.original:
            return [
                GetData(self.data_type, run_period)
                for run_period in RUN_PERIODS
            ]
        elif self.chisqdof is None:
            return [
                AccidentalsAndPolarization(self.data_type, run_period)
                for run_period in RUN_PERIODS
            ]
        elif self.nsig is None and self.nbkg is None:
            return [
                ChiSqDOF(self.data_type, run_period, self.chisqdof)
                for run_period in RUN_PERIODS
            ]
        elif (
            self.splot_method is not None
            and self.nsig is not None
            and self.nbkg is not None
        ):
            return [
                SPlotWeights(
                    self.data_type,
                    run_period,
                    self.chisqdof,
                    self.splot_method,
                    self.nsig,
                    self.nbkg,
                )
                for run_period in RUN_PERIODS
            ]
        else:
            raise Exception('Invalid requirements for PDG plotting!')

    @override
    def output(self):
        path = Paths.plots
        if self.original:
            return [luigi.LocalTarget(path / f'mass_pdg_{self.data_type}.png')]
        elif self.chisqdof is None:
            return [
                luigi.LocalTarget(
                    path / f'mass_pdg_{self.data_type}_accpol.png'
                )
            ]
        elif self.nsig is None and self.nbkg is None:
            return [
                luigi.LocalTarget(
                    path
                    / f'mass_pdg_{self.data_type}_accpol_chisqdof_{self.chisqdof:.1f}.png'
                )
            ]
        elif (
            self.splot_method is not None
            and self.nsig is not None
            and self.nbkg is not None
        ):
            return [
                luigi.LocalTarget(
                    path
                    / f'mass_pdg_{self.data_type}_accpol_chisqdof_{self.chisqdof:.1f}_splot_{self.splot_method}_{self.nsig}s_{self.nbkg}b.png'
                )
            ]
        else:
            raise Exception('Invalid requirements for PDG plotting!')

    @override
    def run(self):
        input_paths = [
            Path(self.input()[i][0].path) for i in range(len(RUN_PERIODS))
        ]
        output_path = self.output()[0].path

        bins = int(self.bins)  # pyright:ignore[reportArgumentType]

        branches = [
            get_branch('M_Resonance'),
            get_branch('Weight'),
        ]
        flat_data = root_io.concatenate_branches(
            input_paths, branches, root=False
        )
        mpl_style.use('thesis_analysis.thesis')
        bar_height: float = 0.8
        bar_spacing: float = 0.3
        # fontsize: int = 3
        fig, (hist_ax, bar_ax) = plt.subplots(
            2,
            1,
            sharex=True,
            gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.0},
        )
        hist_ax.hist(
            flat_data['M_Resonance'],
            weights=flat_data['Weight'],
            bins=bins,
            range=(1.0, 2.0),
            color=colors.blue,
            label=DATA_TYPE_TO_LATEX[str(self.data_type)],
        )

        bar_ax.axhline(
            3 * (bar_height + bar_spacing) + bar_spacing / 2,
            ls=':',
            color=colors.black,
        )
        xmin, xmax = 0.95, 2.05

        @dataclass
        class Particle:
            label: str
            color: str
            center: float
            width: float
            row: int
            established: bool = True

        f0_980 = Particle(r'$f_0(980)$', colors.red, 0.990, 0.100, 0)
        f0_1370 = Particle(r'$f_0(1370)$', colors.red, 1.350, 0.500, 0)
        f0_1500 = Particle(r'$f_0(1500)$', colors.red, 1.522, 0.108, 1)
        f0_1710 = Particle(r'$f_0(1710)$', colors.red, 1.733, 0.150, 0)
        f0_1770 = Particle(
            r'$f_0(1770)$', colors.red, 1.784, 0.161, 1, established=False
        )
        f0_2020 = Particle(r'$f_0(2020)$', colors.red, 1.982, 0.440, 2)
        f0_2100 = Particle(r'$f_0(2100)$', colors.red, 2.095, 0.287, 1)
        f0_2200 = Particle(
            r'$f_0(2200)$', colors.red, 2.187, 0.210, 0, established=False
        )

        a0_980 = Particle(r'$a_0(980)$', colors.green, 0.980, 0.100, 2)
        a0_1450 = Particle(r'$a_0(1450)$', colors.green, 1.439, 0.258, 2)
        a0_1710 = Particle(
            r'$a_0(1710)$', colors.green, 1.713, 0.107, 2, established=False
        )

        f2_1270 = Particle(r'$f_2(1270)$', colors.orange, 1.2754, 0.1866, 3)
        f2_1430 = Particle(
            r'$f_2(1430)$', colors.orange, 1.430, 0.013, 3, established=False
        )
        f2_1525 = Particle(r"$f_2'(1525)$", colors.orange, 1.5173, 0.072, 3)
        f2_1565 = Particle(r'$f_2(1565)$', colors.orange, 1.571, 0.132, 4)
        f2_1640 = Particle(
            r'$f_2(1640)$', colors.orange, 1.639, 0.100, 3, established=False
        )
        f2_1810 = Particle(
            r'$f_2(1810)$', colors.orange, 1.815, 0.197, 3, established=False
        )
        f2_1950 = Particle(r'$f_2(1950)$', colors.orange, 1.936, 0.464, 4)
        f2_2010 = Particle(r'$f_2(2010)$', colors.orange, 2.010, 0.200, 5)
        f2_2150 = Particle(
            r'$f_2(2150)$', colors.orange, 2.157, 0.152, 3, established=False
        )

        a2_1320 = Particle(r'$a_2(1320)$', colors.purple, 1.3181, 0.1098, 5)
        a2_1700 = Particle(r'$a_2(1700)$', colors.purple, 1.706, 0.380, 5)

        particles = [
            f0_980,
            f0_1370,
            f0_1500,
            f0_1710,
            f0_1770,
            f0_2020,
            f0_2100,
            f0_2200,
            a0_980,
            a0_1450,
            a0_1710,
            f2_1270,
            f2_1430,
            f2_1525,
            f2_1565,
            f2_1640,
            f2_1810,
            f2_1950,
            f2_2010,
            f2_2150,
            a2_1320,
            a2_1700,
        ]

        for particle in particles:
            center = particle.center
            width = particle.width
            if center - width < xmin:
                xmin = center - width - 0.05
            if center + width > xmax:
                xmax = center + width + 0.05
            color = particle.color
            row = particle.row

            rect_bottom = row * (bar_height + bar_spacing) + (
                2 * bar_spacing if row > 2 else 0
            )

            rect = patches.Rectangle(
                (center - width / 2, rect_bottom),
                width,
                bar_height,
                edgecolor=color,
                facecolor=color,
                fill=particle.established,
            )
            bar_ax.add_patch(rect)

            # label = particle.label
            # rect_center_x = center
            # rect_center_y = rect_bottom + bar_height / 2
            #
            # bar_ax.text(
            #     rect_center_x,
            #     rect_center_y,
            #     label,
            #     color=color,
            #     ha='center',
            #     va='center',
            #     fontsize=fontsize,
            # )
        bar_ax.set_xlabel(
            f'{BRANCH_NAME_TO_LATEX["M_Resonance"]} ({BRANCH_NAME_TO_LATEX_UNITS["M_Resonance"]})'
        )
        bin_width_mev = int(1000 / bins)
        hist_ax.set_ylabel(f'Counts / {bin_width_mev} MeV/$c^2$')
        existing_handles, existing_labels = hist_ax.get_legend_handles_labels()
        patch_colors = [colors.red, colors.green, colors.orange, colors.purple]
        patch_labels = [r'$f_0$', r'$a_0$', r'$f_2$', r'$a_2$']
        rect_patches = [
            patches.Patch(facecolor=color, edgecolor=color, label=label)
            for color, label in zip(patch_colors, patch_labels)
        ]
        all_handles = existing_handles + rect_patches
        all_labels = existing_labels + patch_labels
        hist_ax.legend(handles=all_handles, labels=all_labels)
        hist_ax.set_xlim(xmin, xmax)
        bar_ax.set_ylim(
            -bar_spacing * 5,
            max(p.row for p in particles) * (bar_height + bar_spacing)
            + bar_height
            + bar_spacing * 7,
        )
        bar_ax.set_yticks([])
        bar_ax.spines['top'].set_visible(False)
        bar_ax.spines['right'].set_visible(False)
        bar_ax.spines['left'].set_visible(False)
        fig.savefig(output_path)
        plt.close()
