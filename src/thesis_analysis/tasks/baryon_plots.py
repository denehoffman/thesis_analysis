# pyright: reportUnnecessaryComparison=false
from pathlib import Path
from typing import Any, final, override

import laddu as ld
import luigi
import matplotlib.pyplot as plt
import matplotlib.style as mpl_style
import numpy as np

import thesis_analysis.colors as colors
from thesis_analysis import root_io
from thesis_analysis.constants import (
    DATA_TYPE_TO_LATEX,
    RUN_PERIODS,
    RootBranchDict,
    get_branch,
)
from thesis_analysis.utils import get_plot_paths, get_plot_requirements


@final
class BaryonPlot(luigi.Task):
    data_type = luigi.Parameter()
    bins = luigi.IntParameter()
    original = luigi.BoolParameter(False)
    chisqdof = luigi.OptionalFloatParameter(None)
    ksb_costheta = luigi.OptionalFloatParameter(None)
    cut_baryons = luigi.OptionalBoolParameter(True)
    splot_method = luigi.OptionalParameter(None)
    nsig = luigi.OptionalIntParameter(None)
    nbkg = luigi.OptionalIntParameter(None)

    @override
    def requires(self):
        return get_plot_requirements(
            self.data_type,
            self.original,
            self.chisqdof,
            self.ksb_costheta,
            self.cut_baryons,
            self.splot_method,
            self.nsig,
            self.nbkg,
        )

    @override
    def output(self):
        return get_plot_paths(
            [
                'baryon_ksb_costheta',
                'baryon_ksb_costheta_v_m_ksbp',
                'baryon_ksb_costheta_v_m_ksks',
                'baryon_m_ksbp',
            ],
            self.data_type,
            self.original,
            self.chisqdof,
            self.ksb_costheta,
            self.cut_baryons,
            self.splot_method,
            self.nsig,
            self.nbkg,
        )

    @override
    def run(self):
        input_paths = [
            Path(self.input()[i][0].path) for i in range(len(RUN_PERIODS))
        ]
        output_path_ksb_costheta = self.output()[0].path
        output_path_ksbp_costheta_v_ksbp = self.output()[1].path
        output_path_ksbp_costheta_v_ksks = self.output()[2].path
        output_path_m_ksbp = self.output()[3].path

        bins = int(self.bins)  # pyright:ignore[reportArgumentType]

        branches = [
            get_branch('E_FinalState', dim=3),
            get_branch('Px_FinalState', dim=3),
            get_branch('Py_FinalState', dim=3),
            get_branch('Pz_FinalState', dim=3),
            get_branch('M_Resonance'),
            get_branch('Weight'),
        ]

        flat_data = root_io.concatenate_branches(
            input_paths, branches, root=False
        )
        flat_data = get_fs_branches(flat_data)
        mpl_style.use('thesis_analysis.thesis')
        # Baryon CosTheta
        fig, ax = plt.subplots()
        ax.hist(
            flat_data['KShortB_CosTheta'],
            weights=flat_data['Weight'],
            bins=100,
            range=(-1, 1),
            color=colors.blue,
            histtype='step',
            density=True,
            label=DATA_TYPE_TO_LATEX[self.data_type],
        )
        bin_width = 1.0 / 100
        ax.set_ylabel(f'Normalized Counts / {bin_width:.2f}')
        ax.legend()
        fig.savefig(output_path_ksb_costheta)
        plt.close()

        # Baryon vs Baryon CosTheta
        fig, ax = plt.subplots()
        ax.hist2d(
            flat_data['M_Baryon'],
            flat_data['KShortB_CosTheta'],
            weights=flat_data['Weight'],
            bins=[bins, 100],
            range=[(1.4, 3.7), (-1, 1)],
        )
        ax.set_xlabel('Invariant Mass of $K_{S,B}^0 p$')
        ax.set_ylabel(r'$\cos\theta$ of $K_{S,B}^0$')
        fig.savefig(output_path_ksbp_costheta_v_ksbp)
        plt.close()

        # Meson v Baryon CosTheta
        fig, ax = plt.subplots()
        ax.hist2d(
            flat_data['M_Meson'],
            flat_data['KShortB_CosTheta'],
            weights=flat_data['Weight'],
            bins=[bins, 100],
            range=[(0.9, 3.0), (-1, 1)],
        )
        ax.set_xlabel('Invariant Mass of $K_{S,1}^0 K_{S,2}^0$')
        ax.set_ylabel(r'$\cos\theta$ of $K_{S,B}^0$')
        fig.savefig(output_path_ksbp_costheta_v_ksks)
        plt.close()

        # Baryon Mass
        fig, ax = plt.subplots()
        ax.hist(
            flat_data['M_Baryon'],
            weights=flat_data['Weight'],
            bins=bins,
            range=(1.4, 3.7),
        )
        ax.set_xlabel('Invariant Mass of $K_{S,B}^0 p$')
        bin_width = bins / ((3.7 - 1.4) * 100)
        ax.set_ylabel(f'Normalized Counts / {bin_width:.2f}')
        fig.savefig(output_path_m_ksbp)
        plt.close()


def get_fs_branches(
    flat_data: RootBranchDict,
) -> dict[str, np.typing.NDArray[Any]]:
    flat_data: dict[str, np.typing.NDArray[Any]] = dict(flat_data)  # pyright:ignore[reportAssignmentType]
    flat_data['Proton_P4'] = np.array(
        [
            ld.Vector4(px[0], py[0], pz[0], e[0])
            for px, py, pz, e in zip(
                flat_data['Px_FinalState'],
                flat_data['Py_FinalState'],
                flat_data['Pz_FinalState'],
                flat_data['E_FinalState'],
            )
        ]
    )
    flat_data['KShort1_P4'] = np.array(
        [
            ld.Vector4(px[1], py[1], pz[1], e[1])
            for px, py, pz, e in zip(
                flat_data['Px_FinalState'],
                flat_data['Py_FinalState'],
                flat_data['Pz_FinalState'],
                flat_data['E_FinalState'],
            )
        ]
    )
    flat_data['KShort2_P4'] = np.array(
        [
            ld.Vector4(px[2], py[2], pz[2], e[2])
            for px, py, pz, e in zip(
                flat_data['Px_FinalState'],
                flat_data['Py_FinalState'],
                flat_data['Pz_FinalState'],
                flat_data['E_FinalState'],
            )
        ]
    )
    flat_data['KShortF_P4'] = np.array(
        [
            ks1 if ks1.vec3.costheta > ks2.vec3.costheta else ks2
            for ks1, ks2 in zip(
                flat_data['KShort1_P4'], flat_data['KShort2_P4']
            )
        ]
    )
    flat_data['KShortB_P4'] = np.array(
        [
            ks2 if ks1.vec3.costheta > ks2.vec3.costheta else ks1
            for ks1, ks2 in zip(
                flat_data['KShort1_P4'], flat_data['KShort2_P4']
            )
        ]
    )
    com_frame = [
        ks1 + ks2 + p
        for ks1, ks2, p in zip(
            flat_data['KShort1_P4'],
            flat_data['KShort2_P4'],
            flat_data['Proton_P4'],
        )
    ]
    flat_data['M_Baryon'] = np.array(
        [
            (ksb + proton).m
            for ksb, proton in zip(
                flat_data['KShortB_P4'], flat_data['Proton_P4']
            )
        ]
    )
    flat_data['KShortB_P4_COM'] = np.array(
        [
            ksb.boost(-com.beta)
            for ksb, com in zip(flat_data['KShortB_P4'], com_frame)
        ]
    )
    flat_data['KShortB_CosTheta'] = np.array(
        [ksb.vec3.costheta for ksb in flat_data['KShortB_P4_COM']]
    )
    flat_data['M_Meson'] = np.array(
        [
            (ks1 + ks2).m
            for ks1, ks2 in zip(
                flat_data['KShort1_P4'], flat_data['KShort2_P4']
            )
        ]
    )
    return flat_data
