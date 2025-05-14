from pathlib import Path
from typing import Any, final, override

import laddu as ld
import luigi
import numpy as np
from numpy.typing import NDArray

from thesis_analysis import root_io
from thesis_analysis.constants import RootBranchDict, get_branch
from thesis_analysis.tasks.chisqdof import ChiSqDOF


@final
class BaryonCut(luigi.Task):
    data_type = luigi.Parameter()
    run_period = luigi.Parameter()
    chisqdof = luigi.FloatParameter()
    ksb_costheta = luigi.FloatParameter()
    cut_baryons = luigi.OptionalBoolParameter(True)

    @override
    def requires(self):
        return [ChiSqDOF(self.data_type, self.run_period, self.chisqdof)]

    @override
    def output(self):
        input_path = Path(self.input()[0][0].path)
        return [
            luigi.LocalTarget(
                Path(str(input_path)).parent
                / f'{input_path.stem}_ksb_costheta_{self.ksb_costheta:.2f}{"mesons" if self.cut_baryons else "baryons"}.root'
            )
        ]

    @override
    def run(self):
        input_path = Path(self.input()[0][0].path)
        output_path = Path(self.output()[0].path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        branches = [
            get_branch('Px_FinalState', 3),
            get_branch('Py_FinalState', 3),
            get_branch('Pz_FinalState', 3),
            get_branch('E_FinalState', 3),
        ]

        def process(
            _i: int,
            px_finalstate: NDArray[np.float32],
            py_finalstate: NDArray[np.float32],
            pz_finalstate: NDArray[np.float32],
            e_finalstate: NDArray[np.float32],
        ) -> bool:
            p_p4 = ld.Vector4(
                px_finalstate[0],
                py_finalstate[0],
                pz_finalstate[0],
                e_finalstate[0],
            )
            ks1_p4 = ld.Vector4(
                px_finalstate[1],
                py_finalstate[1],
                pz_finalstate[1],
                e_finalstate[1],
            )
            ks2_p4 = ld.Vector4(
                px_finalstate[2],
                py_finalstate[2],
                pz_finalstate[2],
                e_finalstate[2],
            )
            com_frame = p_p4 + ks1_p4 + ks2_p4
            ksb_p4 = (
                ks1_p4
                if ks1_p4.vec3.costheta < ks2_p4.vec3.costheta
                else ks2_p4
            )
            ksb_p4_com = ksb_p4.boost(-com_frame.beta)
            if not self.cut_baryons:
                return ksb_p4_com.vec3.costheta <= float(self.ksb_costheta)
            return ksb_p4_com.vec3.costheta > float(self.ksb_costheta)

        root_io.process_root_tree(
            input_path,
            output_path,
            branches,
            process,
        )


def get_ksb_costheta(
    data: RootBranchDict,
) -> dict[str, np.typing.NDArray[Any]]:
    data: dict[str, np.typing.NDArray[Any]] = dict(data)  # pyright:ignore[reportAssignmentType]
    data['Proton_P4'] = np.array(
        [
            ld.Vector4(px[0], py[0], pz[0], e[0])
            for px, py, pz, e in zip(
                data['Px_FinalState'],
                data['Py_FinalState'],
                data['Pz_FinalState'],
                data['E_FinalState'],
            )
        ]
    )
    data['KShort1_P4'] = np.array(
        [
            ld.Vector4(px[1], py[1], pz[1], e[1])
            for px, py, pz, e in zip(
                data['Px_FinalState'],
                data['Py_FinalState'],
                data['Pz_FinalState'],
                data['E_FinalState'],
            )
        ]
    )
    data['KShort2_P4'] = np.array(
        [
            ld.Vector4(px[2], py[2], pz[2], e[2])
            for px, py, pz, e in zip(
                data['Px_FinalState'],
                data['Py_FinalState'],
                data['Pz_FinalState'],
                data['E_FinalState'],
            )
        ]
    )
    data['KShortF_P4'] = np.array(
        [
            ks1 if ks1.vec3.costheta > ks2.vec3.costheta else ks2
            for ks1, ks2 in zip(data['KShort1_P4'], data['KShort2_P4'])
        ]
    )
    data['KShortB_P4'] = np.array(
        [
            ks2 if ks1.vec3.costheta > ks2.vec3.costheta else ks1
            for ks1, ks2 in zip(data['KShort1_P4'], data['KShort2_P4'])
        ]
    )
    com_frame = [
        ks1 + ks2 + p
        for ks1, ks2, p in zip(
            data['KShort1_P4'],
            data['KShort2_P4'],
            data['Proton_P4'],
        )
    ]
    data['KShortB_P4_COM'] = np.array(
        [
            ksb.boost(-com.beta)
            for ksb, com in zip(data['KShortB_P4'], com_frame)
        ]
    )
    data['KShortB_CosTheta'] = np.array(
        [ksb.vec3.costheta for ksb in data['KShortB_P4_COM']]
    )
    return data
