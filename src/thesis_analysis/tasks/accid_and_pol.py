from pathlib import Path
from typing import final, override

import luigi
import numpy as np
from numpy.typing import NDArray

from thesis_analysis import root_io
from thesis_analysis.constants import get_branch
from thesis_analysis.paths import Paths
from thesis_analysis.tasks.ccdb import CCDB
from thesis_analysis.tasks.data import GetData
from thesis_analysis.tasks.rcdb import RCDB
from thesis_analysis.utils import CCDBData, RCDBData


@final
class AccidentalsAndPolarization(luigi.Task):
    data_type = luigi.Parameter()
    run_period = luigi.Parameter()

    @override
    def requires(self):
        return [GetData(self.data_type, self.run_period), CCDB(), RCDB()]

    @override
    def output(self):
        input_path = Path(self.input()[0][0].path)
        return [
            luigi.LocalTarget(
                Path(str(input_path)).parent / f'{input_path.stem}_accpol.root'
            )
        ]

    @override
    def run(self):
        ccdb_data = Paths.ccdb
        rcdb_data = Paths.rcdb
        input_path = Path(self.input()[0][0].path)
        output_path = Path(self.output()[0].path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        is_mc = str(self.data_type) != 'data'

        branches = [
            get_branch('RunNumber'),
            get_branch('EventNumber'),
            get_branch('ComboNumber'),
            get_branch('ChiSqDOF'),
            get_branch('E_Beam'),
            get_branch('Px_Beam'),
            get_branch('Py_Beam'),
            get_branch('Pz_Beam'),
            get_branch('RF'),
            get_branch('Weight'),
        ]

        best_combo_map: dict[tuple[np.uint32, np.ulonglong], np.uint32] = {}
        best_combo_chi2_map: dict[
            tuple[np.uint32, np.ulonglong], np.float64
        ] = {}

        def scan_combos(
            _i: int,
            run_number: NDArray[np.uint32],
            event_number: NDArray[np.ulonglong],
            combo_number: NDArray[np.uint32],
            chisqdof: NDArray[np.float32],
            *_args,
            **_kwargs,
        ):
            best_combo_chi2 = best_combo_chi2_map.get(
                (run_number[0], event_number[0]), np.inf
            )
            if chisqdof[0] < best_combo_chi2:
                best_combo_chi2_map[(run_number[0], event_number[0])] = (
                    chisqdof[0]
                )
                best_combo_map[(run_number[0], event_number[0])] = combo_number[
                    0
                ]

        def process(
            _i: int,
            run_number: NDArray[np.uint32],
            event_number: NDArray[np.ulonglong],
            combo_number: NDArray[np.uint32],
            _chisqdof: NDArray[np.float32],
            e_beam: NDArray[np.float32],
            px_beam: NDArray[np.float32],
            py_beam: NDArray[np.float32],
            pz_beam: NDArray[np.float32],
            rf: NDArray[np.float32],
            weight: NDArray[np.float32],
            ccdb_data: CCDBData,
            rcdb_data: RCDBData,
            *,
            is_mc: bool,
        ) -> bool:
            if (
                best_combo_map.get((run_number[0], event_number[0]), -1)
                != combo_number
            ):
                return False
            new_weight = ccdb_data.get_accidental_weight(
                run_number[0], e_beam[0], rf[0], weight[0], is_mc=is_mc
            )
            eps_x, eps_y, polarized = rcdb_data.get_eps_xy(
                run_number[0], e_beam[0]
            )
            if not polarized:
                return False
            weight[0] = weight[0] * new_weight
            px_beam[0] = eps_x
            py_beam[0] = eps_y
            pz_beam[0] = 0.0
            return True

        root_io.double_process_root_tree(
            input_path,
            output_path,
            branches,
            scan_combos,
            process,
            ccdb_data,
            rcdb_data,
            is_mc=is_mc,
        )
