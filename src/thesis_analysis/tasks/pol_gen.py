from pathlib import Path
from typing import final, override

import luigi
import numpy as np
from numpy.typing import NDArray

from thesis_analysis import root_io
from thesis_analysis.constants import get_branch
from thesis_analysis.paths import Paths
from thesis_analysis.tasks.data import GetData
from thesis_analysis.tasks.rcdb import RCDB
from thesis_analysis.utils import RCDBData


@final
class PolarizeGenerated(luigi.Task):
    run_period = luigi.Parameter()

    @override
    def requires(self):
        return [GetData('genmc', self.run_period), RCDB()]

    @override
    def output(self):
        input_path = Path(self.input()[0][0].path)
        return [
            luigi.LocalTarget(
                Path(str(input_path)).parent / f'{input_path.stem}_pol.root'
            )
        ]

    @override
    def run(self):
        rcdb_data = Paths.rcdb
        input_path = Path(self.input()[0][0].path)
        output_path = Path(self.output()[0].path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        branches = [
            get_branch('RunNumber'),
            get_branch('E_Beam'),
            get_branch('Px_Beam'),
            get_branch('Py_Beam'),
            get_branch('Pz_Beam'),
        ]

        def process(
            _i: int,
            run_number: NDArray[np.uint32],
            e_beam: NDArray[np.float32],
            px_beam: NDArray[np.float32],
            py_beam: NDArray[np.float32],
            pz_beam: NDArray[np.float32],
            rcdb_data: RCDBData,
        ) -> bool:
            eps_x, eps_y, polarized = rcdb_data.get_eps_xy(
                run_number[0], e_beam[0]
            )
            if not polarized:
                return False
            px_beam[0] = eps_x
            py_beam[0] = eps_y
            pz_beam[0] = 0.0
            return True

        root_io.process_root_tree(
            input_path,
            output_path,
            branches,
            process,
            rcdb_data,
        )
