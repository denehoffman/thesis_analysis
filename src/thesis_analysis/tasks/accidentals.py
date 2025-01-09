from pathlib import Path

from numpy.typing import NDArray
from thesis_analysis.constants import get_branch
from thesis_analysis.logger import logger

import luigi
import numpy as np

from thesis_analysis import root_io
from thesis_analysis.paths import Paths
from thesis_analysis.tasks.ccdb import CCDB
from thesis_analysis.tasks.data import GetData
from thesis_analysis.tasks.rcdb import RCDB
from thesis_analysis.utils import CCDBData, RCDBData


class AccidentalsAndPolarization(luigi.Task):
    data_type = luigi.Parameter()
    run_period = luigi.Parameter()

    def requires(self):
        return [GetData(self.data_type, self.run_period), CCDB(), RCDB()]

    def output(self):
        input_path = Path(self.input()[0][0].path)
        return [
            luigi.LocalTarget(
                Path(str(input_path)).parent
                / Path('accidental_subtraction')
                / Path(str(input_path)).name
            )
        ]

    def run(self):
        ccdb_data = Paths.ccdb
        rcdb_data = Paths.rcdb
        input_path = Path(self.input()[0][0].path)
        output_path = (
            Path(str(input_path)).parent
            / Path('accidental_subtraction')
            / Path(str(input_path)).name
        )
        is_mc = self.data_type != 'data'

        branches = [
            get_branch('RunNumber'),
            get_branch('E_Beam'),
            get_branch('Px_Beam'),
            get_branch('Py_Beam'),
            get_branch('Pz_Beam'),
            get_branch('RF'),
            get_branch('Weight'),
        ]

        def process(
            run_number: NDArray,
            e_beam: NDArray,
            px_beam: NDArray,
            py_beam: NDArray,
            pz_beam: NDArray,
            rf: NDArray,
            weight: NDArray,
            ccdb_data: CCDBData,
            rcdb_data: RCDBData,
            *,
            is_mc: bool,
        ) -> bool:
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
            pz_beam[0] = e_beam[0]
            return True

        logger.trace(f'Processing tree {input_path}')
        root_io.process_root_tree(
            input_path,
            output_path,
            branches,
            process,
            ccdb_data,
            rcdb_data,
            is_mc=is_mc,
        )
        logger.trace(f'Finished processing tree {input_path} to {output_path}')
