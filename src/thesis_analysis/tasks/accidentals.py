from pathlib import Path
from thesis_analysis.logger import logger

import luigi
import numpy as np

from thesis_analysis import root_io
from thesis_analysis.paths import Paths
from thesis_analysis.tasks.ccdb import CCDB
from thesis_analysis.tasks.data import GetData
from thesis_analysis.tasks.rcdb import RCDB


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
        logger.debug('reading tree')
        data_in = root_io.read_root_tree(input_path)
        logger.debug('tree loaded')
        is_mc = self.data_type != 'data'
        logger.debug('getting accidental weights')
        weights = ccdb_data.get_accidental_weight(
            data_in['RunNumber'],
            data_in['E_Beam'],
            data_in['RF'],
            data_in['Weight'],
            is_mc=is_mc,
        )
        logger.debug(f'event 3: weight = {weights[2]}')
        eps_x, eps_y, weight_mult = rcdb_data.get_eps_xy(
            data_in['RunNumber'], data_in['E_Beam']
        )
        logger.debug(
            f'EPS for event 3: {eps_x[2]} {eps_y[2]} weight = {weights[2]} * {weight_mult[2]}'
        )
        data_in['Weight'] = np.array(
            weights * weight_mult,
            dtype=np.float32,
        )
        data_in['Px_Beam'] = np.array(
            eps_x,
            dtype=np.float32,
        )
        data_in['Py_Beam'] = np.array(
            eps_y,
            dtype=np.float32,
        )
        data_in['Pz_Beam'] = data_in['E_Beam']
        output_path = (
            Path(str(input_path)).parent
            / Path('accidental_subtraction')
            / Path(str(input_path)).name
        )
        root_io.write_root_tree(data_in, output_path)
