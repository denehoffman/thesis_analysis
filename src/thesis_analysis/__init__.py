import itertools

import luigi

from thesis_analysis.constants import DATA_TYPES, RUN_PERIODS
from thesis_analysis.tasks.accidentals import AccidentalsAndPolarization
from thesis_analysis.tasks.rcdb import RCDB


class RunAll(luigi.WrapperTask):
    def requires(self):
        return [
            AccidentalsAndPolarization(data_type, run_period)
            for data_type, run_period in itertools.product(
                DATA_TYPES, RUN_PERIODS
            )
        ] + [RCDB()]
