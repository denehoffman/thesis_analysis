from pathlib import Path
from typing import final, override

import luigi
import numpy as np
from numpy.typing import NDArray
from thesis_analysis import root_io
from thesis_analysis.constants import get_branch
from thesis_analysis.tasks.accid_and_pol import AccidentalsAndPolarization


@final
class ChiSqDOF(luigi.Task):
    data_type = luigi.Parameter()
    run_period = luigi.Parameter()
    chisqdof = luigi.FloatParameter()

    @override
    def requires(self):
        return [AccidentalsAndPolarization(self.data_type, self.run_period)]

    @override
    def output(self):
        input_path = Path(self.input()[0][0].path)
        return [
            luigi.LocalTarget(
                Path(str(input_path)).parent
                / f'{input_path.stem}_chisqdof_{self.chisqdof:.1f}.root'
            )
        ]

    @override
    def run(self):
        input_path = Path(self.input()[0][0].path)
        output_path = Path(self.output()[0].path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        branches = [
            get_branch('ChiSqDOF'),
            get_branch('M_Resonance'),
        ]

        def process(
            _i: int,
            chisqdof: NDArray[np.float32],
            m_resonance: NDArray[np.float32],
        ) -> bool:
            return chisqdof[0] <= self.chisqdof and (
                1.0 <= m_resonance[0] <= 2.0
            )

        root_io.process_root_tree(
            input_path,
            output_path,
            branches,
            process,
        )
