from pathlib import Path

import luigi
from numpy.typing import NDArray
from thesis_analysis import root_io
from thesis_analysis.constants import get_branch
from thesis_analysis.tasks.accid_and_pol import AccidentalsAndPolarization


class ChiSqDOF(luigi.Task):
    data_type = luigi.Parameter()
    run_period = luigi.Parameter()
    chisqdof = luigi.FloatParameter()

    def requires(self):
        return [AccidentalsAndPolarization(self.data_type, self.run_period)]

    def output(self):
        input_path = Path(self.input()[0][0].path)
        return [
            luigi.LocalTarget(
                Path(str(input_path)).parent
                / Path(f'chisqdof_{self.chisqdof:.1f}')
                / Path(str(input_path)).name
            )
        ]

    def run(self):
        input_path = Path(self.input()[0][0].path)
        output_dir = Path(str(input_path)).parent / Path(
            f'chisqdof_{self.chisqdof:.1f}'
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / Path(str(input_path)).name

        branches = [
            get_branch('ChiSqDOF'),
            get_branch('M_Resonance'),
        ]

        def process(
            _i: int,
            chisqdof: NDArray,
            m_resonance: NDArray,
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
