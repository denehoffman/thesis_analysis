import luigi

from thesis_analysis.paths import Paths
from thesis_analysis.tasks.scp import SCP


class GetData(luigi.Task):
    data_type = luigi.Parameter()
    run_period = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path_map = {
            'data': {
                's17': '/raid3/nhoffman/analysis/S17/ver52/final/flattree_final.root',
                's18': '/raid3/nhoffman/analysis/S18/ver19/final/flattree_final.root',
                'f18': '/raid3/nhoffman/analysis/F18/ver19/final/flattree_final.root',
                's20': '/raid3/nhoffman/analysis/S20/ver04/final/flattree_final.root',
            },
            'accmc': {
                's17': '/raid3/nhoffman/analysis/MCS17/ver52/final/flattree_final.root',
                's18': '/raid3/nhoffman/analysis/MCS18/ver19/final/flattree_final.root',
                'f18': '/raid3/nhoffman/analysis/MCF18/ver19/final/flattree_final.root',
                's20': '/raid3/nhoffman/analysis/MCS20/ver04/final/flattree_final.root',
            },
            'genmc': {
                's17': '/raid3/nhoffman/analysis/MCGS17/ver52/fiducial/flattree_fiducial.root',
                's18': '/raid3/nhoffman/analysis/MCGS18/ver19/fiducial/flattree_fiducial.root',
                'f18': '/raid3/nhoffman/analysis/MCGF18/ver19/fiducial/flattree_fiducial.root',
                's20': '/raid3/nhoffman/analysis/MCGS20/ver04/fiducial/flattree_fiducial.root',
            },
            'bkgmc': {
                's17': '/raid3/nhoffman/analysis/4PIS17/ver52/final/flattree_final.root',
                's18': '/raid3/nhoffman/analysis/4PIS18/ver19/final/flattree_final.root',
                'f18': '/raid3/nhoffman/analysis/4PIF18/ver19/final/flattree_final.root',
                's20': '/raid3/nhoffman/analysis/4PIS20/ver04/final/flattree_final.root',
            },
        }
        self.remote_path = self.path_map[str(self.data_type)][
            str(self.run_period)
        ]
        self.local_path = (
            getattr(Paths, str(self.data_type)) / f'{self.run_period}.root'
        )

    def requires(self):
        return [
            SCP(
                remote_path=self.remote_path,
                local_path=self.local_path,
            )
        ]

    def output(self):
        return [luigi.LocalTarget(self.local_path)]
