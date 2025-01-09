import luigi

from thesis_analysis.paths import Paths
from thesis_analysis.tasks.scp import SCP


class GetDatabases(luigi.Task):
    path_map = [
        (
            '/home/gluex2/gluexdb/ccdb_2024_05_08.sqlite',
            str(Paths.databases / 'ccdb.sqlite'),
        ),
        (
            '/home/gluex2/gluexdb/rcdb_2024_05_08.sqlite',
            str(Paths.databases / 'rcdb.sqlite'),
        ),
        (
            '/raid3/nhoffman/analysis/pol_hists/S17.root',
            str(Paths.databases / 's17.root'),
        ),
        (
            '/raid3/nhoffman/analysis/pol_hists/S18.root',
            str(Paths.databases / 's18.root'),
        ),
        (
            '/raid3/nhoffman/analysis/pol_hists/F18.root',
            str(Paths.databases / 'f18.root'),
        ),
        (
            '/raid3/nhoffman/analysis/pol_hists/S20.root',
            str(Paths.databases / 's20.root'),
        ),
    ]

    def requires(self):
        return [
            SCP(
                remote_path=remote_path,
                local_path=local_path,
            )
            for remote_path, local_path in self.path_map
        ]

    def output(self):
        return [
            luigi.LocalTarget(local_path) for _, local_path in self.path_map
        ]
