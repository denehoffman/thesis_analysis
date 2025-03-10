import pickle
import sqlite3

import luigi
from typing_extensions import override

from thesis_analysis.paths import Paths
from thesis_analysis.tasks.databases import GetDatabases
from thesis_analysis.utils import CCDBData, ScalingFactors


class CCDB(luigi.Task):
    @override
    def requires(self):
        return [GetDatabases()]

    @override
    def output(self):
        return [luigi.LocalTarget(Paths.databases / 'ccdb.pkl')]

    @override
    def run(self):
        with sqlite3.connect(str(Paths.databases / 'ccdb.sqlite')) as ccdb:
            cursor = ccdb.cursor()
            query = """
            SELECT rr.runMin, rr.runMax, cs.vault
            FROM directories d
            JOIN typeTables tt ON d.id = tt.directoryId
            JOIN constantSets cs ON tt.id = cs.constantTypeId
            JOIN assignments a ON cs.id = a.constantSetId
            JOIN runRanges rr ON a.runRangeId = rr.id
            LEFT JOIN variations v ON a.variationId = v.id
            WHERE d.name = 'ANALYSIS'
            AND tt.name = 'accidental_scaling_factor'
            AND v.name IS 'default'
            ORDER BY rr.runMin, a.created DESC
            """
            cursor.execute(query)
            asf_results = cursor.fetchall()
            factors = {}
            for run_min, run_max, vault in asf_results:
                data = [float(v) for v in vault.split('|')]
                fb = tuple(data[:8])
                scale_factors = ScalingFactors(
                    fb[0], fb[2], fb[4], fb[6], fb[7]
                )
                for run in range(run_min, run_max + 1):
                    factors[run] = scale_factors
            pickle.dump(
                CCDBData(factors),
                (Paths.databases / 'ccdb.pkl').open('wb'),
            )
