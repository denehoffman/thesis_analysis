import pickle
import sqlite3

import luigi
import uproot
from typing_extensions import override
from uproot.behaviors.TBranch import HasBranches

from thesis_analysis.constants import get_pol_angle, get_run_period
from thesis_analysis.logger import logger
from thesis_analysis.paths import Paths
from thesis_analysis.tasks.databases import GetDatabases
from thesis_analysis.utils import Histogram, RCDBData


class RCDB(luigi.Task):
    @override
    def requires(self):
        return [GetDatabases()]

    @override
    def output(self):
        return [luigi.LocalTarget(Paths.databases / 'rcdb.pkl')]

    @override
    def run(self):
        angles = {}
        with sqlite3.connect(str(Paths.databases / 'rcdb.sqlite')) as rcdb:
            cursor = rcdb.cursor()
            query = """
            SELECT r.number, c.float_value
            FROM conditions c
            JOIN condition_types ct ON c.condition_type_id = ct.id
            JOIN runs r ON c.run_number = r.number
            WHERE ct.name = 'polarization_angle'
            ORDER BY r.number
            """
            cursor.execute(query)
            pol_angle_results = cursor.fetchall()
            for run_number, angle_deg in pol_angle_results:
                run_period = get_run_period(run_number)
                angle_deg = str(angle_deg)
                pol_angle = get_pol_angle(run_period, angle_deg)
                if pol_angle:
                    angles[run_number] = (
                        run_period,
                        angle_deg.split('.')[0],
                        pol_angle,
                    )
        magnitudes = {}
        pol_hists = {
            's17': Paths.databases / 's17.root',
            's18': Paths.databases / 's18.root',
            'f18': Paths.databases / 'f18.root',
            's20': Paths.databases / 's20.root',
        }
        for rp, hist_path in pol_hists.items():
            hists = {}
            tfile = uproot.open(hist_path)  # pyright:ignore[reportUnknownVariableType]
            for pol in ['0', '45', '90', '135']:
                hist = tfile[f'hPol{pol}']  # pyright:ignore[reportUnknownVariableType]
                if isinstance(hist, HasBranches | uproot.ReadOnlyDirectory):
                    logger.error(f'Error reading histograms from {hist_path}')
                    raise IOError(f'Error reading histograms from {hist_path}')
                mags, edges = hist.to_numpy()  # pyright:ignore[reportUnknownVariableType]
                hists[pol] = Histogram(mags, edges)
            magnitudes[rp] = hists
        pickle.dump(
            RCDBData(angles, magnitudes),
            (Paths.databases / 'rcdb.pkl').open('wb'),
        )
