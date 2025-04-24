import pickle
from pathlib import Path

from thesis_analysis.logger import logger
from thesis_analysis.utils import CCDBData, RCDBData


class PathsSingleton:
    _instance: 'PathsSingleton | None' = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PathsSingleton, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.root: Path = Path.cwd() / 'analysis'
        self.datasets: Path = self.root / 'datasets'
        self.data: Path = self.datasets / 'data'
        self.data_original: Path = self.datasets / 'data_original'
        self.accmc: Path = self.datasets / 'accmc'
        self.accmc_original: Path = self.datasets / 'accmc_original'
        self.genmc: Path = self.datasets / 'genmc'
        self.bkgmc: Path = self.datasets / 'bkgmc'
        self.bkgmc_original: Path = self.datasets / 'bkgmc_original'
        self.bggen: Path = self.datasets / 'bggen'
        self.databases: Path = self.root / 'databases'
        self.fits: Path = self.root / 'fits'
        self.reports: Path = self.root / 'reports'
        self.plots: Path = self.root / 'plots'
        if not self._initialized:
            self.root.mkdir(parents=True, exist_ok=True)
            self.datasets.mkdir(parents=True, exist_ok=True)
            self.data.mkdir(parents=True, exist_ok=True)
            self.data_original.mkdir(parents=True, exist_ok=True)
            self.accmc.mkdir(parents=True, exist_ok=True)
            self.accmc_original.mkdir(parents=True, exist_ok=True)
            self.genmc.mkdir(parents=True, exist_ok=True)
            self.bkgmc.mkdir(parents=True, exist_ok=True)
            self.bkgmc_original.mkdir(parents=True, exist_ok=True)
            self.bggen.mkdir(parents=True, exist_ok=True)
            self.databases.mkdir(parents=True, exist_ok=True)
            self.fits.mkdir(parents=True, exist_ok=True)
            self.reports.mkdir(parents=True, exist_ok=True)
            self.plots.mkdir(parents=True, exist_ok=True)
            PathsSingleton._initialized = True

    @property
    def ccdb(self) -> CCDBData:
        ccdb_data: CCDBData = pickle.load(
            (self.databases / 'ccdb.pkl').open('rb')
        )
        return ccdb_data

    @property
    def rcdb(self) -> RCDBData:
        rcdb_data: RCDBData = pickle.load(
            (self.databases / 'rcdb.pkl').open('rb')
        )
        return rcdb_data


Paths = PathsSingleton()
_ = logger.add(Paths.root / 'analysis.log', level=0)
