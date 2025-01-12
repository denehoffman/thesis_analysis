import pickle
from pathlib import Path

from thesis_analysis.logger import logger
from thesis_analysis.utils import CCDBData, RCDBData


class PathsSingleton:
    _instance = None

    def __new__(cls, working_dir: Path = Path.cwd()):
        if cls._instance is None:
            cls._instance = super(PathsSingleton, cls).__new__(cls)
            cls._instance._initialize(working_dir)
        return cls._instance

    def _initialize(self, working_dir: Path = Path.cwd()):
        self.root = working_dir / 'analysis'
        self.datasets = self.root / 'datasets'
        self.data = self.datasets / 'data'
        self.accmc = self.datasets / 'accmc'
        self.genmc = self.datasets / 'genmc'
        self.bkgmc = self.datasets / 'bkgmc'
        self.databases = self.root / 'databases'
        self.fits = self.root / 'fits'
        self.plots = self.root / 'plots'
        self.root.mkdir(parents=True, exist_ok=True)
        self.datasets.mkdir(parents=True, exist_ok=True)
        self.data.mkdir(parents=True, exist_ok=True)
        self.accmc.mkdir(parents=True, exist_ok=True)
        self.genmc.mkdir(parents=True, exist_ok=True)
        self.bkgmc.mkdir(parents=True, exist_ok=True)
        self.databases.mkdir(parents=True, exist_ok=True)
        self.fits.mkdir(parents=True, exist_ok=True)
        self.plots.mkdir(parents=True, exist_ok=True)

    @property
    def ccdb(self) -> CCDBData:
        return pickle.load((self.databases / 'ccdb.pkl').open('rb'))

    @property
    def rcdb(self) -> RCDBData:
        return pickle.load((self.databases / 'rcdb.pkl').open('rb'))


Paths = PathsSingleton()
logger.add(Paths.root / 'analysis.log', level=0)
