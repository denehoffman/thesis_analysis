from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot
from uproot.behaviors.TBranch import HasBranches

from thesis_analysis.logger import logger


def read_root_tree(path: str | Path, tree: str = 'kin') -> pd.DataFrame:
    ttree_in = uproot.open(f'{path}:{tree}')
    if not isinstance(ttree_in, HasBranches):
        logger.error(f'{path} has no branches')
        raise IOError(f'{path} has no branches')
    return ttree_in.arrays(library='pd')


def write_root_tree(data: pd.DataFrame, path: str | Path, tree: str = 'kin'):
    data_out = data[data['Weight'] != 0.0]
    tfile_out = uproot.recreate(path)
    df_out_typed = {}
    for key, item in data_out.items():
        logger.info(f'{key} : {item.dtype}')
        if item.dtype == 'awkward':
            df_out_typed[key] = ak.values_astype(
                ak.from_iter(np.vstack([item.to_numpy()]).tolist()), np.float32
            )
        elif item.dtype == 'float64':
            df_out_typed[key] = np.astype(item.to_numpy(), np.float32)
        else:
            df_out_typed[key] = item.to_numpy()
    tfile_out[tree] = df_out_typed
    tfile_out.close()
