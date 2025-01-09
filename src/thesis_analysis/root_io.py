from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import awkward as ak
import numpy as np
from numpy.typing import DTypeLike, NDArray
import pandas as pd
import uproot
from uproot.behaviors.TBranch import HasBranches

from thesis_analysis.logger import logger

import ROOT


# def read_root_tree(path: str | Path, tree: str = 'kin') -> pd.DataFrame:
#     ttree_in = uproot.open(f'{path}:{tree}')
#     if not isinstance(ttree_in, HasBranches):
#         logger.error(f'{path} has no branches')
#         raise IOError(f'{path} has no branches')
#     return ttree_in.arrays(library='pd')
#
#
# def write_root_tree(data: pd.DataFrame, path: str | Path, tree: str = 'kin'):
#     data_out = data[data['Weight'] != 0.0]
#     tfile_out = uproot.recreate(path)
#     df_out_typed = {}
#     for key, item in data_out.items():
#         logger.info(f'{key} : {item.dtype}')
#         if item.dtype == 'awkward':
#             df_out_typed[key] = ak.values_astype(
#                 ak.from_iter(np.vstack([item.to_numpy()]).tolist()), np.float32
#             )
#         elif item.dtype == 'float64':
#             df_out_typed[key] = np.astype(item.to_numpy(), np.float32)
#         else:
#             df_out_typed[key] = item.to_numpy()
#     tfile_out[tree] = df_out_typed
#     tfile_out.close()
#


@dataclass
class RootBranch:
    name: str
    dtype: DTypeLike
    dim: int = 1

    def get_array(self):
        return np.zeros(self.dim, dtype=self.dtype)


def process_root_tree(
    in_path: str | Path,
    out_path: str | Path,
    branches: list[RootBranch],
    callback: Callable[..., bool],
    *args,
    tree: str = 'kin',
    **kwargs,
):
    infile = ROOT.TFile.Open(str(in_path), 'READ')
    intree = infile.Get(tree)

    outfile = ROOT.TFile.Open(str(out_path), 'RECREATE')
    outtree = intree.CloneTree(0)

    branch_arrays = []
    for branch in branches:
        arr = branch.get_array()
        intree.SetBranchAddress(branch.name, arr)
        branch_arrays.append(arr)

    for i in range(intree.GetEntries()):
        intree.GetEntry(i)
        fill = callback(*branch_arrays, *args, **kwargs)
        if fill:
            outtree.Fill()

    outtree.Write()
    outfile.Close()
    infile.Close()
