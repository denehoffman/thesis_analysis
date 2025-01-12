from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import DTypeLike

try:
    import ROOT
except:
    print('ROOT not found, expect future errors!')
    pass


@dataclass
class RootBranch:
    name: str
    dtype: DTypeLike
    dim: int = 1

    def get_array(self):
        return np.zeros(self.dim, dtype=self.dtype)


def get_branches(
    in_path: str | Path,
    branches: list[RootBranch],
    *,
    tree: str = 'kin',
) -> dict[str, np.ndarray]:
    output = {branch.name: [] for branch in branches}
    infile = ROOT.TFile.Open(str(in_path), 'READ')
    intree = ROOT.gDirectory.Get(tree)

    branch_arrays = []
    for branch in branches:
        arr = branch.get_array()
        intree.SetBranchAddress(branch.name, arr)
        branch_arrays.append(arr)

    for i in range(intree.GetEntries()):
        intree.GetEntry(i)
        for i, branch in enumerate(branch_arrays):
            output[branches[i].name].append(branch[0])

    infile.Close()
    return {key: np.array(value) for key, value in output.items()}


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
        fill = callback(i, *branch_arrays, *args, **kwargs)
        if fill:
            outtree.Fill()

    outtree.Write()
    outfile.Close()
    infile.Close()
