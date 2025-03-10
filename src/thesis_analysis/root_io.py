# pyright: basic
from pathlib import Path
from typing import Any, Callable

import numpy as np

from thesis_analysis.constants import RootBranch, RootBranchDict

try:
    import ROOT
except Exception:
    print('ROOT not found, expect future errors!')
    pass


def get_branches(
    in_path: str | Path,
    branches: list[RootBranch],
    *,
    tree: str = 'kin',
) -> RootBranchDict:
    data_dict: dict[str, list[np.generic]] = {
        branch.name: [] for branch in branches
    }
    infile = ROOT.TFile.Open(str(in_path), 'READ')  # pyright:ignore
    intree = ROOT.gDirectory.Get(tree)  # pyright:ignore

    branch_arrays = []
    for branch in branches:
        arr = branch.get_array()
        intree.SetBranchAddress(branch.name, arr)
        branch_arrays.append(arr)

    for i in range(intree.GetEntries()):
        intree.GetEntry(i)
        for i, branch in enumerate(branch_arrays):
            data_dict[branches[i].name].append(branch[0])

    infile.Close()
    branch_dict: RootBranchDict = {
        key: np.array(value) for key, value in data_dict.items()
    }  # pyright:ignore[reportAssignmentType]
    return branch_dict


def process_root_tree(
    in_path: str | Path,
    out_path: str | Path,
    branches: list[RootBranch],
    callback: Callable[..., bool],
    *args,
    tree: str = 'kin',
    **kwargs,
):
    infile = ROOT.TFile.Open(str(in_path), 'READ')  # pyright:ignore
    intree = infile.Get(tree)

    outfile = ROOT.TFile.Open(str(out_path), 'RECREATE')  # pyright:ignore
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


def double_process_root_tree(
    in_path: str | Path,
    out_path: str | Path,
    branches: list[RootBranch],
    scan_callback: Callable[..., Any],
    reduce_callback: Callable[..., bool],
    *args,
    tree: str = 'kin',
    **kwargs,
):
    infile = ROOT.TFile.Open(str(in_path), 'READ')  # pyright:ignore
    intree = infile.Get(tree)

    outfile = ROOT.TFile.Open(str(out_path), 'RECREATE')  # pyright:ignore
    outtree = intree.CloneTree(0)

    branch_arrays = []
    for branch in branches:
        arr = branch.get_array()
        intree.SetBranchAddress(branch.name, arr)
        branch_arrays.append(arr)

    for i in range(intree.GetEntries()):
        intree.GetEntry(i)
        scan_callback(i, *branch_arrays, *args, **kwargs)

    for i in range(intree.GetEntries()):
        intree.GetEntry(i)
        fill = reduce_callback(i, *branch_arrays, *args, **kwargs)
        if fill:
            outtree.Fill()

    outtree.Write()
    outfile.Close()
    infile.Close()
