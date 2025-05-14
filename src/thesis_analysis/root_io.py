from pathlib import Path
from typing import Any, Callable

import numpy as np
import uproot
from uproot.behaviors.TBranch import HasBranches

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
    root: bool = True,
) -> RootBranchDict:
    data_dict: dict[str, list[np.generic]] = {
        branch.name: [] for branch in branches
    }
    if root:
        infile = ROOT.TFile.Open(str(in_path), 'READ')
        try:
            intree = ROOT.gDirectory.Get(tree)

            branch_arrays = []
            for branch in branches:
                arr = branch.get_array()
                if arr is not None:
                    intree.SetBranchAddress(branch.name, arr)
                branch_arrays.append(arr)

            for i in range(intree.GetEntries()):
                intree.GetEntry(i)
                for j, branch in enumerate(branch_arrays):
                    if branch is None:  # string type
                        data_dict[branches[j].name].append(
                            str(getattr(intree, branches[j].name))
                        )
                    elif branches[j].dim == 1:
                        data_dict[branches[j].name].append(branch[0])
                    else:
                        data_dict[branches[j].name].append(branch.copy())
        finally:
            infile.Close()
    else:
        infile = uproot.open(in_path)
        intree = infile[tree]
        assert isinstance(intree, HasBranches)
        data_dict = intree.arrays(
            [branch.name for branch in branches], library='np'
        )
    branch_dict: RootBranchDict = {
        key: np.array(value, dtype=branch.dtype)
        if branch.dim == 1
        else np.stack(value, dtype=branch.dtype)
        for branch, (key, value) in zip(branches, data_dict.items())
    }  # pyright:ignore[reportAssignmentType]
    return branch_dict


def concatenate_branches(
    input_paths: list[Path],
    branches: list[RootBranch],
    *,
    tree: str = 'kin',
    root: bool = False,
) -> RootBranchDict:
    dfs = [
        get_branches(
            input_path,
            branches,
            tree=tree,
            root=root,
        )
        for input_path in input_paths
    ]
    res: RootBranchDict = {
        branch.name: np.concatenate([df[branch.name] for df in dfs])
        for branch in branches
    }  # pyright:ignore[reportAssignmentType]
    return res


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
