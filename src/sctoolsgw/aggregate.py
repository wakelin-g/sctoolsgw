from typing import Union
import anndata as ad
from anndata import AnnData
import numpy as np
from numpy.random import RandomState
import decoupler as dc


def aggregate(
    adata: AnnData,
    sample_col: Union[None, str] = None,
    groups_col: Union[None, str] = None,
    use_raw: bool = False,
    mode: str = "sum",
    min_cells: int = 100,
    min_counts: int = 1000,
    n_reps: int = 3,
    random_state: Union[RandomState, int, None] = 0,
) -> AnnData:

    if not isinstance(random_state, np.random.RandomState):
        if type(random_state) == int:
            random_state = np.random.RandomState(random_state)
        elif type(random_state) == None:
            random_state = np.random.RandomState(0)
        else:
            msg = "the type of `random_state` should be one of: int, None, np.random.mtrand.RandomState"
            raise ValueError(msg)

    obs_idx = np.arange(adata.n_obs)
    random_state.shuffle(obs_idx)
    obs_idx = np.array_split(obs_idx, n_reps)

    adata_split = {
        i: dc.get_pseudobulk(
            adata=adata[idx, :],
            sample_col=sample_col,
            groups_col=groups_col,
            use_raw=use_raw,
            mode=mode,
            min_cells=min_cells,
            min_counts=min_counts,
        )
        for i, idx in zip(range(len(obs_idx)), obs_idx)
    }

    res = ad.concat(adatas=adata_split, label="pb_num")  # type: ignore
    return res
