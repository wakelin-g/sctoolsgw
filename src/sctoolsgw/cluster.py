import scanpy as sc
from anndata import AnnData
import numpy as np
from numpy.random.mtrand import RandomState
from typing import Union, List


def cluster(
    adata: AnnData,
    random_state: Union[RandomState, int, None] = 0,
    resolution: Union[List, int] = 1,
    copy: bool = False,
) -> Union[AnnData, None]:
    """Clusters a :py:class:`~anndata.AnnData` object over multiple resolutions.

    Args:
        adata:
            A :py:class:`~anndata.AnnData` object to be processed.

        random_state:
            Integer to be used for seeding of random initialization of PCA, neighbors graph, and UMAP.

            Alternatively, a :py:class:`~numpy.random.mtrand.RandomState` object to be used directly.

            If ``None``, defaults to 0.

        resolution:
            A list of resolutions to cluster at.

            Alternatively, a single resolution to cluster at. By default, this is 1.

        copy:
            Bool indicating whether modifications should be performed inplace (default) or to return a modified copy.

    Returns:
        ``None`` if ``copy = False`` (inplace modification), or the clustered :py:class:`~anndata.AnnData` object if ``copy = True``.
    """

    adata = adata.copy() if copy else adata

    if not isinstance(random_state, np.random.RandomState):
        if type(random_state) == int:
            random_state = np.random.RandomState(random_state)
        elif type(random_state) == None:
            random_state = np.random.RandomState(0)
        else:
            msg = "the type of `random_state` should be one of: int, None, np.random.mtrand.RandomState"
            raise ValueError(msg)

    if isinstance(resolution, int):
        resolution = [resolution]

    for res in resolution:
        sc.tl.leiden(adata, resolution=res, key_added=f"leiden_{res}")

    if copy:
        return adata
