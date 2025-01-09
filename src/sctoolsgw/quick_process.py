import scanpy as sc
from anndata import AnnData
import numpy as np
from numpy.random.mtrand import RandomState
from typing import Optional, Union


def quick_process(
    adata: AnnData,
    random_state: Union[RandomState, int, None] = 0,
    batch_key: Optional[Union[str, None]] = None,
    copy: bool = False,
) -> Union[AnnData, None]:
    """Perform a quick processing workflow on a :py:class:`~anndata.AnnData` object.

    Args:
        adata:
            A :py:class:`~anndata.AnnData` object to be processed.

        random_state:
            Integer to be used for seeding of random initialization of PCA, neighbors graph, and UMAP.

            Alternatively, a :py:class:`~numpy.random.mtrand.RandomState` object to be used directly.

            If ``None``, defaults to 0.

        batch_key:
            Optional :py:attr:`~anndata.AnnData.obs` column name discriminating between batches. Will be used for bbknn.

        copy:
            Bool indicating whether modifications should be performed inplace (default) or to return a modified copy.

    Returns:
        ``None`` if ``copy = False`` (inplace modification), or the processed :py:class:`~anndata.AnnData` object if ``copy = True``.
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

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    sc.pp.pca(adata, random_state=random_state)

    if batch_key is not None:
        if batch_key not in adata.obs.columns:
            msg = (
                "`batch_key` must be a column of .obs in the input AnnData object,"
                f"but {batch_key!r} is not in {adata.obs.keys()!r}"
            )
            raise ValueError(msg)
        sc.external.pp.bbknn(adata, batch_key=batch_key)
    else:
        sc.pp.neighbors(adata, random_state=random_state)

    sc.tl.leiden(adata, random_state=0)
    sc.tl.umap(adata, random_state=random_state)

    if copy:
        return adata
