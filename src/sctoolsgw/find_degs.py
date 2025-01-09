from typing import Union, Optional, List, Literal
from anndata import AnnData
from numpy.random.mtrand import RandomState
import pandas as pd
from pandas import DataFrame
from pydeseq2.default_inference import DefaultInference
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

from .aggregate import aggregate


def find_degs(
    adata: AnnData,
    subset: Optional[List],
    sample_col: Union[str, None] = None,
    groups_col: Union[str, None] = None,
    use_raw: bool = False,
    mode: Literal["sum", "mean"] = "sum",
    min_cells: int = 100,
    min_counts: int = 1000,
    contrast: List = [],
    inference: Union[DefaultInference, None] = None,
    random_state: Union[RandomState, int, None] = 0,
    n_reps: int = 3,
) -> DataFrame:

    if subset:
        if subset[0] not in adata.obs.columns:
            msg = "`subset[0]` must be in .obs of adata"
            raise ValueError(msg)
        else:
            adata = adata[adata.obs[subset[0]] == subset[1], :]

    if sample_col not in adata.obs.columns:
        msg = "`sample_col` must be in .obs of adata"
        raise ValueError(msg)
    if groups_col not in adata.obs.columns:
        msg = "`groups_col` must be in .obs of adata"
        raise ValueError(msg)

    if contrast == []:
        msg = "please specify contrasts in the form of: [obs_column, group_exp, group_ref]"
        raise ValueError(msg)

    adata_agg = aggregate(
        adata=adata,
        sample_col=sample_col,
        groups_col=groups_col,
        use_raw=use_raw,
        mode=mode,
        min_cells=min_cells,
        min_counts=min_counts,
        n_reps=n_reps,
        random_state=random_state,
    )

    if inference is None:
        inference = DefaultInference(n_cpus=2)

    stat_res_list = []
    for g in adata_agg.obs[groups_col].cat.categories:

        idx = adata_agg[groups_col] == g  # type: ignore
        counts = adata_agg[idx, :].to_df()
        metadata = adata_agg[idx, :].obs

        dds = DeseqDataSet(
            counts=counts,
            metadata=metadata,
            design_factors=sample_col,  # type: ignore
            refit_cooks=True,
            inference=inference,
            quiet=False,
        )
        dds.deseq2()

        stat_res = DeseqStats(
            dds, contrast=contrast, inference=inference, quiet=False
        ).results_df
        stat_res["cluster"] = str(g)
        stat_res_list.append(stat_res)

    df = pd.concat(stat_res_list, axis=0)

    return df
