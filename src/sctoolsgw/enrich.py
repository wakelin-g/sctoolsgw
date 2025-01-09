from typing import Union, List
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes
import pandas as pd
from pandas import DataFrame
import gseapy


def _format_df(
    df: DataFrame, condition: str, rank_col: str, pval_cutoff: float
) -> DataFrame:
    df = df.loc[:, [rank_col, "Unnamed: 0", "group"]]
    df = df[df.group == condition]
    return df


def run_enrichr(
    df: DataFrame,
    condition: str,
    gene_sets: List,
    rank_col: str = "bayes_factor",
    pval_cutoff: float = 0.05,
) -> DataFrame:
    df = _format_df(
        df=df, condition=condition, rank_col=rank_col, pval_cutoff=pval_cutoff
    )
    results = gseapy.enrichr(
        gene_list=df["Unnamed: 0"].tolist(),
        gene_sets=gene_sets,
        organism="mouse",
        outdir=None,
    )
    results = results.results[results.results["Adjusted P-value"] < pval_cutoff]

    return results


def barplot_enrichr(
    df: DataFrame,
    pval_cutoff: float = 0.05,
    n_top: int = 25,
    figsize: tuple = (10, 10),
    output_file: Union[str, None] = None,
) -> Axes:
    ax = gseapy.plot.barplot(
        df, cutoff=pval_cutoff, top_term=n_top, figsize=figsize, ofname=output_file
    )
    return ax
