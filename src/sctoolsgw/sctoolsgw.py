import gseapy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

gene_sets = [
    "GO_Molecular_Function_2023",
    "GO_Biological_Process_2023",
    "GO_Cellular_Component_2023",
    "KEGG_2019_Mouse",
    "Reactome_2022",
    "WikiPathways_2019_Mouse",
    "TRANSFAC_and_JASPAR_PWMs",
    "miRTarBase_2017",
    "CORUM",
    "Human_Phenotype_Ontology",
]


def barplot_enrichr(
    df: pd.DataFrame,
    p_val_cutoff: float = 0.05,
    n_top: int = 25,
    figsize: tuple = (10, 10),
    output_file: str = None,
) -> plt.Axes:
    """Barplot of results from enrichment test

    :param df: dataframe containing enrichr results
    :type df: :class:`pandas.DataFrame`
    :param p_val_cutoff: do not plot results below this
    :type p_val_cutoff: float
    :param n_top: number of top genes to plot
    :type n_top: int
    :param figsize: size of figure (in `(width, height)`)
    :type figsize: tuple
    :param output_file: path to output file (or None to not save as file)
    :type output_file: str
    """
    ax = gseapy.plot.barplot(
        df, cutoff=p_val_cutoff, top_term=n_top, figsize=figsize, ofname=output_file
    )
    return ax


def run_enrichr(
    df: pd.DataFrame,
    condition: str,
    gene_sets: list,
    rank_col: str = "bayes_factor",
    p_val_cutoff: float = 0.05,
) -> pd.DataFrame:
    """Performs enrichr gene set test using enrichr API

    :param df: dataframe containing gene sets (scvi DEGs)
    :type df: :class:`pandas.DataFrame`
    :param condition: condition of interest (either db or wt)
    :type condition: str
    :param gene_sets: list of gene sets to test against
    :type gene_sets: list
    :param rank_col: column in df ranking the genes (e.g., pvals)
    :type rank_col: str
    :param p_val_cutoff: filter adjusted p vals above this cutoff
    :type p_val_cutoff: float
    """
    df = df.loc[:, [rank_col, "Unnamed: 0", "group"]]
    df = df[df.group == condition]
    results = gseapy.enrichr(
        gene_list=df["Unnamed: 0"].tolist(),
        gene_sets=gene_sets,
        organism="mouse",
        outdir=None,
    )
    results = results.results[results.results["Adjusted P-value"] < p_val_cutoff]
    return results


def process(adata: sc.AnnData, random_state=None) -> sc.AnnData:
    """Performs a quick processing workflow on :class:`adata <sc.AnnData>`.

    :param adata: object to be processed.
    :type adata: :class:`scanpy.AnnData`
    :param random_state: RandomState object.
    :type random_state: :class:`numpy.random.RandomState`
    :return: procssed object
    :rtype: :class:`scanpy.AnnData`
    """
    adata_cp = adata.copy()

    if random_state is None:
        random_state = np.random.RandomState(42)

    if "counts" not in adata_cp.layers.keys():
        adata_cp.layers["counts"] = adata_cp.X.copy()

    sc.pp.normalize_total(adata_cp, target_sum=1e6)
    sc.pp.log1p(adata_cp)

    sc.pp.pca(adata_cp)
    sc.pp.neighbors(adata_cp)

    sc.tl.leiden(adata_cp)
    sc.tl.umap(adata_cp)

    return adata_cp


def cluster(adata: sc.AnnData, resolution: [int]) -> sc.AnnData:
    """Clusters an :class:`scanpy.AnnData` object.

    Performs leiden clustering on an :class:`~anndata.AnnData`
    object with the option to pass in multiple resolutions as
    a parameter. Here's an example of how that would work:

        adata = cluster(adata, resolution = [0.1, 0.2, 0.3])

    The corresponding resolutions will be placed in columns
    in `adata.obs` named `leiden_{resolution}`.

    :param adata: object to be clustered.
    :type adata: :class:`scanpy.AnnData`
    :param resolution: resolution(s) for the data to be clustered at.
    :type resolution: list
    :return: clustered object
    :rtype: :class:`~scanpy.AnnData`
    """
    adata_cp = adata.copy()

    try:
        len(resolution)
    except TypeError:
        resolution = [resolution]

    for res in resolution:
        sc.tl.leiden(adata_cp, resolution=res, key_added=f"leiden_{res}")

    return adata_cp
