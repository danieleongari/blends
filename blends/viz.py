"""
Functions to visualize Blends and samples.
"""

import pandas as pd
from graphviz import Digraph

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
from umap import UMAP

from .base import Blend, Component, get_base_components

def get_graph(rblend) -> Digraph:
    """Return a graphical representation of the input Root Blend.
    
    rblend: Blend object (root blend)
    
    """

    g = Digraph("Root Blend")
    g.graph_attr["rankdir"] = "LR"
    if rblend.qmin == rblend.qmax:
        rblend_q = f"q= {rblend.qmin}"
    else:
        rblend_q = f"q= {rblend.qmin}-{rblend.qmax}"
    rblend_label = f"RootBlend\n'{rblend.name}'\n{rblend_q}\n(max {rblend.cmax})"
    g.node(rblend_label, shape="diamond")

    for child in rblend.children:
        _add_child_to_graph(g, child, rblend_label)

    return g

def _add_child_to_graph(g, child, parent_name):
    """Recursively add a child to the graph.
    To be used in blends.viz.get_graph().
    """

    if isinstance(child, Component):
        comp_label = f"Component\n'{child.name}'"
        g.node(comp_label, shape="oval")
        g.edge(
            tail_name=f"Component\n'{child.name}'",
            head_name=parent_name,
            label=f"q= {child.qmin}-{child.qmax}",
        )
    else:
        blend_label = f"Blend\n'{child.name}'\n(max {child.cmax})"
        g.node(blend_label, shape="box")
        g.edge(
            tail_name=blend_label,
            head_name=parent_name,
            label=f"q= {child.qmin}-{child.qmax}",
        )
        for subchild in child.children:
            _add_child_to_graph(g, subchild, blend_label)


def plot_design_space(
        rblend: Blend,
        df_samples: pd.DataFrame,
        color: str,
        method: str = "pca-2d",
        verbose: bool = False,
        ):
    """Plot the design space of the Root Blend given the samples.

    rblend: Blend object (root blend)
    samples: pd.DataFrame containing the samples
    color: str, string of the column name in samples to use as color
    method: str, method to use for dimensionality reduction. Options are:
        - 'pca-2d': 2D PCA
        - 'pca-3d': 3D PCA
        - 'tsne-2d': 2D t-SNE
        - 'tsne-3d': 3D t-SNE
        - 'umap-2d': 2D UMAP
        - 'umap-3d': 3D UMAP
    verbose: bool
    """

    base_components_names = [comp.name for comp in get_base_components(rblend)]
    meth, dim = {
        "pca-2d": ("pca", 2),
        "pca-3d": ("pca", 3),
        "tsne-2d": ("tsne", 2),
        "tsne-3d": ("tsne", 3),
        "umap-2d": ("umap", 2),
        "umap-3d": ("umap", 3),
    }[method]

    if meth == "pca":
        pca = PCA(n_components=dim)
        X = pca.fit_transform(df_samples[base_components_names])
        pca_df = pd.DataFrame(X, columns=[f"PC{i+1}" for i in range(dim)])
        pca_df[color] = df_samples[color].values
        if dim == 2:
            fig = px.scatter(
                pca_df,
                x='PC1',
                y='PC2',
                color=color,
            )
        elif dim == 3:
            fig = px.scatter_3d(
                pca_df,
                x='PC1',
                y='PC2',
                z='PC3',
                color=color,
            )
    elif meth == "tsne":
        tsne = TSNE(n_components=dim)
        X = tsne.fit_transform(df_samples[base_components_names])
        tsne_df = pd.DataFrame(X, columns=[f"TSNE{i+1}" for i in range(dim)])
        tsne_df[color] = df_samples[color].values
        if dim == 2:
            fig = px.scatter(
                tsne_df,
                x='TSNE1',
                y='TSNE2',
                color=color,
            )
        elif dim == 3:
            fig = px.scatter_3d(
                tsne_df,
                x='TSNE1',
                y='TSNE2',
                z='TSNE3',
                color=color,
            )
    elif meth == "umap":
        
        reducer = UMAP(n_components=dim)
        X = reducer.fit_transform(df_samples[base_components_names])
        umap_df = pd.DataFrame(X, columns=[f"UMAP{i+1}" for i in range(dim)])
        umap_df[color] = df_samples[color].values
        if dim == 2:
            fig = px.scatter(
                umap_df,
                x='UMAP1',
                y='UMAP2',
                color=color,
            )
        elif dim == 3:
            fig = px.scatter_3d(
                umap_df,
                x='UMAP1',
                y='UMAP2',
                z='UMAP3',
                color=color,
            )
    
    return fig




