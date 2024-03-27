from graphviz import Digraph
from .base import Component  

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