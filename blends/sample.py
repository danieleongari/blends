"""
Functions to generate samples from the constraints of a Blend.
"""

import numpy as np
import pandas as pd
from .base import Blend, get_base_components

#TODO: qmethod should be an attribute of the Blend object not an argument of the function,
#      because the user needs to specify qmin and qmax coherently with the qmethod.

def get_samples_both(rblend, nsamples_fill=1000, nsamples_vert=5000, tollerance=0.001, qmethod="absolute", vertices=False, verbose=False) -> pd.DataFrame:
    """Dummy fanction to pile up the results of get_samples() with vertices=True and False."""
    df_vert = get_samples(rblend, nsamples=nsamples_vert, tollerance=tollerance, qmethod=qmethod, vertices=True, verbose=verbose)
    df_fill = get_samples(rblend, nsamples=nsamples_fill, tollerance=tollerance, qmethod=qmethod, vertices=False, verbose=verbose)
    df = pd.concat([df_vert, df_fill])
    df = df.drop_duplicates()
    return df

def get_samples(rblend, nsamples=100, tollerance=0.001, qmethod="absolute", vertices=False, verbose=False) -> pd.DataFrame: 
    """
    Generate samples from the constraints of a Blend.

    inputs
    ------
    rblend: Blend object (root blend)
    nsamples: int, number of samples to generate
    tollerance: float, tollerance for rounding the quantities
    qmethod: str, either 'absolute' or 'relative'
    vertices: bool, if True the quantities are chosen from the vertices of the search space
    verbose: bool, if True print the number of successful samples generated

    Returns a DataFrame with the samples generated.
    """
    
    samples = []

    for _ in range(nsamples):
        samples.append(generate_children_quantities(rblend, qmethod=qmethod, vertices=vertices))

    # Remove all invalid samples (None)
    samples = [sample for sample in samples if sample is not None]
    nsamples_invalid = nsamples - len(samples)

    base_components_names = [comp.name for comp in get_base_components(rblend)]

    df = pd.DataFrame(samples, columns=base_components_names)
    df = df.fillna(0) # generate_children_quantities leaves NaN for components that were not chosen

    # Round the quantities to the tollerance
    df = np.round(df/tollerance)*tollerance

    # Remove duplicates
    df = df.drop_duplicates()
    nsamples_redundant = nsamples - nsamples_invalid - len(df)

    df["Qtot"] = df.sum(axis=1)


    if verbose:

        print(f"{nsamples} - {nsamples_invalid} invalid - {nsamples_redundant} redundant = {len(df)}: successful samples generated.") 

    return df


def generate_children_quantities(blend, qparent=None, quants=None, qmethod="absolute", vertices=False) -> dict:
    """Made to be called recursively, generates the quantities of the children of a Blend.
    """
    # Validate the input
    if quants is None: # Root blend
        quants = {}
    if qparent is None: # Root blend
        qparent = np.random.uniform(blend.qmin, blend.qmax)
    assert qmethod in ["absolute", "relative"], "qmethod must be either 'absolute' or 'relative'"

    # Populate the quantities of the first offspring
    children_names = [child.name for child in blend.children]
    children_chosen = np.random.permutation(children_names)[:blend.cmax]
    qleft = 1.0 if qmethod == "relative" else qparent
    nchild = len(children_chosen)
    for ichild, child_name in enumerate(children_chosen):
        child = [child for child in blend.children if child.name == child_name][0]
        #print(child.name)
        if qleft < child.qmin:
            return None # there is no quantity left for the constraint
        if ichild==(nchild-1): # last child
            if qleft > child.qmax: # there is too much quantity left for the constraint
                return None
            quants[child_name] = qleft # assign the remaining quantity to the last child
        else:
            if vertices: # Get the extremes of the constraint, i.e., the vertices of the search space
                qchild = np.random.choice([child.qmin, min(qleft, child.qmax)])
            else:
                qchild = np.random.uniform(child.qmin, min(qleft, child.qmax)) 
            quants[child_name] = qchild
            qleft -= qchild

    if qmethod == "relative":
        for child_name in children_chosen:
            quants[child_name] *= qparent


    # Now that the quantities for the first offspring have been generated
    # we can reiterate recursively for the children that are themselves blends
    for child_name in children_chosen:
        child = [child for child in blend.children if child.name == child_name][0]
        #print(child.name)
        if isinstance(child, Blend):
            child_quants = generate_children_quantities(child, qparent=quants[child_name], qmethod=qmethod, vertices=vertices)
            if child_quants is None:
                return None
            quants.update(child_quants)
            del quants[child_name]

    return quants
