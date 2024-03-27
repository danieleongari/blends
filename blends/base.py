import numpy as np
import pandas as pd

class Component:
    """Core component, representing a fluid/material that can be added to a
    formulation in specified amounts.
    """

    def __init__(self, name, description=None, qmin=0, qmax=1):
        self.name = name
        self.description = description
        self.qmin = qmin
        self.qmax = qmax

class Blend:
    """Collection of Component or (sub-)Formulations.
    Note that we limited the recursivity of the operators not to make too complex Formulations:
    a Formulation can have as children Components, or another Formulation, but this child Formulation
    can only have Components as children (i.e., another Formulation).
    """

    def __init__(self, name, children, description=None, qmin=0.0, qmax=1.0, cmax=None):
        self.name = name
        self.description = description
        self.children = children
        self.qmin = qmin
        self.qmax = qmax
        self.cmax = len(self.children) if cmax is None else cmax


def dict_to_blend(children_dict):
    if 'children' in children_dict:
        children = []
        for child_dict in children_dict['children']:
            child = dict_to_blend(child_dict)
            children.append(child)
        
        children_dict_copy = children_dict.copy()
        children_dict_copy.pop('children')
        blend = Blend(**children_dict_copy, children=children)
        
        return blend

    else:
        component = Component(**children_dict)
        
        return component
    

def blend_to_dict(blend):
    if isinstance(blend, Blend):
        children = []
        for child in blend.children:
            child_dict = blend_to_dict(child)
            children.append(child_dict)
        
        blend_dict = blend.__dict__.copy()
        blend_dict['children'] = children
        
        return blend_dict
    else:
        return blend.__dict__
    

def get_base_components(blend):
    if isinstance(blend, Blend):
        components = []
        for child in blend.children:
            components += get_base_components(child)
        
        return components
    else:
        return [blend]
    

def generate_children_quantities(blend, qparent=None, quants=None, qmethod="absolute"):
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
            child_quants = generate_children_quantities(child, qparent=quants[child_name], qmethod=qmethod)
            if child_quants is None:
                return None
            quants.update(child_quants)
            del quants[child_name]

    return quants
