"""
Base classes for defining blends and components 
and converting them to/from dictionaries.
"""

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

    def __init__(self, name, children, description=None, qmin=0.0, qmax=1.0, qmethod="relative", cmax=None):
        self.name = name
        self.description = description
        self.children = children
        self.qmin = qmin
        self.qmax = qmax
        self.qmethod = qmethod
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
    

