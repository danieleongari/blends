"""
Base classes for defining blends and components 
and converting them to/from dictionaries.
"""

class Component:
    """Core component, representing a fluid/material that can be added to a
    formulation in specified amounts.
    """

    def __init__(self, name, description=None, qmin=0, qmax=1, props={}):
        self.name = name
        self.description = description
        self.qmin = qmin
        self.qmax = qmax
        self.props = props

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
    else: # blend is a Component
        return blend.__dict__
    

def get_base_components(blend):
    if isinstance(blend, Blend):
        components = []
        for child in blend.children:
            components += get_base_components(child)
        
        return components
    else: # blend is a Component
        return [blend]
    

def add_prop_to_blend(blend, prop_name, prop_dict):
    """Add a property to each Component in the Blend.
    
    blend: Blend or Component, root blend (unless used ricursively)
    prop_name: str, name of the property to add
    prop_dict: dict, dictionary with the property values for each Component, e.g., {"Comp-A1": 0.5, "Comp-A2": 0.3, ...}

    NOTE: it is a good practice to append a description of the property to the root Blend description.
    """
    if isinstance(blend, Blend):
        for child in blend.children:
            add_prop_to_blend(child, prop_name, prop_dict)
    else: # blend is a Component
        try:
            blend.props[prop_name] = prop_dict[blend.name] 
        except KeyError:
            raise KeyError(f"Component {blend.name} not found in the input prop_dict.")
