# BlenDS

A framework for an intuitive specification of the design space for blends of components.

## Installation

```sh
pip install -r requirements.txt
conda install graphviz # WindowsOS only
```

## Run

To run the streamlit app (activate environment):

streamlit run app.py

## Terminology

- Component
- Blend: combinations of components or other blends that sum up to a certain quantity.
- Root Blend (`rblend`): the top level blend that is the final product.
- Quantity (`q`): the amount of a component or blend in a parent blend. The user needs to contraint it between `qmin` and `qmax`.
- Children of a Blend: components or other blends that are part of a blend.
- Parent of a Blend/Component: the blend that contains the blend or component.
- Maximum components (`cmax`): maximum number of components that can be present in a blend, chosen among its children. For `cmax=1` the blend is simply a choice of one component among its children.
- Properties (`props`): specific attributes of a component that can be weighted by its quantity and summed to obtain the general properties of the root blend.
- Response: unknown property of the root blend that is to be mapped to the quantities of components and optimized
