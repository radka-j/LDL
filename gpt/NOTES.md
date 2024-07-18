## TORCH

- always have to start with nn.Module init (i.e., `super().__init__()`)
- for each module, implement the `forward()` method but this never gets called, just call the module itself 
- `nn.Linear(features_in_dim, features_out_dim)` is a linear transformation
- `nn.Embedding` is a simple look up table populated with values drawn from standard normal (can access as embedding.weight)
- to save parameters that should not be optimised, use `self.register_buffer`
- `nn.ModuleList` can be indexed like a Python list but it also "does the right torch stuff" (presumably that means the parameters within are registered for optimisation etc.)

## ATTENTION

- Attention is a communication mechanism:
    - aggregates information in a directed graph with a weighted sum from all connected nodes
    - the weights are data-dependent 
    - each token emits 3 vectors:
        - key (what do I contain?) 
        - query (what am I looking for?)
        - value (the thing you are looking for)    
    - taking a dot product between the key and query gives us an affinity score - indicating how much these tokens should communicate with each other i.e. to what extent is the key the right response to the query (for example, nouns are looking for adjectives, vowels are looking for consonants)
    - these weights are then multiplied with the value vectors
- There is no notion of space --> need to positionally encode tokens
- The example here implements a decoder block by masking out future tokens 
    - turn this into an ecoder block by removing the single line that applies the mask
- self-attention = keys and values are produced from the same source as queries
- cross-attention = the queries still get produced from x, but the keys and values come from some other, external source (e.g. an encoder module)
- "scaled" attention  divides attention weights by 1/sqrt(head_size) --> ensure Softmax will stay diffuse and not saturate too much
    - without scaling we might get very divergent values which, when passed through Softmax, leads to a very peaky output

## LAYER NORM
TODO

- very similar to batch norm
    - across the batch dimension, make sure any individual neuron has unit Gaussian distribution
    - i.e., it normalises every column in the bach dimension to have 0 mean and 1 SD
- here normalize the rows rather than the columns (TODO: what are the rows here?!)
- original transform paper applied layer norm after the transform but now it's common to do it before ("pre-norm formulation")

## RESIDUAL/SKIP CONNECTIONS

- have a residual pathway from which can fork off and do some computation and then return to it

```
      Xt
      | \
      |  \
      |   \
      |   / <COMPUTATION HAPPES ON THIS PATH>
      |  /
      |/
    <addition>
      Xt+1
```

- addition distributes gradients equally to both branches
    - gradients go through the addition operation directly from "supervision" back to input
    - they also fork off and go through the other branch with the additional computations
- the residual blocks (i.e., computation fork) are intitialised to contribute little at start so the gradient is unimpeded and just flows
- over time, during optimisation, they start to contribute
- this helps dramatically with optimisation

## DROPOUT
- used to prevent overfitting, a regularisation technique
- every forward pass, randomly shut off a subset of neurons (turn them to 0)
- sort of trains an emsenble of subnetworks
- at inference it is not applied
