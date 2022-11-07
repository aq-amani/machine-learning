## Structure of a Neural Network and sizes of each element (as numpy arrays)

<img src="./images/NN_structure.jpg" width="600"/>

## Derivation of backprop equations, for a generic cost function J and activation functions g
- Final goal is to get `dW_l` (`∂J/∂W_l`) and `db_l` (`∂J/∂b_l`) for each layer `l` in order to update the W and b matrices in gradient descent
- Derivation can be done using the chain rule
- For Z matrices dZ_l(`∂J/∂Z_l`) is derived using the Loss function instead of the cost function, hence the absence of the `1/m` term.
### Last layer `K`:
<img src="./images/last_layer.jpg" width="600" />

### Any other layer `l` :
<img src="./images/any_layer.jpg" width="600" />

## Example with a 2-layer network:
2 layers with Linear --> Sigmoid activations

### Last layer (2)
<img src="./images/example_L2.jpg" width="600" />

### First layer (1)
<img src="./images/example_L1.jpg" width="600" />
