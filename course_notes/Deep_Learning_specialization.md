My notes of Andrew Ng's Deep Learning specialization courses.
# Still a WIP

## Deep Learning specialization courses:
- Course 1: Neural Networks and Deep Learning
- Course 2: Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization
- Course 3: Structuring Machine Learning Projects
- Course 4: Convolutional Neural Networks
- Course 5: Sequence Models

# Course1: Neural Networks and Deep Learning
- CNN: Conv NNs / ConvNets --> used for images
- RNN: recurrent NN --> with audio (1D time series/ sequence data)
- Logistic regression(and Linear regressions) are just a `shallow neural network / 1 layer NN`
  - inputs go to one node only and that node gives the output.

#### Matrices and array sizes / broadcasting/ vectorization
- [Link to matrix multiplication notebook](./matrix_multiplication_reference.ipynb)
- shape: `(n,)` : is called a `rank1` array and it's a best practice to avoid using it, always use `(n,1)` shape instead.
  - To avoid rank1 array outputs in numpy functions, use `keep_dims=True`.
  - Use python `assert` statements to make sure shapes are as expected.
    Ex.:
    ```
    assert number > 0, f"number greater than 0 expected, got: {number}"
    ```
    gives nothing when ok, prints message when fails.
- Feeding examples to NN can be vectorized (feed training set at once) only when there is one layer.
  - For multi-layer, we need to loop over the layers(TODO: check)

#### GD and backprop in NNs
- Use the derivate chain rule to simplify derivation.
- For L layers, the model is basically `X --W1,b1--> Z1 --G1(Z1)--> A1 --W2,b2--> Z2 --> G2(Z2) --> ... --> Y_hat/GL(ZL)`
- Identify the cost function, then derive it with respect to each W and b using the chain rule, and update them as done with normal GD.
- One iteration of GD for NN: 1) Forward propagation with caching 2) Backward propagation to get derivatives and update weights.
  - Caching: cache Z values, along with W and b values in Forward propagation to use in backward propagation.

#### Some more activation functions
- **tanh activation** function: gives values between - 1 and 1. works better than sigmoid, but not to be used for the output layer of a binary classification problem.
$$\\frac{exp(z) - exp(-z)} {exp(z) + exp(-z)}$$

- **Leaky ReLU** (`max(0.01*Z, Z`)): small negative values to the left of curve. Better than ReLU cuz faster learning due to non zero derivative(TODO:?) but not used often.

- Derivative of the sigmoid function `g(z) = ??(z)` is `g'(z) = (1 - ??(z)) * ??(z)`

#### NN Parameters/Hyperparameters:
- Parameters: Weights and Biases (W and b)
- Hyperparamters: alpha, iteration count, # hidden layers L, #hidden units, activation function choice, mini-batch size ..etc
- NN Parameter Initialization: Initialize `W` randomly instead of zeros, because zeros wont make any difference over iterations. `b` is ok to be zeros.

# Course2: Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization

#### Regularization
- `L1 regularization`: 
$$\\frac{\\lambda}{2m}\\sum_{}^{} w$$

- `L2 regularization`  (AKA L2 norm/ Euclidean norm/ Weight decay): used more often.
$$\\frac{\\lambda}{2m} \\sum_{}^{} w^2$$

- `Frobenius norm` : L2 regularization on the NN level (for a matrix).. done for all weights in each layer.

- `Drop out regularization`: on each GD iteration, randomly (with a random probability) eliminate some neurons from the network and train/test it.
  - `Inverted dropout`: The most common implementation.
  - at last it divides output by the `keep probability` to not alter outputs(TODO: ?)
  - Predictions are made without dropout.

- Augmentation: considered a regularization technique.
- `Early stopping` : Monitor cost for both train and dev datasets simultaneously and stop before devset cost starts to increase away from train cost.
  - not recommended cuz its like mixing train/dev datasets.

`Vanishing/exploding gradients`: solved by careful choice of initial weights.

#### Batches
Even with vectorized implementations, GD runs slowly when the data size is large. `Mini-batch` is a way to reduce this delay.
- `Batch Gradient descent`: Normal GD with all training dataset processed at once
  - In other words, mini-batch size = full dataset size.
- `Mini-batch GD`: Using mini-batches
  - divide training data into smaller batches and let GD run over them instead of waiting for the whole dataset to process before finishing one iteration)
  - can make progress in GD steps without waiting for whole to process: every epoch gives mini-batch size steps of GD as opposed to 1GD step in batch mode.
  - simply apply GD as always but on the mini-batch, and keep doing on mini-batches until done with whole data

- `Stochastic Gradient Descent`: When mini-batch size is only 1.
  - very noisy as cost goes up and down, and convergence on contour looks like random walk.
  - loses all speedup from vectorization

**Mini-batch size choice**: Better chosen as a power of 2. 64 ~ 512 is common, 1024  is ok too but not so common.

## other optimization algorithms faster than GD:
#### Exponentially weighted averages (moving average)
```
v(t) = ?? * v(t-1) + (1- ??) * ??(t), v(0)=0
```
where v is avg, ?? is data point, 1/(1-??) is averaging window (days, hours..etc), 0< ?? < 1
- the bigger the averaging window, the smoother the curve is but the more shifted to the right(slower to adapt to changes)
- the smaller the window is, the more noisy the curve is but the faster it adapts to changes.
- Bias correction for moving average: in the initial phase, since v0 is 0, first part of the curve wont be accurate.
  - instead of v(t) as is, divide it by (1-??^t) .. this wont have much effect when t is large, but will reduce inaccuracy in early iterations.

#### GD with momentum: use moving average to accelerate movement towards minimum with less vertical movement.
```
w = w - ?? * Vdw
b = b - ?? * Vdb
```
where `Vdw = ?? * Vdw + (1- ??) * dw`,  `Vdb = ?? * Vdb + (1- ??) * db`

usually no one bothers with bias correction when doing GD with momentum.

#### RMSprop (Root mean square prop): another way to speed up GD
```
w = w - ?? * dw / ???Sdw
b = b - ?? * db / ???Sdb
```
where Sdw = ?? * Sdw + (1- ??) * dw^2,  Sdb = ?? * Sdb + (1- ??) * db^2

can add a small epsilon to the denominator to avoid dividing by zero if square root is zero.

#### ADAM optimization: Combines RMSprop and GD with momentum
- initialize Vdw, Vdb, Sdw, Sdb to zero
- for each mini-batch iteration:
  - compute new Vdw, Vdb, Sdw, Sdb  (momentum and RMSprop each has their own ?? -  say ??s and ??v)
  - Bias correct the V and S values by dividing by (1-??^t), each with the corresponding ??
  - update w and b using a hybrid of RMSprop and GD momentum, using the bias corrected S and V values as follows:
```
w = w - ?? * Vdw / (???Sdw + ??)
b = b - ?? * Vdb / (???Sdb + ??)
```
`??` needs to be tuned, common values for `??v` is 0.9, and `??s` is 0.999, and `??` 10^-8

## Learning rate decay: 
- When heading towards minimum on a contour, fixed values of alpha means larger oscillations in a larger area around minimum and not really converging.
- By decaying it, the oscillations happen in a smaller area and it converges better.
- Many different ways/formulas to do that:
  - `?? = ??0 * 1/(1 + decay_rate * epoch_number)`
  - `?? = ??0 * 0.95 ** epoch_num`
  - discrete step functions
  - `?? = ??0 * k/???epoch_num`
  - ...etc.

## Tuning hyper-parameters:
Instead of testing on a continuous space of parameters, choose `randomly distributed` values instead and narrow down then repeat based on that.
- For parameters like #layers, #neurons, `uniformly distributed` random values are ok.
- But for parameters like alpha, beta, a `logarithmic scale` is better.

## Batch normalization (batch norm):
- In case of NNs, normalize inputs to each layer.
- Usually done on Z (before activation) and not on the activation value.
- And is done for each layer.
```
Z_norm = (Z - ??)/??
Z_new = ?? * Z_norm + ??
```
when `??` = `??` and `??` = `??`, then `Z_new` is equal to `Z` (no normalization).

### How keras's BatchNorm with axis=3 works
Assumption for data array shape: (m,h,w,c)
- m is number of examples (images)
- hxw is each image's size
- c is channel count in each image

Say we have 3 channels : R, G, and B
1) All R channels are taken for all examples and scalar mean and variance values are calculated over them. Same is done for G and B channels.
2) Then each pixel in each channel is normalized based on the mean and variance of that channel as calculated in 1

Note: A small value epsilon is added to the variance to avoid division by zero when calculating the normalized values.

# Course3: Structuring Machine Learning Projects

- `Orthogonalization`: variables control single features separately instead of mixed control
- `Satisficing`: having a threshold beyond/under which a metric is considered OK. (ex.: optimize for highest accuracy in a cat classifier, but make sure running time is below certain value. otherwise users wont be satisfied)
  - accuracy: optimizing metric , run time: satisficing metric.
  - Stakeholders must define thresholds for satisficing metrics, leaving the optimizing metric unbounded.
- training sets, dev sets and test sets should come from same distributions (ex.: its not good to use pro photography for one set and amateur photos for another set  )
- For few thousand data examples, train-test would be 70%-30% and train-dev-test would be 60%, 20%, 20% .. but with big data it's ok to go much less than that with just a couple thousands for each.
  - What matters is that the dev and test set have the same distributions

- `Bayes optimal error`: A threshold of performance that model can't surpass (best possible error).
- `Human-level performance`. Usually error decreases rapidly until human-level performance is surpassed, then it slowly flattens.
  - want to minimize difference between human level and model performance.
  - Human-level is roughly close to bayes optimal error. But bayes optimal error is better than human level.
- `Avoidable bias`: difference between bayes error(use human level as proxy for that) and training error.
- variance: diff between training and dev errors.
- human level performance need to be defined based on context: typical human level? expert on the subject human level?

- multitask learining = multi-label learning.

-End to end learning: Instead of breaking the problem into simpler problems, learn all the way using one NN from input to output.
  - ex.: One NN to detect faceID from a video and authorize person as opposed to a Pipeline of: 1)detect person approaching 2) detect/extract face 3) match face to ID

# Course4: Convolutional Neural Networks

`Neural Style Transfer`: repaint an image in another style(picasso style for example)

### Convolution
- Convolution operation, horizontal edge detection and vertical edge detection.
- Each layer has a number of filters (AKA kernels), as opposed to neurons.
- technically it is a cross-correlation operation since no filter flipping takes place first, but in ML literature it is called convolution.
- Convolution operation between matrices is denoted by *
- How is it done : Take filter, scan it over the whole image while multiplying and summing all elements. Each area results in one number.
  - 6x6 matrix convoluted with 3x3 filter, will give a 4x4 matrix (3 element array can be shifted 4 times in a 6 element array ) (outsize= n -f +1)
- many variations of filters, but it's better to let the NN learn the filters by having the fxf filter values as w parameters.
- `Padding`: since image shrinks with convolution, and since corner pixels get used only once in calculations, we can preserve original size by padding(adding a border of 0s to the input)
  - 6x6 img --> 8x8 img after padding by a border of 1 pixel.

### Convolution types
- `valid convolution`: no padding.
  - `out size = n -f +1` 
- `same convolution`: padding to retain original img size after convolution (pad = p)
  - `out size = n +2p -f +1` 
- `Strided convolution`: stride = S, and means how many steps to slide the filter each time.
  - `out size = floor ((n +2p - f)/S + 1)`

### RGB images
- For RGB images, we have 3 matrices for each image, each for one channel R, G and B. (# channels also called depth)
- Filters will have 3 matrices too(each filter is a volume with number of channels equal to that of input channel count).
- Each filter has it's own single bias value. filter values are weights.
  - So conv layer parameters are the count of weights and biases in filters and not the feature count by neuron count like in dense layers
- If want edges only in red channel, then use edge filter for red and set G and B to zeros.
- If want edges regardless of channel, set all 3 filters to the edge filter.
- outputs are only one flat matrix, since we sum over the channels too when doing convolution.
  - when using multiple filters, the output depth will be equal to the filter count. One sheet for each filter.

### Layer types: Conv, Pool and FC
CNNs can be built using `convolutional layers(conv)` only, but many architectures use `Pooling layers(Pool)` and `fully connected layers(FC)`(Dense layers)

- Pooling layer: Max pooling: taking regions on img equal to filter size and taking one max value out of it. (parameters: stride and filter size)
  - same formula for output width and height as that of filters
  - max pooling is done independently on each channel, so output has same num of channels as inputs.
  - Average pooling also exists.
  - in pooling there are no parameters to learn.

- We usually have conv layer and pooling layer after it, and those combined are counted as 1 layer.
- At last, the output is flattened (rolled out) into a single vector to feed to a number of FC layers (normal NN dense layers) that give final output.

### Conv filters and Pooling output sizes
**Conv filters**
- Each conv layer has a number of filters
- Each of these filters has a depth equal to the input's depth (channel count)
- Each filter's output is flat (2D / depth is 1) as output is summed over the channels
- But the layer's output depth is equal to the filter count
- Example
  - 10 filters each of size 4x4x3, where 3 is channel count
  - Output width,height: calculate based on formulas
  - Output depth:  10 (filter count)

**Pooling**
- max, averaging
- usually grouped with conv layers as one layer
- Output for width and height: same formulas as filters
- Output depth: equals input depth, as pooling is applied to each channel separately.

### Classical conv nets:
- LeNet-5
- AlexNet
- VGG-16
- ResNets:
  - output of a layer skips some layers and goes directly to deeper layers.
  - To reduce the impact of vanishing gradients and exploding gradients on the learning speed a problem with very deep plain networks
  - Has CONV `blocks` and Identity `blocks` 
  - More details in the [ResNet jupyter notebook](../jupyter_notebooks/Residual_Networks_ResNets/Residual_Networks.ipynb)

1x1 convolution: AKA Network in Network
**Inception Network**: use different filter sizes and layer types(conv, pool) and stack output together, let NN decide the best combination.

### MobileNet:
- less expensive computationally. 
- Has depthwise separable convolutions and pointwise convolution (normal conv but 1x1)
- [Good MobileNet reference here(Japanese)](https://qiita.com/omiita/items/77dadd5a7b16a104df83)


EfficientNet: adapt architecture to certain device.

ImageNET: A large image database with more than 14 million labelled images.

PCA color augmentation: Principal component analysis?

# Localization/ object detection:
Classification with localization: one object, classify it and locate it with a box.
Detection: Find all such objects and box them
*Localization: Make the classification CNN also output 4 more outputs x,y of the box center, width and height of the box. Also one binary output for the existence of the object.
* Landmark detection: identifying points on the face for example or body to get emotion for example or pose
--> used in things like snapchat filters and VR
* Sliding window detection algorithm:
- train CNN to detect cropped images of the object(exactly fitting the object)
- Use windows of different sizes and certain strides to slide through the image and look for the object. Output will be probability of object existence in that window?
- slow cuz need to run window into the NN every slide --> do convolutional implementation instead.
* convolutional implementation of sliding window: One forward prop pass that takes all windows at once and gives prediction, instead of sequential runs.
main idea: turn FC layers to conv layers (ex.:16 neuron FC layer becomes 1x1x16 conv layer), run image through the conv net once, and each element of the output will be the result of each window.

*Making bounding box predictions more accurate: with sliding window, its possible that no position perfectly has all the car inside, also a rectangle might not be the best shape.
-->solution: YOLO algorithm (You Only Look Once algorithm)
*YOLO algo:
conv implementation
place a grid on the image. Better to have fine grids.
design CNN such that it outputs an object localization/classification vector for each cell in the grid. (output is grid w x grid h x vector size)
--> box location and size is relative to the grid boundary and grid size.
Run image on the CNN.
* Box accuracy measure/ object detection algo evaluation measure: IoU (Intersection over Union): intersection area/union area
--> if over a certain threshold (0.5 usually) then accept box as accurate enough.
* Non-max suppression algorithm: to avoid detecting the same object multiple times when using a fine grid.
- First filter out low probabilities (ex.: p <0.6).
- Take box with highest probability and suppress other boxes that have high IoU with it. Repeat until done.
- Done separately for each class.
* Anchor boxes: for overlapping objects : pre-define a number of box shapes. ex.: one wide box for cars and another slim one for pedestrians.
output will be for bot boxes.
* R-CNNs (Regions with CNNs): Identify regions in image and run predictions on those regions of interest only.
Segementation algorithm to identify regions. Will output box and label for each blob (region)

* Semantic Segmentation (U-net algorithm): Outline objects so you know which pixels belong to it and which do not.
output pixels in colored groups based on predicted label (as in self driving cars: peds, cars, drivable road..)
ex.: segmenting organs in an X-ray or abnormalities.
Ex.: background VS car segmentation: for each pixel, output 1 or zero depending on wheter it belongs to car or background.
Since we need output for all pixels, the U-NET architecture will, as notmal CNNs, shrink the dimensions of the img at the beginning,
but as opposed to other CNNs, it will grow the size back in the latter part.
How to grow size? --> Transpose convolution
The output size is same as the input size, but with number of channels equal to the number of classes we want to identify.
Each pixel has a probability of belonging to that channel (class). we take maximum for classification.

*Transpose convolution:
We have small input, bigger filter, intended size output(original img size).
have stride and padding parameters, but padded area values will be ignored when calculating.
each single input element is multiplied by all filter values --> filter is pasted on the output (ignore pads)
one step in input is one stride in output in same direction
values in areas with overlap in the output are summed.
To calculate needed filter size, assume output as input and the input as filter, then calculate based on the previous formulas that take padding and stride.

U-net implementation contains skip connections.

* One shot learning problem: Using one training example to recognize something(Ex.: faceID where only one pic of each employee is available)
Conv nets with softmax don't work because not enough training examples. Also what if new employee joins, we don't want to retrain the CNN everytime that happens.
--> solution: Learn a similarity function d(img1, img2): degree of difference between them: Low if same person, high if not (threshold?????)
*Siamese network: train a CNN without last softmax layer to output an encoding for each image, and then use distance function to calculate the similarity.
Must be tuned to output values with high difference for different people and low difference for same people.
*Triplet loss: Have an anchor image (comparison base), negative example and positive example. Then loss will be difference between distance of each pair.
Make sure the nn wont get trivial solutions where all encoding are the same or all are zero.
Want d(A,P) <= d(A,N) --> Loss(A,P,N) = max (d(A,P)-d(A,N)+margin, 0) .. if negative then loss is zero, which is what we aim to (negative example distance is larger than positive eample distance)
When training, better to choose triplets that are hard (their distances are close).
*Alternatively: can have two siamese networks feed into a binary classification layer and treat the problem as supervised learning binary classification
where labeled pairs of images (0 or 1 depending if same person or not) are fed to the system. Delta of encoding is fed to the sigmoid function X part in wX+b.

to see what each layer/ neuron is learning, look for training data that maximizes output on that neuron.

*Neural Style Transfer:
generate initial img randomly, and GD to minimize the cost function which is hyper-parameter weighted sum of content cost function(how close to original image) and style cost function(how close to style image)
- content cost function: Use pretrained network (VGG for example), and use a layer somewhere in the middle not too shallow not too deep.
Use the activation output of that layer of both generated image and content image and see the difference--> content cost function
(because layers midway can recognize shapes)
- style cost function: based on correlation between channels of some middle layer (compare correlation value of the style image to the value of the generated image)

# Course 5: RNNs, LSTM and NLP
---------
Sequence models to work with sequence data:
speech recognition, music generation, DNA sequence analysis, machine translation, video activity recognition ...etc
Named entity recognition: determine names (people/companies/countries/currencies..) from sentences

RNNs: Prediction of input Xi in a sequence depends on Xi and on the activation of the previous input(hence recurrent/sequential).
Initial activation can be set to zero
has a weight for input activations Wa and a weight for input data Wx
nth activation = g(Wa * a_n-1 + Wx * Xn + b_a) : g usually tanh or ReLU
nth output y_n = g(Wy * a_n + b_y) : g depends on what type of output
Uni-directional RNNs use inputs from past only.
Bidirectional RNNs (BRNNs) use information from both sides of the sequence.
Many architectures: many-to-one, many-to-many(equal or different size in/out), one-to-many (Sequence Generation)

RNNs can easily have the vanishing gradients problem..also early inputs in the sequence sometimes need to affect later outputs. (The cat/cats, who ate........, was/were .. )
--> solution: Gated Recurrent Units (GRU)
memory cells that get updated and gamma(??) that can be 0 or 1 to act as gate.
2 gates: update gate and relevance gate.

* LSTM : Long Short Term Memory
A more powerful and more general version of GRU
has 3 gates: Forget gate, Update gate, Output gate
Deep RNNs have multiple layers but also it has a temporal dimension (for inputs at each position of sequence, they go through the RNN layers)

*Word embeddings..have features for each word: gender, food, cost, animal..etc
*Beam search: selects best output of multiple possible outputs (ex.: sentence translation)
*Self-attention / multi-head attention
* The Transformer
