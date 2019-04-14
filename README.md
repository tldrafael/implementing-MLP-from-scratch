This small project is for my education purpose, which I try to implement in Python the main concepts of a vanilla **Multilayer Perceptron**.

The concepts worked here were:
  - Feedfoward process
  - Backpropagation
  - Gradient Checking
  - Create distinct neural net structures
  - Use distinct batch sizes

The scripts allow:
  - To work with two problems: `linear-regression` and `logistic-regression`;
  - To instance a neural net with any number of hidden layers and any number of units;
  - To use or not the bias elemnt in the layer;
  - To use gradient checking;
  - To set any size of batch;

The restrictions are:
  - Only work with an output of one dimension;
  - All activation functions are `softmax function`;

***
To see an example of how to train a neural net, check the [main](main.ipynb) notebook.
