# Probabilistic Graphical Inference

### Description
Probabilistic Graphical Inference Algorithms - Commonly used for Graph-based Data Structures

Probabilistic Graphical Models exploit depedencies between random variables to create a graph-based representation of joint distributions. These representations are very flexible and are now frequently used in deep learning networks (check out Graph Convolutional Neural Networks, for instance). For more explanation on Probabilistic Graphical Models, check out: http://6.869.csail.mit.edu/fa19/lectures/L10graphicalModelsInference.pdf 

In this work, I implement belief propagation algorithms on Markov Random Fields in the pgm.py file. 

### Guide

To run the efficient belief propagation algorithm, use:

```bash
       python pgm.py --input "test_graph_00.pickle"
```

To experiment with the brute force algorithm use: 

```bash
       python pgm.py --input "test_graph_00.pickle" --brute_force
```

To generate a large random tree and test the algorithm, use:
```bash
       python pgm.py --input "test_graph_00.pickle" --random
```

One can change the input argument to whichever test graph follows the pickle format found in this repo. Note that for the large random tree we will not be permitting brute force to be used.
