# Artificial Neural Networks

This repository contains Python code that implements artificial neural networks for various machine learning tasks, including binary classification using neural networks with one and two layers. The code is divided into two main parts:

## 1. Part 1: Network Architecture in NumPy
   - **Single Neuron Model**: 
     - Implements a simple neuron model using NumPy, capable of solving linearly separable classification tasks.
     - The neuron uses a predefined activation function (e.g., `hardlim`) and is evaluated on generated linearly separable data.
   - **Two-Layer Network**:
     - Implements a basic two-layer neural network using NumPy, which can handle non-linearly separable data.
     - The network is trained by randomly adjusting weights to achieve 100% classification accuracy on the provided dataset.

## 2. Part 2: Network Training in PyTorch
   - **Multi-Layer Network in PyTorch**:
     - Extends the neural network model to include multiple hidden layers using PyTorch, allowing for more complex model architectures.
     - The network is trained using gradient descent to minimize binary cross-entropy loss, making it suitable for complex tasks like the two-spirals problem.
     - The training process includes mini-batch gradient descent, evaluation, and visualization of the learning process.

## Additional Utilities:
   - **Data Generation**: Functions to generate synthetic datasets such as the two spirals dataset, which is a classic challenge for neural networks.
   - **Visualization**: Tools for visualizing the data, decision boundaries, and activation functions, helping to understand the network's performance and decision-making process.

This code is designed to help understand the implementation and training of artificial neural networks, providing a hands-on approach to building, training, and evaluating neural networks from scratch.
