# Non Linear Least Squares From Scratch

# Training Neural Networks Using Non-Linear Least Squares

This repository contains an implementation of **non-linear least squares algorithms** to train a simple neural network from scratch. The project demonstrates the ability of neural networks as universal approximators by approximating various non-linear functions. The Levenberg-Marquardt algorithm is used to optimize the neural network parameters, with a focus on understanding and implementing non-linear least squares methodologies.

## Features
- **Neural Network Architecture**: 
  - Implements a simple feedforward neural network with three input nodes and a single output node.
  - Features three hidden neurons, with the `tanh` activation function applied at each hidden node.
  - A total of 16 tunable parameters (weights) are optimized to approximate non-linear functions.

- **Non-Linear Least Squares Optimization**:
  - Trains the network by minimizing the sum of squared errors between the predicted output and the target function.
  - Implements the **Levenberg-Marquardt algorithm** for efficient optimization of weights.

- **Regularization**:
  - Adds an optional regularization term \( \lambda ||w||^2_2 \) to the loss function to prevent overfitting.
  - Explores the effect of varying regularization strengths on training performance.

- **Noise Robustness**:
  - Evaluates the model's performance on noisy datasets by adding bounded noise to the target outputs.
  - Analyzes the impact of different noise levels on training and testing errors.

## Implementation Details
1. **Gradient Computation**:
   - Derives the gradient of the neural network output with respect to the weights.
   - Explicitly calculates the derivative of the `tanh` function for backpropagation during optimization.

2. **Training**:
   - Generates 500 random input points \( x \in \mathbb{R}^3 \) and computes target values for the chosen non-linear function \( g(x) \).
   - Minimizes the training loss:
     \[
     l(w) = \sum_{n=1}^{N} \left(f_w(x^{(n)}) - g(x^{(n)})\right)^2 + \lambda ||w||^2_2
     \]
   - Uses the Levenberg-Marquardt algorithm to iteratively update weights.
   - Experiments with various initializations, stopping criteria, and regularization strengths (\( \lambda \)).

3. **Testing**:
   - Generates 100 unseen test points to evaluate the model's generalization performance.
   - Computes error metrics to summarize training and testing accuracy under different conditions.

4. **Noisy Data Experiments**:
   - Trains the network on noisy data where the target outputs are perturbed by bounded noise.
   - Studies the effect of noise levels (\( \epsilon \)) on model robustness and accuracy.

5. **Custom Non-Linear Functions**:
   - Extends the implementation to approximate alternative non-linear functions, providing a flexible framework for experimentation.

6. **Visualization** (Optional):
   - Generates contour plots to compare the actual function \( g(x) \) with its learned approximation.

## Key Results
- Training and testing results include:
  - Loss curves showing the convergence of the Levenberg-Marquardt algorithm.
  - Error metrics for varying regularization strengths, initializations, and noise levels.
  - Observations on how changes in training data bounds (\( \Gamma \)) affect model accuracy.

## Getting Started
1. **Dependencies**:
   - Python or MATLAB (code can be adapted for either language).
   - Libraries for numerical computations (e.g., NumPy for Python).

2. **Run the Code**:
   - Generate training data by sampling input points and computing target outputs using the specified non-linear function.
   - Train the neural network by running the Levenberg-Marquardt optimization routine.
   - Evaluate the model on test data and summarize results with error metrics and visualizations.

3. **Customization**:
   - Modify the target function \( g(x) \) to approximate alternative non-linear relationships.
   - Experiment with different regularization parameters, initializations, and noise levels to study their effects.

## Contributions
This project serves as a foundation for:
- Understanding neural network training via non-linear least squares.
- Learning advanced optimization techniques like the Levenberg-Marquardt algorithm.
- Exploring the effects of regularization and noise in machine learning.

Feel free to contribute by experimenting with additional non-linear functions, enhancing visualization techniques, or integrating alternative optimization algorithms.

---

Dive into the repository to explore the fascinating interplay of neural networks and non-linear optimization. This project offers a hands-on approach to implementing core machine learning techniques from scratch, ideal for gaining a deeper understanding of neural networks and optimization principles.
