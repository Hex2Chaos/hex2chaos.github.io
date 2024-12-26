---
title: Gradient Descent
categories:
  - AI
tags:
  - Algorithm
top: true
mathjax: true
date: 2024-12-26 18:00:00
---

## What is Gradient Descent
> Imagine you’re standing on a mountain, surrounded by thick fog, so you can’t see anything clearly. Your goal is to reach the lowest point in the valley. The only way to find your way is to feel the ground under your feet to determine whether it’s sloping uphill or downhill. To reach the lowest point, you take small steps in the steepest downhill direction, moving closer and closer to your goal.

This is the core idea behind gradient descent.
Gradient descent is an optimization method used to find the minimum value of a function, helping us find the "best solution."
![](/assets/2024-12-26/gradient_descent.png)
## Where is Gradient Descent Used
Gradient descent is widely used in machine learning and deep learning! For example:

1. Training neural networks: To find the best set of parameters for the model to perform optimally.
2. Regression problems (e.g., linear regression): To find parameters that minimize prediction errors.

The goal of gradient descent is to minimize the loss function (or error function) of a problem, such as:
![](/assets/2024-12-26/gradient_descent_used.jpg)

Example of ChatGPT:<br>
![](/assets/2024-12-26/gradient_descent_cases.jpg)
CN:
![](/assets/2024-12-26/gradient_descent_cases_cn.jpg)

## How Does Gradient Descent Work
### What is a Gradient
- Imagine you’re standing somewhere in a valley. The gradient tells you the direction of the steepest uphill slope at your current position.
- Mathematically, the gradient is a vector composed of the partial derivatives of a function. It points in the direction where the function increases the fastest.
Therefore, the negative gradient (-Gradient) points in the steepest downhill direction. Gradient descent uses this direction to progressively "walk toward the lowest point."

### The Algorithm Steps
1. Initialize a Starting Point
Start from a random point (randomly initialize the parameters), such as $\theta\omicron$,where $\theta\omicron$ represents the parameters.
2. Compute the Gradient
Calculate the gradient of the loss function (or objective function) at the current point,denoted as $\nabla$L($\theta$). This tells us which direction to move.
3. Update Parameters Along the Gradient
Update the parameters based on the gradient using the formula:
$\theta$t+1 = $\theta$t - $\eta$ * $\nabla$L($\theta$t)
![](/assets/2024-12-26/update_parameters_along_the_gradient.jpg)
4. Repeat the Process
Iterate this process repeatedly, reducing the loss function step by step until we reach a point close to the "minimum."

### Visualizing Gradient Descent
Imagine the objective function is shaped like a bowl (e.g., a quadratic function):

At the start, gradient descent begins at a random position.
At each step, it calculates the gradient direction and moves toward the bottom of the bowl.
**As it gets closer to the lowest point, the gradient becomes smaller, and the steps become shorter until it stops.**

### The Importance of Learning Rate
If the learning rate $\eta$ is too large, you might overshoot the minimum and never converge to the correct answer.
If the learning rate $\eta$ is too small, the steps will be tiny, and convergence will be very slow.
It’s important to tune the step size to make gradient descent both efficient and stable.

## Advantages and Disadvantages of Gradient Descent
### Advantages:
- Simple and efficient: Gradient descent is very practical, especially for large-scale data problems.
- Applicable to various types of objective functions (as long as they are differentiable).

### Disadvantages:
- Can get stuck in local minima: Some functions have multiple minima, and not all of them are globally optimal.
- Requires careful tuning of the learning rate; otherwise, it may diverge or converge too slowly.

## Summary
Start from an initial point, calculate the slope (gradient), and take small steps in the direction that reduces the objective function, repeating this process until you find the optimal solution.