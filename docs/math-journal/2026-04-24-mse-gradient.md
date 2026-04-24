# MSE gradient — derivation from scratch

**Date:** 2026-04-24
**Objective:** derive the gradient of mean squared error with respect to the weight vector,
in full index notation, then reassemble into matrix form and solve for the closed-form optimum.

## Setup

- Data matrix $X \in \mathbb{R}^{n \times d}$ (n samples, d features).
- Weight vector $w \in \mathbb{R}^d$.
- Targets $y \in \mathbb{R}^n$.
- Prediction $\hat{y} = Xw \in \mathbb{R}^n$.
- Loss:
$$
L(w) = \frac{1}{n} \| Xw - y \|_2^2 = \frac{1}{n} \sum_{i=1}^{n} (Xw - y)_i^2.
$$

## Step 1 — Expand into a sum over samples using index notation

(your work here — rewrite $L(w)$ so every index is explicit)

## Step 2 — Compute $\partial L / \partial w_k$ for a single scalar parameter

(your work here — show chain rule, Kronecker delta, every step)

## Step 3 — Reassemble $\nabla_w L$ in matrix form

(your work — rewrite the indexed expression as a matrix/vector product, confirm dimensions)

## Step 4 — Solve $\nabla_w L = 0$ for $w^*$

(your work — end with $w^* = ...$)

## Step 5 — When does the closed form break down?

(one sentence — name the condition and what is done in practice)

## Reflection

(2–3 sentences: what was hard, what clicked, what you'd teach someone else)