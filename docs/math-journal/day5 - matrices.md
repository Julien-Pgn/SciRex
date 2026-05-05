## 8. Your exercises

Show your work. State dimensions before computing — make this a habit.

### Exercise 5.1 — matrix-vector multiplication

Let $A = \begin{bmatrix} 2 & 1 \\ 0 & 3 \\ 1 & -1 \end{bmatrix}$ and $x = \begin{bmatrix} 4 \ 2 \end{bmatrix}$.

(a) State the dimensions of $A$, $x$, and $Ax$. Use the inner-dimensions-match rule to verify the product is defined.
$$A \in \mathbb{R}^{3 \times 2}, x \in \mathbb{R}^{2}$$
(b) Compute $Ax$ row by row. Show each dot product separately.
$$Ax = [10, 6, 2]$$
### Exercise 5.2 — the linear model on a small dataset

You have 3 customers, each with 2 features (say, normalized age and income):

$$ X = \begin{bmatrix} 1 & 3 \\ 2 & 1 \\ 0 & 4 \end{bmatrix} $$

The model has weights $w = (0.5, 1.0)$.

(a) State the dimensions of $X$, $w$, and $\hat{y}$.
$$X \in \mathbb{R}^{3 \times 2}, w \in \mathbb{R}^{2}, \hat{y}  \in \mathbb{R}^{3}$$

(b) Compute $\hat{y} = Xw$. You should end up with three predictions, one per customer.
$$\hat{y} = [3.5, 2, 4]$$
(c) The true targets are $y = (3.0, 2.5, 4.0)$. Compute the **error vector** $e = \hat{y} - y$ component by component.
$$e = [0.5, 0.5, 0]$$

(d) Compute the squared error norm $|e|^2 = e \cdot e$. This is the unscaled MSE — the actual MSE divides by $n = 3$.
$$|e|^2 = [0.25, 0.25, 0]$$

### Exercise 5.3 — transpose and dimension chaining

Let $A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}$.

(a) Write out $A^T$ and state its dimensions.
$$A^T = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}$$
$A^T \in \mathbb{R}^{3\times 2}$

(b) Let $v = \begin{bmatrix} 1 \ 1 \ 1 \end{bmatrix}$. Compute $Av$, stating dimensions before and after.
$A \in \mathbb{R}^{2\times 3}, Av \in \mathbb{R}^{2}$
$$Av = [6, 15]$$
(c) Now compute $A^T (Av)$ — apply $A^T$ to the result you got in (b). State dimensions before and after.
$A^T \in \mathbb{R}^{3\times 2}, A^T(Av) \in \mathbb{R}^{3}$
$$A^T (Av) = [66, 87, 108]$$

This pattern — "apply $A$, then apply $A^T$" — is the beginning of the algebra behind the closed-form linear regression solution. You don't need to interpret it yet, just compute it.

### Exercise 5.4 — small matrix-matrix product

Let $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ and $B = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}$.

(a) State the dimensions of $A$, $B$, and $AB$. Confirm the inner dimensions match.
$A \in \mathbb{R}^{2\times 2}, B \in \mathbb{R}^{2\times 2}$
This question is way too simple to confirm the inner dimensions based on mathematical formulas. 

(b) Compute the four entries of $AB$ one by one, using the row-of-$A$-dot-column-of-$B$ rule. Show each dot product.
$$AB = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}$$

---

## 9. Self-check questions

Answer in one or two sentences each.

1. The matrix-vector product $Ax$ where $A \in \mathbb{R}^{m \times d}$ and $x \in \mathbb{R}^d$ produces a vector in $\mathbb{R}^m$. In your own words, why is the output dimension $m$ and not $d$? Connect your answer to the "stacked dot products" picture.
	1. Because you compute the dot product of an entire row composed of several columns (representing the features). In this example, A is the DL network with m layers each composed of d neurons.  
    
2. In $\hat{y} = Xw$ with $X \in \mathbb{R}^{n \times d}$ and $w \in \mathbb{R}^d$: what does each _row_ of $X$ represent in ML terms, and what does each _entry_ of $\hat{y}$ represent?
	1. Each row of X is a layer (n layers composed of d neurons) and each entry of $\hat{y}$ is the prediction made by each neuron. 

3. A neural network layer takes a 1000-dimensional input and produces a 50-dimensional output. What are the dimensions of the weight matrix for this layer? Using the formula for matrix-vector multiplication, how many scalar multiplications happen in the forward pass through this layer?
	1. The layer produces a vector of stacked scalars of each neurons, thus we have 50 neurons. The weight matrix has a dimension of 20 rows and 50 columns which gives a 1000 dimension. During the forward pass through this layer we will have 1000 scalar multiplications. 
    

---

## 10. Reflection (brief)

Two sentences:

- One thing that clicked today.
	- The importance of matrices in DL.
- One thing still fuzzy.
	- Matrices products.