## 9. Your exercises

Today's exercises focus on translating between index notation and matrix notation, and reading sums.
### Exercise 6.1 — basic sum manipulation

Compute each sum explicitly.

(a) Let $a = (2, -1, 4, 3)$. Compute $\sum_{j=1}^{4} a_j$.
$$a = 8$$
(b) Let $a = (1, 2, 3)$ and $b = (4, 5, 6)$. Compute $\sum_{j=1}^{3} a_j b_j$. (You should recognize this.)
$$ab = 4 + 10 + 18 = 32$$
(c) Let $c = 5$ and $a = (1, 2, 3, 4)$. Verify the linearity rule: compute both $\sum_{j=1}^{4} c , a_j$ and $c \sum_{j=1}^{4} a_j$, and confirm they're equal.
$$ac = 5 + 10 + 15 + 20 = 50$$
$$ac = 5 * 10 = 50$$
### Exercise 6.2 — translating between notations

**Goal.** Translate $\hat{y} = Xw$ (where $X \in \mathbb{R}^{n \times d}$, $w \in \mathbb{R}^d$, $\hat{y} \in \mathbb{R}^n$) into index notation, and verify dimensions.
$$ \hat{y}_i = \sum_{j=1}^{d} X_{ij} , w_j. $$

For each matrix expression, write the equivalent index-notation equation. State which indices are free and which are summed.

(a) $\hat{y} = Xw$ where $X \in \mathbb{R}^{n \times d}$, $w \in \mathbb{R}^d$.
$$ \hat{y}_i = \sum_{j=1}^{d} X_{ij} , w_j. $$
i is free and j summed.

(b) $z = u \cdot v$ where $u, v \in \mathbb{R}^d$ and $z$ is a scalar.
$$ z = \sum_{i=1}^{d}u_iv_i $$
Here i is not free, it sums, z is a scalar.

(c) $C = AB$ where $A \in \mathbb{R}^{m \times d}$, $B \in \mathbb{R}^{d \times p}$, $C \in \mathbb{R}^{m \times p}$. _Hint:_ this one has two free indices.

$$C_{ik} = \sum_j A_{ij} B_{jk}$$
Free indexes are i and k while j is a summed index. C is a mtrix.
### Exercise 6.3 — reading index notation

For each of the following, identify (i) the free indices, (ii) the summed indices, (iii) the resulting object's shape (scalar, vector, or matrix).

(a) $f_i = \sum_{j} A_{ij} x_j$.
i is free, j is summed. f is a vector.

(b) $s = \sum_{i} \sum_{j} A_{ij} B_{ij}$.
i and j are summed so s is a scalar.

(c) $M_{ik} = \sum_{j} A_{ij} B_{jk}$.
i and k are free, j is summed. M is a matrix.


### Exercise 6.4 — fully indexed concrete example

Let $X = \begin{bmatrix} 2 & 1 \\ 0 & 3 \\ 1 & -1 \end{bmatrix}$ and $w = (4, 2)$ (these are the same numbers as Exercise 5.1 yesterday).

(a) Write the index-notation equation for $\hat{y}_i$ as a sum over $j$.
$$ \hat{y}_i = \sum_{j=1}^{2} X_{ij} , w_j. $$
(b) Unpack the sum for each value of $i$ (i.e., write $\hat{y}_1$, $\hat{y}_2$, $\hat{y}_3$ as fully expanded sums of products) and compute each.

For $i = 1$: $$ \hat{y}_1 = (2)(4) + (1)(2) = 8 + 2 = 10 $$
For $i = 2$: $$ \hat{y}_2 = (0)(4) + (3)(2) = 6 $$
For $i = 3$: $$ \hat{y}_3 = (1)(4) + (-1)(2) = 2 $$
(c) Confirm your answers match what you got in Exercise 5.1.
Yes it is the same : 
$$Xw = [10, 6, 2]$$
### Exercise 6.5 — the loss function in index notation

The MSE loss for a dataset is:

$$ L(w) = \frac{1}{n} \sum_{i=1}^{n} \left( \sum_{j=1}^{d} X_{ij} w_j - y_i \right)^2. $$

(a) Identify all free indices and all summed indices in this expression. What is the shape of $L(w)$ (scalar, vector, or matrix)?

(b) For $n = 2$, $d = 2$, with $X = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$, $y = (5, 11)$, and $w = (1, 1)$, compute $L(w)$ by carefully unpacking both sums.

Let's start with 
$$ (\sum_{j=1}^{d} X_{ij} w_j - y_i )^2$$
for $i =1$: $(3 - 5)^2=4$
for $i =12: $(7 - 11)^2=16$

Then we do this:
$$ L(w) = \frac{1}{n} \sum_{i=1}^{n}$$
$1/2(4+16) = 10$

So $L(w) = 10$

---

## 10. Self-check questions

1. In index notation, what is the difference between a **free** index and a **summed** index? Give one example of each.
	1. The free index is on the result. The number of free indexes defines the type of results (0 = scalar, 1 = vector, 2 = matrix). The summed term is on the right.
    
2. The matrix-vector product $\hat{y} = Xw$ in matrix notation becomes $\hat{y}_i = \sum_j X_{ij} w_j$ in index notation. Why does this version make differentiation easier?
	1. Because it is a lot more literal. You know you have to multiply X and w and sum all combinations until you reached j. In other words, it tells you to treat your matrix as a n vectors and do you dot product and sum it all.  
    
3. In the expression $C_{ik} = \sum_j A_{ij} B_{jk}$, why does $C$ end up with two indices $(i, k)$? Connect your answer to the dimension rule from Day 5.
	1. Because during the multiplication of 2 matrices, you multiply the rows of A with the columns of B, j times. So the final matrix will be i rows and k columns. 
    
4. The Kronecker delta $\delta_{jk}$ has the "collapsing" property $\sum_j a_j \delta_{jk} = a_k$. In one sentence, explain why this is true.
	1. I don't know. 
