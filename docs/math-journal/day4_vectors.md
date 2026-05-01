## 10. Your exercises

### Exercise 4.1 — vector arithmetic

Let $a = \begin{bmatrix} 2 \ -1 \ 3 \end{bmatrix}$, $b = \begin{bmatrix} 0 \ 4 \ -2 \end{bmatrix}$.

(a) Compute $a + b$.
$$a + b = \begin{bmatrix} 2 \ 3 \ 1 \end{bmatrix}$$
(b) Compute $a - b$.
$$a - b = \begin{bmatrix} 2 \ -5 \ 5 \end{bmatrix}$$
(c) Compute $5a$.
$$5a = \begin{bmatrix} 10 \ -5 \ 15 \end{bmatrix}$$
(d) Compute $-2b$.
$$-2b = \begin{bmatrix} 0 \ -8 \ 4 \end{bmatrix}$$
(e) Compute $3a + 2b$.
$$3a + 2b = \begin{bmatrix} 6 \ -3 \ 9 \end{bmatrix} + \begin{bmatrix} 0 \ 8 \ -4 \end{bmatrix} = \begin{bmatrix} 6 \ 5 \ 5 \end{bmatrix}$$
### Exercise 4.2 — dot products

(a) Compute $a \cdot b$ for the vectors above.
$$a \cdot b = 0 - 4 - 6 = -10$$
(b) Let $u = (1, 2, 3)$ and $v = (4, 5, 6)$. Compute $u \cdot v$.
$$u \cdot v = 4 + 10 + 18 = 32$$
(c) Let $p = (1, 0, 0)$ and $q = (0, 1, 0)$. Compute $p \cdot q$. (Pay attention to this answer — it's important.)
$$p \cdot q = 0 + 0 + 0 = 0$$
(d) Let $r = (3, 4)$ and $s = (3, 4)$. Compute $r \cdot s$.
$$r \cdot s = 9 + 16 = 25$$
(e) Compute $|s|$ for the vector $s$ above.
$$|s| = \sqrt{9 + 16} = 5$$
### Exercise 4.3 — norms

Compute the norm of each.

(a) $u = (3, 4)$.
$$|u| = \sqrt{9 + 16} = 5$$
(b) $v = (1, 1, 1, 1)$.
$$|v| = \sqrt{4} = 2$$
(c) $w = (2, -2, 1)$.
$$|w| = \sqrt{4 + 4 +1} = 3$$
(d) $z = (5, 0, 0, 0)$. _Comment in one sentence on why this answer makes intuitive sense._
$$|z| = \sqrt{25} = 5$$
This makes sense because all other values composing the vector are 0 so the vector norm is actually its lenght or magnitude. 
### Exercise 4.4 — ML connection: prediction as dot product

Let the weight vector be $w = (0.5, -1.0, 2.0)$ and three different feature vectors be:

$$ x_A = (1, 1, 1), \quad x_B = (2, 0, 1), \quad x_C = (-1, 3, 0). $$

(a) Compute $\hat{y}_A = w \cdot x_A$, $\hat{y}_B = w \cdot x_B$, $\hat{y}_C = w \cdot x_C$.
$$\hat{y}_A = 0.5 - 1 + 2 = 1.5$$
$$\hat{y}_B = 1 + 2 = 3$$
$$\hat{y}_C = -0.5 - 3 = 2.5$$
(b) The true target for $x_A$ is $y_A = 0$. Compute the squared loss $(\hat{y}_A - y_A)^2$.
2,25
(c) Geometrically (using the alignment intuition), why might the model predict $\hat{y}_C$ as the smallest of the three? _No proof needed, just one sentence._
- I don't know.
### Exercise 4.5 — perpendicular vectors

Two vectors are **perpendicular** (also called _orthogonal_) when their dot product is zero. This corresponds geometrically to a 90° angle.

(a) Verify that $u = (1, 2)$ and $v = (-2, 1)$ are perpendicular.
$$u \cdot v = -2 + 2 = 0$$ The dot product is 0, thus u and v are orthogonal. 

(b) Find a vector $w$ that is perpendicular to $a = (3, 4)$. _Hint: there are infinitely many; pick any one and verify the dot product is zero.
- b = (-4, 3) -> $$a \cdot b = -12 + 12 = 0$$
(c) In one sentence, why does it make geometric sense that perpendicular vectors are "as different as possible" in direction?
That is due to the angle between them. 180° is th exact opposite which is juts an inverted verctor thus perpendicular vectors are pointing to direction that are different as possible as 90° is the angle of maximum different direction. Below or above they get closer to one another or to their opposite direction.  
### Exercise 4.6 — gradient descent in vector form

Let $w_{\text{old}} = (1.0, 2.0, -0.5)$ be the current weights. The gradient of the loss is $\nabla L = (0.4, -0.2, 0.1)$. The learning rate is $\eta = 0.1$.

(a) Compute $w_{\text{new}} = w_{\text{old}} - \eta \nabla L$. Show every component.
$$w_{\text{new}} = (1, 2, -0.5) - (0.04, -0.02, 0.01) = (0.96, 1.98, -0.51) $$
(b) For each component, in one phrase, say whether it increased, decreased, or stayed roughly the same, and connect this to the sign of the corresponding gradient component.
- Each component decreased a little but remain roughly the same which suggest that the derivative of the loss, the gradient is positive? 

---

## 11. Self-check questions

1. In one sentence, what does it mean for two vectors to be "the same dimension," and why is this required for addition?
	1. It means that they have the same number of values. It is important as otherwise you cannot add a value to nothing.

2. The dot product $u \cdot v$ produces a scalar from two vectors. The norm $|u|$ also produces a scalar from a single vector. Express the norm in terms of the dot product. (Don't just look at Section 7 — say it in your own words.)
	1. The norm is the squarre root of the vector's dot product itself.
    
3. The forward pass of a linear model with $d$ features can be written as a single dot product $w^T x$. If we instead had a network with 100 neurons in the first layer, how many dot products are computed in the forward pass of that one layer? (One sentence, just the count and reasoning.)
	1. We would have 100 dot product as each neuronal output is a vector. 
    
4. Suppose two word embeddings $u$ and $v$ have a large positive dot product. In plain English, what does that say about the meanings of the two words?
	1. If they have a large positive dot production it means that they have the same direction and thus their meaning is very close in the embedded space. 
    
5. The gradient descent update $w_{\text{new}} = w_{\text{old}} - \eta \nabla L$ is a vector equation. If the network has 1 million weights, how many scalar subtractions are happening in this single line of math? Why is it efficient to write it this way?
	1. 1 millions scalar substraction because every wheight will be updated. 
    

---

## 12. Reflection (required)

Write 3-5 sentences:

- Where on a 1-5 scale do you sit on vectors today?
	- 4, I like it a lot. I understand better and better the concepts and the maths behind ML and DL more specifically. 
- Did the "list view" or the "arrow view" come more naturally to you? (No wrong answer — most people lean one way.)
	- The list view is more natural. But sometimes, I try to represent the vectors in space to visualize a little better their differences. 
- In your own words, what is the relationship between dot products and the predictions of a linear model?
	- The dots products is the mathematical operations for updating the weights after an epoch. 