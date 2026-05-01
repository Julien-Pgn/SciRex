## 9. Your exercises

Show every step. For each partial derivative, state which variable is being held constant before you start.

### Exercise 3.1 — basic partials of polynomials

Compute $\partial f / \partial x$ and $\partial f / \partial y$ for each.

(a) $f(x, y) = x^3 + 4y^2$
$$\frac{\partial f}{\partial x} = 3x^2$$
$$\frac{\partial f}{\partial y} = 8y$$
(b) $f(x, y) = 5xy + 7$
$$\frac{\partial f}{\partial x} = 5y$$
$$\frac{\partial f}{\partial y} = 5x$$
(c) $f(x, y) = x^2 y + xy^2$
$$\frac{\partial f}{\partial x} = 2xy + y^2$$
$$\frac{\partial f}{\partial y} = x^2 + 2xy$$
### Exercise 3.2 — three variables

Compute all three partials for:

$f(x, y, z) = x^2 y - 3yz + z^4.$

$$\frac{\partial f}{\partial x} = 2xy$$
$$\frac{\partial f}{\partial y}= x^2 -3z$$
$$\frac{\partial f}{\partial z} = -3y + 4z^3$$
### Exercise 3.3 — partial chain rule

Compute both partials for $f(x, y) = (x + y^2)^4$.

$$\frac{\partial f}{\partial x} = 4(x + y^2)^3 * 1 = 4(x + y^2)^3$$

$$\frac{\partial f}{\partial y} = 4(x +y^2)^3 * 2y = 8y(c + y^2)^3$$
### Exercise 3.4 — partial chain rule with two-variable inner

Compute both partials for $f(x, y) = \sqrt{x^2 + y^2}$.

This function appears constantly in ML — it's the Euclidean distance from the origin to the point $(x, y)$. _Hint:_ rewrite as $(x^2 + y^2)^{1/2}$ first.

$$
\frac{\partial f}{\partial x} = \frac{1}{2}(x^2 + y^2)^{-1/2} \cdot 2x = \frac{x}{(x^2 + y^2)^{1/2}} = \frac{x}{\sqrt{x^2 + y^2}}
$$
$$
\frac{\partial f}{\partial y} = \frac{1}{2}(x^2 + y^2)^{-1/2} \cdot 2y = \frac{y}{(x^2 + y^2)^{1/2}} = \frac{y}{\sqrt{x^2 + y^2}}
$$
### Exercise 3.5 — ML connection: two-feature squared loss

Let the data point be $(x_1, x_2) = (1, 4)$ with target $y = 10$. The model is $\hat{y} = w_1 x_1 + w_2 x_2$ and the loss is:

$$ L(w_1, w_2) = (w_1 x_1 + w_2 x_2 - y)^2. $$

(a) Substitute the data point in to get $L(w_1, w_2)$ as a function of weights only.
$$
L(w1,w2) = (w1 + 4w2 - 10)^2
$$
(b) Compute $\partial L / \partial w_1$ using the chain rule. Do not expand the square.
$$\frac{\partial f}{\partial w1} = 2(w1 +4w2-10)$$
(c) Compute $\partial L / \partial w_2$ using the chain rule.
$$\frac{\partial f}{\partial w2} = 8(w1 +4w2-10)$$
(d) At the point $(w_1, w_2) = (2, 1)$, what are the numerical values of $\partial L / \partial w_1$ and $\partial L / \partial w_2$?
$$\frac{\partial f}{\partial w1} = -8$$
$$\frac{\partial f}{\partial w1} = -32$$
(e) In one sentence each, explain in plain English: (i) what the _sign_ of $\partial L / \partial w_1$ at $(2, 1)$ tells gradient descent to do, and (ii) why $\partial L / \partial w_2$ is four times $\partial L / \partial w_1$ in magnitude (think about the input features).
- The negative sign of $\frac{\partial f}{\partial w1}$ tells gradient descent to increase the values to minimize the loss for the next epoch. $\frac{\partial f}{\partial w2}$ being 4 times $\frac{\partial f}{\partial w1}$ means that a small changes in the weights of 2 will move the loss function 4 times more than modifying w1. 
- Correction: Gradient descent moves the weight in the opposite direction of the gradient. Concerning the 4 times difference: **The deep ML principle**: weights connected to large-magnitude features receive proportionally larger gradient updates. This is why **feature scaling** (standardizing features to similar magnitudes) is one of the first preprocessing steps in any ML pipeline — without it, weights tied to large features dominate training and weights tied to small features barely move. You'll see this in scikit-learn as `StandardScaler` or `MinMaxScaler`. 
### Exercise 3.6 — find a critical point

Let $f(x, y) = x^2 + y^2 - 4x + 6y + 5$.

(a) Compute $\partial f / \partial x$ and $\partial f / \partial y$.
$$
\frac{\partial f}{\partial x} = 2x -4 $$
$$
\frac{\partial f}{\partial y} = 2y + 6$$
(b) A **critical point** is a point where _both_ partials equal zero simultaneously. Set both to zero and solve the resulting system of two equations for $(x^_, y^_)$.

$$
\begin{aligned}
x=2 \\
y=-3
\end{aligned}
$$

(c) Compute $f(x^_, y^_)$.
$$f(2, -3) = 4 +9 -8 - 18 +5 = -8$$
(d) In one sentence, explain why setting all partial derivatives to zero is the multivariable generalization of "set the derivative to zero" from Day 1. _Hint:_ think about when none of the inputs has any direction of decrease.
- I wish I knew but fatigue is getting me. 
- Correction: In single-variable calculus, "derivative = 0" means the function is *flat* at that point — the slope is zero, so neither moving left nor right changes ff f to first order. That's the defining condition of a minimum, maximum, or inflection. In multivariable calculus, the function has many "directions to move" — not just left/right but every direction in the input space. Each partial derivative measures the rate of change in _one specific axis direction_. So "all partials = 0" means **the function doesn't change to first order in any axis direction.** But why does that catch all minima? It turns out (and this is a real theorem you'll use later) that if a function changes in any direction, it must change in at least one axis direction too. So **all partials = 0 ⟺ no direction of immediate change ⟺ candidate for minimum/maximum**.
- In ML terms: at a minimum of the loss, the gradient (vector of all partials) is zero. Gradient descent is, in fact, defined to halt when the gradient is sufficiently close to zero — because at that point, no weight update would meaningfully reduce the loss. **"Gradient = 0" is the multivariable generalization of "derivative = 0," and it is the convergence criterion for training.**

---

## 10. Self-check questions

Answer each in one or two sentences in your own words.

1. The notation switches from $d/dx$ to $\partial/\partial x$ when we go from one variable to many. What does the curly $\partial$ signal that the straight $d$ does not?
	1. It signal that there is another variable that you should treat as a constant.
    
2. When computing $\partial f / \partial x$ for $f(x, y, z) = x^2 yz$, which variables are treated as constants and what is the answer?
	1. Y and z are treated as constants and $\frac{\partial f}{\partial x} = 2xyz$
    
3. A neural network has 1 million weights. How many partial derivatives appear in the gradient of its loss function? In plain English, what does each one tell us?
	1. It has 1 million partial derivatives. Each one tells us how important they are in the loss function relatively to the others. 
	2. Correction: Each partial derivative tells gradient descent how much (and in what direction) to update one specific weight at the current point in training. The partial derivative doesn't tell how important a weight is but tells the local sensitivity at the current weight setting how much the loss would change if you nudged w slightly. 
    
4. Why is the answer for $\partial f / \partial y$ in Section 7's example ($3(x^2+y)^2$) the same as the _outer factor_ of $\partial f / \partial x$? What does this say about the efficiency of computing many partials at once?
	1. It is the same due to the chain rule where we are computing derivatives of a function in a function. Thus we have the same outer function in both partial derivatives which saves computing as it is calculated once and placed in cache for the other partial derivatives computing. 
    
5. Suppose at some weight setting in a neural network, $\partial L / \partial w_1 = +0.5$ and $\partial L / \partial w_2 = -0.3$. In one sentence, what should gradient descent do to each weight, and why?
	1. Gradient descent should move w1 lower, in the negatives to reduce the loss and move w2 in the positives to reduce the loss. The partial derivative being negative, means that the loss decreases as w2 increases so gradient descent increases w2. 
    

---

## 11. Reflection (required)

Write 3-5 sentences:

- Where on a 1-5 scale do you sit on partial derivatives today? Be honest.
	- 4, I feel the compound effect of day 1 and day 2 of math exercises. 
- What was the most confusing thing? Was the "treat others as constants" rule intuitive or did it take a few examples to click?
	- It was not intuitive but a bit of attention solves it easily. 
- In your own words, why are partial derivatives the natural object for training neural networks (as opposed to ordinary derivatives)?
	- Because there are many weights, not one so you cannot calculate 1 derivative as the loss functions takes all weights. 