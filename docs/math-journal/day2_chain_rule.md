## 10. Your exercises

Show every step. Identify inner and outer functions explicitly in each chain rule problem.

### Exercise 2.1 — basic chain rule

Compute the derivative of each. For each, write down the inner and outer function before applying the rule.

(a) $f(x) = (5x + 2)^3$
$$f'(x) = 3(5x + 2)^2 * 5 = 15(5x + 2)^2$$

(b) $f(x) = (x^2 + 1)^4$
$$f'(x) = 4(x^2 + 1)^3 * 2x = 8x(x^2 + 1)^3$$

(c) $f(x) = (7 - 2x)^6$
$$f'(x) = 6(7 - 2x)^5 * -2 = -12(7 - 2x)^5$$

### Exercise 2.2 — chain rule with non-integer powers

Compute the derivative. Recall $\sqrt{u} = u^{1/2}$ and $\frac{1}{u^n} = u^{-n}$.

(a) $f(x) = \sqrt{x^2 + 9}$
$$
\begin{aligned}
f(x) = (x^2 + 9)^1/2 \\
f'(x) = 1/2(x^2 + 9)^-1/2 * 2x = x(x^2 + 9)^-1/2 = x/\sqrt{x^2 + 9}
\end{aligned}
$$

(b) $f(x) = \dfrac{1}{(2x + 3)^2}$ 
$$
\begin{aligned}
f(x) = (2x + 3)^-2 \\
f'(x) = -2(2x + 3)^-3 * 2 = -4(2x + 3)
\end{aligned}
$$

### Exercise 2.3 — product rule

Compute the derivative.

(a) $h(x) = x \cdot (x^2 + 1)$ 
$$
h'(x) = (x^2+1) + 2x^2
$$

(b) $h(x) = (2x + 1)(3x - 4)$
$$
h'(x) = 2(3x - 4) + 3(2x + 1) = (6x -8) + (6x + 3) = (3x - 4) + (x + 1)
$$
### Exercise 2.4 — chain rule meets product rule

Compute the derivative.

$h(x) = x \cdot (x + 1)^3$
$$
h'(x) = (x + 1)^3 + x * 3(x+1)^2 * 1 = (x+1)^3 + 3x(x+1)^2
$$

_Strategy:_ this is a product of two functions, where the second function is itself a composition. Use the product rule, and inside it, use the chain rule on the cubed term.

### Exercise 2.5 — ML connection: one-sample squared loss

This is a baby version of the MSE gradient you'll derive on Day 10. Take it seriously — it is the single most important exercise in this lesson.

Let $x = 2$, $y = 7$ be a fixed data point (input $x$, target $y$), and let $w$ be a parameter. Define the loss:

$$ L(w) = (wx - y)^2. $$

(a) Substitute $x=2, y=7$ to get $L(w)$ as a function of $w$ alone.
$$
L(w) = (2w - 7)^2
$$
(b) Compute $L'(w)$ in two different ways:

- **Way 1:** Expand the square first, then differentiate term by term.
$$
\begin{aligned}
L(w) = 4w^2 - 28w + 49
L'(w) = 8w - 28 = 2w -7
\end{aligned}
$$
- **Way 2:** Apply the chain rule directly without expanding (inner = $wx - y$, outer = $u^2$). Verify the two methods give the same answer.
$$
L'(w) = 2(2w-7) * 2 = 8w - 28 = 2w -7
$$
(c) Solve $L'(w) = 0$ for $w^*$.
$$
w^* = 7/2
$$
(d) In one sentence, explain in plain English what $w^*$ represents in terms of the data point $(x, y) = (2, 7)$.

w* is the lowest value that the parameter w can have and it seems that at this data point (2,7) the derivative is increasing suggesting that to minimize the loss, the gradient descent should move to the opposite direction.
### Exercise 2.6 — three-level chain rule

Compute the derivative of $f(x) = ((x^2 + 1)^3 + 5)^2$.
$$
f'(x) = 2((x^2 + 1)^3 + 5) * (3(x^2 + 1)^2 * 2x)^2 * 2x = ((2x^2 + 2) + 10) * 36x^2(x^2 + 1) * 2x
$$
_Strategy:_ write it as a three-level composition. Outer: $u^2$. Middle: $v^3 + 5$. Inner: $x^2 + 1$. Apply the chain rule with three factors.

---

## 11. Self-check questions

Answer each in one or two sentences in your own words.

1. State the chain rule in plain English without using any mathematical notation.
	1. The chain rule dictate how to compute the derivative of a function inside a function which is important to know as layer of DL model are all inside each other. To learn the model has to compute the revitaive of the loss so the gradient descent can update the weights accoridngly to learn. 
	
2. In the expression $(2x + 5)^7$, what is the inner function and what is the outer function? Why does it matter to identify them in this order?
	1. The outer function is $u^7$ while the inner function is $2x+5$. It is important to knwo this to compute the derivative correctly. 
    
3. Backpropagation in a neural network applies the chain rule once per layer. If a network has 50 layers, how many chain-rule factors will appear in the derivative of the loss with respect to the first layer's weights? Roughly, why might it be a problem if all those factors are smaller than 1?
	1. If the model has 50 layers, then the chain rule will have 50 factors. 
    
4. The Leibniz form $\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$ looks like fractions cancelling. In one sentence, what is misleading about that mental picture, and why is it still useful?
	1. This notation fasten the calculation ad dy/du is the derivative of the outer function while du/dx is the derivative of the inner function. Multiply both and replace u by the inner function. 
	
5. The product rule has the structure "derivative of A times B, plus A times derivative of B." Why is the answer NOT simply "derivative of A times derivative of B"? Use a specific example to show the wrong rule fails.
	1. I don't know
    

---

## 12. Reflection (required)

Write 3-5 sentences:

- Where on a 1-5 scale do you sit on the chain rule today? Be honest.
	- 3 for simple chain rule calculation
- What was the most confusing thing? (If nothing was confusing, push yourself: redo Exercise 2.6 from scratch without looking, and report on that.)
	- Caluclation of chain rule of 3 factors and more would be too complicated
- In your own words, what is the link between the chain rule and training neural networks?
	- This is where my biggest learning is. I finally understood a little bit more how to train neural netwerk by computing the derivative of the loss function accross all layers to then adjust the wieght and start a new round of training. Plus I now knwo where the activation function lies in all of this and how it helps learning (by introducing non linearity). 