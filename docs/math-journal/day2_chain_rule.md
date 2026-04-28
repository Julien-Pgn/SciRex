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
f'(x) = -2(2x + 3)^-3 * 2 = -4(2x + 3)^-3 = \dfrac{-4}{(2x + 3)^3}
\end{aligned}
$$

### Exercise 2.3 — product rule

Compute the derivative.

(a) $h(x) = x \cdot (x^2 + 1)$ 
$$
h'(x) = (x^2+1) + 2x^2 = 3x^2 + 1
$$

(b) $h(x) = (2x + 1)(3x - 4)$
$$
h'(x) = 2(3x - 4) + 3(2x + 1) = (6x -8) + (6x + 3) = 12x -5
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
L'(w) = 8w - 28
\end{aligned}
$$
- **Way 2:** Apply the chain rule directly without expanding (inner = $wx - y$, outer = $u^2$). Verify the two methods give the same answer.
$$
L'(w) = 2(2w-7) * 2 = 8w - 28
$$
(c) Solve $L'(w) = 0$ for $w^*$.
$$
w^* = 7/2
$$
(d) In one sentence, explain in plain English what $w^*$ represents in terms of the data point $(x, y) = (2, 7)$.

w* is the value that minimizes the loss. It is also the slope of the rangent at lowest value that the parameter w can have and it seems that at this data point (2,7) the derivative is increasing suggesting that to minimize the loss, the gradient descent should move to the opposite direction.
### Exercise 2.6 — three-level chain rule

Compute the derivative of $f(x) = ((x^2 + 1)^3 + 5)^2$.
inner: $h(x) = x^2 + 1$, $h'(x) = 2x$
middle: $g(v) = v^3 + 5$, $g'(v) = 3v^2$
outer: $f(u) = u^2$, $f'(u) = 2u$

So $f'(x) = f'(g(h(x))).g'(h(x)).h'(x)$
$$
f'(x) = 2((x^2 + 1)^3 + 5) * 3(x^2 + 1)^2 * 2x = 12x((x^2 + 1)^3 +5) * (x^2 + 1)
$$
_Strategy:_ write it as a three-level composition. Outer: $u^2$. Middle: $v^3 + 5$. Inner: $x^2 + 1$. Apply the chain rule with three factors.

Let's redo it with the Leibniz style:

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dv}\cdot \frac{dv}{dx} =  2u . 3v^2 . 2x = 12x((x^2 + 1)^3 + 5) * (x^2 + 1)^2$$

---

## 11. Self-check questions

Answer each in one or two sentences in your own words.

1. State the chain rule in plain English without using any mathematical notation.
	1. To compute the derivative of a function inside a function, take the derivative of the outer function, treat the inner function as a variable and multiply by the derivative of the inner function. 
	
2. In the expression $(2x + 5)^7$, what is the inner function and what is the outer function? Why does it matter to identify them in this order?
	1. The outer function is $u^7$ while the inner function is $2x+5$. It is important to know this to compute the derivative correctly. 
    
3. Backpropagation in a neural network applies the chain rule once per layer. If a network has 50 layers, how many chain-rule factors will appear in the derivative of the loss with respect to the first layer's weights? Roughly, why might it be a problem if all those factors are smaller than 1?
	1. If the model has 50 layers, then the chain rule will have 50 factors. 
	2. Correction: Let's assume that the factors are 0.5 in 50 layers it will be 0.5^50 ≈ 10^-15 ≈ 0. This means that the derivative of the loss after passing all layers is close to 0 thus, teh early layers receive no gradient signal during training, so they don't learn. **That is called the vanishing gradient problem.** Which is why deep networks were not trainable before the introduction of ResNets with residual connections to keep gradients from collapsing though many layers and chains.Conversely, factors greater that 1 results in the  **exploding gradient**
    
4. The Leibniz form $\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$ looks like fractions cancelling. In one sentence, what is misleading about that mental picture, and why is it still useful?
	1. These are not numbers nor fraction, they are symbols representing infinitesimal limits. They bring clarity 
	
5. The product rule has the structure "derivative of A times B, plus A times derivative of B." Why is the answer NOT simply "derivative of A times derivative of B"? Use a specific example to show the wrong rule fails.
	1. I don't know
	2. Correction: That fails because when a product changes, both factors contribute changes simultaneously, not independently. 
Let f(x)=x2f(x) = x^2 f(x)=x2 and g(x)=x3g(x) = x^3 g(x)=x3. Then h(x)=x2⋅x3=x5h(x) = x^2 \cdot x^3 = x^5 h(x)=x2⋅x3=x5.

- *Wrong rule:* $h′(x)=f′(x)⋅g′(x)=2x⋅3x2=6x3h'(x) = f'(x) \cdot g'(x) = 2x \cdot 3x^2 = 6x^3 h′(x)=f′(x)⋅g′(x)=2x⋅3x2=6x3$
- *Correct product rule:* $h′(x)=f′(x)g(x)+f(x)g′(x)=2x⋅x3+x2⋅3x2=2x4+3x4=5x4h'(x) = f'(x) g(x) + f(x) g'(x) = 2x \cdot x^3 + x^2 \cdot 3x^2 = 2x^4 + 3x^4 = 5x^4 h′(x)=f′(x)g(x)+f(x)g′(x)=2x⋅x3+x2⋅3x2=2x4+3x4=5x4$
The wrong rule gave 6x36x^3 6x3, the correct rule gave 5x45x^4 5x4 — these aren't even the same kind of thing.
---

## 12. Reflection (required)

Write 3-5 sentences:

- Where on a 1-5 scale do you sit on the chain rule today? Be honest.
	- 3 for simple chain rule calculation
- What was the most confusing thing? (If nothing was confusing, push yourself: redo Exercise 2.6 from scratch without looking, and report on that.)
	- Calculation of chain rule of 3 factors and more would be too complicated. I should use the Leibniz notation to simplify the calculation. 
- In your own words, what is the link between the chain rule and training neural networks?
	- This is where my biggest learning is. I finally understood a little bit more how to train neural netwerk by computing the derivative of the loss function accross all layers to then adjust the wieght and start a new round of training. Plus I now knwo where the activation function lies in all of this and how it helps learning (by introducing non linearity). 