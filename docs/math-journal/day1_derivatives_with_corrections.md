## 7. Your exercises

Do these by hand on paper or directly in this file under each prompt. Show every step — the grader (me) wants to see your thinking, not just the final answer.
### Exercise 1.1 — power rule warm-up

Compute the derivative of each:

(a) $f(x) = x^4$
$$4x^3$$
(b) $f(x) = x^7$
$$f'(x) = 7x^6$$

(c) $f(x) = x^{-1}$
$$f'(x) = -x^-2$$
### Exercise 1.2 — linearity

Compute the derivative of each, showing the intermediate steps:

(a) $f(x) = 3x^2 + 2x - 7$
$$f'(x) = 6x + 2$$

(b) $f(x) = \tfrac{1}{2} x^2 - 5x + 100$
$$f'(x) = x - 5$$

(c) $f(x) = (2x + 1)^2$ 
$$f'(x) = 4x^2 + 4x + 1$$
$$f'(x) = 8x + 4$$

### Exercise 1.3 — find a minimum

Consider $f(x) = x^2 - 6x + 10$.

(a) Compute $f'(x)$.
$$f'(x) = 2x - 6$$
(b) Solve $f'(x) = 0$ for $x$. Call this value $x^*$.
$$2x - 6 = 0$$
$$x =3$$

(c) What is $f(x^*)$? Is it a minimum or maximum? Justify your answer in one sentence.
$$f(3) = 9 - 18 + 10 = 1$$

It is a minimum as both f(2) = 2 and f(4) = 2 are higher tha, f(3) = 1.
### Exercise 1.4 — ML connection

Pretend I have a one-parameter loss function $L(w) = (w - 5)^2$.

(a) Compute $L'(w)$ (remember to expand the square first).
$$
\begin{aligned}
L(w) &= w^2 - 10w + 25 \\
L'(w) &= 2w - 10
\end{aligned}
$$

(b) Solve $L'(w) = 0$. This gives you $w^*$, the optimal weight.
$$w^* = 5$$

(c) In one sentence, explain in plain English what this optimization problem is doing. 

This optimization problem measures how far off our parameter w is from the truth (5). All of this in the power of 2, so the result is always positive and it penalizes values that are far from each other. 

---

## 8. Self-check questions

Answer each in one or two sentences, in your own words. No peeking at the lecture above.

1. If I tell you $f'(2) = -0.5$, what does that mean about the function $f$ near $x = 2$? In which direction should I move $x$ to make $f$ smaller?

Near x = 2 the function is decreasing thus you should move to higher values to get lower. 

1. Why is the derivative of a constant function always zero? Explain using the geometric (slope) intuition.

Because the function of a constant is a line, and thus flat with a slope of zero (no rate fo change). 

1. In machine learning we _minimize_ a loss function rather than maximize it. What would happen, conceptually, if we maximized the loss instead? (One sentence.)

Maximizing the loss will modify the weights towards the worst possible predictions. Thus the model would learn the exact opposite of what you want it to learn and predict. 

1. The power rule says $\frac{d}{dx} x^n = n x^{n-1}$. What does the rule say about $\frac{d}{dx} x^0$? Does this match your expectation given that $x^0 = 1$?

I don't know. Correction: Usually you can solve these types of questions by trying simple cases with n=0 or n=1 and abstract things. Let's try it with n=0 : 
$$0 * x^-1 = 0$$
$$1 * x^0 = 1$$

2. In your own words, what is the relationship between the derivative of a loss function and the gradient descent algorithm?

At each epochs, the loss is computed. Gradient descent computes the derivatives of the loss and then modifies the weights and recomputes the derivative of the loss. If the difference is decreasing, then we knwo we have moved to the right direction (to a local miminum). 

Gradient descent computes the derivative of the loss (calculated at each epoch) to modify the weights for the next epoch. The weights will be multiplied by a value of the opposite sign of the derivative (to reduce it). However, the magnitde of this value will be proportional to the magnitude of the derivative (a small step on a big slope modifies a lot while a small step on a sight slope doesn't move you a lot). 

---
## 9. Reflection (required)

Write 3-5 sentences addressing:

- What was the single most confusing thing today?
	- The calculation of difficult derivatives as I don't remember all the nuances and all the rules. 
- What finally clicked?
	- The definition of the loss function and its link to gradient descent and the role of derivatives in DL. 
- How would you explain "derivative" to a friend, in one sentence, without using the word "slope"?
	- It is the prediction of the output of the function when you change the input. Tells you where you are heading in the future if you move like this or that. 
- Where does today's material sit on a scale from 1 (completely lost) to 5 (I could teach this)?
	- 3, it took me an hour for basic maths, but everyone starts somewhere before climbing high. Let's continue!
