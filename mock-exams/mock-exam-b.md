# Machine Learning 1 - Mock Exam B

**Duration:** 2 hours  
**Total:** 100 points

## Instructions

- Show all important intermediate steps. Answers without justification can receive reduced credit.
- State any additional assumptions that you use.
- Give matrix and vector dimensions where the question asks for them.
- This course uses the numerator-layout convention: if `f: R^m -> R^n`, then its Jacobian has shape `n x m`.
- Unless a question states otherwise, vectors are column vectors.
- The suggested working times total 110 minutes. Reserve about 10 minutes for checking your answers.

---

## Exercise 1: Normalization and a nonsmooth loss (20 points, 20 minutes)

### 1.1 Vector normalization

Define `h: R^d \ {0} -> R^d` by

```text
h(x) = x / ||x||_2.
```

1. Derive `partial h_i / partial x_j` using index notation. **(5 points)**
2. Write the full Jacobian `J_h(x)` in matrix notation and state its dimensions. **(4 points)**
3. Show that `J_h(x) x = 0`. Explain why this agrees with the behavior of `h(c x)` for `c > 0`. **(3 points)**

### 1.2 Absolute-error regression

Let `a, w in R^p`, with `a != 0`, let `y in R`, and let `lambda > 0`. Define

```text
f(w) = |a^T w - y| + (lambda/2) w^T w.
```

4. When `a^T w - y != 0`, derive the `1 x p` derivative of `f(w)` using the course's numerator-layout convention. **(4 points)**
5. State why the derivative does not exist when `a^T w-y = 0`. Give one valid subgradient at such a point, and write one subgradient-descent update with step size `eta > 0`. **(4 points)**

---

## Exercise 2: A transformed waiting-time model (24 points, 25 minutes)

Let `X` have density

```text
p(x | lambda) = r lambda x^(r-1) exp(-lambda x^r),    x > 0,
p(x | lambda) = 0,                                    x <= 0,
```

where `r > 0` is known and `lambda > 0` is unknown.

1. Show that this is a valid probability density. **(3 points)**
2. Define `Y = X^r`. Derive the density of `Y`, identify its distribution, and hence find `E[X^r | lambda]`. **(4 points)**
3. Derive `P(X > c | lambda)` for `c >= 0`. Evaluate it for `r = 2`, `lambda = 0.25`, and `c = 2`. **(3 points)**

Now let `X_1, ..., X_N` be independent observations and define `S = sum_{n=1}^N X_n^r`. Assume `S > 0`.

4. Derive the maximum-likelihood estimator of `lambda`. Verify that it maximizes the likelihood. **(4 points)**

Use a Gamma prior with shape `a` and rate `b`:

```text
p(lambda) proportional to lambda^(a-1) exp(-b lambda),
```

where `a > 0` and `b > 0`.

5. Derive the posterior distribution and identify its parameters. **(4 points)**
6. Derive the MAP estimator. **(3 points)**
7. Give the posterior mean and compare it with the MAP estimator. What happens to their difference as `N` increases while `S/N` remains fixed and positive? **(3 points)**

---

## Exercise 3: Bias, variance, and shrinkage (15 points, 16 minutes)

Let `X_1, ..., X_N` be independent random variables with

```text
E[X_n] = mu,    Var(X_n) = sigma^2,
```

where the known value `sigma^2` is strictly positive. Let `m_0` be a fixed reference value and define, for `0 <= c <= 1`,

```text
mu_tilde_c = c X_bar + (1-c) m_0.
```

1. Derive the bias of `mu_tilde_c` as an estimator of `mu`. State when it is unbiased. **(3 points)**
2. Derive its variance. **(3 points)**
3. Derive its mean squared error. You may use `MSE = variance + bias^2`. **(3 points)**
4. For fixed `mu`, find the value of `c` in `[0,1]` that minimizes the mean squared error. **(4 points)**
5. Explain why the minimizing value from part 4 is called an oracle choice. Describe its limiting behavior as `N -> infinity`, distinguishing `mu = m_0` from `mu != m_0`. **(2 points)**

---

## Exercise 4: Multi-output MAP regression (22 points, 24 minutes)

Let `T in R^(N x k)`, `Phi in R^(N x m)`, and `W in R^(m x k)`. Consider the likelihood

```text
p(T | W) proportional to exp(-(beta/2) ||T - Phi W||_F^2),
```

where `beta > 0`. Let `M_0 in R^(m x k)` and let `R in R^(m x m)` be symmetric positive definite. Use the prior

```text
p(W) proportional to
    exp(-(alpha/2) tr((W-M_0)^T R (W-M_0))),
```

where `alpha > 0`.

1. Write the negative log-posterior, up to additive constants independent of `W`. **(4 points)**
2. Derive the MAP estimator of `W`. Define `lambda = alpha/beta` and give your final answer in terms of `lambda`. **(7 points)**
3. Show that the MAP estimator is unique, even if `Phi` does not have full column rank. **(3 points)**
4. Describe the limits of `W_MAP` as `lambda -> 0` and `lambda -> infinity`. State any rank assumption needed for the first limit. **(4 points)**
5. As `lambda` increases, describe the changes in `||T-Phi W_MAP||_F^2` and in `tr((W_MAP-M_0)^T R (W_MAP-M_0))`. Under the usual correctly specified regression assumptions, explain the general effect of stronger shrinkage on variance and bias, including the special case in which `M_0` equals the true coefficient matrix. **(4 points)**

---

## Exercise 5: Two sequential Bayesian updates (19 points, 25 minutes)

The current posterior in a Bayesian linear regression model is

```text
p(w | D_N) = N(w | mu_N, Sigma_N).
```

Assume that `Sigma_N` is symmetric positive definite.

Two new observations are

```text
(phi_a, t_a, beta_a) and (phi_b, t_b, beta_b),
```

where each `phi` is in `R^m`, each target is scalar, and each positive `beta` is the precision of that observation's Gaussian noise.

Define the current natural parameter

```text
eta_N = Sigma_N^(-1) mu_N.
```

1. Derive the posterior precision after incorporating both observations. **(5 points)**
2. Derive the final natural parameter and use it to give the final posterior mean. **(4 points)**
3. Use your answers to show that the final posterior does not depend on the order in which the two observations are processed. State the probabilistic assumption that makes this result valid. **(4 points)**
4. Suppose `phi_b = phi_a`. Show that the two observations have the same effect on the posterior as one observation with precision `beta_a + beta_b`. Derive the equivalent target value. **(4 points)**
5. If `phi_b = c phi_a` for a nonzero scalar `c`, does the second observation provide information in a new parameter-space direction? Explain briefly. **(2 points)**

---

**End of Mock Exam B**
