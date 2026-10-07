# Machine Learning 1 - Mock Exam A

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

## Exercise 1: Multivariate derivatives and regularization (20 points, 18 minutes)

### 1.1 Temperature-scaled log-softmax

Let `x in R^d` and let `tau > 0`. Define `s: R^d -> R^d` and `ell: R^d -> R^d` by

```text
s_i(x) = exp(x_i / tau) / sum_{k=1}^d exp(x_k / tau).
ell_i(x) = log(s_i(x)).
```

1. Derive `partial ell_i / partial x_j` using index notation. Express the answer using `s_j`, `tau`, and the Kronecker delta `delta_ij`. **(5 points)**
2. Write the full Jacobian `J_ell(x)` in matrix notation and state its dimensions. **(4 points)**
3. Show that `J_ell(x) 1 = 0`, where `1` is the all-ones vector. Explain this result using a transformation of `x` that leaves `ell(x)` unchanged. **(4 points)**

### 1.2 Generalized ridge regression

Let `X in R^(n x p)`, `y in R^n`, `theta in R^p`, `gamma > 0`, and let `R in R^(p x p)` be symmetric positive definite. Consider

```text
L(theta) = (1/n) ||X theta - y||_2^2 + gamma theta^T R theta.
```

4. Derive the stationary point of `L(theta)`. Then show that it is the unique global minimizer, even when `X` does not have full column rank. **(7 points)**

---

## Exercise 2: Waiting times and Bayesian estimation (24 points, 25 minutes)

A monitoring system checks a component repeatedly. Each check succeeds independently with probability `q`, where `0 < q < 1`. Let `K` be the number of failed checks before the first success. Its probability mass function is

```text
p(K = k | q) = q (1-q)^k,    k = 0, 1, 2, ...
```

You may use

```text
sum_{k=0}^infinity r^k = 1/(1-r),    |r| < 1.
```

1. Derive `E[K | q]`. Do not only state the result. **(3 points)**
2. Compute `P(K = 2 | q = 0.25)`. Give an exact value or decimal value. **(3 points)**

Now suppose that `K_1, ..., K_N` are independent observations and define `S = sum_{n=1}^N K_n`. Assume `S > 0`.

3. Write the likelihood and log-likelihood of `q`, up to constants that do not depend on `q`. **(3 points)**
4. Derive the maximum-likelihood estimator of `q`. Verify that your stationary point is a maximum. **(4 points)**

Place a `Beta(a,b)` prior on `q`, with density

```text
p(q) proportional to q^(a-1) (1-q)^(b-1),    0 < q < 1,
```

where `a > 1` and `b > 1`.

5. Derive the posterior distribution of `q` and identify its parameters. **(4 points)**
6. Derive the MAP estimator of `q`. **(4 points)**
7. Identify a one-dimensional statistic of the data that contains all data dependence in the posterior. Briefly justify your answer. **(3 points)**

---

## Exercise 3: Estimating an unknown range (14 points, 14 minutes)

Measurements `X_1, ..., X_N` are independent and uniformly distributed on `[-M, M]`, where `M > 0` is unknown. Define

```text
A = max_{1 <= n <= N} |X_n|.
```

Assume that at least one observed value is nonzero, so `A > 0`.

1. Write the likelihood `p(x_1, ..., x_N | M)`, including the support constraint. **(4 points)**
2. Derive the maximum-likelihood estimator of `M`. For the observations

```text
-3.2, 1.1, 2.4, -2.8, 0.5,
```

give its numerical value. **(3 points)**

You may use

```text
E[A]   = N M/(N+1),
E[A^2] = N M^2/(N+2).
```

3. Derive a multiplicative correction of the maximum-likelihood estimator that is unbiased for `M`. **(3 points)**
4. Derive the variance of your unbiased estimator and simplify the result. **(4 points)**

---

## Exercise 4: Multi-output regression with correlated noise (23 points, 29 minutes)

For each observation, let `phi_n in R^m` be a feature vector and `t_n in R^k` be a target vector. Let `W in R^(m x k)`. The model is

```text
t_n = W^T phi_n + epsilon_n,
epsilon_n ~ N(0, sigma^2 C),
```

where `C in R^(k x k)` is known and symmetric positive definite, while `sigma^2 > 0` is unknown. The noise vectors are independent.

Let `Phi in R^(N x m)` have rows `phi_n^T`, and let `T in R^(N x k)` have rows `t_n^T`. Assume that `Phi` has full column rank, `N > m`, and the fitted residual is nonzero.

1. State the dimensions of `Phi W`, `T - Phi W`, and `(T-Phi W)^T(T-Phi W)`. **(2 points)**
2. Derive the log-likelihood of `W` and `sigma^2`, including all terms that depend on either parameter. You may express the quadratic term using a trace. **(6 points)**
3. Derive the maximum-likelihood estimator of `W`. Show why it does not depend on `C` or `sigma^2`. **(7 points)**
4. Using the fitted value of `W`, derive the maximum-likelihood estimator of `sigma^2`. **(5 points)**
5. Let `E = T - Phi W_ML`. Show that `Phi^T E = 0` and explain the geometric meaning of this identity. **(3 points)**

---

## Exercise 5: Information from one Bayesian observation (19 points, 24 minutes)

After observing a data set `D_N`, the posterior over the regression weights is

```text
p(w | D_N) = N(w | mu_N, Sigma_N),
```

where `w, mu_N in R^m` and `Sigma_N in R^(m x m)` is symmetric positive definite. A new observation has feature vector `phi in R^m`, target `t in R`, and likelihood

```text
p(t | w, phi) = N(t | w^T phi, beta^(-1)),
```

where `beta > 0` is the observation-noise precision.

The standard precision and natural-parameter updates are

```text
Sigma_(N+1)^(-1) = Sigma_N^(-1) + beta phi phi^T,
Sigma_(N+1)^(-1) mu_(N+1) = Sigma_N^(-1) mu_N + beta phi t.
```

You may use the Sherman-Morrison identity

```text
(A + u v^T)^(-1)
    = A^(-1) - A^(-1) u v^T A^(-1)/(1 + v^T A^(-1) u).
```

1. Derive the rank-one covariance update

```text
Sigma_(N+1) = Sigma_N
    - Sigma_N phi phi^T Sigma_N/(beta^(-1) + phi^T Sigma_N phi).
```

**(4 points)**

2. Derive the innovation form of the mean update

```text
mu_(N+1) = mu_N
    + K (t - phi^T mu_N),
```

and give the gain vector `K`. Explain the roles of the direction and the scalar residual in this update. **(5 points)**

3. Before observing `t`, derive the posterior predictive distribution `p(t | phi, D_N)`. Give both its mean and variance, and identify the two sources of predictive variance. **(4 points)**

4. For any direction `a in R^m`, derive

```text
a^T (Sigma_N - Sigma_(N+1)) a.
```

Show that it is nonnegative. State exactly when it is zero and explain what this says about the parameter directions learned from the observation. **(6 points)**

---

**End of Mock Exam A**
