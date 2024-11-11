**Derivation of Simple Logistic Regression**

---

**Introduction**

Logistic regression is a statistical method used for modeling a binary outcome variable (dependent variable $Y$ that can take on two possible outcomes, typically coded as 0 or 1) based on one or more predictor variables (independent variables $X$). In simple logistic regression, we have one predictor variable. The goal is to model the probability that $Y$ equals 1 given $X$.

---

**The Logistic Function**

To model the probability, we use the logistic (sigmoid) function, which maps any real-valued number into a value between 0 and 1:

$P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X)}}$

where:

- $P(Y=1|X)$ is the probability that $Y = 1$ given $X$.
- $\beta_0$ is the intercept term.
- $\beta_1$ is the coefficient for the predictor variable $X$.
- $e$ is the base of the natural logarithm.

This function ensures that the predicted probabilities are between 0 and 1.

---

**The Logit Function**

By rearranging the logistic function, we can express it in terms of the log-odds (logit):

$\text{Logit}(P(Y=1|X)) = \ln\left( \frac{P(Y=1|X)}{1 - P(Y=1|X)} \right) = \beta_0 + \beta_1 X$

This linear relationship between the log-odds and the predictor variable simplifies the analysis and estimation of the parameters.

---

**Likelihood Function**

Given $n$ independent observations $(X_i, Y_i)$, where $Y_i \in \{0,1\}$, the likelihood function represents the probability of observing the data as a function of the parameters $\beta_0$ and $\beta_1$:

$L(\beta_0, \beta_1) = \prod_{i=1}^n P(Y_i | X_i)$

Since $Y_i$ is binary, we can express $P(Y_i | X_i)$ as:

$P(Y_i | X_i) = [P(Y_i=1|X_i)]^{Y_i} [1 - P(Y_i=1|X_i)]^{1 - Y_i}$

Substituting the logistic function into the likelihood:

$L(\beta_0, \beta_1) = \prod_{i=1}^n \left( \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_i)}} \right)^{Y_i} \left( \frac{e^{-(\beta_0 + \beta_1 X_i)}}{1 + e^{-(\beta_0 + \beta_1 X_i)}} \right)^{1 - Y_i}$

Simplifying further:

$L(\beta_0, \beta_1) = \prod_{i=1}^n \left( \frac{e^{\beta_0 + \beta_1 X_i}}{1 + e^{\beta_0 + \beta_1 X_i}} \right)^{Y_i} \left( \frac{1}{1 + e^{\beta_0 + \beta_1 X_i}} \right)^{1 - Y_i}$

---

**Log-Likelihood Function**

To simplify calculations, we take the natural logarithm of the likelihood function to obtain the log-likelihood function:

$\ell(\beta_0, \beta_1) = \ln L(\beta_0, \beta_1) = \sum_{i=1}^n \left[ Y_i (\beta_0 + \beta_1 X_i) - \ln(1 + e^{\beta_0 + \beta_1 X_i}) \right]$

This form is derived by applying logarithm properties and simplifies the product into a sum, which is easier to differentiate.

---

**Maximum Likelihood Estimation (MLE)**

To estimate $\beta_0$ and $\beta_1$, we maximize the log-likelihood function. This involves taking partial derivatives of $\ell(\beta_0, \beta_1)$ with respect to $\beta_0$ and $\beta_1$, setting them to zero, and solving for the parameters.

**Partial Derivatives**

1. **Derivative with respect to $\beta_0$:**

$\frac{\partial \ell}{\partial \beta_0} = \sum_{i=1}^n \left[ Y_i - \frac{e^{\beta_0 + \beta_1 X_i}}{1 + e^{\beta_0 + \beta_1 X_i}} \right]$

2. **Derivative with respect to $\beta_1$:**

$\frac{\partial \ell}{\partial \beta_1} = \sum_{i=1}^n \left[ Y_i - \frac{e^{\beta_0 + \beta_1 X_i}}{1 + e^{\beta_0 + \beta_1 X_i}} \right] X_i$

**Simplifying the Derivatives**

Define $\hat{P_i} = \frac{e^{\beta_0 + \beta_1 X_i}}{1 + e^{\beta_0 + \beta_1 X_i}} = P(Y_i = 1 | X_i)$.

The partial derivatives become:

1. **For $\beta_0$:**

$\frac{\partial \ell}{\partial \beta_0} = \sum_{i=1}^n (Y_i - \hat{P_i})$

2. **For $\beta_1$:**

$\frac{\partial \ell}{\partial \beta_1} = \sum_{i=1}^n (Y_i - \hat{P_i}) X_i$

---

**Setting the Derivatives to Zero**

To find the maximum of the log-likelihood function, set the partial derivatives to zero:

1. **For $\beta_0$:**

$\sum_{i=1}^n (Y_i - \hat{P_i}) = 0$

2. **For $\beta_1$:**

$\sum_{i=1}^n (Y_i - \hat{P_i}) X_i = 0$

These equations are known as the **score equations**.

---

**Numerical Methods for Solving**

The score equations are nonlinear and typically cannot be solved analytically. Therefore, we use iterative numerical methods such as:

- **Newton-Raphson Method**
- **Fisher Scoring Method**
- **Gradient Ascent**

These methods iteratively update the parameter estimates until convergence is achieved.

---

**Gradient Ascent Algorithm**

1. **Initialize** $\beta_0^{(0)}$ and $\beta_1^{(0)}$.

2. **Iterative Update:**

   - For each iteration $t$:

     - Compute $\hat{P_i}^{(t)} = \frac{1}{1 + e^{-(\beta_0^{(t)} + \beta_1^{(t)} X_i)}}$ for all $i$.

     - Update parameters:

       $\beta_0^{(t+1)} = \beta_0^{(t)} + \alpha \sum_{i=1}^n (Y_i - \hat{P_i}^{(t)})$

       $\beta_1^{(t+1)} = \beta_1^{(t)} + \alpha \sum_{i=1}^n (Y_i - \hat{P_i}^{(t)}) X_i$

     - $\alpha$ is the learning rate.

3. **Convergence Check:**

   - Stop the iterations when the change in the log-likelihood or parameters is below a predefined threshold.

---

**Alternative: Minimizing the Negative Log-Likelihood**

In machine learning, logistic regression is often implemented by minimizing the negative log-likelihood (also known as the cross-entropy loss):

$J(\beta_0, \beta_1) = -\ell(\beta_0, \beta_1) = -\sum_{i=1}^n \left[ Y_i (\beta_0 + \beta_1 X_i) - \ln(1 + e^{\beta_0 + \beta_1 X_i}) \right]$

The gradients of $J(\beta_0, \beta_1)$ are:

1. **For $\beta_0$:**

$\frac{\partial J}{\partial \beta_0} = -\frac{\partial \ell}{\partial \beta_0} = -\sum_{i=1}^n (Y_i - \hat{P_i})$

2. **For $\beta_1$:**

$\frac{\partial J}{\partial \beta_1} = -\frac{\partial \ell}{\partial \beta_1} = -\sum_{i=1}^n (Y_i - \hat{P_i}) X_i$

**Gradient Descent Algorithm**

1. **Initialize** $\beta_0^{(0)}$ and $\beta_1^{(0)}$.

2. **Iterative Update:**

   - For each iteration $t$:

     - Compute $\hat{P_i}^{(t)}$ as before.

     - Update parameters:

       $\beta_0^{(t+1)} = \beta_0^{(t)} - \alpha \left( -\sum_{i=1}^n (Y_i - \hat{P_i}^{(t)}) \right)$

       $\beta_1^{(t+1)} = \beta_1^{(t)} - \alpha \left( -\sum_{i=1}^n (Y_i - \hat{P_i}^{(t)}) X_i \right)$

     - Simplify:

       $\beta_0^{(t+1)} = \beta_0^{(t)} + \alpha \sum_{i=1}^n (Y_i - \hat{P_i}^{(t)})$

       $\beta_1^{(t+1)} = \beta_1^{(t)} + \alpha \sum_{i=1}^n (Y_i - \hat{P_i}^{(t)}) X_i$

3. **Convergence Check:**

   - Similar to gradient ascent.

---

**Summary**

- **Modeling**: The logistic regression model relates the log-odds of the probability $P(Y=1|X)$ to the predictor $X$ linearly.
  
- **Estimation**: Parameters $\beta_0$ and $\beta_1$ are estimated using maximum likelihood estimation by maximizing the log-likelihood function.

- **Optimization**: Due to the nonlinearity, iterative numerical methods are employed to find the parameter estimates.

- **Implementation**: In practice, algorithms like gradient ascent (for maximizing the log-likelihood) or gradient descent (for minimizing the negative log-likelihood) are used.

---

**Conclusion**

The derivation of simple logistic regression involves modeling a binary outcome using the logistic function, formulating the likelihood and log-likelihood functions, computing the gradients, and applying numerical optimization methods to estimate the model parameters. This process allows us to understand the relationship between the predictor variable and the probability of the outcome, providing valuable insights in various fields such as medicine, finance, and social sciences.

---


To derive simple logistic regression, follow these steps:

### 1. Define the Logistic Function
The logistic regression model is built upon the logistic function:
$p(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x)}}$
where $p(y=1|x)$ represents the probability that $y = 1$ given $x$, and $\beta_0$ and $\beta_1$ are parameters.

### 2. Log-Odds Transformation
For logistic regression, we model the log-odds (logit) as a linear function:
$\log\left(\frac{p}{1 - p}\right) = \beta_0 + \beta_1 x$
where $p$ is shorthand for $p(y=1|x)$.

### 3. Likelihood Function
For $n$ observations, we can write the likelihood function as:
$L(\beta_0, \beta_1) = \prod_{i=1}^{n} p_i^{y_i} (1 - p_i)^{1 - y_i}$
where $p_i = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_i)}}$.

### 4. Log-Likelihood
Taking the logarithm of the likelihood function gives us the log-likelihood:
$\ell(\beta_0, \beta_1) = \sum_{i=1}^{n} \left( y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right)$

### 5. Derivatives with Respect to Parameters
To maximize the log-likelihood, we find derivatives with respect to $\beta_0$ and $\beta_1$:
$\frac{\partial \ell}{\partial \beta_j} = \sum_{i=1}^{n} \left( y_i - p_i \right) x_{ij}$
where $x_{ij}$ is 1 for $\beta_0$ and $x_i$ for $\beta_1$.

### 6. Solving for Parameters
Set $\frac{\partial \ell}{\partial \beta_j} = 0$ for each $j$ and use numerical methods (like gradient ascent) to solve for $\beta_0$ and $\beta_1$.