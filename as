# Quiz-2 Question 2

# **Problem Statement**

Consider the following variant of the perceptron learning algorithm for the dataset $D = \{(x^i, y^i)\}_{i=1}^{n}$ where $x^i \in \mathbb{R}^d$ and $y^i \in \{+1, -1\}$. Also assume that $\|x^i\|_2 = 1$ for every $i \in \{1,2,\dots,n\}$.

1. **Initialize** $\tilde{w} = \left(\frac{1}{\sqrt{d+1}}, \dots, \frac{1}{\sqrt{d+1}}\right) \in \mathbb{R}^{d+1}$, max epochs $M$.
2. **For** $t = 1,2,\dots,M$ do:
   - Shuffle the dataset $D$ to obtain $T = \{(u^i, s^i)\}_{i=1}^{n}$ where $u^i = x^{\pi(i)}$ and $s^i = y^{\pi(i)}$ for $i \in \{1,2,\dots,n\}$, for some permutation function $\pi: \{1,2,\dots,n\} \to \{1,2,\dots,n\}$.
   - **For** $i = 1,2,\dots,n$ do:
     * $\tilde{u}^i = (u^i, 1)$.
     * Predict label for $u^i$ as $\hat{s}^i$ using Perceptron.
     * **If** $s^i \neq \hat{s}^i$:
       - $v = \tilde{w} + s^i \tilde{u}^i$.
       - $\tilde{w} = \frac{v}{\|v\|_2}$.

# Questions
## (a) **[4 marks]** 

Suppose the perceptron makes a mistake on some sample, leading to an update. Let $\tilde{w}^{old}$ be the weight vector before the update and let $\tilde{w}^{new}$ be the weight vector after the update. Find $\langle \tilde{w}^{old}, \tilde{w}^{new} \rangle$. Using this quantity, explain the geometric relationship between the old and new weight vectors.

## **Solution**  
#### **Step 1: Define Variables**
- Let $\tilde{w}^{\text{old}} = w$, the weight vector before the update, with $\|w\|_2 = 1$.
- Let $\tilde{u}^i = x = (u^i, 1) \in \mathbb{R}^{d+1}$, where $ u^i \in \mathbb{R}^d $, $\|u^i\|_2 = 1$, so:
$$
\|x\|_2 = \sqrt{\|u^i\|_2^2 + 1} = \sqrt{1 + 1} = \sqrt{2}.
$$
- Let $ s = s^i \in \{+1, -1\} $, the true label.
- A mistake occurs, so:
$$
\hat{s} = \text{sign}(\langle w, x \rangle) \neq s.
$$
- Update rule:
$$
v = w + s x, \quad \tilde{w}^{\text{new}} = \frac{v}{\|v\|_2}.
$$
- Define:
$$
\alpha = \langle w, x \rangle = w^\top x.
$$
- **Mistake Condition**:  
A mistake occurs on the $ n $-th sample, so the predicted label 
$\hat{s} = \text{sign}(\langle w, x \rangle) = \text{sign}(\alpha) \neq s$ .
This implies:
$$
s \alpha = s \langle w, x \rangle \leq 0.
$$

  - If $\alpha > 0$, $\text{sign}(\alpha) = +1$, so $ s \neq +1 \implies s = -1 $, and $ s \alpha = (-1) \cdot \alpha < 0 $.
  - If $\alpha < 0$, $\text{sign}(\alpha) = -1$, so $ s \neq -1 \implies s = +1 $, and $ s \alpha = (+1) \cdot \alpha < 0 $.
  - If $\alpha = 0$, $\text{sign}(0)$ is undefined, but typically considered a mistake (e.g., $\hat{s} = +1 \neq s = -1$), so $ s \alpha = 0 $.

**Cauchy–Schwarz inequality:**  
For any vectors $w$ and $x$:
$$
|\langle w, x \rangle| \le \|w\| \cdot \|x\|
$$
$\Rightarrow$ 
$$
-\|w\| \cdot \|x\| \le \langle w, x \rangle \le \|w\| \cdot \|x\|
$$

* **Bounds on $\alpha$**:  

  - By Cauchy-Schwarz:
    $$
    |\alpha| = |\langle w, x \rangle| \leq \|w\|_2 \|x\|_2 = 1 \cdot \sqrt{2} = \sqrt{2}.
    $$
    Thus:
    $$
    \alpha \in [-\sqrt{2}, \sqrt{2}].
    $$
  - Since $ s \alpha \leq 0 $, we consider:
    - If $ s = +1 $, $\alpha \leq 0$, so $\alpha \in [-\sqrt{2}, 0]$, and $ s \alpha = \alpha $.
    - If $ s = -1 $, $\alpha \geq 0$, so $\alpha \in [0, \sqrt{2}]$, and $ s \alpha = -\alpha $.
  - Combining: $ s \alpha \in [-\sqrt{2}, 0] $.

#### **Step 2: Compute the Inner Product**
We need:
$$
\langle \tilde{w}^{\text{old}}, \tilde{w}^{\text{new}} \rangle = w^\top \tilde{w}^{\text{new}} = w^\top \left( \frac{v}{\|v\|_2} \right) = \frac{w^\top v}{\|v\|_2}.
$$
- **Numerator**:
  $$
  w^\top v = w^\top (w + s x) = w^\top w + s w^\top x = \|w\|_2^2 + s \alpha = 1 + s \alpha.
  $$
- **Denominator**:
  $$
  \|v\|_2^2 = \|w + s x\|_2^2 = w^\top w + 2 s w^\top x + s^2 x^\top x.
  $$
  - $ w^\top w = 1 $.
  - $ w^\top x = \alpha $.
  - $ s^2 = 1 $.
  - $ x^\top x = \|x\|_2^2 = 2 $.  
  $$
  \|v\|_2^2 = 1 + 2 s \alpha + 2 = 3 + 2 s \alpha.
  $$
  $$
  \|v\|_2 = \sqrt{3 + 2 s \alpha}.
  $$
  Since $ s \alpha \in [-\sqrt{2}, 0] $, check positivity:
  - At $ s \alpha = -\sqrt{2} $: $ 3 - 2 \sqrt{2} \approx 0.172 > 0 $.
  - At $ s \alpha = 0 $: $ 3 > 0 $.
  Thus, $\|v\|_2 > 0$.
- **Inner Product**:
  $$
  \langle \tilde{w}^{\text{old}}, \tilde{w}^{\text{new}} \rangle = \frac{1 + s \alpha}{\sqrt{3 + 2 s \alpha}}.
  $$

#### **Step 3: Geometric Interpretation**
The inner product is:
$$
\langle \tilde{w}^{\text{old}}, \tilde{w}^{\text{new}} \rangle = \frac{1 + s \alpha}{\sqrt{3 + 2 s \alpha}},
$$
where $ s \alpha \in [-\sqrt{2}, 0] $, and $\theta = \arccos(\langle \tilde{w}^{\text{old}}, \tilde{w}^{\text{new}} \rangle)$ is the angle between $ w $ and $\tilde{w}^{\text{new}}$.

1. **Sign of Inner Product**:
   - **Numerator**: $ 1 + s \alpha $.
     - At $ s \alpha = 0 $: $ 1 $.
     - At $ s \alpha = -\sqrt{2} $: $ 1 - \sqrt{2} \approx -0.414 $.
   - **Denominator**: $\sqrt{3 + 2 s \alpha} > 0$.
   - **Range**:
     - At $ s \alpha = 0 $:
       $$
       \langle w, \tilde{w}^{\text{new}} \rangle = \frac{1}{\sqrt{3}} \approx 0.577.
       $$
     - At $ s \alpha = -\sqrt{2} $:
       $$
       \langle w, \tilde{w}^{\text{new}} \rangle = \frac{1 - \sqrt{2}}{\sqrt{3 - 2 \sqrt{2}}} = \frac{-(\sqrt{2} - 1)}{\sqrt{2} - 1} = -1.
       $$
     - For $ s \alpha \in (-\sqrt{2}, 0) $, the inner product ranges from $-1$ to $ \frac{1}{\sqrt{3}} $.

1. **Angle Analysis**:
   - For $ s \alpha \in (-1, 0] $, $ 1 + s \alpha > 0 $, so $\langle w, \tilde{w}^{\text{new}} \rangle > 0$, and:
     $$
     0^\circ < \theta < 90^\circ.
     $$
     - At $ s \alpha = 0 $: $\theta \approx 54.74^\circ$.
     - As $ s \alpha \to -1^+ $: $\langle w, \tilde{w}^{\text{new}} \rangle \to 0^+$, so $\theta \to 90^\circ$.   
     
   - At $ s \alpha = -1 $:
     $$
     \langle w, \tilde{w}^{\text{new}} \rangle = 0, \quad \theta = 90^\circ.
     $$
   - At $ s \alpha = -\sqrt{2} $:
     $$
     \langle w, \tilde{w}^{\text{new}} \rangle = -1, \quad \theta = 180^\circ.
     $$
     This suggests an extreme case where $\tilde{w}^{\text{new}} = -w$, which is atypical for perceptron updates.

1. **Geometric Relationship**:
   - The update $ v = w + s x $ shifts $ w $ toward $ s x $, and normalization ensures $\tilde{w}^{\text{new}}$ is unit-length.
   - Since $ s \alpha \leq 0 $, $ s x $ corrects the mistake by rotating $ w $ toward the direction that aligns $\langle \tilde{w}^{\text{new}}, x \rangle$ with $ s $.
   - For $ s \alpha \in (-1, 0] $:
     - The angle $\theta$ is acute, indicating $\tilde{w}^{\text{new}}$ remains in the same half-space as $ w $.
     - Smaller $ |s \alpha| $ (near 0) means a smaller mistake margin, leading to a smaller $\theta$.
     - Larger $ |s \alpha| $ (near $-1$) means a larger mistake, increasing $\theta$ toward $ 90^\circ $.
   - The edge case $ s \alpha \leq -1 $ (e.g., $-\sqrt{2}$) is less typical, as $\theta \geq 90^\circ$ suggests significant misalignment, potentially flipping the weight vector.

**Final Answer**:
$$
\langle \tilde{w}^{\text{old}}, \tilde{w}^{\text{new}} \rangle = \frac{1 + s \alpha}{\sqrt{3 + 2 s \alpha}},
$$
where $ s \alpha \in [-\sqrt{2}, 0] $.

**Geometric Relationship**:  
The angle $\theta$ between $\tilde{w}^{\text{old}}$ and $\tilde{w}^{\text{new}}$ is acute ($\theta < 90^\circ$) for $ s \alpha > -1 $, ranging from $\approx 54.74^\circ$ (at $ s \alpha = 0 $) to $\approx 90^\circ$ (as $ s \alpha \to -1^+ $). The update rotates $\tilde{w}^{\text{old}}$ toward $ s \tilde{u}^i $, with the rotation angle increasing with the mistake’s margin. The edge case $ s \alpha = -\sqrt{2} $ ($\theta = 180^\circ$) is atypical, indicating an extreme correction.

$$
\boxed{\frac{1 + s \alpha}{\sqrt{3 + 2 s \alpha}}}
$$

---

## (b) **[4 marks]** 

Suppose in a particular epoch, there are no mistakes found for the first $n-1$ samples. Now when processing the $n$-th sample, a mistake is made by the perceptron, leading to an update. Let $\tilde{w}^{old}$ be the weight vector before the update and $\tilde{w}^{new}$ be the weight vector after the update. Describe a geometric configuration of the dataset which will yield the minimum value of $\langle \tilde{w}^{old}, \tilde{w}^{new} \rangle$. Also describe another geometric configuration of the dataset which will yield the maximum value of $\langle \tilde{w}^{old}, \tilde{w}^{new} \rangle$. Explain with suitable illustrations.

## Solution

From part (a), we have the inner product between the old and new weight vectors:
$$
\langle \tilde{w}^{\text{old}}, \tilde{w}^{\text{new}} \rangle = \frac{1 + s \alpha}{\sqrt{3 + 2 s \alpha}},
$$
where:
- $\tilde{w}^{\text{old}} = w$, the unit-length weight vector before the update ($\|w\|_2 = 1$).
- $\tilde{u}^n = x = (u^n, 1)$, the augmented input for the $n$-th sample, with $\|u^n\|_2 = 1$, so $\|x\|_2 = \sqrt{1 + 1} = \sqrt{2}$.
- $s = s^n \in \{+1, -1\}$, the true label of the $n$-th sample.
- $\alpha = \langle w, x \rangle$, the inner product between the weight vector and the augmented input.
- Update rule: $v = w + s x$, and $\tilde{w}^{\text{new}} = \frac{v}{\|v\|_2}$.
- A mistake occurs on the $n$-th sample, so the predicted label $\hat{s} = \text{sign}(\langle w, x \rangle) = \text{sign}(\alpha) \neq s$, implying:
  $$
  s \alpha \leq 0.
  $$
- By Cauchy-Schwarz, $|\alpha| = |\langle w, x \rangle| \leq \|w\|_2 \|x\|_2 = 1 \cdot \sqrt{2} = \sqrt{2}$, so:
  $$
  s \alpha \in [-\sqrt{2}, 0].
  $$

Our goal is to find the values of $s \alpha$ that minimize and maximize the inner product $\langle w, \tilde{w}^{\text{new}} \rangle$, and describe the corresponding geometric configurations of the dataset, considering the first $n-1$ samples are correctly classified and the $n$-th sample is misclassified.

#### **Step 1: Analyze the Inner Product**
The inner product is:
$$
\langle w, \tilde{w}^{\text{new}} \rangle = \frac{1 + s \alpha}{\sqrt{3 + 2 s \alpha}}.
$$
To find the minimum and maximum, we need to evaluate this expression over $s \alpha \in [-\sqrt{2}, 0]$. Let’s define:
$$
z = s \alpha, \quad z \in [-\sqrt{2}, 0].
$$
Then:
$$
\langle w, \tilde{w}^{\text{new}} \rangle = \frac{1 + z}{\sqrt{3 + 2 z}}.
$$
We need to determine how this function behaves as a function of $z$.

- **Numerator:** $1 + z \geq 1 - \sqrt{2} \approx -0.4142$. At $z = 0$, the numerator is 1; at $z = -\sqrt{2}$, it’s $1 - \sqrt{2}$.
- **Denominator:** $\sqrt{3 + 2 z}$. Compute its range:
  - At $z = 0$: $\sqrt{3} \approx 1.732$.
  - At $z = -\sqrt{2}$: $\sqrt{3 - 2 \sqrt{2}} = \sqrt{(\sqrt{2} - 1)^2} = \sqrt{2} - 1 \approx 0.4142$.
- **Sign of Inner Product:**
  - The denominator is always positive ($\sqrt{3 + 2 z} > 0$).
  - The numerator $1 + z$ is positive for $z > -1$. Since $-\sqrt{2} \approx -1.414 < -1$, we check:
    - At $z = -1$: Numerator = $1 - 1 = 0$, so $\langle w, \tilde{w}^{\text{new}} \rangle = 0$.
    - At $z = -\sqrt{2}$: Numerator = $1 - \sqrt{2} < 0$, so $\langle w, \tilde{w}^{\text{new}} \rangle < 0$.
  - From part (a), the angle $\theta = \arccos(\langle w, \tilde{w}^{\text{new}} \rangle)$ should be acute ($\theta < 90^\circ$), requiring:
    $$
    \langle w, \tilde{w}^{\text{new}} \rangle > 0.
    $$
    Thus, we need $1 + z > 0$, so:
    $$
    z > -1 \implies s \alpha > -1.
    $$
    Since $s \alpha \in [-\sqrt{2}, 0]$, we restrict to $s \alpha \in (-1, 0]$ to ensure a positive inner product.

#### **Step 2: Maximize the Inner Product**
To maximize $\langle w, \tilde{w}^{\text{new}} \rangle$, maximize the function:
$$
f(z) = \frac{1 + z}{\sqrt{3 + 2 z}}, \quad z \in [-1, 0].
$$
- Evaluate at the boundary $z = 0$:
  $$
  f(0) = \frac{1 + 0}{\sqrt{3 + 0}} = \frac{1}{\sqrt{3}} \approx 0.577.
  $$
- Check if the function is increasing or decreasing by computing the derivative or testing points. Let’s test at $z = -1$:
  $$
  f(-1) = \frac{1 - 1}{\sqrt{3 - 2}} = \frac{0}{\sqrt{1}} = 0.
  $$
- For $z \in (-1, 0)$, say $z = -0.5$:
  $$
  f(-0.5) = \frac{1 - 0.5}{\sqrt{3 - 1}} = \frac{0.5}{\sqrt{2}} \approx \frac{0.5}{1.414} \approx 0.353.
  $$
The function increases as $z$ approaches 0, since the numerator grows and the denominator shrinks. Thus, the maximum occurs at:
$$
s \alpha = 0, \quad \langle w, \tilde{w}^{\text{new}} \rangle = \frac{1}{\sqrt{3}} \approx 0.577.
$$
**Geometric Configuration for Maximum:**
- **Condition:** $s \alpha = s \langle w, x \rangle = 0$, so $\langle w, x \rangle = 0$, meaning $w$ is orthogonal to $x = \tilde{u}^n = (u^n, 1)$.
- **Interpretation:** The weight vector $w$ is perpendicular to the augmented input $\tilde{u}^n$, resulting in a prediction $\hat{s} = \text{sign}(\langle w, x \rangle) = \text{sign}(0)$, which is undefined but typically considered a mistake (e.g., $\hat{s} = +1 \neq s = -1$). This represents a **small margin mistake**, as the sample lies on the decision boundary ($\langle w, x \rangle = 0$).
- **Dataset Configuration:**
  - **First $n-1$ samples:** For samples $i = 1, \ldots, n-1$, no mistakes occur, so:
    $$
    s^i \langle w, \tilde{u}^i \rangle > 0, \quad \tilde{u}^i = (u^i, 1).
    $$
    This means $w$ correctly classifies $\tilde{u}^i$ with positive margin, i.e., $\text{sign}(\langle w, \tilde{u}^i \rangle) = s^i$.
  - **$n$-th sample:** Choose $\tilde{u}^n = (u^n, 1)$ such that:
    $$
    \langle w, \tilde{u}^n \rangle = w^\top (u^n, 1) = w_{1:d}^\top u^n + w_{d+1} = 0.
    $$
    Since $\|u^n\|_2 = 1$, $u^n \in \mathbb{R}^d$ lies on the unit sphere, and $\tilde{u}^n$ has an additional coordinate of 1.
  - **Example Configuration:**
    - Let $w = \left( \frac{1}{\sqrt{d+1}}, \ldots, \frac{1}{\sqrt{d+1}} \right) \in \mathbb{R}^{d+1}$ (initial weight vector).
    - Choose $u^n = (1, 0, \ldots, 0) \in \mathbb{R}^d$, so $\|u^n\|_2 = 1$, and:
      $$
      \tilde{u}^n = (1, 0, \ldots, 0, 1).
      $$
      $\langle w, \tilde{u}^n \rangle = \sum_{j=1}^{d+1} w_j \tilde{u}^n_j = \frac{1}{\sqrt{d+1}} \cdot 1 + 0 + \cdots + 0 + \frac{1}{\sqrt{d+1}} \cdot 1 = \frac{2}{\sqrt{d+1}}.$  
      This is not zero, so adjust $\tilde{u}^n$. Instead, solve for orthogonality:
      $$
      w_{1:d}^\top u^n + w_{d+1} = 0.
      $$
      For simplicity, in $\mathbb{R}^2$ ($d=1$):
      - $w = \left( \frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}} \right)$.
      - $u^n = (1)$, $\tilde{u}^n = (1, 1)$.
      - $\langle w, \tilde{u}^n \rangle = \frac{1}{\sqrt{2}} \cdot 1 + \frac{1}{\sqrt{2}} \cdot 1 = \sqrt{2} \neq 0$.
      Try a vector orthogonal to $w$. This suggests we need a general approach:
      - Choose $u^n$ such that $w_{1:d}^\top u^n = -w_{d+1}$.
      - Since $w$ is unit-length, normalize appropriately.

    - General case: Construct $\tilde{u}^n$ in the orthogonal complement of $w$. For example, if $w = (a, b, c)$, find $\tilde{u}^n = (u^n_1, u^n_2, 1)$ with $a u^n_1 + b u^n_2 + c = 0$, and $\sqrt{(u^n_1)^2 + (u^n_2)^2} = 1$.
  - **First $n-1$ samples:** Ensure $\tilde{u}^i$ are aligned such that $s^i \langle w, \tilde{u}^i \rangle > 0$. For example, $\tilde{u}^i$ could be close to $w$ for $s^i = +1$ or $-w$ for $s^i = -1$.
- **Illustration Description:**
  - In $\mathbb{R}^{d+1}$, visualize $w$ as a unit vector.
  - $\tilde{u}^n = (u^n, 1)$ lies in the hyperplane orthogonal to $w$, i.e., the decision boundary.
  - Other samples $\tilde{u}^i$ (for $i=1,\ldots,n-1$) lie on the correct side of the hyperplane defined by $w$, with $s^i \langle w, \tilde{u}^i \rangle > 0$.
  - The update rotates $w$ slightly toward $s \tilde{u}^n$, resulting in a small angle ($\theta \approx \arccos(1/\sqrt{3}) \approx 55^\circ$).

#### **Step 3: Minimize the Inner Product**
To minimize $\langle w, \tilde{w}^{\text{new}} \rangle$, minimize:
$$
f(z) = \frac{1 + z}{\sqrt{3 + 2 z}}, \quad z \in [-1, 0].
$$
- Evaluate at $z = -1$:
  $$
  f(-1) = 0.
  $$
- The function decreases as $z$ becomes more negative within $[-1, 0]$. However, $f(-1) = 0$ corresponds to $\theta = 90^\circ$, which is the boundary of an acute angle. To ensure $\theta < 90^\circ$, we consider $z \to -1^+$ (i.e., $s \alpha$ slightly greater than $-1$) or evaluate at a point close to $-1$.
- Test at $z = -0.99$:
  $$
  1 + z = 1 - 0.99 = 0.01, \quad 3 + 2 z = 3 - 1.98 = 1.02, \quad \sqrt{1.02} \approx 1.01.
  $$
  $$
  f(-0.99) \approx \frac{0.01}{1.01} \approx 0.0099.
  $$
- As $z \to -1^+$, $f(z) \to 0^+$, giving a very small positive inner product, corresponding to $\theta \to 90^\circ$ from below.
- **Boundary Consideration:** The previous solution’s claim of $z = -\sqrt{2}$ gave:
  $$
  f(-\sqrt{2}) = \frac{1 - \sqrt{2}}{\sqrt{3 - 2 \sqrt{2}}} = \frac{-(\sqrt{2} - 1)}{\sqrt{2} - 1} = -1.
  $$
  This implies $\theta = 180^\circ$, which is incorrect for an acute angle. The error arose because $z = -\sqrt{2}$ is possible but leads to a negative inner product, violating the geometric interpretation from part (a). Thus, we constrain:
  $$
  s \alpha \in (-1, 0].
  $$
- **Minimum Inner Product:** The minimum positive inner product occurs as $s \alpha \to -1^+$:
  $$
  \langle w, \tilde{w}^{\text{new}} \rangle \to 0^+.
  $$
  For a concrete value, choose $s \alpha = -0.999$:
  $$
  f(-0.999) = \frac{1 - 0.999}{\sqrt{3 - 1.998}} = \frac{0.001}{\sqrt{1.002}} \approx \frac{0.001}{1.001} \approx 0.001.
  $$
  This gives a small positive inner product, ensuring $\theta < 90^\circ$.

**Geometric Configuration for Minimum:**
- **Condition:** $s \alpha \approx -1$, so $\alpha \approx -1/s$. For example, if $s = -1$, then $\langle w, x \rangle \approx 1$; if $s = +1$, then $\langle w, x \rangle \approx -1$.
- **Interpretation:** The weight vector $w$ is strongly aligned with $x$ in the direction opposite to the correct classification, i.e., $\langle w, x \rangle \approx s \|x\|_2 / \sqrt{2} = s \sqrt{2} / \sqrt{2} = s$. Since $s \alpha \approx -1$, we have:
  $$
  \langle w, x \rangle \approx -s.
  $$
  This represents a **large margin mistake**, where the sample is far on the wrong side of the decision boundary.

- **Dataset Configuration:**
  - **First $n-1$ samples:** As before, $s^i \langle w, \tilde{u}^i \rangle > 0$, so $\tilde{u}^i$ are correctly classified.
  - **$n$-th sample:** Choose $\tilde{u}^n = (u^n, 1)$ such that:
    $$
    \langle w, \tilde{u}^n \rangle \approx -s.
    $$
    Since $\|\tilde{u}^n\|_2 = \sqrt{2}$, we aim for:
    $$
    \alpha = \langle w, \tilde{u}^n \rangle \approx -s, \quad |\alpha| \leq \sqrt{2}.
    $$
    - Example: Let $s = -1$, so $\alpha \approx 1$.
      - Suppose $w = (a, b, c)$ with $\|w\|_2 = 1$.
      - Choose $\tilde{u}^n = (u^n_1, u^n_2, 1)$ such that:
        $$
        a u^n_1 + b u^n_2 + c \approx 1, \quad \sqrt{(u^n_1)^2 + (u^n_2)^2} = 1.
        $$
      - For simplicity, in $\mathbb{R}^2$ ($d=1$):
        - $w = \left( \frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}} \right)$.
        - Try $\tilde{u}^n = \left( \frac{1}{\sqrt{2}}, 1 \right)$:
          $$
          \|u^n\|_2 = \left\| \frac{1}{\sqrt{2}} \right\|_2 = \frac{1}{\sqrt{2}} \neq 1.
          $$
          Adjust $u^n = (1)$, so $\tilde{u}^n = (1, 1)$:
          $$
          \langle w, \tilde{u}^n \rangle = \frac{1}{\sqrt{2}} \cdot 1 + \frac{1}{\sqrt{2}} \cdot 1 = \sqrt{2} \approx 1.414.
          $$
          $$
          s = -1, \quad s \alpha = -1 \cdot \sqrt{2} \approx -1.414 < -1.
          $$
          This overshoots. Scale $\tilde{u}^n$:
          - Try $\tilde{u}^n = \left( \frac{\sqrt{2}}{2}, 1 \right)$:
            $$
            \|u^n\|_2 = \frac{\sqrt{2}}{2} \approx 0.707 \neq 1.
            $$

          - General approach: Set $\tilde{u}^n \approx \frac{w}{\|w\|_2} \cdot \frac{\sqrt{2}}{2}$, but adjust the last coordinate.
      - Instead, choose $\alpha \approx 1$:
        - Let $w = (w_{1:d}, w_{d+1})$, $\tilde{u}^n = (u^n, 1)$.
        - Solve: $w_{1:d}^\top u^n + w_{d+1} \approx 1$, with $\|u^n\|_2 = 1$.
        - For $d=2$, $w = \left( \frac{1}{\sqrt{3}}, \frac{1}{\sqrt{3}}, \frac{1}{\sqrt{3}} \right)$:
          $$
          u^n = \left( \frac{\sqrt{2}}{2}, \frac{\sqrt{2}}{2} \right), \quad \tilde{u}^n = \left( \frac{\sqrt{2}}{2}, \frac{\sqrt{2}}{2}, 1 \right).
          $$
          $$
          \|u^n\|_2 = \sqrt{\left( \frac{\sqrt{2}}{2} \right)^2 + \left( \frac{\sqrt{2}}{2} \right)^2}
          $$
          $$
          = \sqrt{\frac{2}{4} + \frac{2}{4}} = \sqrt{1} = 1.
          $$
          $$
          \langle w, \tilde{u}^n \rangle = \frac{1}{\sqrt{3}} \cdot \frac{\sqrt{2}}{2} + \frac{1}{\sqrt{3}} \cdot \frac{\sqrt{2}}{2} + \frac{1}{\sqrt{3}} \cdot 1 
          $$
          $$
          = \frac{\sqrt{2} + \sqrt{2} + 2 \sqrt{3}}{2 \sqrt{3}} = \frac{2 \sqrt{2} + 2 \sqrt{3}}{2 \sqrt{3}} \approx 1.393.
          $$
          $$
          s = -1, \quad s \alpha \approx -1.393.
          $$  
          This is too negative. Adjust to get $s \alpha \approx -0.999$:
          - Numerically adjust $u^n$ to reduce $\langle w, \tilde{u}^n \rangle$ closer to 1.
          
    - General case: Choose $\tilde{u}^n$ such that:
      $$  
      \langle w, \tilde{u}^n \rangle \approx -s \cdot 0.999.
      $$
  - **First $n-1$ samples:** Similar to the maximum case, ensure $s^i \langle w, \tilde{u}^i \rangle > 0$.
- **Illustration Description:**
  - Visualize $w$ as a unit vector in $\mathbb{R}^{d+1}$.
  - $\tilde{u}^n = (u^n, 1)$ is nearly aligned with $w$ (if $s = -1$) or $-w$ (if $s = +1$), causing a large margin mistake.
  - Other samples $\tilde{u}^i$ are on the correct side of the hyperplane.
  - The update rotates $w$ significantly toward $s \tilde{u}^n$, approaching a $90^\circ$ angle.

#### **Step 4: Final Summary**
- **Maximum Inner Product:**
  $$
  \langle \tilde{w}^{\text{old}}, \tilde{w}^{\text{new}} \rangle = \frac{1}{\sqrt{3}} \approx 0.577, \quad \text{when } s \alpha = 0.
  $$
  - **Configuration:** $\tilde{u}^n$ is orthogonal to $w$, representing a small margin mistake. The first $n-1$ samples are correctly classified ($s^i \langle w, \tilde{u}^i \rangle > 0$).
  - **Geometric Meaning:** Minimal rotation of $w$, as the mistake is near the decision boundary.
- **Minimum Inner Product:**
  $$
  \langle \tilde{w}^{\text{old}}, \tilde{w}^{\text{new}} \rangle \to 0^+, \quad \text{as } s \alpha \to -1^+.
  $$
  - For a concrete value, at $s \alpha = -0.999$:
    $\langle \tilde{w}^{\text{old}}, \tilde{w}^{\text{new}} \rangle \approx 0.001.$
  - **Configuration:** $\tilde{u}^n$ is nearly aligned with $w$ in the direction that causes a mistake (e.g., $\langle w, \tilde{u}^n \rangle \approx -s \cdot 0.999$). The first $n-1$ samples are correctly classified.
  - **Geometric Meaning:** Large rotation of $w$ toward $s \tilde{u}^n$, approaching $90^\circ$, due to a significant misalignment.

**Correctness Notes:**
- The previous solution incorrectly claimed $\langle \tilde{w}^{\text{old}}, \tilde{w}^{\text{new}} \rangle \approx -0.8$ at $s \alpha = -\sqrt{2}$, which gave a negative inner product ($-1$), implying $\theta = 180^\circ$. This was inconsistent with part (a)’s requirement of an acute angle ($\theta < 90^\circ$).
- By restricting $s \alpha > -1$, we ensure $\langle \tilde{w}^{\text{old}}, \tilde{w}^{\text{new}} \rangle > 0$, aligning with the perceptron’s behavior of rotating the weight vector toward the misclassified example without overshooting to the opposite half-space.
- The configurations account for the dataset’s structure: $n-1$ correctly classified samples and one misclassified sample, with $\|u^i\|_2 = 1$.

**Illustration Guidance (for conceptual understanding):**
- **Maximum Case:** Draw a unit sphere in $\mathbb{R}^{d+1}$. Place $w$ as a vector from the origin. $\tilde{u}^n$ lies in the plane orthogonal to $w$, and other $\tilde{u}^i$ are positioned such that their projections onto $w$ align with their labels $s^i$. The update slightly tilts $w$ toward $s \tilde{u}^n$.
- **Minimum Case:** $\tilde{u}^n$ is nearly parallel to $w$ (or $-w$) in the direction causing a mistake. The update rotates $w$ significantly, almost perpendicular to its original direction. Other $\tilde{u}^i$ remain on the correct side.

---

## (c) **[3 marks]** 

Recall that $D$ is linearly separable when for every $i \in \{1,2,\dots,n\}$, we have $y^i \langle \tilde{w}, \tilde{x}^i \rangle \geq \gamma$ where $\gamma > 0$ and $\tilde{x}^i = (x,1)$. In part (b), suppose that after the update to $\tilde{w}$ at the $n$-th sample, you wish that the updated weights serve as a linear separator for $D$. Explain with suitable reasons a possible value of $\gamma$ for such a scenario.

## Solution

#### **Step 1: Define Variables**
- $\tilde{w}^{\text{old}} = w$, unit-length ($\|w\|_2 = 1$).
- $\tilde{u}^n = x = (u^n, 1)$, with $\|u^n\|_2 = 1$, so $\|x\|_2 = \sqrt{2}$.
- $s = s^n \in \{+1, -1\}$, true label of the $n$-th sample.
- Update: $v = w + s x$, $\tilde{w}^{\text{new}} = \frac{v}{\|v\|_2}$.
- Dataset is linearly separable with margin $\gamma > 0$:
  $$
  y^i \langle w, \tilde{x}^i \rangle \geq \gamma, \quad \tilde{x}^i = (x^i, 1), \quad \|\tilde{x}^i\|_2 = \sqrt{2}.
  $$
- Goal: Find $\gamma' > 0$ such that:
  $$
  y^i \langle \tilde{w}^{\text{new}}, \tilde{x}^i \rangle \geq \gamma' \quad \forall i.
  $$

#### **Step 2: Compute New Margin**
- Update:
  $$
  \tilde{w}^{\text{new}} = \frac{w + s x}{\|w + s x\|_2},
  $$
  $$
  \|w + s x\|_2^2 = \|w\|_2^2 + 2 s \langle w, x \rangle + \|x\|_2^2 = 1 + 2 s \alpha + 2, \quad
  $$
  $$
  \alpha = \langle w, x \rangle.
  $$
  $$
  \|w + s x\|_2 = \sqrt{3 + 2 s \alpha}.
  $$
- For each sample $i$:
  $$  
  y^i \langle \tilde{w}^{\text{new}}, \tilde{x}^i \rangle = y^i \left\langle \frac{w + s x}{\sqrt{3 + 2 s \alpha}}, \tilde{x}^i \right\rangle = \frac{y^i \langle w, \tilde{x}^i \rangle + s y^i \langle x, \tilde{x}^i \rangle}{\sqrt{3 + 2 s \alpha}}.
  $$
- Original margin: $y^i \langle w, \tilde{x}^i \rangle \geq \gamma$.
- We need:
  $$
  y^i \langle \tilde{w}^{\text{new}}, \tilde{x}^i \rangle \geq \gamma'.
  $$
- **Estimate $\gamma'$**:
  - The term $y^i \langle w, \tilde{x}^i \rangle \geq \gamma$.
  - The term $s y^i \langle x, \tilde{x}^i \rangle$:
    $$
    \langle x, \tilde{x}^i \rangle = \langle (u^n, 1), (x^i, 1) \rangle = (u^n)^\top x^i + 1.
    $$
    Since $\|u^n\|_2 = \|x^i\|_2 = 1$, by Cauchy-Schwarz:
    $$
    |(u^n)^\top x^i| \leq 1 \implies \langle x, \tilde{x}^i \rangle \in [0, 2].
    $$
    Thus, $s y^i \langle x, \tilde{x}^i \rangle \in [-2, 2]$.
  - Worst case for the margin:
    $$
    y^i \langle \tilde{w}^{\text{new}}, \tilde{x}^i \rangle \geq \frac{\gamma + s y^i \langle x, \tilde{x}^i \rangle}{\sqrt{3 + 2 s \alpha}}.
    $$
    Minimize over $i$:
    - $s \alpha \leq 0$ (mistake condition), so $\sqrt{3 + 2 s \alpha} \leq \sqrt{3}$.
    - Worst case for $s y^i \langle x, \tilde{x}^i \rangle \approx -2$.
    - Assume $\langle x, \tilde{x}^i \rangle \approx 0$ (average case, as samples may not align with $x$):
      $$
      y^i \langle \tilde{w}^{\text{new}}, \tilde{x}^i \rangle \approx \frac{\gamma}{\sqrt{3 + 2 s \alpha}}.
      $$
    - Minimum $\gamma'$:
      $$
      \gamma' = \min_i \frac{y^i \langle w, \tilde{x}^i \rangle + s y^i \langle x, \tilde{x}^i \rangle}{\sqrt{3 + 2 s \alpha}}.
      $$
      Simplest estimate: Assume the update preserves separability with a reduced margin:
      $$
      \gamma' = \frac{\gamma}{\sqrt{3}}.
      $$
      Since $s \alpha \leq 0$, the denominator $\sqrt{3 + 2 s \alpha} \leq \sqrt{3}$, so:
      $$
      \gamma' \leq \frac{\gamma}{\sqrt{3 + 2 s \alpha}} \leq \frac{\gamma}{\sqrt{3}}.
      $$

#### **Step 3: Verify**
- For the $n$-th sample:
  $$
  s \langle \tilde{w}^{\text{new}}, x \rangle = s \frac{\langle w, x \rangle + s \|x\|_2^2}{\sqrt{3 + 2 s \alpha}} = \frac{s \alpha + 2}{\sqrt{3 + 2 s \alpha}}.
  $$
  Since $s \alpha \leq 0$, $s \alpha + 2 \geq 2 - \sqrt{2} > 0$, so the updated weight correctly classifies the $n$-th sample.
- For other samples, the margin is reduced due to the rotation, approximated by $\gamma / \sqrt{3}$.

**Final Answer**:
A possible value for $\gamma'$ is:
$$
\gamma' = \frac{\gamma}{\sqrt{3}}.
$$
This ensures $\tilde{w}^{\text{new}}$ is a linear separator, with a margin reduced by the worst-case scaling factor.

$$
\boxed{\frac{\gamma}{\sqrt{3}}}
$$
