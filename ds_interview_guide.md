# Complete Data Science Interview Preparation Guide
## Swiggy Data Scientist Role - Comprehensive Coverage

---

# TABLE OF CONTENTS

1. [Statistics & Probability](#statistics--probability)
2. [Linear Algebra](#linear-algebra)
3. [Python Programming](#python-programming)
4. [Data Structures & Algorithms](#data-structures--algorithms)
5. [Pandas & NumPy](#pandas--numpy)
6. [SQL](#sql)
7. [Machine Learning](#machine-learning)
8. [Deep Learning](#deep-learning)
9. [Model Evaluation](#model-evaluation)
10. [Practice Problems](#practice-problems)

---

# STATISTICS & PROBABILITY

## 1. Measures of Central Tendency

### Mean
**Definition**: Average of all values
```
Mean (μ) = (Σ xi) / n
```

**Properties**:
- Sensitive to outliers
- Used when data is symmetric
- Sample mean (x̄) estimates population mean (μ)

### Median
**Definition**: Middle value when data is sorted
- For odd n: middle element
- For even n: average of two middle elements

**Properties**:
- Robust to outliers
- Better for skewed distributions

### Mode
**Definition**: Most frequently occurring value
- Can have multiple modes (bimodal, multimodal)
- Used for categorical data

---

## 2. Measures of Dispersion

### Variance

**Population Variance**:
```
σ² = Σ(xi - μ)² / N
```

**Sample Variance**:
```
s² = Σ(xi - x̄)² / (n-1)
```

**Why (n-1)? - Bessel's Correction**

Using n-1 instead of n makes the sample variance an **unbiased estimator** of population variance.

**Proof**:
```
E[s²] = E[Σ(xi - x̄)² / (n-1)]

Step 1: Σ(xi - x̄)² = Σ(xi - μ + μ - x̄)²
                    = Σ[(xi - μ) - (x̄ - μ)]²
                    = Σ(xi - μ)² - n(x̄ - μ)²

Step 2: E[Σ(xi - μ)²] = n·σ²

Step 3: E[(x̄ - μ)²] = Var(x̄) = σ²/n

Step 4: E[Σ(xi - x̄)²] = n·σ² - n·(σ²/n) = (n-1)σ²

Step 5: E[s²] = E[(n-1)σ² / (n-1)] = σ²
```

Therefore, s² is unbiased!

### Standard Deviation
```
σ = √(variance)
```

**Standard Error of the Mean**:
```
SE = σ / √n
```

**Why σ/√n?**

**Derivation**:
```
Given: X₁, X₂, ..., Xₙ are independent, each with variance σ²

Sample mean: x̄ = (X₁ + X₂ + ... + Xₙ) / n

Var(x̄) = Var[(X₁ + X₂ + ... + Xₙ) / n]
        = (1/n²) · Var(X₁ + X₂ + ... + Xₙ)
        
Since variables are independent:
        = (1/n²) · [Var(X₁) + Var(X₂) + ... + Var(Xₙ)]
        = (1/n²) · [σ² + σ² + ... + σ²]  (n times)
        = (1/n²) · n·σ²
        = σ²/n

Therefore: SD(x̄) = √(σ²/n) = σ/√n
```

**Interpretation**: As sample size increases, sample mean becomes more precise.

---

## 3. Variance of Sample Mean

**Problem**: Prove that Var(x̄) = σ²/n

**Full Proof**:
```
Given:
- X₁, X₂, ..., Xₙ ~ iid with E[Xi] = μ, Var(Xi) = σ²
- Sample mean: x̄ = (1/n)Σ Xi

Step 1: Express variance using definition
Var(x̄) = E[(x̄ - E[x̄])²]

Step 2: Find E[x̄]
E[x̄] = E[(1/n)Σ Xi]
     = (1/n)Σ E[Xi]
     = (1/n)·n·μ
     = μ

Step 3: Compute variance
Var(x̄) = Var[(1/n)Σ Xi]
        = (1/n²)Var[Σ Xi]

Step 4: Use independence
For independent variables: Var(X + Y) = Var(X) + Var(Y)

Var(x̄) = (1/n²)[Var(X₁) + Var(X₂) + ... + Var(Xₙ)]
        = (1/n²)[σ² + σ² + ... + σ²]
        = (1/n²)·n·σ²
        = σ²/n
```

---

## 4. Central Limit Theorem (CLT)

**Statement**: 
For a random sample of size n from any distribution with mean μ and variance σ², as n → ∞:

```
(x̄ - μ) / (σ/√n) → N(0, 1)

Or equivalently: x̄ ~ N(μ, σ²/n)
```

**Key Points**:
- Works for ANY distribution (uniform, exponential, etc.)
- Sample size n ≥ 30 is often sufficient
- Explains why many real-world phenomena are normal

**Intuitive Explanation**:
- Individual observations can be wild/skewed
- Their average smooths out extremes
- Sum of many random effects → Normal distribution

**Proof Sketch** (using MGF):
```
1. Let Yi = (Xi - μ)/σ, then E[Yi] = 0, Var(Yi) = 1

2. Define Zn = √n · ȳ = (1/√n)Σ Yi

3. MGF of Zn: Mn(t) = [M₁(t/√n)]ⁿ

4. Taylor expand M₁(t/√n):
   M₁(t/√n) ≈ 1 + 0·(t/√n) + (1/2)·(t/√n)² + o(1/n)
              = 1 + t²/(2n) + o(1/n)

5. Therefore:
   Mn(t) = [1 + t²/(2n) + o(1/n)]ⁿ → e^(t²/2) as n→∞

6. e^(t²/2) is the MGF of N(0,1)
```

**Applications**:
- Confidence intervals
- Hypothesis testing
- Quality control
- A/B testing

---

## 5. Probability Distributions

### Bernoulli Distribution
**Use**: Single trial with success/failure
```
P(X = 1) = p
P(X = 0) = 1-p

E[X] = p
Var(X) = p(1-p)
```

### Binomial Distribution
**Use**: Number of successes in n independent trials
```
P(X = k) = C(n,k) · p^k · (1-p)^(n-k)

E[X] = np
Var(X) = np(1-p)
```

**Example**: Probability of exactly 3 heads in 5 coin flips

### Poisson Distribution
**Use**: Number of events in fixed interval
```
P(X = k) = (λ^k · e^(-λ)) / k!

E[X] = λ
Var(X) = λ
```

**Example**: Number of customers arriving per hour

### Normal Distribution
```
f(x) = (1/(σ√(2π))) · e^(-(x-μ)²/(2σ²))

E[X] = μ
Var(X) = σ²
```

**68-95-99.7 Rule**:
- 68% of data within μ ± σ
- 95% within μ ± 2σ
- 99.7% within μ ± 3σ

### Exponential Distribution
**Use**: Time between events in Poisson process
```
f(x) = λe^(-λx)  for x ≥ 0

E[X] = 1/λ
Var(X) = 1/λ²
```

**Memoryless Property**: P(X > s+t | X > s) = P(X > t)

---

## 6. Hypothesis Testing

### Type I and Type II Errors

|  | H₀ True | H₀ False |
|---|---------|----------|
| **Reject H₀** | Type I Error (α) | Correct |
| **Fail to Reject H₀** | Correct | Type II Error (β) |

**Type I Error (False Positive)**:
- Rejecting a true null hypothesis
- Significance level α (commonly 0.05)
- "Convicting an innocent person"

**Type II Error (False Negative)**:
- Failing to reject a false null hypothesis
- Probability β
- "Acquitting a guilty person"

**Power of a Test**:
```
Power = 1 - β = P(Reject H₀ | H₀ is false)
```

**Trade-off**:
- Decreasing α increases β
- Increasing sample size decreases both

### P-value
**Definition**: Probability of observing data as extreme as (or more extreme than) what was observed, assuming H₀ is true.

**Interpretation**:
- p < 0.05: Reject H₀ (statistically significant)
- p > 0.05: Fail to reject H₀

**Common Mistake**: p-value is NOT the probability that H₀ is true!

### Confidence Intervals

**95% CI for population mean**:
```
x̄ ± 1.96 · (σ/√n)  [when σ known]
x̄ ± t₀.₀₂₅ · (s/√n)  [when σ unknown, use t-distribution]
```

**Interpretation**: 
- NOT "95% probability μ is in this interval"
- Correct: "If we repeat this process many times, 95% of intervals will contain μ"

### t-test vs z-test

**z-test**:
- Population variance σ² known
- Large sample (n > 30)
- Test statistic: z = (x̄ - μ₀)/(σ/√n)

**t-test**:
- Population variance unknown
- Small sample (n < 30)
- Test statistic: t = (x̄ - μ₀)/(s/√n)
- Uses t-distribution with (n-1) degrees of freedom

---

## 7. Probability Fundamentals

### Conditional Probability
```
P(A|B) = P(A ∩ B) / P(B)
```

### Bayes' Theorem
```
P(A|B) = [P(B|A) · P(A)] / P(B)

Where:
P(B) = P(B|A)·P(A) + P(B|A')·P(A')  [Law of Total Probability]
```

**Example Problem**: Medical Test
```
Disease prevalence: P(D) = 0.01
Test sensitivity: P(+|D) = 0.95
Test specificity: P(-|D') = 0.90

Find: P(D|+) = ?

P(+) = P(+|D)·P(D) + P(+|D')·P(D')
     = 0.95·0.01 + 0.10·0.99
     = 0.0095 + 0.099
     = 0.1085

P(D|+) = [P(+|D)·P(D)] / P(+)
       = (0.95·0.01) / 0.1085
       = 0.0876 ≈ 8.76%
```

**Insight**: Even with a positive test, only 8.76% chance of disease due to low prevalence!

### Independence
```
Events A and B are independent if:
P(A ∩ B) = P(A) · P(B)

Equivalently: P(A|B) = P(A)
```

### Expected Value
```
E[X] = Σ x·P(X=x)  [discrete]
E[X] = ∫ x·f(x)dx  [continuous]
```

**Properties**:
- E[aX + b] = aE[X] + b
- E[X + Y] = E[X] + E[Y]  [always true]
- E[XY] = E[X]·E[Y]  [only if independent]

### Variance Properties
```
Var(X) = E[X²] - (E[X])²

Var(aX + b) = a²·Var(X)

Var(X + Y) = Var(X) + Var(Y) + 2·Cov(X,Y)

If X, Y independent: Var(X + Y) = Var(X) + Var(Y)
```

### Covariance and Correlation
```
Cov(X,Y) = E[(X - E[X])(Y - E[Y])]
         = E[XY] - E[X]·E[Y]

Correlation: ρ = Cov(X,Y) / (σx·σy)
```

**Properties**:
- -1 ≤ ρ ≤ 1
- ρ = 0 means uncorrelated (not necessarily independent!)
- ρ = ±1 means perfect linear relationship

---

# LINEAR ALGEBRA

## 1. Matrices and Vectors

### Matrix Basics
**Matrix**: m × n array of numbers

```
A = [a₁₁  a₁₂  a₁₃]
    [a₂₁  a₂₂  a₂₃]
```

**Dimensions**: m rows × n columns

### Special Matrices

**Square Matrix**: m = n

**Identity Matrix**: 
```
I = [1  0  0]
    [0  1  0]
    [0  0  1]
```
Property: A·I = I·A = A

**Zero Matrix**:
```
O = [0  0]
    [0  0]
```

**Diagonal Matrix**:
```
D = [d₁  0   0 ]
    [0   d₂  0 ]
    [0   0   d₃]
```

**Symmetric Matrix**: A = Aᵀ
```
[1  2  3]
[2  5  4]
[3  4  6]
```

---

## 2. Matrix Operations

### Addition
Only if dimensions match:
```
[a  b] + [e  f] = [a+e  b+f]
[c  d]   [g  h]   [c+g  d+h]
```

### Scalar Multiplication
```
k·[a  b] = [ka  kb]
  [c  d]   [kc  kd]
```

### Matrix Multiplication
For A(m×n) and B(n×p), result is C(m×p):
```
cᵢⱼ = Σ(aᵢₖ · bₖⱼ) for k=1 to n
```

**Example**:
```
[1  2] · [5  6] = [1·5+2·7  1·6+2·8] = [19  22]
[3  4]   [7  8]   [3·5+4·7  3·6+4·8]   [43  50]
```

**Properties**:
- NOT commutative: AB ≠ BA (generally)
- Associative: (AB)C = A(BC)
- Distributive: A(B+C) = AB + AC

### Transpose
Flip rows and columns:
```
      [1  2  3]ᵀ   [1  4]
A =   [4  5  6]  = [2  5]
                   [3  6]
```

**Properties**:
- (Aᵀ)ᵀ = A
- (A + B)ᵀ = Aᵀ + Bᵀ
- (AB)ᵀ = BᵀAᵀ

---

## 3. Determinants

### 2×2 Matrix
```
det([a  b]) = ad - bc
   [c  d]
```

### 3×3 Matrix (Cofactor Expansion)
```
     [a  b  c]
det  [d  e  f] = a·det[e f] - b·det[d f] + c·det[d e]
     [g  h  i]         [h i]       [g i]       [g h]

                = a(ei-fh) - b(di-fg) + c(dh-eg)
```

### Properties of Determinants

1. **det(AB) = det(A)·det(B)**

2. **det(Aᵀ) = det(A)**

3. **det(kA) = kⁿ·det(A)** for n×n matrix

4. **det(A⁻¹) = 1/det(A)**

5. **Swapping two rows multiplies det by -1**

6. **Adding multiple of one row to another doesn't change det**

7. **If two rows are identical, det = 0**

8. **det(I) = 1**

### Geometric Interpretation
- For 2×2: Area of parallelogram formed by column vectors
- For 3×3: Volume of parallelepiped
- Sign indicates orientation

**Example**:
```
Vectors: v₁ = [3, 0], v₂ = [0, 2]

det = 3·2 - 0·0 = 6

Area of rectangle = 3 × 2 = 6 ✓
```

---

## 4. Matrix Inverse

### Definition
For square matrix A, if there exists A⁻¹ such that:
```
A·A⁻¹ = A⁻¹·A = I
```

Then A⁻¹ is the inverse of A.

### Conditions for Invertibility
**A is invertible if and only if det(A) ≠ 0**

Such matrices are called **non-singular**.

### 2×2 Inverse Formula
```
     [a  b]              1      [ d  -b]
A =  [c  d]    A⁻¹ = --------- [-c   a]
                     ad - bc
```

**Example**:
```
A = [4  7]    det(A) = 4·2 - 7·1 = 1
    [1  2]

A⁻¹ = [2  -7]
      [-1  4]

Verify: [4  7]·[2  -7] = [1  0]
        [1  2] [-1  4]   [0  1] ✓
```

### Properties
- (A⁻¹)⁻¹ = A
- (AB)⁻¹ = B⁻¹A⁻¹
- (Aᵀ)⁻¹ = (A⁻¹)ᵀ
- det(A⁻¹) = 1/det(A)

---

## 5. Rank of a Matrix

### Definition
**Rank**: Maximum number of linearly independent rows (or columns)

**Full Rank**: rank(A) = min(m, n) for m×n matrix

### Methods to Find Rank

**Row Echelon Form**:
Convert matrix to row echelon form, count non-zero rows

**Example**:
```
A = [1  2  3]
    [2  4  6]
    [1  1  2]

R₂ → R₂ - 2R₁:  [1  2  3]
                [0  0  0]
                [1  1  2]

R₃ → R₃ - R₁:   [1  2  3]
                [0  0  0]
                [0 -1 -1]

Swap R₂, R₃:    [1  2  3]
                [0 -1 -1]
                [0  0  0]

Rank = 2 (two non-zero rows)
```

### Properties
- rank(A) = rank(Aᵀ)
- rank(AB) ≤ min(rank(A), rank(B))
- rank(A + B) ≤ rank(A) + rank(B)

---

## 6. Eigenvalues and Eigenvectors

### Definition
For square matrix A, if:
```
A·v = λ·v
```

where v ≠ 0, then:
- **λ** is an eigenvalue
- **v** is the corresponding eigenvector

### Finding Eigenvalues

Solve the **characteristic equation**:
```
det(A - λI) = 0
```

**Example** (2×2):
```
A = [4  1]
    [2  3]

A - λI = [4-λ   1  ]
         [2    3-λ ]

det(A - λI) = (4-λ)(3-λ) - 2
            = λ² - 7λ + 10
            = (λ-5)(λ-2)

Eigenvalues: λ₁ = 5, λ₂ = 2
```

### Finding Eigenvectors

For each λ, solve (A - λI)v = 0:

**For λ₁ = 5**:
```
[4-5  1 ][v₁]   [0]     [-1  1][v₁]   [0]
[2   3-5][v₂] = [0]  →  [2  -2][v₂] = [0]

-v₁ + v₂ = 0  →  v₂ = v₁

Eigenvector: v₁ = [1]  (or any scalar multiple)
                  [1]
```

**For λ₂ = 2**:
```
[4-2  1 ][v₁]   [0]     [2  1][v₁]   [0]
[2   3-2][v₂] = [0]  →  [2  1][v₂] = [0]

2v₁ + v₂ = 0  →  v₂ = -2v₁

Eigenvector: v₂ = [ 1]
                  [-2]
```

### Properties
- Sum of eigenvalues = Trace(A)
- Product of eigenvalues = det(A)
- Eigenvectors from different eigenvalues are linearly independent

---

## 7. Applications in Machine Learning

### Principal Component Analysis (PCA)

**Goal**: Reduce dimensionality while preserving variance

**Steps**:
1. Center data: X' = X - mean(X)
2. Compute covariance matrix: C = (1/n)·X'ᵀX'
3. Find eigenvalues and eigenvectors of C
4. Sort eigenvalues (λ₁ ≥ λ₂ ≥ ... ≥ λₙ)
5. Select top k eigenvectors as principal components
6. Project data: Z = X'·Vₖ

**Variance Explained**:
```
Variance explained by PC_i = λᵢ / Σλⱼ
```

### Linear Regression (Matrix Form)

**Model**: y = Xβ + ε

**Normal Equation**:
```
β = (XᵀX)⁻¹Xᵀy
```

**Derivation**:
```
Minimize: ||y - Xβ||²

Take derivative w.r.t. β and set to 0:
-2Xᵀ(y - Xβ) = 0
XᵀXβ = Xᵀy
β = (XᵀX)⁻¹Xᵀy
```

### Singular Value Decomposition (SVD)

Any matrix A (m×n) can be decomposed:
```
A = UΣVᵀ
```

where:
- U (m×m): Left singular vectors (eigenvectors of AAᵀ)
- Σ (m×n): Diagonal matrix of singular values
- V (n×n): Right singular vectors (eigenvectors of AᵀA)

**Applications**:
- Dimensionality reduction
- Matrix compression
- Collaborative filtering (recommendation systems)
- Latent Semantic Analysis

---

# PYTHON PROGRAMMING

## 1. Functions

### Basic Function
```python
def greet(name):
    """Returns a greeting message"""
    return f"Hello, {name}!"

result = greet("Alice")  # "Hello, Alice!"
```

### Default Arguments
```python
def power(base, exponent=2):
    return base ** exponent

power(3)      # 9 (3²)
power(3, 3)   # 27 (3³)
```

### Variable Arguments

**`*args`** - Variable positional arguments:
```python
def sum_all(*args):
    """Accepts any number of arguments"""
    return sum(args)

sum_all(1, 2, 3)        # 6
sum_all(1, 2, 3, 4, 5)  # 15
```

**`**kwargs`** - Variable keyword arguments:
```python
def print_info(**kwargs):
    """Accepts any number of keyword arguments"""
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Alice", age=25, city="NYC")
# Output:
# name: Alice
# age: 25
# city: NYC
```

**Combining all**:
```python
def complex_function(arg1, arg2, *args, kwarg1=None, **kwargs):
    print(f"arg1: {arg1}")
    print(f"arg2: {arg2}")
    print(f"args: {args}")
    print(f"kwarg1: {kwarg1}")
    print(f"kwargs: {kwargs}")

complex_function(1, 2, 3, 4, kwarg1="test", extra1="a", extra2="b")
```

### Lambda Functions
```python
# Anonymous one-line functions
square = lambda x: x ** 2
add = lambda x, y: x + y

square(5)     # 25
add(3, 4)     # 7

# Common use: with map, filter, sorted
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))  # [1, 4, 9, 16, 25]

even = list(filter(lambda x: x % 2 == 0, numbers))  # [2, 4]

people = [("Alice", 25), ("Bob", 30), ("Charlie", 20)]
sorted_people = sorted(people, key=lambda x: x[1])  # Sort by age
```

---

## 2. Decorators

### What are Decorators?
Functions that modify the behavior of other functions.

### Basic Decorator
```python
def my_decorator(func):
    def wrapper():
        print("Before function call")
        func()
        print("After function call")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()
# Output:
# Before function call
# Hello!
# After function call
```

### Decorator with Arguments
```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        print(f"Result: {result}")
        return result
    return wrapper

@my_decorator
def add(a, b):
    return a + b

add(3, 5)
# Output:
# Calling add with args=(3, 5), kwargs={}
# Result: 8
```

### Practical Decorators

**Timing Decorator**:
```python
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end-start:.4f} seconds")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(2)
    return "Done"

slow_function()  # "slow_function took 2.0001 seconds"
```

**Memoization Decorator**:
```python
def memoize(func):
    cache = {}
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper

@memoize
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

fibonacci(100)  # Fast due to caching!
```

**Built-in Decorators**:
```python
class MyClass:
    
    @staticmethod
    def static_method():
        """No self or cls parameter"""
        return "I'm static"
    
    @classmethod
    def class_method(cls):
        """Receives class as first argument"""
        return f"I'm a method of {cls.__name__}"
    
    @property
    def my_property(self):
        """Access like attribute, not method"""
        return self._value
```

---

## 3. Exception Handling

### Basic Try-Except
```python
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")
```

### Multiple Exceptions
```python
try:
    # Some risky code
    value = int(input("Enter a number: "))
    result = 10 / value
except ValueError:
    print("Invalid input! Please enter a number.")
except ZeroDivisionError:
    print("Cannot divide by zero!")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Try-Except-Else-Finally
```python
try:
    file = open("data.txt", "r")
    data = file.read()
except FileNotFoundError:
    print("File not found!")
else:
    # Runs only if no exception occurred
    print(f"File read successfully: {len(data)} characters")
finally:
    # Always runs, regardless of exceptions
    if 'file' in locals():
        file.close()
    print("Cleanup completed")
```

### Raising Exceptions
```python
def validate_age(age):
    if age < 0:
        raise ValueError("Age cannot be negative!")
    if age > 150:
        raise ValueError("Age seems unrealistic!")
    return True

try:
    validate_age(-5)
except ValueError as e:
    print(f"Validation error: {e}")
```

### Custom Exceptions
```python
class InsufficientFundsError(Exception):
    """Raised when withdrawal exceeds balance"""
    pass

class BankAccount:
    def __init__(self, balance):
        self.balance = balance
    
    def withdraw(self, amount):
        if amount > self.balance:
            raise InsufficientFundsError(
                f"Cannot withdraw ${amount}. Balance: ${self.balance}"
            )
        self.balance -= amount

account = BankAccount(100)
try:
    account.withdraw(150)
except InsufficientFundsError as e:
    print(e)
```

---

## 4. List Comprehensions

### Basic List Comprehension
```python
# Traditional way
squares = []
for i in range(10):
    squares.append(i**2)

# List comprehension
squares = [i**2 for i in range(10)]
```

### With Conditionals
```python
# Even numbers only
evens = [i for i in range(20) if i % 2 == 0]

# If-else
labels = ["even" if i % 2 == 0 else "odd" for i in range(10)]
```

### Nested List Comprehensions
```python
# Flatten a matrix
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [num for row in matrix for num in row]
# [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Cartesian product
pairs = [(x, y) for x in [1, 2, 3] for y in ['a', 'b']]
# [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b'), (3, 'a'), (3, 'b')]
```

### Dictionary Comprehension
```python
# Square numbers
squares_dict = {i: i**2 for i in range(5)}
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# Filter dictionary
prices = {'apple': 0.5, 'banana': 0.3, 'orange': 0.6}
cheap = {k: v for k, v in prices.items() if v < 0.5}
```

### Set Comprehension
```python
# Unique even numbers
unique_evens = {i % 10 for i in range(100) if i % 2 == 0}
```

---

## 5. Generators

### What are Generators?
Functions that yield values one at a time, useful for large datasets.

### Basic Generator
```python
def count_up_to(n):
    count = 1
    while count <= n:
        yield count
        count += 1

counter = count_up_to(5)
print(next(counter))  # 1
print(next(counter))  # 2

# Or iterate
for num in count_up_to(5):
    print(num)
```

### Generator vs List
```python
# List: Creates entire list in memory
numbers_list = [i**2 for i in range(1000000)]  # Uses lots of memory

# Generator: Computes on-demand
numbers_gen = (i**2 for i in range(1000000))   # Memory efficient

# Generator expression (like list comprehension with parentheses)
gen = (x**2 for x in range(10))
```

### Practical Example: Reading Large Files
```python
def read_large_file(file_path):
    """Generator to read file line by line"""
    with open(file_path) as file:
        for line in file:
            yield line.strip()

# Memory efficient for huge files
for line in read_large_file("huge_file.txt"):
    process(line)
```

---

## 6. Iterators

### What is an Iterator?
Object that implements `__iter__()` and `__next__()` methods.

### Custom Iterator
```python
class Countdown:
    def __init__(self, start):
        self.current = start
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current <= 0:
            raise StopIteration
        self.current -= 1
        return self.current + 1

for num in Countdown(5):
    print(num)  # 5, 4, 3, 2, 1
```

---

## 7. Context Managers

### Using `with` Statement
```python
# Automatic resource cleanup
with open("file.txt", "r") as file:
    data = file.read()
# File automatically closed after block
```

### Custom Context Manager
```python
class Timer:
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        self.end = time.time()
        print(f"Elapsed: {self.end - self.start:.4f}s")

with Timer():
    # Code to time
    time.sleep(2)
# Automatically prints elapsed time
```

### Using `contextlib`
```python
from contextlib import contextmanager

@contextmanager
def timer():
    start = time.time()
    yield
    end = time.time()
    print(f"Elapsed: {end - start:.4f}s")

with timer():
    time.sleep(1)
```

---

## 8. Multithreading vs Multiprocessing

### Threading (I/O-bound tasks)
```python
import threading
import time

def worker(name, delay):
    print(f"{name} starting")
    time.sleep(delay)
    print(f"{name} finished")

# Create threads
threads = []
for i in range(3):
    t = threading.Thread(target=worker, args=(f"Thread-{i}", 2))
    threads.append(t)
    t.start()

# Wait for all threads to complete
for t in threads:
    t.join()
```

**Global Interpreter Lock (GIL)**:
- Only one thread executes Python bytecode at a time
- Good for I/O-bound tasks (network, file operations)
- NOT good for CPU-bound tasks

### Multiprocessing (CPU-bound tasks)
```python
from multiprocessing import Process
import os

def worker(name):
    print(f"{name} - Process ID: {os.getpid()}")

processes = []
for i in range(3):
    p = Process(target=worker, args=(f"Process-{i}",))
    processes.append(p)
    p.start()

for p in processes:
    p.join()
```

**Key Differences**:
| Feature | Threading | Multiprocessing |
|---------|-----------|-----------------|
| **Use for** | I/O-bound | CPU-bound |
| **GIL** | Shared | Separate |
| **Memory** | Shared | Separate |
| **Overhead** | Low | Higher |

### Thread Pool Executor
```python
from concurrent.futures import ThreadPoolExecutor
import requests

urls = [
    "https://api.github.com",
    "https://api.twitter.com",
    "https://api.reddit.com"
]

def fetch_url(url):
    response = requests.get(url)
    return len(response.content)

# Execute in parallel
with ThreadPoolExecutor(max_workers=3) as executor:
    results = executor.map(fetch_url, urls)

for result in results:
    print(f"Size: {result} bytes")
```

---

## 9. Important Built-in Functions

### map()
```python
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
# [1, 4, 9, 16, 25]
```

### filter()
```python
numbers = [1, 2, 3, 4, 5, 6]
evens = list(filter(lambda x: x % 2 == 0, numbers))
# [2, 4, 6]
```

### reduce()
```python
from functools import reduce

numbers = [1, 2, 3, 4, 5]
product = reduce(lambda x, y: x * y, numbers)
# 120 (1*2*3*4*5)
```

### zip()
```python
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]
combined = list(zip(names, ages))
# [('Alice', 25), ('Bob', 30), ('Charlie', 35)]
```

### enumerate()
```python
fruits = ["apple", "banana", "cherry"]
for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")
# 0: apple
# 1: banana
# 2: cherry
```

---

# DATA STRUCTURES & ALGORITHMS

## 1. Time Complexity

### Big O Notation

**O(1)** - Constant:
```python
def get_first(arr):
    return arr[0]  # Always one operation
```

**O(log n)** - Logarithmic:
```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

**O(n)** - Linear:
```python
def find_max(arr):
    max_val = arr[0]
    for num in arr:  # Iterate through all elements
        if num > max_val:
            max_val = num
    return max_val
```

**O(n log n)** - Linearithmic:
```python
def merge_sort(arr):
    # Most efficient sorting algorithms
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)
```

**O(n²)** - Quadratic:
```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n-1):  # Nested loops
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
```

**O(2ⁿ)** - Exponential:
```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)  # Two recursive calls
```

### Common Data Structure Operations

| Operation | Array | Linked List | Hash Table | BST (avg) |
|-----------|-------|-------------|------------|-----------|
| Access | O(1) | O(n) | - | O(log n) |
| Search | O(n) | O(n) | O(1) | O(log n) |
| Insert | O(n) | O(1)* | O(1) | O(log n) |
| Delete | O(n) | O(1)* | O(1) | O(log n) |

*At known position

---

## 2. Arrays and Strings

### Two Pointers Technique

**Example: Reverse String**
```python
def reverse_string(s):
    s = list(s)  # Convert to list (strings are immutable)
    left, right = 0, len(s) - 1
    
    while left < right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1
    
    return ''.join(s)

reverse_string("hello")  # "olleh"
```

**Example: Two Sum (Sorted Array)**
```python
def two_sum_sorted(arr, target):
    """Find two numbers that sum to target"""
    left, right = 0, len(arr) - 1
    
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    
    return None

two_sum_sorted([1, 2, 3, 4, 5], 7)  # [1, 4] (2+5=7)
```

### Sliding Window

**Example: Maximum Sum Subarray of Size K**
```python
def max_sum_subarray(arr, k):
    """Find maximum sum of any k consecutive elements"""
    n = len(arr)
    if n < k:
        return None
    
    # Compute sum of first window
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    # Slide the window
    for i in range(k, n):
        window_sum = window_sum - arr[i-k] + arr[i]
        max_sum = max(max_sum, window_sum)
    
    return max_sum

max_sum_subarray([1, 4, 2, 10, 23, 3, 1, 0, 20], 4)  # 39
```

**Example: Longest Substring Without Repeating Characters**
```python
def length_of_longest_substring(s):
    char_set = set()
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        
        char_set.add(s[right])
        max_length = max(max_length, right - left + 1)
    
    return max_length

length_of_longest_substring("abcabcbb")  # 3 ("abc")
```

---

## 3. Hash Tables (Dictionaries)

### Common Patterns

**Counting Frequency**
```python
def character_frequency(s):
    freq = {}
    for char in s:
        freq[char] = freq.get(char, 0) + 1
    return freq

# Or using defaultdict
from collections import defaultdict

def character_frequency(s):
    freq = defaultdict(int)
    for char in s:
        freq[char] += 1
    return freq

# Or using Counter
from collections import Counter
freq = Counter("hello")  # {'h': 1, 'e': 1, 'l': 2, 'o': 1}
```

**Two Sum Problem**
```python
def two_sum(nums, target):
    """Find indices of two numbers that sum to target"""
    seen = {}
    
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    
    return None

two_sum([2, 7, 11, 15], 9)  # [0, 1]
```

**Group Anagrams**
```python
def group_anagrams(strs):
    anagrams = defaultdict(list)
    
    for s in strs:
        # Sort string as key
        key = ''.join(sorted(s))
        anagrams[key].append(s)
    
    return list(anagrams.values())

group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
# [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]
```

---

## 4. Linked Lists

### Implementation
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, val):
        """Add node to end"""
        new_node = ListNode(val)
        if not self.head:
            self.head = new_node
            return
        
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
    
    def print_list(self):
        current = self.head
        while current:
            print(current.val, end=" -> ")
            current = current.next
        print("None")
```

### Reverse Linked List
```python
def reverse_list(head):
    prev = None
    current = head
    
    while current:
        next_temp = current.next  # Save next
        current.next = prev       # Reverse link
        prev = current            # Move prev forward
        current = next_temp       # Move current forward
    
    return prev
```

### Detect Cycle (Floyd's Algorithm)
```python
def has_cycle(head):
    """Detect if linked list has a cycle"""
    if not head or not head.next:
        return False
    
    slow = head
    fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
        if slow == fast:
            return True
    
    return False
```

---

## 5. Stacks and Queues

### Stack Implementation
```python
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if not self.is_empty():
            return self.items.pop()
    
    def peek(self):
        if not self.is_empty():
            return self.items[-1]
    
    def is_empty(self):
        return len(self.items) == 0

# Or simply use a list
stack = []
stack.append(1)  # push
stack.append(2)
top = stack.pop()  # 2
```

### Valid Parentheses
```python
def is_valid_parentheses(s):
    """Check if brackets are balanced"""
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:
            top = stack.pop() if stack else '#'
            if mapping[char] != top:
                return False
        else:
            stack.append(char)
    
    return not stack

is_valid_parentheses("()[]{}"))  # True
is_valid_parentheses("([)]")     # False
```

### Queue Implementation
```python
from collections import deque

class Queue:
    def __init__(self):
        self.items = deque()
    
    def enqueue(self, item):
        self.items.append(item)
    
    def dequeue(self):
        if not self.is_empty():
            return self.items.popleft()
    
    def is_empty(self):
        return len(self.items) == 0

# Or use deque directly
queue = deque()
queue.append(1)     # enqueue
queue.append(2)
first = queue.popleft()  # 1
```

---

## 6. Trees

### Binary Tree Node
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

### Tree Traversals

**Inorder (Left-Root-Right)**
```python
def inorder(root):
    if not root:
        return []
    return inorder(root.left) + [root.val] + inorder(root.right)

# Iterative
def inorder_iterative(root):
    result = []
    stack = []
    current = root
    
    while current or stack:
        while current:
            stack.append(current)
            current = current.left
        
        current = stack.pop()
        result.append(current.val)
        current = current.right
    
    return result
```

**Preorder (Root-Left-Right)**
```python
def preorder(root):
    if not root:
        return []
    return [root.val] + preorder(root.left) + preorder(root.right)
```

**Postorder (Left-Right-Root)**
```python
def postorder(root):
    if not root:
        return []
    return postorder(root.left) + postorder(root.right) + [root.val]
```

**Level Order (BFS)**
```python
from collections import deque

def level_order(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    return result
```

### Maximum Depth of Binary Tree
```python
def max_depth(root):
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))
```

---

## 7. Dynamic Programming

### Key Concepts
1. **Optimal Substructure**: Solution can be built from solutions to subproblems
2. **Overlapping Subproblems**: Same subproblems computed multiple times

### Fibonacci (Memoization)
```python
def fibonacci(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    
    memo[n] = fibonacci(n-1, memo) + fibonacci(n-2, memo)
    return memo[n]

# Time: O(n), Space: O(n)
```

### Fibonacci (Tabulation)
```python
def fibonacci(n):
    if n <= 1:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]

# Time: O(n), Space: O(n)
```

### Coin Change Problem
```python
def coin_change(coins, amount):
    """Minimum coins needed to make amount"""
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

coin_change([1, 2, 5], 11)  # 3 (5+5+1)
```

### Longest Common Subsequence
```python
def lcs(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

lcs("abcde", "ace")  # 3 ("ace")
```

---

## 8. Most Common Interview Problems

### 1. Best Time to Buy and Sell Stock
```python
def max_profit(prices):
    """Find maximum profit from one transaction"""
    min_price = float('inf')
    max_profit = 0
    
    for price in prices:
        min_price = min(min_price, price)
        max_profit = max(max_profit, price - min_price)
    
    return max_profit

max_profit([7, 1, 5, 3, 6, 4])  # 5 (buy at 1, sell at 6)
```

### 2. Merge Intervals
```python
def merge_intervals(intervals):
    """Merge overlapping intervals"""
    if not intervals:
        return []
    
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last = merged[-1]
        
        if current[0] <= last[1]:
            # Overlapping - merge
            merged[-1] = [last[0], max(last[1], current[1])]
        else:
            merged.append(current)
    
    return merged

merge_intervals([[1,3], [2,6], [8,10], [15,18]])
# [[1,6], [8,10], [15,18]]
```

### 3. Product of Array Except Self
```python
def product_except_self(nums):
    """Product of all elements except current (no division)"""
    n = len(nums)
    result = [1] * n
    
    # Left products
    left = 1
    for i in range(n):
        result[i] = left
        left *= nums[i]
    
    # Right products
    right = 1
    for i in range(n-1, -1, -1):
        result[i] *= right
        right *= nums[i]
    
    return result

product_except_self([1, 2, 3, 4])  # [24, 12, 8, 6]
```

### 4. Valid Palindrome
```python
def is_palindrome(s):
    """Check if string is palindrome (ignore case and non-alphanumeric)"""
    left, right = 0, len(s) - 1
    
    while left < right:
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        
        if s[left].lower() != s[right].lower():
            return False
        
        left += 1
        right -= 1
    
    return True

is_palindrome("A man, a plan, a canal: Panama")  # True
```

---

# PANDAS & NUMPY

## 1. NumPy Fundamentals

### Array Creation
```python
import numpy as np

# From list
arr = np.array([1, 2, 3, 4, 5])

# Zeros, ones, empty
zeros = np.zeros((3, 4))        # 3x4 array of zeros
ones = np.ones((2, 3))          # 2x3 array of ones
empty = np.empty((2, 2))        # Uninitialized

# Range
arange = np.arange(0, 10, 2)    # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5) # [0., 0.25, 0.5, 0.75, 1.]

# Random
random = np.random.rand(3, 3)   # Uniform [0, 1)
randn = np.random.randn(3, 3)   # Standard normal
randint = np.random.randint(0, 10, size=(3, 3))  # Random integers

# Identity matrix
identity = np.eye(3)
```

### Array Attributes
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

arr.shape      # (2, 3)
arr.ndim       # 2 (dimensions)
arr.size       # 6 (total elements)
arr.dtype      # dtype('int64')
```

### Indexing and Slicing
```python
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

arr[0]         # 0
arr[-1]        # 9
arr[2:5]       # [2, 3, 4]
arr[::2]       # [0, 2, 4, 6, 8] (every 2nd)

# 2D arrays
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

arr_2d[0, 0]   # 1
arr_2d[1, :]   # [4, 5, 6] (row 1)
arr_2d[:, 1]   # [2, 5, 8] (column 1)
arr_2d[0:2, 1:3]  # [[2, 3], [5, 6]]
```

### Boolean Indexing
```python
arr = np.array([1, 2, 3, 4, 5, 6])

# Boolean mask
mask = arr > 3
arr[mask]      # [4, 5, 6]

# Direct
arr[arr > 3]   # [4, 5, 6]
arr[(arr > 2) & (arr < 5)]  # [3, 4]
```

### Array Operations

**Broadcasting**
```python
# Scalar operation
arr = np.array([1, 2, 3])
arr + 10       # [11, 12, 13]
arr * 2        # [2, 4, 6]

# Array operations
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr1 + arr2    # [5, 7, 9]

# Broadcasting with different shapes
arr = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
row = np.array([10, 20, 30])             # (3,)
arr + row      # [[11, 22, 33], [14, 25, 36]]
```

**Aggregations**
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

arr.sum()         # 21
arr.mean()        # 3.5
arr.std()         # Standard deviation
arr.min()         # 1
arr.max()         # 6

# Along axis
arr.sum(axis=0)   # [5, 7, 9] (column sums)
arr.sum(axis=1)   # [6, 15] (row sums)
```

**Reshaping**
```python
arr = np.arange(12)

arr.reshape(3, 4)     # 3x4 array
arr.reshape(2, 2, 3)  # 3D array
arr.reshape(-1, 4)    # Infer dimension: 3x4

# Flatten
arr_2d.flatten()      # 1D array
arr_2d.ravel()        # 1D view (faster)
```

**Stacking**
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

np.vstack([a, b])     # [[1,2,3], [4,5,6]] (vertical)
np.hstack([a, b])     # [1,2,3,4,5,6] (horizontal)
np.column_stack([a, b])  # [[1,4], [2,5], [3,6]]
```

### Linear Algebra
```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication
np.dot(A, B)       # or A @ B
A.T                # Transpose

# Inverse and determinant
np.linalg.inv(A)   # Inverse
np.linalg.det(A)   # Determinant

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

# Solve linear system Ax = b
b = np.array([1, 2])
x = np.linalg.solve(A, b)
```

---

## 2. Pandas Fundamentals

### Series
```python
import pandas as pd

# Create Series
s = pd.Series([1, 2, 3, 4, 5])
s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])

# From dictionary
s = pd.Series({'a': 1, 'b': 2, 'c': 3})

# Indexing
s['a']         # 1
s[0]           # 1
s[['a', 'c']]  # Multiple

# Operations
s + 10
s * 2
s.mean()
s.max()
```

### DataFrame Creation
```python
# From dictionary
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['NYC', 'LA', 'Chicago']
})

# From list of dictionaries
data = [
    {'name': 'Alice', 'age': 25},
    {'name': 'Bob', 'age': 30}
]
df = pd.DataFrame(data)

# From NumPy array
arr = np.random.randn(3, 4)
df = pd.DataFrame(arr, columns=['A', 'B', 'C', 'D'])

# Read from file
df = pd.read_csv('data.csv')
df = pd.read_excel('data.xlsx')
df = pd.read_json('data.json')
```

### DataFrame Exploration
```python
df.head()          # First 5 rows
df.tail(10)        # Last 10 rows
df.info()          # Column types, non-null counts
df.describe()      # Statistical summary
df.shape           # (rows, columns)
df.columns         # Column names
df.dtypes          # Data types
df.index           # Row indices
```

### Indexing and Selection
```python
# Column selection
df['name']              # Single column (Series)
df[['name', 'age']]     # Multiple columns (DataFrame)

# Row selection
df.loc[0]               # By label
df.iloc[0]              # By position
df.loc[0:2]             # Rows 0, 1, 2
df.iloc[0:2]            # Rows 0, 1

# Both rows and columns
df.loc[0:2, ['name', 'age']]
df.iloc[0:2, 0:2]

# Conditional selection
df[df['age'] > 25]
df[(df['age'] > 25) & (df['city'] == 'NYC')]
df[df['name'].isin(['Alice', 'Bob'])]
```

### Adding/Removing Columns
```python
# Add column
df['salary'] = [50000, 60000, 70000]
df['bonus'] = df['salary'] * 0.1

# Drop column
df = df.drop('bonus', axis=1)
df = df.drop(columns=['bonus'])

# Rename columns
df = df.rename(columns={'name': 'full_name'})
```

### Sorting
```python
# Sort by column
df.sort_values('age')                    # Ascending
df.sort_values('age', ascending=False)   # Descending
df.sort_values(['city', 'age'])          # Multiple columns

# Sort by index
df.sort_index()
```

### GroupBy Operations
```python
# Basic groupby
df.groupby('city').mean()
df.groupby('city')['age'].mean()

# Multiple aggregations
df.groupby('city').agg({
    'age': ['mean', 'min', 'max'],
    'salary': 'sum'
})

# Custom aggregations
df.groupby('city')['age'].agg(['mean', 'std', 'count'])

# Multiple grouping columns
df.groupby(['city', 'department']).mean()

# Apply custom function
df.groupby('city')['age'].apply(lambda x: x.max() - x.min())
```

### Handling Missing Data
```python
# Detect missing
df.isnull()        # Boolean DataFrame
df.isnull().sum()  # Count per column

# Drop missing
df.dropna()                    # Drop rows with any NaN
df.dropna(how='all')           # Only if all values are NaN
df.dropna(subset=['age'])      # Based on specific columns

# Fill missing
df.fillna(0)                   # Fill with 0
df.fillna(df.mean())           # Fill with mean
df.fillna(method='ffill')      # Forward fill
df.fillna(method='bfill')      # Backward fill

# Interpolate
df.interpolate()
```

### Merging and Joining
```python
df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['B', 'C', 'D'], 'value2': [4, 5, 6]})

# Inner join (default)
pd.merge(df1, df2, on='key')

# Left join
pd.merge(df1, df2, on='key', how='left')

# Right join
pd.merge(df1, df2, on='key', how='right')

# Outer join
pd.merge(df1, df2, on='key', how='outer')

# Multiple keys
pd.merge(df1, df2, on=['key1', 'key2'])
```

### Concatenation
```python
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})

# Vertical (rows)
pd.concat([df1, df2], axis=0)

# Horizontal (columns)
pd.concat([df1, df2], axis=1)

# Ignore index
pd.concat([df1, df2], ignore_index=True)
```

### Pivot Tables
```python
df = pd.DataFrame({
    'date': ['2024-01', '2024-01', '2024-02', '2024-02'],
    'city': ['NYC', 'LA', 'NYC', 'LA'],
    'sales': [100, 200, 150, 250]
})

# Pivot
pivot = df.pivot(index='date', columns='city', values='sales')

# Pivot table with aggregation
pivot_table = df.pivot_table(
    values='sales',
    index='date',
    columns='city',
    aggfunc='sum',
    fill_value=0
)
```

### Apply Functions
```python
# Apply to column
df['age_squared'] = df['age'].apply(lambda x: x**2)

# Apply to row
df['total'] = df.apply(lambda row: row['price'] * row['quantity'], axis=1)

# Apply to entire DataFrame
df_normalized = df.apply(lambda x: (x - x.mean()) / x.std())

# Map (for Series)
df['grade'] = df['score'].map({
    100: 'A',
    90: 'B',
    80: 'C'
})
```

### String Operations
```python
# Access string methods
df['name'].str.lower()
df['name'].str.upper()
df['name'].str.len()
df['name'].str.contains('Alice')
df['name'].str.startswith('A')
df['name'].str.split(' ')
df['name'].str.replace('Alice', 'Alicia')
df['name'].str.strip()  # Remove whitespace
```

### DateTime Operations
```python
df['date'] = pd.to_datetime(df['date'])

# Extract components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek
df['day_name'] = df['date'].dt.day_name()

# Date arithmetic
df['next_week'] = df['date'] + pd.Timedelta(days=7)
df['days_since'] = (pd.Timestamp.now() - df['date']).dt.days
```

### Window Functions
```python
# Rolling mean
df['rolling_mean'] = df['value'].rolling(window=3).mean()

# Cumulative sum
df['cumsum'] = df['value'].cumsum()

# Rank
df['rank'] = df['score'].rank(ascending=False)

# Shift
df['prev_value'] = df['value'].shift(1)   # Previous row
df['next_value'] = df['value'].shift(-1)  # Next row

# Percentage change
df['pct_change'] = df['value'].pct_change()
```

### MultiIndex
```python
# Create MultiIndex
arrays = [
    ['A', 'A', 'B', 'B'],
    ['one', 'two', 'one', 'two']
]
index = pd.MultiIndex.from_arrays(arrays, names=['first', 'second'])
df = pd.DataFrame(np.random.randn(4, 2), index=index, columns=['X', 'Y'])

# Accessing
df.loc['A']
df.loc[('A', 'one')]

# Reset index
df.reset_index()

# Set index
df.set_index(['col1', 'col2'])
```

---

# SQL

## 1. Basic Queries

### SELECT
```sql
-- Select all columns
SELECT * FROM employees;

-- Select specific columns
SELECT first_name, last_name, salary FROM employees;

-- With alias
SELECT first_name AS fname, salary * 12 AS annual_salary
FROM employees;

-- DISTINCT values
SELECT DISTINCT department FROM employees;

-- LIMIT
SELECT * FROM employees LIMIT 10;
```

### WHERE Clause
```sql
-- Comparison operators
SELECT * FROM employees WHERE salary > 50000;
SELECT * FROM employees WHERE department = 'Sales';
SELECT * FROM employees WHERE hire_date >= '2020-01-01';

-- Logical operators
SELECT * FROM employees 
WHERE salary > 50000 AND department = 'Sales';

SELECT * FROM employees 
WHERE department = 'Sales' OR department = 'Marketing';

SELECT * FROM employees WHERE NOT department = 'IT';

-- IN operator
SELECT * FROM employees 
WHERE department IN ('Sales', 'Marketing', 'IT');

-- BETWEEN
SELECT * FROM employees 
WHERE salary BETWEEN 40000 AND 60000;

-- LIKE (pattern matching)
SELECT * FROM employees WHERE first_name LIKE 'J%';  -- Starts with J
SELECT * FROM employees WHERE email LIKE '%@gmail.com';  -- Ends with
SELECT * FROM employees WHERE phone LIKE '555-____';  -- Pattern
```

### ORDER BY
```sql
-- Ascending (default)
SELECT * FROM employees ORDER BY salary;

-- Descending
SELECT * FROM employees ORDER BY salary DESC;

-- Multiple columns
SELECT * FROM employees 
ORDER BY department ASC, salary DESC;
```

### AGGREGATE Functions
```sql
-- COUNT
SELECT COUNT(*) FROM employees;
SELECT COUNT(DISTINCT department) FROM employees;

-- SUM
SELECT SUM(salary) FROM employees;

-- AVG
SELECT AVG(salary) FROM employees;

-- MIN/MAX
SELECT MIN(salary), MAX(salary) FROM employees;

-- Multiple aggregates
SELECT 
    COUNT(*) as num_employees,
    AVG(salary) as avg_salary,
    MIN(salary) as min_salary,
    MAX(salary) as max_salary
FROM employees;
```

---

## 2. GROUP BY and HAVING

### GROUP BY
```sql
-- Group by single column
SELECT department, COUNT(*) as num_employees
FROM employees
GROUP BY department;

-- Group by multiple columns
SELECT department, job_title, AVG(salary) as avg_salary
FROM employees
GROUP BY department, job_title;

-- With aggregations
SELECT 
    department,
    COUNT(*) as num_employees,
    AVG(salary) as avg_salary,
    MAX(salary) as max_salary
FROM employees
GROUP BY department;
```

### HAVING
```sql
-- Filter groups (use HAVING, not WHERE)
SELECT department, AVG(salary) as avg_salary
FROM employees
GROUP BY department
HAVING AVG(salary) > 50000;

-- WHERE filters rows, HAVING filters groups
SELECT department, COUNT(*) as num_employees
FROM employees
WHERE hire_date >= '2020-01-01'
GROUP BY department
HAVING COUNT(*) > 5;
```

---

## 3. JOINS

### Sample Tables
```sql
-- employees table
| emp_id | name    | dept_id |
|--------|---------|---------|
| 1      | Alice   | 10      |
| 2      | Bob     | 20      |
| 3      | Charlie | NULL    |

-- departments table
| dept_id | dept_name  |
|---------|------------|
| 10      | Sales      |
| 20      | Marketing  |
| 30      | IT         |
```

### INNER JOIN
```sql
-- Returns only matching rows
SELECT e.name, d.dept_name
FROM employees e
INNER JOIN departments d ON e.dept_id = d.dept_id;

-- Result:
| name  | dept_name  |
|-------|------------|
| Alice | Sales      |
| Bob   | Marketing  |
```

### LEFT JOIN (LEFT OUTER JOIN)
```sql
-- Returns all rows from left table
SELECT e.name, d.dept_name
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.dept_id;

-- Result:
| name    | dept_name  |
|---------|------------|
| Alice   | Sales      |
| Bob     | Marketing  |
| Charlie | NULL       |
```

### RIGHT JOIN
```sql
-- Returns all rows from right table
SELECT e.name, d.dept_name
FROM employees e
RIGHT JOIN departments d ON e.dept_id = d.dept_id;

-- Result:
| name  | dept_name  |
|-------|------------|
| Alice | Sales      |
| Bob   | Marketing  |
| NULL  | IT         |
```

### FULL OUTER JOIN
```sql
-- Returns all rows from both tables
SELECT e.name, d.dept_name
FROM employees e
FULL OUTER JOIN departments d ON e.dept_id = d.dept_id;

-- Result:
| name    | dept_name  |
|---------|------------|
| Alice   | Sales      |
| Bob     | Marketing  |
| Charlie | NULL       |
| NULL    | IT         |
```

### Multiple Joins
```sql
SELECT 
    e.name,
    d.dept_name,
    p.project_name
FROM employees e
INNER JOIN departments d ON e.dept_id = d.dept_id
LEFT JOIN projects p ON e.emp_id = p.emp_id;
```

---

## 4. Subqueries

### Subquery in WHERE
```sql
-- Find employees with above-average salary
SELECT name, salary
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);

-- Using IN
SELECT name
FROM employees
WHERE dept_id IN (
    SELECT dept_id 
    FROM departments 
    WHERE dept_name IN ('Sales', 'Marketing')
);
```

### Subquery in FROM
```sql
-- Subquery as derived table
SELECT dept, avg_salary
FROM (
    SELECT department as dept, AVG(salary) as avg_salary
    FROM employees
    GROUP BY department
) AS dept_averages
WHERE avg_salary > 50000;
```

### Correlated Subquery
```sql
-- Subquery references outer query
SELECT e1.name, e1.salary
FROM employees e1
WHERE e1.salary > (
    SELECT AVG(e2.salary)
    FROM employees e2
    WHERE e2.department = e1.department
);
```

### EXISTS
```sql
-- Check if subquery returns any rows
SELECT name
FROM employees e
WHERE EXISTS (
    SELECT 1
    FROM orders o
    WHERE o.emp_id = e.emp_id
);
```

---

## 5. Window Functions

### ROW_NUMBER
```sql
-- Assign unique row number
SELECT 
    name,
    department,
    salary,
    ROW_NUMBER() OVER (ORDER BY salary DESC) as row_num
FROM employees;

-- Partition by department
SELECT 
    name,
    department,
    salary,
    ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) as dept_rank
FROM employees;
```

### RANK and DENSE_RANK
```sql
-- RANK: Gaps in ranking for ties
SELECT 
    name,
    salary,
    RANK() OVER (ORDER BY salary DESC) as rank
FROM employees;

-- DENSE_RANK: No gaps
SELECT 
    name,
    salary,
    DENSE_RANK() OVER (ORDER BY salary DESC) as dense_rank
FROM employees;

-- Example:
| salary | RANK | DENSE_RANK |
|--------|------|------------|
| 100    | 1    | 1          |
| 100    | 1    | 1          |
| 90     | 3    | 2          |
| 80     | 4    | 3          |
```

### LAG and LEAD
```sql
-- Access previous row
SELECT 
    date,
    sales,
    LAG(sales, 1) OVER (ORDER BY date) as prev_sales,
    sales - LAG(sales, 1) OVER (ORDER BY date) as sales_diff
FROM daily_sales;

-- Access next row
SELECT 
    date,
    sales,
    LEAD(sales, 1) OVER (ORDER BY date) as next_sales
FROM daily_sales;
```

### Running Totals
```sql
-- Cumulative sum
SELECT 
    date,
    sales,
    SUM(sales) OVER (ORDER BY date) as running_total
FROM daily_sales;

-- Moving average
SELECT 
    date,
    sales,
    AVG(sales) OVER (
        ORDER BY date 
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) as moving_avg_3day
FROM daily_sales;
```

---

## 6. Common Table Expressions (CTEs)

### Basic CTE
```sql
WITH high_earners AS (
    SELECT * FROM employees WHERE salary > 70000
)
SELECT department, COUNT(*) as num_high_earners
FROM high_earners
GROUP BY department;
```

### Multiple CTEs
```sql
WITH 
sales_dept AS (
    SELECT * FROM employees WHERE department = 'Sales'
),
high_performers AS (
    SELECT * FROM sales_dept WHERE salary > 60000
)
SELECT * FROM high_performers;
```

### Recursive CTE
```sql
-- Example: Organization hierarchy
WITH RECURSIVE employee_hierarchy AS (
    -- Base case: top-level managers
    SELECT emp_id, name, manager_id, 1 as level
    FROM employees
    WHERE manager_id IS NULL
    
    UNION ALL
    
    -- Recursive case: employees reporting to previous level
    SELECT e.emp_id, e.name, e.manager_id, eh.level + 1
    FROM employees e
    INNER JOIN employee_hierarchy eh ON e.manager_id = eh.emp_id
)
SELECT * FROM employee_hierarchy;
```

---

## 7. Advanced Topics

### CASE Statements
```sql
SELECT 
    name,
    salary,
    CASE 
        WHEN salary < 40000 THEN 'Low'
        WHEN salary BETWEEN 40000 AND 70000 THEN 'Medium'
        ELSE 'High'
    END as salary_category
FROM employees;
```

### UNION vs UNION ALL
```sql
-- UNION: Removes duplicates
SELECT name FROM employees_2020
UNION
SELECT name FROM employees_2021;

-- UNION ALL: Keeps duplicates (faster)
SELECT name FROM employees_2020
UNION ALL
SELECT name FROM employees_2021;
```

### Date Functions
```sql
-- Current date/time
SELECT CURRENT_DATE, CURRENT_TIMESTAMP;

-- Extract parts
SELECT 
    EXTRACT(YEAR FROM hire_date) as hire_year,
    EXTRACT(MONTH FROM hire_date) as hire_month
FROM employees;

-- Date arithmetic
SELECT hire_date + INTERVAL '1 year' FROM employees;
SELECT DATEDIFF(CURRENT_DATE, hire_date) as days_employed FROM employees;
```

### String Functions
```sql
-- Concatenation
SELECT CONCAT(first_name, ' ', last_name) as full_name FROM employees;

-- Substring
SELECT SUBSTRING(email, 1, POSITION('@' IN email) - 1) as username FROM employees;

-- Upper/Lower
SELECT UPPER(name), LOWER(email) FROM employees;

-- TRIM
SELECT TRIM(name) FROM employees;
```

---

## 8. Performance Optimization

### Indexes
```sql
-- Create index
CREATE INDEX idx_salary ON employees(salary);

-- Composite index
CREATE INDEX idx_dept_salary ON employees(department, salary);

-- Unique index
CREATE UNIQUE INDEX idx_email ON employees(email);
```

### Query Optimization Tips
1. **Use WHERE instead of HAVING when possible**
2. **Avoid SELECT ***, specify columns
3. **Use LIMIT for large tables**
4. **Create indexes on frequently queried columns**
5. **Use EXISTS instead of IN for subqueries**
6. **Avoid functions on indexed columns in WHERE**

---

# MACHINE LEARNING

## 1. Linear Regression

### Concept
Predict continuous target variable using linear relationship with features.

**Equation**:
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε

Or in matrix form: y = Xβ + ε
```

### Cost Function (Mean Squared Error)
```
J(β) = (1/2m) Σ(hβ(x⁽ⁱ⁾) - y⁽ⁱ⁾)²

Where hβ(x) = βᵀx (hypothesis function)
```

### Normal Equation
```
β = (XᵀX)⁻¹Xᵀy
```

**Pros**: Direct solution, no iterations
**Cons**: Computationally expensive for large n (features)

### Gradient Descent
```
Repeat until convergence:
    β := β - α·∇J(β)
    
Where ∇J(β) = (1/m)·Xᵀ(Xβ - y)
```

**Learning rate α**: 
- Too large → doesn't converge
- Too small → slow convergence

### Assumptions
1. **Linearity**: Relationship is linear
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant variance of errors
4. **Normality**: Errors are normally distributed
5. **No multicollinearity**: Features not highly correlated

### Python Implementation
```python
from sklearn.linear_model import LinearRegression

# Train
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# R² score
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
```

---

## 2. Logistic Regression

### Concept
Binary classification using sigmoid function.

**Sigmoid Function**:
```
σ(z) = 1 / (1 + e⁻ᶻ)

where z = βᵀx
```

**Properties**:
- Output between 0 and 1
- Interpretation: probability
- Decision boundary: z = 0

### Cost Function (Log Loss)
```
J(β) = -(1/m) Σ[y⁽ⁱ⁾·log(hβ(x⁽ⁱ⁾)) + (1-y⁽ⁱ⁾)·log(1-hβ(x⁽ⁱ⁾))]
```

**Why not MSE?** 
- MSE is non-convex for logistic regression
- Log loss is convex → guaranteed global minimum

### Gradient Descent
```
∇J(β) = (1/m)·Xᵀ(σ(Xβ) - y)

Update: β := β - α·∇J(β)
```

### Multi-class Classification
**One-vs-Rest (OvR)**: Train k binary classifiers
**Softmax Regression**: Generalization for k classes

```
P(y=k|x) = exp(βₖᵀx) / Σⱼ exp(βⱼᵀx)
```

### Python Implementation
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

# Predict classes
y_pred = model.predict(X_test)

# Predict probabilities
y_proba = model.predict_proba(X_test)
```

---

## 3. Regularization

### Why Regularization?
- Prevent overfitting
- Handle multicollinearity
- Feature selection

### L1 Regularization (Lasso)
```
J(β) = MSE + λΣ|βⱼ|

Penalty: L1 norm
```

**Properties**:
- **Sparse solutions**: Many coefficients → 0
- **Feature selection**: Automatically selects important features
- **Not differentiable** at 0

### L2 Regularization (Ridge)
```
J(β) = MSE + λΣβⱼ²

Penalty: L2 norm
```

**Properties**:
- **Shrinks coefficients** but doesn't zero them
- **Handles multicollinearity** well
- **Differentiable** everywhere

### Elastic Net
```
J(β) = MSE + λ₁Σ|βⱼ| + λ₂Σβⱼ²

Combines L1 and L2
```

**Use when**: Many correlated features

### λ (Lambda) Selection
```python
from sklearn.linear_model import RidgeCV

# Cross-validation to find best λ
model = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100])
model.fit(X_train, y_train)
print("Best alpha:", model.alpha_)
```

### Comparison Table
| Aspect | L1 (Lasso) | L2 (Ridge) | Elastic Net |
|--------|------------|------------|-------------|
| **Sparsity** | Yes | No | Yes |
| **Feature Selection** | Yes | No | Yes |
| **Multicollinearity** | Not robust | Robust | Robust |
| **Computation** | Slower | Faster | Moderate |

---

## 4. Bias-Variance Tradeoff

### Definitions

**Bias**: Error from wrong assumptions
- **High bias** → Underfitting
- Model too simple

**Variance**: Error from sensitivity to training data
- **High variance** → Overfitting  
- Model too complex

### Total Error Decomposition
```
Expected Error = Bias² + Variance + Irreducible Error
```

### Visual Representation
```
High Bias, Low Variance:    Low Bias, High Variance:
   Simple model                Complex model
   Underfitting                Overfitting
        
   🎯 •••                      🎯•  •
                                    •
                                   •
                               
Low Bias, Low Variance (Goal):
   🎯 ••
     ••
```

### How to Address

**High Bias (Underfitting)**:
- Add more features
- Use more complex model
- Reduce regularization (decrease λ)
- Train longer

**High Variance (Overfitting)**:
- Get more training data
- Simplify model
- Increase regularization (increase λ)
- Use dropout (for neural networks)
- Early stopping

### Learning Curves
```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)
)

# Plot
plt.plot(train_sizes, train_scores.mean(axis=1), label='Training')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation')
```

**Interpretation**:
- **High bias**: Both curves plateau at low performance
- **High variance**: Large gap between training and validation

---

## 5. Decision Trees

### How They Work
1. Start with entire dataset
2. Choose best feature to split on
3. Create branches for each value
4. Repeat recursively for each branch
5. Stop when criteria met (max depth, min samples, etc.)

### Splitting Criteria

**Gini Impurity** (Classification):
```
Gini = 1 - Σ(pᵢ)²

where pᵢ = proportion of class i
```

**Example**:
```
Node with [10 positive, 5 negative]:
Gini = 1 - (10/15)² - (5/15)² = 0.444
```

**Pure node** (all same class): Gini = 0
**Most impure** (50-50 split): Gini = 0.5

**Entropy** (Information Gain):
```
Entropy = -Σ(pᵢ·log₂(pᵢ))

Information Gain = Entropy(parent) - Weighted_Avg(Entropy(children))
```

**MSE** (Regression):
```
MSE = (1/n)Σ(yᵢ - ȳ)²
```

### Advantages
- Easy to interpret and visualize
- Non-linear relationships
- No feature scaling needed
- Handles missing values
- Mixed data types (numerical + categorical)

### Disadvantages
- **Prone to overfitting**
- High variance (small data changes → different tree)
- Biased toward features with more levels
- Not good at extrapolation

### Hyperparameters
```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    max_depth=5,              # Maximum tree depth
    min_samples_split=20,     # Min samples to split node
    min_samples_leaf=10,      # Min samples in leaf
    max_features='sqrt',      # Features to consider per split
    criterion='gini'          # Splitting criterion
)
```

---

## 6. Random Forest

### Concept
**Ensemble method** using multiple decision trees.

**Key Ideas**:
1. **Bagging**: Bootstrap Aggregating
2. **Random feature selection**: Each split considers random subset of features

### Algorithm
```
For each tree (n_estimators):
    1. Create bootstrap sample (sample with replacement)
    2. Build decision tree:
       - At each split, consider random subset of features
       - Choose best split from this subset
    3. Grow tree fully (no pruning typically)

Prediction:
    - Classification: Majority vote
    - Regression: Average
```

### Why It Works

**Bagging reduces variance**:
```
Var(Average of n trees) ≈ Var(single tree) / n
```

**Random features** → Decorrelates trees
- Prevents same strong features dominating all trees
- Increases diversity

### Advantages
- **Reduces overfitting** compared to single tree
- **Handles high-dimensional data**
- **Feature importance** built-in
- **Robust to outliers**
- **Parallel training**

### Disadvantages
- Less interpretable than single tree
- Slower prediction than single tree
- Larger memory footprint

### Feature Importance
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Feature importance
importances = model.feature_importances_
for i, imp in enumerate(importances):
    print(f"Feature {i}: {imp:.4f}")
```

**Calculated by**: Mean decrease in impurity across all trees

---

## 7. Gradient Boosting

### Concept
**Sequential ensemble**: Each tree corrects errors of previous trees.

**Key Difference from Random Forest**:
- RF: Trees independent (parallel)
- Boosting: Trees dependent (sequential)

### Algorithm (Simplified)
```
1. Initialize: F₀(x) = average(y)

2. For m = 1 to M (number of trees):
   a. Compute residuals: rᵢ = yᵢ - Fₘ₋₁(xᵢ)
   b. Fit tree hₘ(x) to residuals
   c. Update: Fₘ(x) = Fₘ₋₁(x) + α·hₘ(x)
   
3. Final model: F(x) = Σ α·hₘ(x)
```

**Learning rate α**: Controls contribution of each tree
- Smaller α → more trees needed, but better performance

### Types

**Gradient Boosting (sklearn)**:
```python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)
```

**XGBoost** (Extreme Gradient Boosting):
```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,          # Sample fraction per tree
    colsample_bytree=0.8    # Feature fraction per tree
)
```

**LightGBM** (Light Gradient Boosting):
```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)
```

### Advantages
- **Best performance** on structured data (often)
- **Feature importance**
- **Handles missing values** (XGBoost, LightGBM)

### Disadvantages
- **Prone to overfitting** (careful tuning needed)
- **Sequential** → can't parallelize tree building
- **Sensitive to hyperparameters**

### Key Hyperparameters
- **n_estimators**: Number of trees
- **learning_rate**: Shrinkage factor
- **max_depth**: Tree complexity
- **subsample**: Fraction of samples per tree
- **min_child_weight**: Minimum sum of weights in leaf

---

## 8. Support Vector Machines (SVM)

### Concept
Find hyperplane that maximizes margin between classes.

**Margin**: Distance from hyperplane to nearest data point

### Linear SVM

**Objective**:
```
Minimize: (1/2)||w||²
Subject to: yᵢ(wᵀxᵢ + b) ≥ 1 for all i
```

**Decision boundary**: wᵀx + b = 0
**Margin**: 2/||w||

### Soft Margin SVM
Allow some misclassification with penalty C:
```
Minimize: (1/2)||w||² + C·Σξᵢ

where ξᵢ = slack variables (violation amount)
```

**C parameter**:
- **Large C**: Hard margin, less tolerance → overfitting risk
- **Small C**: Soft margin, more tolerance → underfitting risk

### Kernel Trick
Transform data to higher dimensions where it's linearly separable.

**Common Kernels**:

**Linear**: K(x, x') = xᵀx'

**Polynomial**: K(x, x') = (γxᵀx' + r)ᵈ

**RBF (Gaussian)**: K(x, x') = exp(-γ||x - x'||²)
- Most common
- γ controls influence of single training example

**Sigmoid**: K(x, x') = tanh(γxᵀx' + r)

### Python Implementation
```python
from sklearn.svm import SVC

# Linear kernel
model = SVC(kernel='linear', C=1.0)

# RBF kernel
model = SVC(kernel='rbf', C=1.0, gamma='scale')

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### Advantages
- **Effective in high dimensions**
- **Memory efficient** (uses support vectors only)
- **Versatile** (different kernels)

### Disadvantages
- **Slow for large datasets** (O(n²) to O(n³))
- **Sensitive to feature scaling**
- **No probability estimates** (by default)
- **Hard to interpret**

---

## 9. K-Means Clustering

### Algorithm
```
1. Initialize: Randomly select k centroids

2. Repeat until convergence:
   a. Assignment: Assign each point to nearest centroid
   b. Update: Recalculate centroids as mean of assigned points

3. Stop when centroids don't change (or max iterations)
```

### Distance Metric
**Euclidean Distance**:
```
d(x, μ) = √(Σ(xᵢ - μᵢ)²)
```

### Cost Function (Inertia)
```
J = Σᵢ Σⱼ₌₁ᵏ ||xᵢ - μⱼ||²

where μⱼ = centroid of cluster j
```

### Choosing k

**Elbow Method**:
```python
inertias = []
K = range(1, 11)

for k in K:
    model = KMeans(n_clusters=k)
    model.fit(X)
    inertias.append(model.inertia_)

# Plot and look for "elbow"
plt.plot(K, inertias)
```

**Silhouette Score**:
```python
from sklearn.metrics import silhouette_score

for k in range(2, 11):
    model = KMeans(n_clusters=k)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f"k={k}: {score:.3f}")
```

**Silhouette Score** (per sample):
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))

where:
- a(i) = avg distance to points in same cluster
- b(i) = avg distance to points in nearest other cluster

Range: [-1, 1]
- Close to 1: Well clustered
- Close to 0: On boundary
- Negative: Might be in wrong cluster
```

### K-Means++
Better initialization:
```python
model = KMeans(n_clusters=3, init='k-means++')
```

**Algorithm**:
1. Choose first centroid randomly
2. For each subsequent centroid:
   - Choose point with probability ∝ distance² from nearest centroid
3. Proceed with standard K-means

**Advantage**: Faster convergence, better results

### Limitations
- **Assumes spherical clusters**
- **Sensitive to initialization**
- **Requires k to be specified**
- **Sensitive to outliers**
- **Struggles with different sized/density clusters**

### Alternative: DBSCAN
```python
from sklearn.cluster import DBSCAN

model = DBSCAN(eps=0.5, min_samples=5)
labels = model.fit_predict(X)
```

**Advantages over K-means**:
- No need to specify k
- Finds arbitrarily shaped clusters
- Identifies outliers

---

## 10. Dimensionality Reduction

### Principal Component Analysis (PCA)

**Goal**: Find directions of maximum variance

**Algorithm**:
```
1. Standardize data: X' = (X - μ) / σ

2. Compute covariance matrix: C = (1/n)X'ᵀX'

3. Compute eigenvalues and eigenvectors of C

4. Sort eigenvalues (λ₁ ≥ λ₂ ≥ ... ≥ λₙ)

5. Select top k eigenvectors as principal components

6. Transform data: Z = X'·Vₖ
```

**Variance Explained**:
```
Variance explained by PC_i = λᵢ / Σλⱼ
```

### Python Implementation
```python
from sklearn.decomposition import PCA

# Keep components explaining 95% variance
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X)

print("Original shape:", X.shape)
print("Reduced shape:", X_reduced.shape)
print("Variance explained:", pca.explained_variance_ratio_)

# Or specify number of components
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
```

### When to Use PCA
- **Visualization** (reduce to 2-3 dimensions)
- **Noise reduction**
- **Speed up learning** (fewer features)
- **Multicollinearity** in features

### Limitations
- **Linear transformation** only
- **Assumes** directions of max variance are most important
- **Loses interpretability**

### t-SNE (t-Distributed Stochastic Neighbor Embedding)

**Use**: Visualization (non-linear dimensionality reduction)

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30)
X_embedded = tsne.fit_transform(X)

plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
```

**Key Parameter**: `perplexity` (5-50)
- Roughly: expected number of neighbors
- Smaller → local structure
- Larger → global structure

**Note**: Only for visualization, not for preprocessing!

---

# DEEP LEARNING

## 1. Neural Network Basics

### Perceptron (Single Neuron)
```
Output = Activation(Σ(wᵢxᵢ) + b)

where:
- wᵢ = weights
- xᵢ = inputs
- b = bias
```

### Multi-Layer Perceptron (MLP)

**Architecture**:
```
Input Layer → Hidden Layer(s) → Output Layer
```

**Forward Propagation**:
```
For each layer l:
    z⁽ˡ⁾ = W⁽ˡ⁾·a⁽ˡ⁻¹⁾ + b⁽ˡ⁾
    a⁽ˡ⁾ = g(z⁽ˡ⁾)

where:
- z = weighted sum
- a = activation
- g = activation function
```

### Activation Functions

**Sigmoid**:
```
σ(z) = 1 / (1 + e⁻ᶻ)

Range: (0, 1)
Use: Binary classification (output layer)
Problem: Vanishing gradients
```

**Tanh**:
```
tanh(z) = (e^z - e⁻ᶻ) / (e^z + e⁻ᶻ)

Range: (-1, 1)
Use: Hidden layers (zero-centered)
Problem: Vanishing gradients
```

**ReLU** (Rectified Linear Unit):
```
ReLU(z) = max(0, z)

Range: [0, ∞)
Use: Hidden layers (most common)
Advantages:
  - No vanishing gradient for z > 0
  - Computationally efficient
Problem: Dying ReLU (neurons always output 0)
```

**Leaky ReLU**:
```
LeakyReLU(z) = max(αz, z)  where α = 0.01

Fixes dying ReLU problem
```

**Softmax** (Multi-class output):
```
softmax(zᵢ) = e^zᵢ / Σⱼ e^zⱼ

Properties:
- Output sums to 1
- Interpreted as probabilities
```

### Loss Functions

**Mean Squared Error** (Regression):
```
MSE = (1/n)Σ(ŷᵢ - yᵢ)²
```

**Binary Cross-Entropy** (Binary Classification):
```
BCE = -(1/n)Σ[yᵢ·log(ŷᵢ) + (1-yᵢ)·log(1-ŷᵢ)]
```

**Categorical Cross-Entropy** (Multi-class):
```
CCE = -(1/n)ΣΣ yᵢⱼ·log(ŷᵢⱼ)
```

---

## 2. Backpropagation

### Concept
Calculate gradients using chain rule to update weights.

### Chain Rule
```
∂L/∂w⁽ˡ⁾ = ∂L/∂a⁽ᴸ⁾ · ∂a⁽ᴸ⁾/∂z⁽ᴸ⁾ · ... · ∂z⁽ˡ⁺¹⁾/∂a⁽ˡ⁾ · ∂a⁽ˡ⁾/∂z⁽ˡ⁾ · ∂z⁽ˡ⁾/∂w⁽ˡ⁾
```

### Algorithm
```
1. Forward pass: Compute all activations

2. Compute output error: δ⁽ᴸ⁾ = ∂L/∂z⁽ᴸ⁾

3. Backpropagate error:
   For l = L-1 down to 1:
       δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾)ᵀδ⁽ˡ⁺¹⁾ ⊙ g'(z⁽ˡ⁾)
   
4. Compute gradients:
   ∂L/∂W⁽ˡ⁾ = δ⁽ˡ⁾(a⁽ˡ⁻¹⁾)ᵀ
   ∂L/∂b⁽ˡ⁾ = δ⁽ˡ⁾

5. Update weights:
   W⁽ˡ⁾ := W⁽ˡ⁾ - α·∂L/∂W⁽ˡ⁾
```

---

## 3. Optimization Algorithms

### Gradient Descent Variants

**Batch Gradient Descent**:
```
- Uses entire dataset per update
- Slow for large datasets
- Smooth convergence
```

**Stochastic Gradient Descent (SGD)**:
```
- Uses single sample per update
- Fast but noisy
- Can escape local minima
```

**Mini-Batch Gradient Descent**:
```
- Uses batch of samples (e.g., 32, 64, 128)
- Balance between batch and SGD
- Most commonly used
```

### Advanced Optimizers

**Momentum**:
```
v_t = βv_{t-1} + (1-β)∇L
θ_t = θ_{t-1} - α·v_t

- Accumulates past gradients
- Smooths updates
- β typically 0.9
```

**RMSprop**:
```
s_t = β·s_{t-1} + (1-β)·(∇L)²
θ_t = θ_{t-1} - α·∇L/√(s_t + ε)

- Adapts learning rate per parameter
- Good for non-stationary problems
```

**Adam** (Adaptive Moment Estimation):
```
m_t = β₁·m_{t-1} + (1-β₁)·∇L        # First moment
v_t = β₂·v_{t-1} + (1-β₂)·(∇L)²     # Second moment

m̂_t = m_t/(1-β₁ᵗ)                   # Bias correction
v̂_t = v_t/(1-β₂ᵗ)

θ_t = θ_{t-1} - α·m̂_t/√(v̂_t + ε)

Common values: β₁=0.9, β₂=0.999, ε=10⁻⁸
```

**Adam is most commonly used** - combines benefits of Momentum and RMSprop.

### Learning Rate Scheduling

**Step Decay**:
```python
# Reduce LR every N epochs
lr = initial_lr * (decay_rate)^(epoch // step_size)
```

**Exponential Decay**:
```python
lr = initial_lr * e^(-decay_rate * epoch)
```

**ReduceLROnPlateau**:
```python
from tensorflow.keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,        # Multiply LR by 0.5
    patience=5,        # Wait 5 epochs
    min_lr=1e-7
)
```

---

## 4. Regularization Techniques

### Dropout

**Concept**: Randomly drop neurons during training

**Implementation**:
```
During training:
    - Randomly set neurons to 0 with probability p
    - Scale remaining neurons by 1/(1-p)

During inference:
    - Use all neurons (no dropout)
```

**Why it works**:
- Prevents co-adaptation of neurons
- Forces network to learn redundant representations
- Ensemble effect

**Typical values**: p = 0.2 to 0.5

**Important**: 
```python
# Keras/TensorFlow
model.add(Dropout(0.5))  # Drops 50% of neurons
```

**What happens with stacked dropout?**
```python
# Example: Two dropout layers back-to-back
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dropout(0.5))  # WRONG - redundant!

# Correct:
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))  # RIGHT - after each layer
```

**Stacking dropout layers** (back-to-back without a dense layer between) is redundant and doesn't add benefit.

**Placement matters**:
```python
# Common practice:
Dense → Activation → Dropout → Dense → Activation → Dropout
```

### Batch Normalization

**Concept**: Normalize inputs to each layer

**Algorithm**:
```
For mini-batch B = {x₁, ..., xₘ}:

1. Compute mean: μ_B = (1/m)Σxᵢ
2. Compute variance: σ²_B = (1/m)Σ(xᵢ - μ_B)²
3. Normalize: x̂ᵢ = (xᵢ - μ_B)/√(σ²_B + ε)
4. Scale and shift: yᵢ = γx̂ᵢ + β

where γ, β are learnable parameters
```

**Benefits**:
- Allows higher learning rates
- Reduces sensitivity to initialization
- Acts as regularization
- Speeds up training

**Placement**:
```python
# Before or after activation (both used)
Dense → BatchNorm → Activation  # More common
Dense → Activation → BatchNorm
```

### L1/L2 Regularization
```python
from tensorflow.keras import regularizers

model.add(Dense(
    64,
    kernel_regularizer=regularizers.l2(0.01)  # L2
))
```

### Early Stopping
```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

model.fit(X_train, y_train, 
          validation_data=(X_val, y_val),
          callbacks=[early_stop])
```

---

## 5. Convolutional Neural Networks (CNN)

### Convolutional Layer

**Operation**:
```
For each position (i, j):
    Output[i, j] = Σ Σ Input[i+m, j+n] · Kernel[m, n]
```

**Key Parameters**:
- **Kernel size**: e.g., 3×3, 5×5
- **Stride**: Step size (usually 1)
- **Padding**: Border handling

### Output Size Formula

**CRITICAL FORMULA**:
```
Output_size = ⌊(Input_size - Kernel_size + 2·Padding) / Stride⌋ + 1
```

**Derivation - Why each term?**

**1. (Input_size - Kernel_size)**:
```
- Kernel slides across input
- At edge, kernel extends beyond input
- Number of valid positions = Input_size - Kernel_size + 1
- Example: 5×5 input, 3×3 kernel → 5-3+1 = 3 positions
```

**2. + 2·Padding**:
```
- Padding adds border pixels
- 2· because padding on both sides
- Increases valid positions
- Example: Padding=1 adds 2 pixels → +2 to input
```

**3. / Stride**:
```
- Stride > 1 skips positions
- Reduces output size
- Example: Stride=2 → half the positions
```

**4. + 1**:
```
- Converts from "gaps" to "positions"
- Example: 3 gaps = 4 positions (like fence posts)
```

**Examples**:

```
Example 1: Input=28, Kernel=3, Padding=0, Stride=1
Output = (28 - 3 + 0)/1 + 1 = 26

Example 2: Input=28, Kernel=3, Padding=1, Stride=1
Output = (28 - 3 + 2)/1 + 1 = 28  # Same size!

Example 3: Input=28, Kernel=5, Padding=2, Stride=2
Output = (28 - 5 + 4)/2 + 1 = 14  # Half size
```

### Padding Types

**Valid** (No padding):
- Output size < Input size
- Loses boundary information

**Same** (Padding preserves size):
```
Padding = (Kernel_size - 1) / 2

For 3×3 kernel: Padding = 1
For 5×5 kernel: Padding = 2
```

### Pooling Layers

**Max Pooling**:
```
Takes maximum value in each window
Common: 2×2 with stride 2
Effect: Reduces size by half
```

**Average Pooling**:
```
Takes average value in each window
Less common than max pooling
```

**Why Pooling?**
- Reduces spatial dimensions
- Translation invariance
- Reduces parameters
- Prevents overfitting

### Typical CNN Architecture
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    # Conv Block 1
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    
    # Conv Block 2
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Conv Block 3
    Conv2D(64, (3, 3), activation='relu'),
    
    # Flatten and Dense
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
```

**Layer Progression**:
```
Input: 28×28×1

Conv2D(32, 3×3):     28×28×32  (same padding)
MaxPool(2×2):        14×14×32

Conv2D(64, 3×3):     14×14×64
MaxPool(2×2):        7×7×64

Conv2D(64, 3×3):     7×7×64

Flatten:             3136
Dense(64):           64
Dense(10):           10
```

### Why CNNs for Images?

1. **Parameter Sharing**: Same filter across image
   - Fully connected: millions of parameters
   - CNN: thousands of parameters

2. **Translation Invariance**: Detects features anywhere

3. **Hierarchical Learning**:
   - Early layers: Edges, textures
   - Middle layers: Parts, patterns  
   - Deep layers: Objects

---

## 6. Recurrent Neural Networks (RNN)

### Basic RNN

**Concept**: Process sequential data with hidden state

**Equations**:
```
h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b_h)
y_t = W_hy · h_t + b_y

where:
- h_t = hidden state at time t
- x_t = input at time t
- y_t = output at time t
```

### Vanishing Gradient Problem

**Issue**: Gradients become very small in deep/long sequences

**Why?**
```
∂L/∂h_1 = ∂L/∂h_T · ∂h_T/∂h_{T-1} · ... · ∂h_2/∂h_1

Each term: ∂h_t/∂h_{t-1} = W · diag(tanh'(z))

After T steps: (W · tanh')^T

If ||W|| < 1: Exponential decay → vanishing
If ||W|| > 1: Exponential growth → exploding
```

**Consequence**: Hard to learn long-term dependencies

### Long Short-Term Memory (LSTM)

**Purpose**: Solve vanishing gradient problem

**Key Idea**: Cell state with gates to control information flow

**Gates**:

**1. Forget Gate**:
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)

Decides what to forget from cell state
Output: 0 (forget) to 1 (keep)
```

**2. Input Gate**:
```
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)

i_t: Decides what new info to add
C̃_t: Candidate values to add
```

**3. Update Cell State**:
```
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t

⊙ = element-wise multiplication
```

**4. Output Gate**:
```
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
h_t = o_t ⊙ tanh(C_t)
```

**Why LSTM Solves Vanishing Gradients**:

1. **Additive updates** to cell state:
   ```
   C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
   ```
   Addition (not multiplication) → gradient flows better

2. **Gradient highway**: Cell state provides path for gradients
   ```
   ∂C_t/∂C_{t-1} = f_t  (close to 1 if forget gate open)
   ```

3. **Controlled forgetting**: Gates decide when to forget
   - Prevents irrelevant old info from dominating

### GRU (Gated Recurrent Unit)

**Simpler than LSTM**: Combines forget and input gates

**Gates**:
```
r_t = σ(W_r · [h_{t-1}, x_t])     # Reset gate
z_t = σ(W_z · [h_{t-1}, x_t])     # Update gate

h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

**LSTM vs GRU**:
- GRU: Fewer parameters, faster
- LSTM: More expressive, better for complex tasks
- Both solve vanishing gradient

### Python Implementation
```python
from tensorflow.keras.layers import LSTM, GRU

# LSTM
model.add(LSTM(
    units=128,
    return_sequences=True,  # Return full sequence
    dropout=0.2
))

# GRU
model.add(GRU(
    units=128,
    return_sequences=False  # Return only last output
))
```

---

## 7. Transfer Learning

### Concept
Use pre-trained model as starting point

**Why?**
- Leverage knowledge from large datasets
- Faster training
- Better performance with less data

### Approaches

**1. Feature Extraction**:
```python
# Load pre-trained model
base_model = VGG16(weights='imagenet', include_top=False)

# Freeze base model
base_model.trainable = False

# Add custom layers
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(num_classes, activation='softmax')
])
```

**2. Fine-Tuning**:
```python
# Unfreeze some layers
base_model.trainable = True

# Freeze only early layers
for layer in base_model.layers[:15]:
    layer.trainable = False

# Train with low learning rate
model.compile(optimizer=Adam(lr=1e-5), ...)
```

### Common Pre-trained Models
- **VGG16/VGG19**: Simple, effective
- **ResNet**: Deeper with skip connections
- **Inception**: Multi-scale features
- **MobileNet**: Lightweight for mobile
- **EfficientNet**: State-of-the-art efficiency

---

# MODEL EVALUATION

## 1. Classification Metrics

### Confusion Matrix
```
              Predicted
              Pos    Neg
Actual  Pos   TP     FN
        Neg   FP     TN
```

### Accuracy
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**When to use**: Balanced classes
**Problem**: Misleading for imbalanced data

**Example**: 95% spam → predict all spam → 95% accuracy!

### Precision
```
Precision = TP / (TP + FP)

"Of all predicted positive, how many are actually positive?"
```

**High precision**: Few false positives
**Use when**: Cost of FP is high (e.g., spam filter)

### Recall (Sensitivity, True Positive Rate)
```
Recall = TP / (TP + FN)

"Of all actual positive, how many did we find?"
```

**High recall**: Few false negatives
**Use when**: Cost of FN is high (e.g., disease detection)

### F1-Score
```
F1 = 2 · (Precision · Recall) / (Precision + Recall)

Harmonic mean of precision and recall
```

**Use when**: Balance between precision and recall needed

### Precision-Recall Tradeoff
```
High threshold → High precision, Low recall
Low threshold → Low precision, High recall
```

### ROC Curve
```
ROC = Plot of TPR vs FPR at different thresholds

TPR (Recall) = TP / (TP + FN)
FPR = FP / (FP + TN)
```

### AUC (Area Under ROC Curve)
```
AUC = 1.0: Perfect classifier
AUC = 0.5: Random guessing
AUC < 0.5: Worse than random
```

**Interpretation**: Probability that model ranks random positive example higher than random negative example

**When to use**:
- **Imbalanced data**: Better than accuracy
- **Binary classification**: Clear threshold-agnostic metric
- **Compare models**: Single number comparison

### ROC-AUC vs Precision-Recall

**Use ROC-AUC when**:
- Balanced classes
- Care about both classes equally

**Use Precision-Recall when**:
- Imbalanced classes (rare positive class)
- Care more about positive class
- Example: Disease detection (1% have disease)

### Python Implementation
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

# Basic metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# ROC-AUC (needs probabilities)
auc = roc_auc_score(y_true, y_proba)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Comprehensive report
print(classification_report(y_true, y_pred))
```

---

## 2. Regression Metrics

### Mean Absolute Error (MAE)
```
MAE = (1/n)Σ|yᵢ - ŷᵢ|

Average absolute difference
```

**Properties**:
- Same units as target
- Less sensitive to outliers
- Easy to interpret

### Mean Squared Error (MSE)
```
MSE = (1/n)Σ(yᵢ - ŷᵢ)²

Average squared difference
```

**Properties**:
- Penalizes large errors more
- Not in same units (squared)
- Sensitive to outliers

### Root Mean Squared Error (RMSE)
```
RMSE = √MSE = √[(1/n)Σ(yᵢ - ŷᵢ)²]

Same units as target
```

### R² Score (Coefficient of Determination)
```
R² = 1 - (SS_res / SS_tot)

where:
SS_res = Σ(yᵢ - ŷᵢ)²  (residual sum of squares)
SS_tot = Σ(yᵢ - ȳ)²   (total sum of squares)
```

**Interpretation**:
- R² = 1: Perfect predictions
- R² = 0: As good as predicting mean
- R² < 0: Worse than predicting mean

**Percentage of variance explained**: R² = 0.85 means model explains 85% of variance

### Python Implementation
```python
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
r2 = r2_score(y_true, y_pred)
```

---

## 3. Cross-Validation

### K-Fold Cross-Validation
```
1. Split data into k folds
2. For each fold:
   - Use as validation set
   - Train on remaining k-1 folds
   - Evaluate
3. Average k scores
```

**Benefits**:
- Better estimate of model performance
- Uses all data for training and validation
- Reduces variance

### Python Implementation
```python
from sklearn.model_selection import cross_val_score

# 5-fold CV
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print(f"Scores: {scores}")
print(f"Mean: {scores.mean():.3f}")
print(f"Std: {scores.std():.3f}")
```

### Stratified K-Fold
```python
from sklearn.model_selection import StratifiedKFold

# Preserves class distribution in each fold
skf = StratifiedKFold(n_splits=5)

for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
```

**Use when**: Imbalanced classes

### Leave-One-Out Cross-Validation (LOOCV)
```
k = n (number of samples)
Each sample used once as validation
```

**Pros**: Maximum use of data
**Cons**: Computationally expensive

---

# PRACTICE PROBLEMS

## Statistics Problems

### Problem 1: Hypothesis Testing
```
A company claims average customer satisfaction is 8/10.
Sample of 100 customers: mean = 7.5, std = 1.2

Test at α = 0.05 if claim is valid.

Solution:
H₀: μ = 8
H₁: μ ≠ 8

Test statistic: z = (x̄ - μ) / (σ/√n)
                  = (7.5 - 8) / (1.2/√100)
                  = -0.5 / 0.12
                  = -4.17

Critical value: ±1.96 (two-tailed, α=0.05)

Since |-4.17| > 1.96, reject H₀
Conclusion: Claim is not supported
```

### Problem 2: Bayes' Theorem
```
Disease prevalence: 1%
Test sensitivity (P(+|D)): 95%
Test specificity (P(-|¬D)): 90%

Person tests positive. Probability of having disease?

Solution:
P(D|+) = P(+|D)·P(D) / P(+)

P(+) = P(+|D)·P(D) + P(+|¬D)·P(¬D)
     = 0.95·0.01 + 0.10·0.99
     = 0.0095 + 0.099
     = 0.1085

P(D|+) = 0.95·0.01 / 0.1085
       = 0.0876 ≈ 8.8%
```

---

## Machine Learning Problems

### Problem 3: Bias-Variance
```
You have:
- Training accuracy: 98%
- Validation accuracy: 75%

What's the problem and how to fix?

Solution:
High variance (overfitting)
- Gap between train and val is large
- Model memorizes training data

Fixes:
1. Get more training data
2. Reduce model complexity
3. Add regularization (L1/L2)
4. Use dropout
5. Early stopping
6. Data augmentation
```

### Problem 4: Choosing k in K-Means
```
Given dataset, how to choose k?

Solution:
1. Elbow method:
   - Plot inertia vs k
   - Look for "elbow" point

2. Silhouette score:
   - Compute for different k
   - Choose k with highest score

3. Domain knowledge:
   - Sometimes k is known (e.g., customer segments)

4. Try multiple k and evaluate:
   - Business metrics
   - Interpretability
```

---

## Python/Pandas Problems

### Problem 5: GroupBy Aggregation
```python
# Given sales data, find top 3 products by revenue per region

import pandas as pd

df = pd.DataFrame({
    'region': ['East', 'East', 'West', 'West', 'East'],
    'product': ['A', 'B', 'A', 'C', 'A'],
    'quantity': [10, 5, 8, 12, 15],
    'price': [100, 200, 100, 150, 100]
})

# Calculate revenue
df['revenue'] = df['quantity'] * df['price']

# Top 3 products per region
result = (df.groupby(['region', 'product'])['revenue']
          .sum()
          .reset_index()
          .sort_values(['region', 'revenue'], ascending=[True, False])
          .groupby('region')
          .head(3))

print(result)
```

### Problem 6: Missing Data
```python
# Handle missing data appropriately

import pandas as pd
import numpy as np

df = pd.DataFrame({
    'age': [25, np.nan, 30, 35, np.nan],
    'income': [50000, 60000, np.nan, 70000, 55000],
    'category': ['A', 'B', np.nan, 'A', 'B']
})

# Numerical: Fill with mean/median
df['age'].fillna(df['age'].median(), inplace=True)
df['income'].fillna(df['income'].mean(), inplace=True)

# Categorical: Fill with mode or 'Unknown'
df['category'].fillna(df['category'].mode()[0], inplace=True)

# Or drop rows with any missing
df_clean = df.dropna()
```

---

## SQL Problems

### Problem 7: Find Nth Highest Salary
```sql
-- Find 3rd highest salary

SELECT DISTINCT salary
FROM employees
ORDER BY salary DESC
LIMIT 1 OFFSET 2;  -- Skip top 2, get next

-- Or using window function
SELECT salary
FROM (
    SELECT 
        salary,
        DENSE_RANK() OVER (ORDER BY salary DESC) as rank
    FROM employees
) ranked
WHERE rank = 3;
```

### Problem 8: Self Join
```sql
-- Find employees earning more than their manager

SELECT 
    e.name as employee,
    e.salary as emp_salary,
    m.name as manager,
    m.salary as mgr_salary
FROM employees e
INNER JOIN employees m ON e.manager_id = m.emp_id
WHERE e.salary > m.salary;
```

---

## Deep Learning Problems

### Problem 9: CNN Output Size
```
Input: 224×224×3 image
Conv1: 64 filters, 7×7 kernel, stride=2, padding=3
MaxPool: 3×3, stride=2

What's output size after MaxPool?

Solution:
After Conv1:
Output = (224 - 7 + 2×3)/2 + 1
       = (224 - 7 + 6)/2 + 1
       = 223/2 + 1
       = 111.5 + 1 = 112
Size: 112×112×64

After MaxPool:
Output = (112 - 3)/2 + 1
       = 109/2 + 1
       = 54.5 + 1 = 55
Size: 55×55×64
```

### Problem 10: Dropout Placement
```
Which is better?

A) Dense(128) → Dropout(0.5) → Dropout(0.3) → Dense(64)
B) Dense(128) → Dropout(0.5) → Dense(64) → Dropout(0.3)

Solution: B is better

Reason:
- Stacking dropout layers (A) is redundant
- Dropout should be between dense layers
- Each dropout after a processing layer makes sense
- Back-to-back dropout wastes computation
```

---

# INTERVIEW TIPS

## 1. Problem-Solving Approach

### For Coding Questions:
```
1. Clarify the problem
   - Ask about edge cases
   - Confirm input/output format

2. Think out loud
   - Explain your approach
   - Discuss tradeoffs

3. Start simple
   - Brute force first
   - Then optimize

4. Test your code
   - Walk through examples
   - Consider edge cases

5. Analyze complexity
   - Time: O(?)
   - Space: O(?)
```

### For ML Questions:
```
1. Understand the problem
   - Supervised vs unsupervised?
   - Classification vs regression?
   - What's the metric?

2. Discuss data
   - Size of dataset?
   - Features available?
   - Label quality?

3. Propose approach
   - Start with baseline
   - Suggest improvements
   - Explain reasoning

4. Address challenges
   - Imbalanced data?
   - Missing values?
   - Scaling needed?

5. Discuss evaluation
   - How to validate?
   - What metric to use?
   - How to tune?
```

## 2. Common Mistakes to Avoid

1. **Not asking clarifying questions**
2. **Jumping to code without planning**
3. **Ignoring edge cases**
4. **Not explaining your thought process**
5. **Giving up too quickly**
6. **Not testing your solution**
7. **Being dishonest about what you know**

## 3. Resources for Practice

**Coding**:
- LeetCode (focus on easy-medium)
- HackerRank
- Company-tagged problems

**ML Concepts**:
- Andrew Ng's Machine Learning course
- StatQuest YouTube channel
- Hands-on Kaggle competitions

**Statistics**:
- Khan Academy Statistics
- Practice problems online

**SQL**:
- Mode Analytics SQL tutorial
- LeetCode SQL problems
- StrataScratch

---

**Good luck with your interview! 🚀**

Remember: Interviewers want to see how you think, not just whether you know the answer. Communicate clearly, ask questions, and show your problem-solving process!
