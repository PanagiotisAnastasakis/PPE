# Probabilistic Predictive Elicitation (PPE)

This repository contains a Python implementation of **Probabilistic Predictive Elicitation (PPE)**, following [“Flexible Prior Elicitation via the Prior Predictive Distribution”](https://proceedings.mlr.press/v124/hartmann20a.html) by Hartmann et al. The aim is to calibrate prior-distribution hyperparameters using an expert's probabilistic judgements about a model's target variable.

The implementation offers the following functionality:

- Flexible target variables: Supports both discrete and continuous target variables. Expert judgements can be specified as probabilities assigned to individual classes or to intervals of the target space, respectively.
- Multiple covariate settings: Allows expert judgements to be incorporated across multiple target instances, each potentially associated with different covariate values and target-space partitions.
- Gradient-based optimization: Implements gradient-based optimization for models where prior predictive probabilities can be evaluated in closed form. JAX is used for automatic differentiation to calculate the gradients of the objective function with respect to the model hyperparameters.
- Bayesian optimization: Provides an alternative for models whose prior predictive distributions are not available in closed form. Prior predictive probabilities are approximated from samples generated from the model with PyMC and Bayesian optimization is employed to estimate the hyperparameters using Ax.

## Repository Structure

    .
    ├── ppe/
    │   ├── __init__.py
    │   ├── bayesian_optimization.py
    │   ├── computing_probabilities.py
    │   ├── dirichlet.py
    │   ├── gradient_optimization.py
    │   ├── optimization_loop.py
    │   └── stochastic_optimization.py
    ├── tests/
    │   ├── Tests_Bayesian_Optimization.ipynb
    │   ├── Tests_dirichlet_probabilities.ipynb
    │   ├── Tests_optimization.ipynb
    │   ├── optimization_normflow_prior.py
    │   ├── optimization_simple_gaussian.py
    │   ├── stochastic_gradient_bernoulli.py
    │   └── stochastic_gradient_continous.py
    ├── BO_simulations/
    │   ├── functions.py
    │   ├── gaussian_models.py
    │   ├── linear_models.py
    │   ├── logistic_models.py
    │   └── simulations.ipynb
    ├── requirements.txt
    └── PPE_final_report.pdf


## Method overview

The prior distribution plays a crucial role in Bayesian inference. It encapsulates information about an unknown parameter $\boldsymbol{\theta}$, which is then integrated with the probability distribution of new data $Y$, leading to the posterior distribution. Specifying a suitable prior distribution can be challenging, especially when there is lack of prior information about the parameters of the model. One way to obtain informed priors is by deriving information about the parameters from an expert, which can then be turned into a prior distribution. This approach, however, has the downside that the expert may be unfamiliar with statistical measures used in probabilistic distributions, thus the elicitation process can lead to unreasonable priors.

Motivated by this limitation, Probabilistic Predictive Elicitation (PPE) was introduced as an alternative methodology for capturing expert knowledge and incorporating it into prior distributions. The main idea is that instead of asking questions about the parameters, the modeller can extract probabilistic judgements from an expert about the target, which can then be used to obtain the hyperparameters of the prior distribution.

Suppose an expert provides probabilities describing a target variable Y. For a **discrete** target, the expert's input is probabilities about each class. For a **continuous** target, the probabilities are for intervals (partitions) of the target range. Let $\boldsymbol{q}$ be the expert's probability vector and $p(\boldsymbol{\lambda})$ the corresponding probabilities implied by the model's prior predictive distribution, whose hyperparameters are $\boldsymbol{\lambda}$. PPE compares these vectors using a Dirichlet log-likelihood, which is maximized with respect to the hyperparameters, making the prior predictive probabilities compatible with the expert's judgements.

In most cases, the expert will provide data about multiple instances of the target $Y_1, ..., Y_J$, usually under the presence of covariate sets $\boldsymbol{X}_1, ..., \boldsymbol{X}_J$. Each covariate setting is represented by separate probability vectors. The Dirichlet likelihood is then summed over every setting and the optimal hyperparameters are determined by maximizing this joint log-likelihood.


## Optimization

Optimizing the Dirichlet log-likelihood is relatively straightforward when the prior predictive distribution is available in closed form. Then, the chain rule can be used to derive the gradient of the objective function with respect to $\boldsymbol{\lambda}$, allowing gradient-based optimization methods to be employed. However, in most practical applications, the prior predictive distribution is not available in closed form, making the analytical computation of the gradient difficult or infeasible. In such cases, gradient-based optimization may still be possible through stochastic gradient estimation. Alternatively, general-purpose global optimization methods can be employed. In this work, Bayesian optimization was chosen as it is well suited to optimizing computationally expensive black-box objective functions. Furthermore, its probabilistic formulation is conceptually aligned with the Bayesian framework underlying PPE.


## Implementation

This repository contains a Python framework for PPE that accommodates both discrete and continuous target variables and allows expert judgements to be incorporated across multiple covariate settings. Two complementary optimization approaches are implemented. First, gradient-based optimization is used when prior predictive probabilities can be evaluated in closed form, with JAX providing automatic differentiation to compute the required gradients. Second, Bayesian optimization is employed as a simulation-based alternative when the prior predictive distribution is not available in closed form. In this case, prior predictive probabilities are approximated through sampling from the model and the resulting objective function is optimized using the Ax framework.


## Installation

Create an isolated Python environment and install the declared requirements:

    git clone https://github.com/PanagiotisAnastasakis/ppe.git
    cd ppe
    python -m venv .venv
    # macOS / Linux:
    source .venv/bin/activate
    # Windows PowerShell:
    # .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt


## How to Run

To run PPE, the user needs to provide the following inputs:

1. The covariates $\boldsymbol{X}_1, ..., \boldsymbol{X}_J$, if any, for each instance $Y_1, ..., Y_J$ of the target.
2. The expert's probabilistic judgements $\boldsymbol{q}_1, ..., \boldsymbol{q}_J$ for each observation.
3. The partition of the target space for which each $\boldsymbol{q}_j$ is given. Different observations may correspond to different partitions.
4. Either the prior predictive distribution in closed form or a probabilistic model from which prior predictive samples can be generated.

### 1. The Prior Predictive Distribution is Available in Closed-Form

First, we consider models for which the prior predictive distribution can be obtained analytically. In this case, the user must define a function that evaluates the prior predictive probabilities for the specified target-space partitions. The function should accept the following three arguments:

`partition`: An array specifying the target-space partition for one target instance $Y_j$.

`lambda`: An array containing the model hyperparameters (\boldsymbol{\lambda}) to be estimated.

`X`: An array containing the covariate values associated with the given target instance. If the model does not include covariates, this argument can be omitted.

The function should return the prior predictive probabilities corresponding to the specified partitions, given the hyperparameter values and covariate settings.

```
def prior_predictive_distribution(
    partition,
    lambda,
    X):
  
  # Extract model hyperparameters
  # Use the covariate values, if applicable
  # Calculate prior predictive probabilities for each partition
  
  return(probabilities)

```

The user must then initialize the `optimize_ppe` class, implemented in `ppe/gradient_optimization.py`:

```
from ppe.gradient_optimization import optimize_ppe

opt_class = optimize_ppe(
    alpha,
    J,
    ppd
)
```

Here, `alpha` determines the concentration of the Dirichlet likelihood, which can be interpreted as a quantification of uncertainty of how well we can represent the expert’s probabilistic judgements with the prior predictive probability. It can be either set to a fixed value, or assigned to `None`, in which case it is treated as an additional hyperparameter. `J` specifies the number of target instances for which expert judgements are available and `ppd` links the optimization procedure to the user-defined `prior_predictive_distribution` function for evaluating prior predictive probabilities.

After initializing the `optimize_ppe` class, the user can estimate the model hyperparameters by calling its `gradient_descent` method:

```
lam_new, lik_progression, grad_progression = opt_class.gradient_descent(
    partitions,
    expert_probs,
    lam_0,
    iters,
    step_size,
    tol,
    total_covariates
)
```

The method takes the following inputs:

`partitions`: A list of arrays of partitions corresponding to each target instance $Y_j$.
`expert_probs`: A list of arrays of expert-provided probabilities for each target instance $Y_j$ and corresponding partition.
`lam_0`: An array of initial values for the hyperparameters.
`iters`: The maximum number of gradient descent iterations.
`step_size`: The learning rate, which controls the magnitude of the updates to the hyperparameters at each iteration.
`tol`: The convergence tolerance used as a stopping criterion.
`total_covariates`: A list of arrays of covariate values associated with the different target instances.

The method returns three objects:

`lam_new`: An array of the estimated hyperparameter vector obtained through optimization.

`lik_progression`: An array capturing the value of the objective function over the optimization iterations.

`grad_progression`: An array of the gradient values over optimization iterations.

The likelihood and gradient values recorded across iterations can be used to monitor the convergence of the optimization procedure.

The estimated hyperparameters, stored in `lam_new`, represent the values that optimize the agreement between the model-implied prior predictive probabilities and the expert-provided judgements, according to the specified Dirichlet likelihood. The corresponding prior predictive probabilities can subsequently be obtained using the user-defined `prior_predictive_distribution` function.

If the Dirichlet concentration parameter `alpha` was not specified during the initialization of `optimize_ppe`, its optimized value can be returned by calling the `get_alpha` method:

```
opt_class.get_alpha(
    partitions,
    lam_new,
    expert_probs,
    total_covariates
)
```

Comprehensive examples of PPE in this scenario can be found in `tests/Tests_optimization.ipynb`.


### 2. The Prior Predictive Distribution is Not Available in Closed-Form

In most practical applications, the prior predictive distribution cannot be derived analytically. PPE can still be applied by approximating the prior predictive probabilities using samples generated from the model's prior predictive distribution. The implementation provides the `Bayesian_Optimization` class, defined in `ppe/bayesian_optimization.py`, which uses Bayesian optimization to estimate the model hyperparameters.

Unlike the previous approach, the user does not need to define a function that directly evaluates the prior predictive probabilities. Instead, they must provide a probabilistic model as a function, from which prior predictive samples can be generated. The function is assumed to be implemented using PyMC and accepts two arguments:

`lam`: A dictionary containing the model hyperparameters, with parameter names as keys and their corresponding values.
`n_samples`: The number of prior predictive samples to generate.

A simple example is a model where $Y \sim \mathcal{N}(\theta, \sigma)$, with $\theta \sim \mathcal{N}(\mu_1, \sigma_1)$. The hyperparameter vector is $\[\sigma, \mu_1, \sigma_1\]$, according to which we define the probabilistic model:

```
import pymc as pm

def gaussian_model(lam, n_samples):

    gaussian_model = pm.Model()

    with gaussian_model:

        theta = pm.Normal("theta", mu=lam["mu_1"], sigma=lam["sigma_1"])

        Y_obs = pm.Normal("Y_obs", mu=theta, sigma=lam["sigma"], observed=1.)

    with gaussian_model:

        idata = pm.sample_prior_predictive(draws = n_samples)

    return idata
```


The user then initializes the `Bayesian_Optimization` class, implemented in `ppe/bayesian_optimization.py`:

```
from ppe.bayesian_optimization import Bayesian_Optimization

BO = Bayesian_Optimization(
    pymc_sampling_func,
    J,
    alpha,
    target_type,
    target_samples
)
```

The initialization requires the following arguments:

`pymc_sampling_func`: The user-defined PyMC function that generates prior predictive samples.
`J`: The number of target instances for which expert judgements are provided.
`alpha`: The Dirichlet concentration parameter. It can be fixed to a specified value or set to `None` if it is to be estimated as part of the optimization.
`target_type`: Specifies whether the target variable is `"discrete"` or `"continuous"`.
`target_samples`: The number of prior predictive samples used to approximate the prior predictive probabilities.

The user can run the optimization procedure by calling `optimize_hyperparams`:

```
best_params = BO.optimize_hyperparams(
    param_names,
    param_types,
    param_bounds,
    param_expected_vals,
    param_weights,
    partitions,
    expert_probs,
    n_trials
)
```

The method takes the following arguments:

`param_names`: List of names of the model hyperparameters and, if applicable, the Dirichlet concentration parameter `alpha` to be optimized.
`param_types`: List describing the type of search for each hyperparameter. It is either `"range"` if the search space is a pre-defined range, or `"choice"` if the search space is constrained to specific values.
`param_bounds`: List of arrays for either lower and upper bounds defining the search space for each parameter (if `param_types == "range"`), or specific values (if `param_types == "search"`).
`param_expected_vals`: A list of the expert's estimates for the hyperparameters. It also accepts `None` if no such input can be given.
`param_weights`: List of weights associated with the expected parameter values. The weights must range from 0 to 1.
`partitions`: A list of arrays of partitions corresponding to each target instance $Y_j$. If the target is discrete, then it is an array of all possible values.
`expert_probs`: A list of arrays of expert-provided probabilities for each target instance $Y_j$ and corresponding partition.
`n_trials`: The number of Bayesian optimization trials to perform.

The arguments `param_expected_vals` and `param_weights` correspond to an additional functionality of this implementation that supports the use of hyperpriors for the hyperparameters. Details can be found in the report.

During optimization, the prior predictive samples generated by the user-defined PyMC model are used to approximate the probabilities implied by each candidate set of hyperparameters. These probabilities are then compared with the expert judgements through the Dirichlet likelihood. Bayesian optimization uses the resulting objective-function to guide the search towards promising hyperparameter values.

The procedure returns `best_params`, containing the parameter values identified by the optimization procedure.

After the optimized hyperparameters have been obtained, the corresponding prior predictive probabilities can be computed:

```
BO.get_model_probs(best_params, partitions)
```

Comprehensive examples of PPE with Bayesian Optimization can be found in `tests/Tests_Bayesian_Optimization.ipynb`.



## Stochastic Gradients

The use of stochastic gradients was also investigated as a potential alternative for cases where the prior predictive distribution is not available in closed form. In this setting, gradients can be estimated using samples from the prior predictive distribution, potentially allowing gradient-based optimization without requiring an analytical expression for the objective function. However, this approach was not developed into a general optimization procedure within the current implementation. Instead, the repository includes experimental utilities for estimating stochastic gradients in specific examples, namely Bernoulli and continuous-target models. These experiments provide a basis for further investigation into the applicability of stochastic-gradient methods to PPE.


## Reference

Hartmann, M., et al. [Flexible Prior Elicitation via the Prior Predictive Distribution](https://proceedings.mlr.press/v124/hartmann20a.html). Proceedings of Machine Learning Research, 2020.

---
