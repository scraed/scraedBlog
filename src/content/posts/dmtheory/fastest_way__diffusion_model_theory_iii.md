---
title: The Fastest Way to Diffusion Model Theory - III
published: 2025-06-23
tags: [Diffusion Model, Theory]
category: Diffusion Model Theory
draft: false
---

:::note[Recap]
[Previous section](../fastest_way__diffusion_model_theory_ii/) introduced **Forward Process** and **Backward Process** of Denoising Diffusion Probabilistic Model (DDPM). 

**Forward Process** 

$$
d \mathbf{x}_t = - \frac{1}{2} \mathbf{x}_t dt + d\mathbf{W}_t, \label{Forward Process}
$$

where $t \in [0,T]$ is the forward diffusion time. This process describes a gradual noising operation that transforms clean images into Gaussian noise.

$$
d\mathbf{x}_{t'} = \left( \frac{1}{2} \mathbf{x}_{t'}+ \mathbf{s}(\mathbf{x}_{t'}, T-t') \right) dt' + d\mathbf{W}_{t'}, \label{Backward Process}
$$

where $t' = T - t$ is the backward diffusion time, $\mathbf{s}(\mathbf{x}, t) = \nabla_{\mathbf{x}} \log p_t(\mathbf{x})$ is the score function of the density of $\mathbf{x}_{t}$ in the forward process.

:::

In this section, we will show how to train a neural network that models the score function $\mathbf{s}(\mathbf{x}, t)$.

**Prerequisites**: Calculus.

# Implementation of the Denoising Diffusion Probabilistic Model (DDPM)


## Numerical Implementation of the Forward Process

To numerically simulate the forward diffusion process, we divide the time range $[0, T]$ into intervals of length $\beta_i$, where $i = 1, \ldots, n$. We denote the intermediate times as $t_i = \sum_{j=0}^{i-1} \beta_j$.

The vanilla discretization of the $\ref{Forward Process}$ is given by:

$$
\mathbf{x}_{i} = \mathbf{x}_{i-1} - \frac{1}{2} \mathbf{x}_{i-1} \beta_{i-1} + \sqrt{\beta_{i-1}} \boldsymbol{\epsilon}_{i-1},
$$

where we approximate $d\mathbf{W}_t$ as $\sqrt{dt} \, \boldsymbol{\epsilon}_{i}$, $\boldsymbol{\epsilon}_{i}$ is standard Gaussian random variable. (refer to the [previous section](../fastest_way__diffusion_model_theory_i/)).

A more subtle but equivalent implementation is the variance-preserving (VP) form [^Song2020ScoreBasedGM]:

$$
\mathbf{x}_{i} =  \sqrt{1-\beta_{i-1}} \mathbf{x}_{i-1}  + \sqrt{\beta_{i-1}}\boldsymbol{\epsilon}_{i-1},
$$

This formulation ensures that if $\mathbf{x}_{0}$ is initialized with unit variance, then the variance of $\mathbf{x}_{i}$ remains equal to 1. It gradually adds a small amount of Gaussian noise to the image at each time step $i$, gradually contaminating the image until $\mathbf{x}_n \sim \mathcal{N}(\mathbf{0},I)$.

:::warning
Note that our interpretation of $\beta$ differs from that in [^Song2020ScoreBasedGM], treating $\beta$ as a varying time-step size to solve the autonomous SDE (1.5 OU process noise) instead of a time-dependent SDE. Our interpretation greatly simplifies future analysis, but it holds only if every $\beta_i$ is sufficiently small. 
:::


Instead of expressing the iterative relationship between $\mathbf{x}_{i}$ and $\mathbf{x}_{i-1}$, we can directly represent the dependency of $\mathbf{x}_{i}$ on $\mathbf{x}_{0}$ using the following forward relation:

$$
\mathbf{x}_{i} = \sqrt{\bar{\alpha}_i} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_i} \bar{\boldsymbol{\epsilon}}_i; \quad 1 \leq i \leq n, \label{discrete forward diffusion}
$$

where $\bar{\alpha}_i = \prod_{j=0}^{i-1} (1 - \beta_j)$ denotes the contamination weight, and $\bar{\boldsymbol{\epsilon}}_i$ represents standard Gaussian noise.

:::tip
An useful property we shall exploit later is that for **infinitesimal** time steps $\beta$, the contamination weight $\bar{\alpha}_i$ is the exponential of the diffusion time $t_i$

$$
\lim_{\max_j \beta_j \xrightarrow[]{}0} \bar{\alpha}_i  \xrightarrow[]{} e^{-t_i}.
$$
:::

## Numerical Implementation of the Backward Process

The **backward diffusion process** is used to sample from the DDPM by removing the noise of an image step by step. It is the time reversed version of the OU process, starting at $x_{0'} \sim \mathcal{N}(\mathbf{x}|\mathbf{0}, I)$, using the reverse of the OU process (1.5 reverse diffusion process). 

The vanilla discretization of the $\ref{Backward Process}$ is given by:

$$
\mathbf{x}_{i'+1} = (1 + \frac{1}{2} \beta_{n-i'}) \mathbf{x}_{i'} + \mathbf{s}(\mathbf{x}_{i'}, T-t'_{i'}) \beta_{n-i'} + \sqrt{\beta_{n-i'}}\boldsymbol{\epsilon}_{i'},
$$

where $i' = 0, \ldots, n$ represents the backward time step, and $\mathbf{x}_{i'}$ is the image at the $i'$th step with time $t_{i'}' = \sum_{j=0}^{i'-1} \beta_{n-1-j} = T - t_{n-i'}$. 

A more common discretization is:

$$
\mathbf{x}_{i'+1} = \frac{\mathbf{x}_{i'} + \mathbf{s}(\mathbf{x}_{i'}, T-t'_{i'}) \beta_{n-i'}}{\sqrt{1-\beta_{n-i'}}} + \sqrt{\beta_{n-i'}}\boldsymbol{\epsilon}_{i'}, \label{discrete backward process}
$$

This formulation is equivalent to the vanilla discretization when $\beta_i$ is small. The score function $\mathbf{s}(\mathbf{x}_{i'}, T-t'_{i'})$ is typically modeled by a neural network trained using a denoising objective.

## Training the Score Function

Training the score function requires a training objective. We will show that the score function could be trained with a denoising objective.

DDPM is trained to removes the noise $\bar{\boldsymbol{\epsilon}}_i$ from $\mathbf{x}_i$ in the forward diffusion process, by training a denoising neural network $\boldsymbol{\epsilon}_\theta( \mathbf{x}, t_i  )$ to predict and remove the noise $\bar{\boldsymbol{\epsilon}}_i $. This means that DDPM minimizes the **denoising objective** [^Ho2020DenoisingDP]:

$$
L_{denoise}(\boldsymbol{\epsilon}_\theta) =\frac{1}{n}\sum_{i=1}^n \mathbf{E}_{\mathbf{x}_0 \sim p_0(\mathbf{x})}  \mathbf{E}_{\bold{x}_i \sim p(\mathbf{x}_i | \mathbf{x}_0) }\| \bar{\boldsymbol{\epsilon}}_i  -  \boldsymbol{\epsilon}_\theta( \mathbf{x}_i, t_i  )\|_2^2, \label{denoising objective}
$$
where $\bar{\boldsymbol{\epsilon}}_i$ is determined from $\mathbf{x}_i$ according to the $\ref{discrete forward diffusion}$ process.

Now we show that $\boldsymbol{\epsilon}_\theta$ trained with the above objective is proportional to the score function $\mathbf{s}$. There are two important properties regarding the relationship between the noise $\bar{\boldsymbol{\epsilon}}_i$ and the score function $\mathbf{s}$:

1. **Gaussian Distribution of $\mathbf{x}_i$**:  
    According to the $\ref{discrete forward diffusion}$, the distribution of $\mathbf{x}_i$ given $\mathbf{x}_0$ is a Gaussian distribution, expressed as:
    $$
    p(\mathbf{x}_i | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_i|\sqrt{\bar{\alpha}_i}\mathbf{x}_0, (1-\bar{\alpha}_i) I).
    $$

2. **Proportionality of Noise to Score Function**:  
    The noise $\bar{\boldsymbol{\epsilon}}_i$ is directly proportional to a score function, given by:
    $$
    \bar{\boldsymbol{\epsilon}}_i = -\sqrt{1-\bar{\alpha}_i}  \mathbf{s}(\mathbf{x}_i | \mathbf{x}_0, t_i), \label{score-noise relationship}
    $$
    where $\mathbf{s}(\mathbf{x}_i | \mathbf{x}_0, t_i)=\nabla_{\mathbf{x}_i} \log p(\mathbf{x}_i | \mathbf{x}_0)$ represents the score of the conditional probability density $p(\mathbf{x}_i | \mathbf{x}_0)$ at $\mathbf{x}_i$.

These properties indicate that the noise $\bar{\boldsymbol{\epsilon}}_i$ is directly related to a conditional score function, which connects to the score function $\mathbf{s}(\mathbf{x}, t)$ through the above equations.


Now we are very close to our target. The conditional score function $\mathbf{s}(\mathbf{x}_i | \mathbf{x}_0, t_i)$ is connected to the score function $\mathbf{s}(\mathbf{x}, t)$ through the following equation:

$$
\begin{aligned}
\mathbf{E}_{\mathbf{x}_0 \sim p_0(\mathbf{x})}  \mathbf{E}_{\mathbf{x}_i\sim p(\mathbf{x}_i | \mathbf{x}_0)}  f(\mathbf{x}_i) \mathbf{s} (\mathbf{x}_i | \mathbf{x}_0) &= \int \int f(\mathbf{x}_i) \nabla_{\mathbf{x}_i} p(\mathbf{x}_i | \mathbf{x}_0) p_0(\mathbf{x}_0) \, d\mathbf{x}_i \, d\mathbf{x}_0
 \\
&= \int f(\mathbf{x}_i) \nabla_{\mathbf{x}_i} \int p(\mathbf{x}_i | \mathbf{x}_0) p_0(\mathbf{x}_0) d\mathbf{x}_0 \, d\mathbf{x}_i
 \\
&= \mathbf{E}_{\mathbf{x}_i \sim p_{t_i}(\mathbf{x})} f(\mathbf{x}_i) \mathbf{s}(\mathbf{x}, t_i) \\
\end{aligned}
$$

where $f$ is an arbitrary function and $ \mathbf{s}(\mathbf{x}, t) =\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$ is the score function of the probability density of $\mathbf{x}_t$.

Substituting the $\ref{score-noise relationship}$ into the $\ref{denoising objective}$, expanding the squares, and utilizing the above equation, we can derive that the denoising objective is equivalent to a denoising score matching objective:

$$
L_{denoise}(\boldsymbol{\epsilon}_\theta) =\frac{1}{n}\sum_{i=1}^{n}   \mathbf{E}_{\mathbf{x_i}\sim p_{t_i}(\mathbf{x})} \| \sqrt{1-\bar{\alpha}_i}  \mathbf{s}(\mathbf{x}_i, t_i)  + \boldsymbol{\epsilon}_\theta( \bold{x}_i, t_i  )\|_2^2,
$$

This objectives says that the denoising neural network $\boldsymbol{\epsilon}_\theta( \mathbf{x}, t_i  )$ is trained to approximate a scaled score function $\boldsymbol{\epsilon}( \mathbf{x}, t_i  )$ [^Yang2022DiffusionMA]

$$
\boldsymbol{\epsilon}_\theta( \mathbf{x}, t_i  ) \approx  -\sqrt{1-\bar{\alpha}_i}\mathbf{s}(\mathbf{x}, t_i). \label{eps-score relation}
$$


## Summary:

We have covered all aspects of the DDPM theory. You can now find a suitable dataset, perform the $\ref{discrete forward diffusion}$, train a denoising neural network using the $\ref{denoising objective}$, and subsequently generate new samples with the $\ref{eps-score relation}$ and the $\ref{discrete backward process}$.


## What is Next
In the [next section](../fastest_way__diffusion_model_theory_iv/), we will discuss an alternative version of the backward diffusion process: ordinary differential equation (ODE) based backward sampling. This approach serves as the foundation for several modern architectures, such as rectified flow diffusion models.

Stay tuned for the next installment!

## Discussion
If you have questions, suggestions, or ideas to share, please visit the [discussion post](https://github.com/scraed/scraedBlog/discussions/4).



[^Song2020ScoreBasedGM]: Yang Song, et al. "Score-Based Generative Modeling through Stochastic Differential Equations." _ArXiv_ (2020).
[^Ho2020DenoisingDP]: Jonathan Ho, et al. "Denoising Diffusion Probabilistic Models." _ArXiv_ (2020).
[^Yang2022DiffusionMA]: Ling Yang, et al. "Diffusion Models: A Comprehensive Survey of Methods and Applications." _ACM Computing Surveys_ (2022).



---
