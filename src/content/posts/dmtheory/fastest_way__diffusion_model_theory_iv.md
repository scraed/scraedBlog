---
title: The Fastest Way to Diffusion Model Theory - IV
published: 2025-07-06
tags: [Diffusion Model, Theory]
category: Diffusion Model Theory
draft: false
---



:::note[Recap]
[Section I](../fastest_way__diffusion_model_theory_i/) introduced **Langevin Dynamics** for sampling from $p(\mathbf{x})$:  
$$
d\mathbf{x}_t = \mathbf{s}(\mathbf{x}_t) dt + \sqrt{2} d\mathbf{W}_t
\quad\text{or}\quad
d\mathbf{x}_t = \frac{1}{2}\mathbf{s}(\mathbf{x}_t) dt + d\mathbf{W}_t
$$  
where $\mathbf{s}(\mathbf{x}) = \nabla_{\mathbf{x}} \log p(\mathbf{x})$ is the score function, and $d\mathbf{W}_t \sim \sqrt{dt}\,\mathcal{N}(0,\mathbf{I})$.

[Section II](../fastest_way__diffusion_model_theory_ii/) defined DDPM's processes:  
- **Forward Process** ($t \in [0,T]$):  
  $$
  d\mathbf{x}_t = - \frac{1}{2} \mathbf{x}_t dt + d\mathbf{W}_t \label{Forward Process}
  $$  
  The discrete version adds noise through:  
  $$
  \mathbf{x}_{i} = \sqrt{\bar{\alpha}_i} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_i} \bar{\boldsymbol{\epsilon}}_i, \quad 1 \leq i \leq n \label{discrete backward process}
  $$  
  where $\bar{\boldsymbol{\epsilon}}_i \sim \mathcal{N}(0,\mathbf{I})$ is the noise at step $i$, and $\bar{\alpha}_i$ controls the noise schedule.

- **Backward Process** ($t' = T - t$):  
  $$
  d\mathbf{x}_{t'} = \left( \frac{1}{2} \mathbf{x}_{t'} + \mathbf{s}(\mathbf{x}_{t'}, T-t') \right) dt' + d\mathbf{W}_{t'}  \label{Backward Process}
  $$  
  where $\mathbf{s}(\mathbf{x},t) = \nabla_{\mathbf{x}} \log p_t(\mathbf{x})$.

[Section III](../fastest_way__diffusion_model_theory_iii/) covered the **denoising objective**:  
$$
\min_{\boldsymbol{\epsilon}_\theta} \frac{1}{n}\sum_{i=1}^n \mathbb{E}_{\substack{\mathbf{x}_0 \sim p_0 \\ \mathbf{x}_i \sim p(\mathbf{x}_i|\mathbf{x}_0)}} \| \bar{\boldsymbol{\epsilon}}_i - \boldsymbol{\epsilon}_\theta( \mathbf{x}_i, t_i )\|_2^2 \label{denoising objective}
$$  
which trains $\boldsymbol{\epsilon}_\theta$ to predict $\bar{\boldsymbol{\epsilon}}_i$ (approximating $-\sqrt{1-\bar{\alpha}_i}\mathbf{s}(\mathbf{x}, t_i)$).
:::

# ODE and Flow-Based Diffusion Model

## The ODE Based Backward Diffusion Process

The backward diffusion process $\ref{Backward Process}$ is not the only reverse process for the forward process $\ref{Forward Process}$. We can derive a deterministic ordinary differential equation (ODE) as an alternative, removing the stochastic term $d\mathbf{W}$ in the backward process.

To obtain this ODE reverse process, consider the Langevin dynamics with a rescaled time ($d\tau \rightarrow \frac{1}{2} d\tau$):  

$$
\begin{split}
    d\mathbf{x}_\tau &= \frac{1}{2} \mathbf{s}(\mathbf{x}_\tau, t) d\tau + \, d\mathbf{W}_\tau, \\
&= \underbrace{-\frac{1}{2} \mathbf{x}_\tau d\tau + d\mathbf{W}_\tau}_{\text{Forward}} + \underbrace{\frac{1}{2} \mathbf{x}_t d\tau + \frac{1}{2} \mathbf{s}(\mathbf{x}_\tau, t) d\tau }_{\text{Backward}},  
\end{split}
$$

Following the same logic used to derive the backward diffusion process in [Section II](../fastest_way__diffusion_model_theory_ii/), we could read from this splitting the backward ODE (known as the probability flow ODE [^Song2020ScoreBasedGM]):  

$$
d\mathbf{x}_{t'} =  \left( \frac{1}{2} \mathbf{x}_{t'} + \frac{1}{2} \mathbf{s}(\mathbf{x}, T-t') \right) dt',  \label{Probability flow ODE}
$$

where $t' \in [0,T]$ is backward time, and $\mathbf{s}(\mathbf{x}, t) = \nabla_{\mathbf{x}_{t}} \log p_t(\mathbf{x})$ is the score function of the density of $\mathbf{x}_{t}$ in the forward process. This ODE maintains the same forward-backward duality as the SDE reverse process $\ref{Backward Process}$.  

Since the ODE is deterministic, it enables faster sampling than the SDE version. Established ODE solvers—such as higher-order methods and exponential integrators—can further reduce computational steps while maintaining accuracy.

## Variance Perserving, Variance Exploding, and Rectified Flow

With the ODE based backward process, we can discuss three common formulations of ODE based diffusion models: variance-preserving (VP), variance-exploding (VE), and rectified flow (RF). We demonstrate their mathematical equivalence and show how they can be transformed into one another.

To simplify notation, we now use continuous time $t$ and its corresponding state $\mathbf{x}_t$, rather than discrete notations like $t_i$ and $\mathbf{x}_i$.

### Variance Preserving (VP)

The 'variance-preserving' formulation is very similar to what we have introduced in the previous section, just replacing the SDE backward process to the ODE version.

The forward diffusion process in continuous time $t$ is:

$$
\mathbf{x}_{t} = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t} \bar{\boldsymbol{\epsilon}}_t, \label{VP forward}
$$

where $\bar{\alpha}_t = e^{-t}$ and $\bar{\boldsymbol{\epsilon}}_t \sim \mathcal{N}(0, \mathbf{I})$. It is the same as $\ref{discrete backward process}$ introduced in the previous section.

#### Forward and Backward Processes

The forward and backward processes in VP notation are consistent with the previously introduced DDPM model, with the only difference being the substitution of the SDE backward process with the ODE version.

- **Forward SDE** ($\ref{Forward Process}$):

$$
d\mathbf{x}_t = -\frac{1}{2}\mathbf{x}_t dt + d\mathbf{W}_t  \label{VP forward SDE}
$$

- **Backward ODE** ($\ref{Probability flow ODE}$):

$$
d\mathbf{x}_{t'} = \frac{1}{2} \left(\mathbf{x}_{t'} + \mathbf{s}(\mathbf{x}_{t'}, T-t')\right)dt',  \label{VP backward ODE}
$$

where $t' \in [0,T]$ is reversed time, and the score function $\mathbf{s}(\mathbf{x}, t) = \nabla_{\mathbf{x}} \log p_t(\mathbf{x})$ is learned via the denoising objective.

#### Score Matching Objective

While we previously trained the denoising network $\boldsymbol{\epsilon}_\theta$ using the $\ref{denoising objective}$, we can alternatively model the score function $\mathbf{s}_\theta$ directly. This yields the equivalent **score matching objective**:

$$
L_{score}(\mathbf{s}_\theta) = \mathbf{E}_{t \sim \mathcal{U}[0,1]} \mathbf{E}_{\mathbf{x}_0 \sim p_0(\mathbf{x})} \mathbf{E}_{\bar{\boldsymbol{\epsilon}}_t \sim \mathcal{N}(\mathbf{0},I)} \left\| \frac{\bar{\boldsymbol{\epsilon}}_t}{\sqrt{1-\bar{\alpha}_t}} + \mathbf{s}_\theta(\mathbf{x}_t, t) \right\|_2^2,
$$

where $\mathbf{x}_t$ follows the forward process. This represents an equivalent but reweighted version of the original denoising objective.


### Variance Exploding (VE)

The variance exploding formulation provides an alternative to variance preserving. Define:

$$
\sigma = \sqrt{\frac{1 - \bar{\alpha}_t}{\bar{\alpha}_t}}; \quad  
\sigma' = \sqrt{\frac{1 - \bar{\alpha}_{T-t'}}{\bar{\alpha}_{T-t'}}}; \quad  
\mathbf{z}_{\sigma} = \frac{\mathbf{x}_t}{\sqrt{\bar{\alpha}_t}}; \quad  
\mathbf{z}_{\sigma'} = \frac{\mathbf{x}_{t'}}{\sqrt{\bar{\alpha}_{T-t'}}};\quad  
\boldsymbol{\epsilon}( \mathbf{z}_\sigma, \sigma  ) = -\sqrt{1-\bar{\alpha}_t}\mathbf{s}( \mathbf{x}_t, t  ), \label{VE notations}
$$

Substituting the definitions above and rewriting the $\ref{VP forward}$ process in VE notation yields:

$$
\mathbf{z}_{\sigma} = \mathbf{z}_0 + \sigma \bar{\boldsymbol{\epsilon}}_\sigma, \label{VE forward}
$$

where $\mathbf{z}_0$ is the clean image corrupted by noise of magnitude $\sigma$.

#### Forward and Backward Processes

Substituting the definitions from $\ref{VE notations}$ and rewriting both the $\ref{VP forward SDE}$ and $\ref{VP backward ODE}$ yields:

- **Forward SDE**:

$$
d\mathbf{z}_{\sigma} = \sqrt{2\sigma} d\mathbf{W}_{\sigma}, \quad \sigma \in \left[0, \sqrt{\tfrac{1 - \bar{\alpha}_T}{\bar{\alpha}_T}}\right] \label{VE forward SDE}
$$

- **Backward ODE**:

$$
d\mathbf{z}_{\sigma'} = \boldsymbol{\epsilon}(\mathbf{z}_{\sigma'}, \sigma')d\sigma', \quad \sigma' \in \left[\sqrt{\tfrac{1 - \bar{\alpha}_T}{\bar{\alpha}_T}}, 0\right] \label{VE backward ODE}
$$

The advantage of the VE notation lies in its simpler backward ODE compared to the VP notation. In practice, directly discretizing the $\ref{VE backward ODE}$ using an Euler solver tends to yield greater accuracy than the $\ref{VP backward ODE}$, which includes an additional $\frac{1}{2} \mathbf{x}$ term that can introduce numerical errors. However, a notable disadvantage of the VE notation is that $\sigma$ can become quite large at time $T$, potentially leading to numerical instability.

#### Denoising Objective

To directly model $\boldsymbol{\epsilon}_\theta(\mathbf{z}, \sigma)$, we adapt the $\ref{denoising objective}$ to VE coordinates by replacing $\mathbf{x}_t$ with $\mathbf{z}_\sigma$:

$$
L_{denoise}(\boldsymbol{\epsilon}_\theta) = \mathbf{E}_{\sigma \sim \mathcal{U}[0, \sigma_{max}] } \mathbf{E}_{\mathbf{z}_0 \sim p_0(\mathbf{x})}  \mathbf{E}_{\bar{\boldsymbol{\epsilon}}_\sigma \sim \mathcal{N}(\mathbf{0},I)} \| \bar{\boldsymbol{\epsilon}}_\sigma - \boldsymbol{\epsilon}_\theta(\mathbf{z}_\sigma, \sigma) \|_2^2, \label{VE denoising objective}
$$

where $\sigma_{max} = \sqrt{(1 - \bar{\alpha}_T)/\bar{\alpha}_T}$ and $\mathbf{z}_\sigma$ follows the VE forward process. This preserves the denoising objective's structure while operating in VE space.

### Rectified Flow (RF)

While often presented as a distinct framework from DDPMs, rectified flows are mathematically equivalent [^Gao2025DiffusionGFM] to DDPMs. We now provide a much simpler proof via the following transformations:

$$
s = \frac{\sigma}{1+\sigma}; \quad  
s' = \frac{\sigma'}{1+\sigma'}; \quad  
\mathbf{r}_{s} = \frac{\mathbf{z}_\sigma}{1+\sigma}; \quad  
\mathbf{r}_{s'} = \frac{\mathbf{z}_{\sigma'}}{1+\sigma'}; \quad  
\mathbf{v}(\mathbf{r}_s, s) = \frac{\boldsymbol{\epsilon}(\mathbf{z}_{\sigma}, \sigma) - \mathbf{r}_{s}}{1-s} \label{RF notation}
$$

Rewriting the $\ref{VE forward}$ process in $\ref{RF notation}$ yields:

$$
\mathbf{r}_{s} = (1-s)\mathbf{r}_0 + s\bar{\boldsymbol{\epsilon}}_s, \label{RF forward}
$$

which linearly interpolates between clean data ($\mathbf{r}_0$) and noise.

#### Forward and Backward Processes

The forward and backward process of rectified flow model could be derived from the $\ref{VE forward SDE}$ and $\ref{VE backward ODE}$ by substituting the $\ref{RF notations}$. 

- **Forward SDE**:

$$
d\mathbf{r}_{s} = -\frac{\mathbf{r}_s}{1-s}ds + \sqrt{\frac{2s}{1-s}}d\mathbf{W}_{s}, \quad s\in[0,1] \label{RF forward SDE}
$$

- **Backward ODE**:

$$
d\mathbf{r}_{s'} = \mathbf{v}(\mathbf{r}_{s'}, s')ds', \quad s'\in[1,0] \label{RF backward ODE}
$$

The advantage of the rectified flow notation is its simple backward ODE, which eliminates the diverging behavior of $\sigma$ at time $T$ found in the VE notation, ensuring that $s$ remains within a finite range of $[0, 1]$.

#### Flow Matching Objective

To directly model $\mathbf{v}_\theta(\mathbf{r}_{s'}, s')$, we transform the $\ref{VE denoising objective}$ by substituting $\bar{\boldsymbol{\epsilon}}_\sigma$ and $\boldsymbol{\epsilon}_\theta$ with $\bar{\boldsymbol{\epsilon}}_s$ and $\mathbf{v}_\theta$, respectively. This transformation utilizes the $\ref{VE forward}$ and $\ref{RF forward}$ processes and the definitions from $\ref{RF notation}$, while also neglecting a constant scaling factor of $(1-s)$. As a result, we obtain the flow matching objective:

$$
L_{flow}(\mathbf{v}_\theta) = \mathbf{E}_{s\sim \mathcal{U}[0,1]}\mathbf{E}_{\mathbf{r}_0 \sim p_0(\mathbf{x})}\mathbf{E}_{\bar{\boldsymbol{\epsilon}}_s\sim \mathcal{N}(\mathbf{0},I)}\|\bar{\boldsymbol{\epsilon}}_s - \mathbf{r}_0 - \mathbf{v}_\theta(\mathbf{r}_s, s)\|_2^2,
$$

where $\mathbf{r}_s$ follows the RF forward process. This represents a re-weighted equivalent of the denoising objective, interpreted in the flow matching framework where $\bar{\boldsymbol{\epsilon}}$ corresponds to the endpoint $\mathbf{r}_1$ and $\mathbf{v}_\theta$ models the velocity field transporting $\mathbf{r}_0$ to $\mathbf{r}_1$.


:::important
## The equivalence between VP, VE, and RF notation
The three notations (VP, VE, and RF) are mutually transformable through the mappings defined above. Models trained with score matching, denoising, and flow matching objectives can be converted into other notations. This implies that samplers or guidance designed for one notation can be easily transformed and adapted to the others.
:::

## What is Next
Now that we have covered the major theories of diffusion models, including DDPMs, ODE-based diffusion models, and flow models, it is important to note that these models are primarily unconditional. In the next section, we will explore how diffusion models can be utilized to model conditional distributions.

Stay tuned for the next installment!


## Discussion
If you have questions, suggestions, or ideas to share, please visit the [discussion post](https://github.com/scraed/scraedBlog/discussions/4).

[^Song2020ScoreBasedGM]: Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2020). Score-Based Generative Modeling through Stochastic Differential Equations. *arXiv preprint arXiv:2011.13456*.
[^Gao2025DiffusionGFM]: Gao, R., Hoogeboom, E., Heek, J., De Bortoli, V., Murphy, K. P., & Salimans, T. (2025). Diffusion Models and Gaussian Flow Matching: Two Sides of the Same Coin. *The Fourth Blogpost Track at ICLR 2025*. https://openreview.net/forum?id=C8Yyg9wy0s
