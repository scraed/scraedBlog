---
category: LanPaint
tags: [LanPaint]
---

# LanPaint: Training-Free Partial Conditional Sampling with Langevin Dynamics

## Training-Free Conditional Sampling for Diffusion and Rectified-Flow Models

**LanPaint is a training-free and efficient partial conditional sampling method based on Langevin Dynamics Monte Carlo, designed for ODE-based diffusion samplers and rectified-flow models.**

Its goal is to sample from a conditional distribution represented by a pretrained generative model without training an additional conditional model and without using inference-time backpropagation.

LanPaint introduces two main components:

1. **Bidirectional Guided (BiG) Score**, designed to mitigate local maxima trapping by enabling mutual adaptation between the unknown and observed components.
2. **Fast Langevin Dynamics (FLD)**, an accelerated Langevin sampling scheme designed to reach the stationary distribution efficiently while remaining stable at larger numerical step sizes.

## At a Glance

- **Problem:** Sample unknown variables conditioned on observed variables from a pretrained joint generative model.
- **Core method:** Combine the Bidirectional Guided (BiG) Score with Fast Langevin Dynamics (FLD).
- **Model setting:** ODE-based diffusion samplers and Rectified Flow models.
- **Training requirement:** No additional conditional model, fine-tuning, or inference-time backpropagation.
- **Main application:** Diffusion inpainting and other partial conditional generation tasks.

The method was introduced in:

**LanPaint: Training-Free Diffusion Inpainting with Asymptotically Exact and Fast Conditional Sampling**

Candi Zheng, Yuan Lan, Yang Wang

Transactions on Machine Learning Research, 2025

**Paper:** [TMLR / OpenReview](https://openreview.net/forum?id=JPC8JyOUSW)
**arXiv:** [arXiv:2502.03491](https://arxiv.org/abs/2502.03491)
**Code:** [Official GitHub repository](https://github.com/scraed/LanPaint)
**Project Hub:** [LanPaint research hub](/scraedBlog/lanpaint/)

---

## What is partial conditional sampling?

Consider a pretrained generative model that represents a joint distribution

$$
p(z), \qquad z=(x,y),
$$

where \(x\) and \(y\) denote an arbitrary split of the variables.

LanPaint studies the problem of sampling

$$
x \sim p(x\mid y=y_o),
$$

when \(y_o\) is observed and \(x\) is unknown.

For image inpainting, \(x\) can represent the pixels or latent variables inside the masked region, while \(y_o\) represents the known region.

The important point is that the pretrained diffusion model was trained to represent the **joint distribution** \(p(x,y)\). It was not necessarily trained as a specialized conditional model for directly sampling \(p(x\mid y)\).

LanPaint therefore asks:

> How can a pretrained diffusion model perform partial conditional sampling in a training-free way?

This formulation is more general than the image-inpainting application itself. Inpainting is the principal task studied in the LanPaint paper, while **partial conditional sampling** is the underlying probabilistic problem.

---

## Why is partial conditional sampling difficult for pretrained diffusion models?

Modern diffusion models are highly effective at sampling an entire variable \(z\) from the learned joint distribution.

Partial conditioning is different.

Given a pretrained model for

$$
p(z)=p(x,y),
$$

the desired distribution is instead

$$
p(x\mid y=y_o).
$$

The LanPaint paper distinguishes this goal from several common approaches.

### Sequential Monte Carlo

Sequential Monte Carlo methods can perform exact partial conditional sampling. Representative examples discussed in the LanPaint paper include diffusion-based conditional sampling for motif scaffolding by Trippe et al. (2022) [1] and the practical asymptotically exact method of Wu et al. (2024) [2]. The paper notes that these approaches require expensive stochastic sampling with many diffusion steps and particles.

This dependence on the probabilistic DDPM framework makes them difficult to combine with the fast deterministic ODE-based samplers commonly used by modern diffusion models.

### Linear inverse-problem methods

Another family of approaches formulates the task as an inverse problem, typically using observations of the form

$$
y=Hz+\epsilon.
$$

The LanPaint paper emphasizes that this is a different objective from direct partial conditional sampling.

These approaches construct a posterior or approximate solution intended to produce a plausible reconstruction, but that approximation is not required to exactly follow the joint distribution \(p(z)\) represented by the pretrained diffusion model.

The original paper discusses diffusion inverse-problem methods including variational approaches such as DDRM [3], CoPaint [4], and MMPS [5], as well as heuristic-loss methods such as MCG [6], DPS [7], GradPaint [8], DCPS [9], and D-Flow [10]. Many such methods rely on approximate objectives or inference-time gradient computation.

LanPaint instead focuses directly on **partial conditional sampling from the pretrained model's distribution**.

---

## How does Langevin dynamics enable conditional sampling?

LanPaint builds on **Langevin Dynamics Monte Carlo (LMC)**.

For a target distribution \(p(z)\), define its score function as

$$
s(z)=\nabla_z \log p(z).
$$

The original Langevin dynamics follows

$$
dz_\tau=s(z_\tau)d\tau+\sqrt{2}\,dW_\tau,
$$

where \(W_\tau\) is Brownian motion and \(\tau\) denotes Langevin time.

Under the conditions discussed in the paper, this dynamics asymptotically converges to the stationary distribution

$$
z_\tau \sim p(z).
$$

This makes Langevin dynamics attractive for conditional sampling: if an appropriate score for the desired target distribution can be constructed, Langevin Monte Carlo can be used to sample that distribution.

LanPaint builds its partial conditional sampler around this idea.

---

## How is diffusion inpainting connected to Langevin dynamics?

The connection between diffusion inpainting and Langevin dynamics predates LanPaint.

RePaint [11] introduced iterative denoising and renoising—often described as a “time travel” procedure—for training-free diffusion inpainting.

Subsequent work, including Training-free guidance of diffusion models for generalised inpainting [12], showed that this procedure can be interpreted as Langevin Dynamics Monte Carlo.

LanPaint adopts this Langevin perspective and adapts it to the setting of **fast ODE-based diffusion samplers and rectified-flow models**.

This is important because modern generative models frequently use deterministic ODE-based sampling with relatively few diffusion steps rather than the long stochastic DDPM trajectories used by earlier inpainting methods.

However, directly combining conventional Langevin-based inpainting with fast ODE sampling creates two practical problems addressed by LanPaint:

* **local maxima trapping**, and
* **slow or unstable Langevin convergence**.

These motivate the two main components of LanPaint: BiG Score and FLD.

---

## Bidirectional Guided (BiG) Score

### What problem does the BiG Score solve?

The LanPaint paper identifies **local maxima trapping** as an important limitation of conventional Langevin-based inpainting, particularly when using fast ODE samplers with large diffusion step sizes.

Let

* \(x\) denote the inpainted or unknown component,
* \(y\) denote the observed component,
* \(y_o\) denote the original observation.

In conventional Langevin-based inpainting, the sampling of \(x\) is guided by \(y\).

The paper observes that information flow can become effectively one-directional: \(x\) adapts to \(y\), but there is no corresponding mechanism through which the evolving \(x\) penalizes an incompatible state of \(y\).

In multimodal distributions, this can cause samples to become trapped around local maxima of

$$
p(x\mid y),
$$

even when the corresponding joint state \((x,y)\) has low likelihood.

The BiG Score is introduced to create **bidirectional information flow between the inpainted and observed components**.

---

### What is the Bidirectional Guided Score?

LanPaint defines an auxiliary guided target distribution of the form

$$
p_t(x,y\mid y_o)
\approx
q_{\lambda,t}(x,y\mid y_o)
=
\frac{1}{Z}
p_t(x\mid y)
\frac{
p_t(y\mid y_o)^{1+\lambda}
}{
p_t(y)^\lambda
},
$$

where \(\lambda>-1\) is the guidance scale and \(Z\) is a normalizing constant.

Here:

* \(p_t(x\mid y)\) couples the unknown region to the current observed component,
* \(p_t(y\mid y_o)\) anchors the observed component to the original observation,
* \(p_t(y)\) provides the additional guidance term,
* \(\lambda\) controls the strength of bidirectional guidance.

The corresponding BiG score used for the observed component is written in the paper as

$$
g_\lambda(x,y,t)
=
-
\left(
(1+\lambda)
\frac{
y-\sqrt{\bar{\alpha}_t}y_o
}{
1-\bar{\alpha}_t
}
+
\lambda s_y(x,y,t)
\right),
$$

where \(s_y(x,y,t)\) is the \(y\)-component of the diffusion model's joint score.

The resulting coupled Langevin dynamics uses the ordinary joint score for the unknown component and the BiG score for the observed component.

The purpose of this construction is not simply to force \(y\) toward \(y_o\). It allows information from the evolving \(x\) to influence the dynamics of \(y\), while maintaining conditioning on the observation.

This is the “bidirectional” aspect of the BiG Score.

---

## Why is LanPaint described as asymptotically exact?

The wording **asymptotically exact** is important.

LanPaint does not claim that every intermediate noisy distribution at finite diffusion time is represented without approximation.

The formal analysis in the paper shows that the Langevin dynamics defined using the BiG Score converges to a target distribution whose deviation from the desired construction becomes negligible as the diffusion time approaches the clean-data limit.

In the paper's analysis, the approximation error vanishes with the noise scale as

$$
t\rightarrow0.
$$

Therefore the final clean sample approaches the desired partial conditional distribution.

This is the basis for describing LanPaint as providing:

**asymptotically exact partial conditional sampling.**

On the analytically tractable conditional-Gaussian benchmark, LanPaint approaches near-zero KL divergence as the diffusion and inner Langevin iteration counts increase; a separate Gaussian-mixture experiment evaluates local-maxima trapping.

---

## Why does the BiG Score matter more with fast ODE samplers?

Fast ODE samplers reduce the number of diffusion steps by taking larger steps through diffusion time.

The LanPaint experiments show that this makes local maxima trapping particularly visible for conventional Langevin-based inpainting.

On a multimodal Gaussian-mixture benchmark, the paper demonstrates that samples can concentrate around local maxima of \(p(x\mid y)\) that correspond to low joint likelihood.

The BiG Score propagates information from \(x\) back into the dynamics of \(y\), helping steer samples away from these low-joint-likelihood states.

This addresses one of the key difficulties in combining Langevin-based conditional sampling with fast ODE samplers.

---

## Fast Langevin Dynamics (FLD)

### Why is ordinary Langevin dynamics too slow?

Standard Langevin dynamics presents a practical trade-off.

Small numerical steps are stable but require many iterations before reaching the stationary distribution.

Large numerical steps can accelerate convergence, but simple discretizations such as Euler-Maruyama accumulate larger numerical errors. In image generation this can appear as white-noise artifacts in pixel space or blurriness in latent space.

LanPaint therefore introduces **Fast Langevin Dynamics (FLD)** to improve convergence while maintaining numerical stability.

---

### What is Fast Langevin Dynamics?

FLD introduces an auxiliary momentum variable \(q_\tau\).

The dynamics is defined in the paper as

$$
dz_\tau=q_\tau d\tau,
$$

$$
dq_\tau
=
\Gamma
\left(
-q_\tau d\tau
+
s(z_\tau,t)d\tau
+
\sqrt{2}\,dW_\tau
\right),
$$

where:

* \(z_\tau\) is the state being sampled,
* \(q_\tau\) is the momentum variable,
* \(s(z_\tau,t)\) is the diffusion-model score,
* \(\Gamma\) is the friction coefficient.

The paper describes the momentum term as introducing a time-averaging effect over previous Langevin states, accelerating convergence toward the stationary distribution.

FLD is related to underdamped Langevin dynamics, but is designed specifically around the requirements of LanPaint's diffusion sampling setting.

---

### What is the diffusion damping force?

Momentum alone does not solve the numerical stability problem caused by large Langevin steps.

LanPaint therefore introduces what the paper calls a **diffusion damping force** in the FLD numerical solver.

The score is decomposed as

$$
s(z_\tau,t)
=
C_t(z_\tau)-A_tz_\tau.
$$

During an FLD numerical interval, \(C_t(z_\tau)\) is treated as constant while the term

$$
-A_tz_\tau
$$

acts as the diffusion damping force.

For the VP diffusion formulation, setting

$$
A_t=(1-\bar{\alpha}_t)^{-1}
$$

connects this damping term directly to the forward diffusion process.

The paper shows that even as the Langevin numerical interval becomes large, the state remains finite and approaches

$$
\mathcal{N}
\left(
\sqrt{\bar{\alpha}_t}\hat z_0,
1-\bar{\alpha}_t
\right),
$$

where \(\hat z_0\) is the Tweedie estimator of the clean sample.

This matches the form of the forward diffusion process and provides the stability property needed to use larger Langevin steps.

---

### Does FLD change the target distribution?

A central result of the paper is that FLD preserves the stationary distribution of the original Langevin dynamics.

Theorem 4.2 states that under Fast Langevin Dynamics, the joint state \((z,q)\) has stationary distribution

$$
(z,q)
\sim
p(z)\,
\mathcal{N}(q\mid0,\Gamma).
$$

Consequently,

$$
z\sim p(z)
$$

retains the same stationary distribution as the original Langevin dynamics.

This is important because FLD is intended to **accelerate sampling**, not replace the target distribution with a different heuristic objective.

In the experiments, FLD substantially accelerates convergence relative to the original Langevin dynamics. The paper highlights high-fidelity results with five inner iterations per diffusion step in its practical configuration.

---

## How do BiG Score and FLD work together?

The two components solve different problems.

### BiG Score

Addresses the **quality and correctness of the conditional sampling trajectory**, especially the local maxima trapping caused by insufficient information flow between the unknown and observed components.

### Fast Langevin Dynamics

Addresses the **efficiency and numerical stability of Langevin sampling**, allowing rapid movement toward the stationary distribution without the instability caused by simply enlarging Euler-Maruyama step sizes.

Together they form the core LanPaint conditional-sampling procedure:

$$
\text{Partial conditioning}
\rightarrow
\text{BiG Score}
\rightarrow
\text{Fast Langevin Dynamics}
\rightarrow
\text{ODE diffusion step}.
$$

The paper's ablation experiments evaluate the contributions of BiG Score and FLD separately.

---

## Why is LanPaint backpropagation-free?

LanPaint's sampling updates are constructed from the diffusion model's score—or an equivalent prediction parameterization—rather than by defining an external image-space loss and differentiating that loss through the entire diffusion network.

As a result, the method does not require inference-time backpropagation through the pretrained model.

This is why the paper describes LanPaint as enabling:

**fast, backpropagation-free Monte Carlo sampling.**

The distinction is particularly relevant compared with training-free inverse-problem methods that guide generation by repeatedly differentiating heuristic reconstruction losses during inference.

---

## How does LanPaint work with ODE-based diffusion samplers?

LanPaint is specifically designed around modern **ODE-based diffusion sampling**.

Earlier stochastic inpainting methods often depend directly on the probabilistic DDPM transition process.

LanPaint instead separates the conditional-sampling operation from the outer diffusion solver.

At each diffusion time, Langevin dynamics performs the conditional sampling update; the generative trajectory can then continue using a deterministic ODE-based diffusion sampler.

This lets LanPaint operate with the fast deterministic samplers widely used in modern diffusion systems.

---

## How does LanPaint work with Rectified Flow?

The paper derives the method using **variance-preserving (VP) diffusion notation**, but explicitly states that LanPaint is not restricted to VP diffusion models.

The score formulation can be converted to mathematically equivalent parameterizations.

The paper describes conversions among:

* variance-preserving diffusion,
* variance-exploding diffusion,
* rectified-flow notation.

For rectified-flow models, the required score is obtained through conversion to the model's **velocity-prediction** formulation.

Therefore LanPaint does not require a rectified-flow model to expose a separate score network.

Instead, the paper uses the mathematical equivalence between these parameterizations to apply the same conditional-sampling formulation.

This is the basis for LanPaint's compatibility with rectified-flow models such as the model families demonstrated in the paper and official implementation.

---

## Is LanPaint an inverse-problem solver?

The LanPaint paper makes an explicit distinction between its goal and the common **linear inverse-problem** formulation.

Linear inverse-problem approaches typically model an observation as

$$
y=Hz+\epsilon
$$

and construct a posterior or optimization objective that yields a plausible reconstruction.

LanPaint instead begins from a pretrained joint generative distribution

$$
p(z)=p(x,y)
$$

and targets the partial conditional distribution

$$
p(x\mid y).
$$

The paper therefore characterizes many inverse-problem approaches as solving a related but fundamentally different objective: their approximate posterior does not necessarily require the reconstructed joint sample to follow the exact joint distribution represented by the pretrained diffusion model.

LanPaint's theoretical goal is **partial conditional sampling**, not optimization of an external reconstruction loss.

---

## How is LanPaint different from RePaint and earlier Langevin methods?

LanPaint builds directly on the line of work connecting diffusion inpainting to Langevin Dynamics Monte Carlo.

**RePaint [11]** introduced repeated denoising and renoising for training-free diffusion inpainting.

Later work, including Training-free guidance of diffusion models for generalised inpainting [12], reformulated this “time travel” process as an independent Langevin dynamics procedure.

LanPaint retains the Langevin conditional-sampling perspective but focuses on two limitations that become particularly important with modern fast samplers:

1. **local maxima trapping**, addressed by the BiG Score;
2. **slow or unstable convergence**, addressed by FLD.

The method is therefore designed to make Langevin-based partial conditional sampling practical with fast ODE-based samplers and rectified-flow models.

---

## What evidence supports asymptotic exactness?

The paper includes a synthetic two-dimensional conditional Gaussian experiment for which the ground-truth conditional distribution and score are analytically available.

This allows sampling quality to be evaluated independently of diffusion-model training error.

The experiment conditions on the \(y\) component and samples the unknown \(x\) component.

LanPaint's sample mean and covariance are compared with the analytical ground truth using KL divergence.

The experiments show that LanPaint approaches near-zero KL divergence with increasing diffusion and inner iteration steps on this benchmark.

A separate Gaussian-mixture experiment is used to study the **local maxima trapping** problem and the effect of the BiG Score.

---

## Is the method restricted to 2D images?

No mathematical 2D-image assumption is built into the core BiG Score and FLD formulations.

The paper describes these formulations as **dimension-agnostic**, operating on a joint variable

$$
z=(x,y)\in\mathbb{R}^d.
$$

The TMLR paper uses image inpainting as its main application and also demonstrates video inpainting by treating video as a spatio-temporal tensor.

The same conditional-sampling formulation can therefore be expressed for arbitrary-dimensional variables, although practical applicability still depends on having an appropriate pretrained generative model and score-equivalent prediction function.

---

## Summary

LanPaint approaches diffusion inpainting as a **partial conditional sampling problem**.

Given a pretrained model for

$$
p(x,y),
$$

the objective is to sample

$$
x\sim p(x\mid y=y_o)
$$

without training a new conditional model.

The method combines:

**Langevin Dynamics Monte Carlo**

for sampling from score-defined distributions,

**Bidirectional Guided (BiG) Score**

for mutual adaptation between unknown and observed components and mitigation of local maxima trapping,

and

**Fast Langevin Dynamics (FLD)**

for accelerated, stable sampling while preserving the Langevin stationary distribution.

Together these components provide the basis for LanPaint's **training-free, asymptotically exact partial conditional sampling** with ODE-based and rectified-flow diffusion models.

For applications of this method to mask-constrained local editing, see:

**[LanPaint for Training-Free Local Editing](/scraedBlog/lanpaint/local-editing/)**

For the original paper, implementation, benchmark, and subsequent research usage, see:

**[LanPaint Research Resources](/scraedBlog/lanpaint/)**

---

## Original Paper

**LanPaint: Training-Free Diffusion Inpainting with Asymptotically Exact and Fast Conditional Sampling**

Candi Zheng, Yuan Lan, Yang Wang

Transactions on Machine Learning Research, 2025

[TMLR / OpenReview](https://openreview.net/forum?id=JPC8JyOUSW) · [arXiv:2502.03491](https://arxiv.org/abs/2502.03491)

---

## Citation

```bibtex
@article{
zheng2025lanpaint,
title={LanPaint: Training-Free Diffusion Inpainting with Asymptotically Exact and Fast Conditional Sampling},
author={Candi Zheng and Yuan Lan and Yang Wang},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=JPC8JyOUSW}
}
```

---

## References

The following references are drawn from the original LanPaint paper and correspond to the numbered citations above.

[1] Brian L. Trippe et al. [Diffusion probabilistic modeling of protein backbones in 3d for the motif-scaffolding problem](https://arxiv.org/abs/2206.04119). arXiv preprint, 2022.

[2] Luhuan Wu et al. [Practical and asymptotically exact conditional sampling in diffusion models](https://arxiv.org/abs/2306.17775). NeurIPS, 2024.

[3] Bahjat Kawar et al. [Denoising diffusion restoration models](https://arxiv.org/abs/2201.11793). 2022.

[4] Guanhua Zhang et al. [Towards coherent image inpainting using denoising diffusion implicit models](https://arxiv.org/abs/2304.03322). 2023.

[5] François Rozet et al. [Learning diffusion priors from observations by expectation maximization](https://arxiv.org/abs/2405.13712). NeurIPS, 2024.

[6] Hyungjin Chung et al. [Improving diffusion models for inverse problems using manifold constraints](https://arxiv.org/abs/2206.00941). 2022.

[7] Hyungjin Chung et al. [Diffusion posterior sampling for general noisy inverse problems](https://arxiv.org/abs/2209.14687). 2022.

[8] Asya Grechka et al. [GradPaint: Gradient-guided inpainting with diffusion models](https://arxiv.org/abs/2309.09614). 2024.

[9] Yazid Janati et al. [Divide-and-conquer posterior sampling for denoising diffusion priors](https://arxiv.org/abs/2403.11407). 2024.

[10] Heli Ben-Hamu et al. [D-Flow: Differentiating through flows for controlled generation](https://arxiv.org/abs/2402.14017). 2024.

[11] Andreas Lugmayr et al. [RePaint: Inpainting using denoising diffusion probabilistic models](https://arxiv.org/abs/2201.09865). CVPR, 2022.

[12] Lewis Cornwall et al. [Training-free guidance of diffusion models for generalised inpainting](https://openreview.net/forum?id=AC1QLOJK7l). 2024.
