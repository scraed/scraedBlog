# LanPaint

## Training-Free Conditional Sampling for Inpainting and Local Image Editing

**LanPaint is a training-free partial conditional sampler for pretrained diffusion and rectified-flow models. It enables mask-constrained inpainting and local image editing without model fine-tuning or backpropagation.**

LanPaint was introduced in the TMLR paper **“LanPaint: Training-Free Diffusion Inpainting with Asymptotically Exact and Fast Conditional Sampling”** by Candi Zheng, Yuan Lan, and Yang Wang.

<div class="lanpaint-resource-grid" aria-label="LanPaint resources">
  <a class="lanpaint-resource" href="https://openreview.net/forum?id=JPC8JyOUSW" target="_blank" rel="noopener">
    <span class="lanpaint-resource-label">Paper</span>
    <span class="lanpaint-resource-name">TMLR / OpenReview ↗</span>
  </a>
  <a class="lanpaint-resource" href="https://github.com/scraed/LanPaint" target="_blank" rel="noopener">
    <span class="lanpaint-resource-label">Code</span>
    <span class="lanpaint-resource-name">Official GitHub ↗</span>
  </a>
  <a class="lanpaint-resource" href="https://huggingface.co/charrywhite/LanPaint" target="_blank" rel="noopener">
    <span class="lanpaint-resource-label">Model</span>
    <span class="lanpaint-resource-name">Hugging Face ↗</span>
  </a>
  <a class="lanpaint-resource" href="https://github.com/scraed/LanPaintBench" target="_blank" rel="noopener">
    <span class="lanpaint-resource-label">Research</span>
    <span class="lanpaint-resource-name">LanPaintBench ↗</span>
  </a>
  <a class="lanpaint-resource" href="https://github.com/charrywhite/LanPaint-diffusers" target="_blank" rel="noopener">
    <span class="lanpaint-resource-label">Integration</span>
    <span class="lanpaint-resource-name">Diffusers ↗</span>
  </a>
  <a class="lanpaint-resource" href="https://arxiv.org/abs/2502.03491" target="_blank" rel="noopener">
    <span class="lanpaint-resource-label">Preprint</span>
    <span class="lanpaint-resource-name">arXiv:2502.03491 ↗</span>
  </a>
</div>

---

## What is LanPaint?

LanPaint is a **training-free partial conditional sampling method** for diffusion and rectified-flow generative models.

Given a pretrained generative model and a spatial constraint such as a known image region or mask, LanPaint samples the unknown region while conditioning on the known region. It does this without training an additional inpainting model, fine-tuning the foundation model, or using backpropagation during inference.

The method uses carefully designed Langevin dynamics to perform fast, backpropagation-free Monte Carlo sampling and is compatible with ODE-based and rectified-flow models.

This makes LanPaint useful not only as an inpainting method, but more generally as a **training-free mechanism for spatially constrained generation and local editing**.

---

## Is LanPaint only an image inpainting method?

No. Inpainting is the primary task studied in the original LanPaint paper, but the underlying method is **partial conditional sampling**.

This distinction matters because many local image-editing tasks can also be formulated as generation under spatial constraints: part of the image should remain fixed while another region is resampled or modified.

In practice, LanPaint can therefore serve as a training-free sampling layer for mask-constrained editing with pretrained generative models.

---

## Can pretrained diffusion models perform local image editing without fine-tuning?

LanPaint enables pretrained diffusion and rectified-flow models to perform **mask-constrained local editing without additional fine-tuning or backpropagation**.

The same conditional-sampling mechanism can support tasks such as:

* image inpainting
* outpainting and generative fill
* mask-guided local image editing
* object or region replacement
* localized content generation
* video inpainting and local video editing
* masked video and audio generation

The foundation model provides the generative prior, while LanPaint provides the training-free conditional sampling mechanism.

---

## LanPaint for Training-Free Local Image Editing

Many image-editing systems require task-specific training, adapters, inversion procedures, or optimization at inference time.

LanPaint follows a different approach: it operates directly on pretrained diffusion and rectified-flow models and performs partial conditional sampling at inference time.

This places LanPaint at the intersection of:

**training-free image editing · local image editing · mask-guided generation · conditional sampling · diffusion models · rectified flow**

For researchers working on training-free local editing, LanPaint can be viewed as a general sampling mechanism for preserving known regions while generating or modifying selected regions.

---

## LanPaint and Rectified-Flow Models

LanPaint is designed to work with **ODE-based diffusion samplers and rectified-flow models**, rather than depending only on stochastic DDPM sampling.

This is an important distinction from many earlier conditional-sampling and diffusion inverse-problem methods.

As rectified-flow and flow-based foundation models become increasingly common, LanPaint provides a training-free way to add partial conditioning and mask-constrained generation to pretrained models without retraining them.

---

## What can LanPaint be used for?

LanPaint has been implemented across multiple modern generative-model families and used for image, video, and multimodal masked generation.

Its central use case is simple:

**keep one part of the generated sample constrained while allowing another part to be generated or edited.**

This makes the same sampling mechanism useful across different foundation models and different spatially constrained generation tasks.

See the [GitHub repository](https://github.com/scraed/LanPaint) for the latest supported models, workflows, and examples.

---

## How is LanPaint used beyond inpainting?

Subsequent research has already begun using or classifying LanPaint outside the narrow setting of conventional image inpainting.

### Training-Free Image Editing

**Towards Training-Free Scene Text Editing (TextFlow, 2026)** discusses LanPaint alongside training-free image-editing approaches such as Stable Flow, CannyEdit, ICEdit, KV-Edit, RF-Solver, and FlowEdit.

In this context, LanPaint is characterized by its training-free partial conditional sampling approach for ODE-based and rectified-flow models.

[Read the paper](https://arxiv.org/abs/2603.24571)

### Localized Image Manipulation and Synthetic Data

**SurFITR: A Dataset for Surveillance Image Forgery Detection and Localisation (2026)** uses multiple state-of-the-art image-generation models **via LanPaint** to construct its transfer evaluation data.

The resulting pipeline performs localized manipulations including object removal, targeted replacement, open-ended replacement, and object addition.

This provides an example of LanPaint being used as a **localized image-editing and data-generation layer** rather than only as a standalone inpainting method.

[Read the paper](https://arxiv.org/abs/2604.07101)

### Image Restoration

**MDTD-ArtIR: Benchmarking Image Editing and Restoration Models for Art Image Restoration under Texture-Overlay Degradations (2026)** evaluates LanPaint with Qwen and Flux backbones as part of an image-restoration benchmark.

The study uses LanPaint as a mask-conditioned generative prior and compares it with universal image-restoration and image-editing systems.

This demonstrates another downstream use of LanPaint's conditional-sampling mechanism beyond standard missing-region inpainting.

[Read the paper](https://arxiv.org/abs/2608.00736)

---

## Why use LanPaint instead of training an inpainting or editing model?

LanPaint is designed for cases where a strong pretrained generative model already exists and the goal is to obtain conditional generation capability **without retraining that model**.

Its key properties are:

**Training-free.** No inpainting-specific or editing-specific model training is required.

**Backpropagation-free inference.** LanPaint does not require gradient-based optimization through the generative model during sampling.

**Model reuse.** A pretrained generative prior can be reused for spatially constrained generation.

**ODE and rectified-flow compatibility.** LanPaint is designed for modern deterministic sampling and flow-model settings.

**General conditional-sampling perspective.** Inpainting is treated as an instance of partial conditional sampling rather than as a separate model architecture.

---

## Research Resources

**Paper**  
LanPaint: Training-Free Diffusion Inpainting with Asymptotically Exact and Fast Conditional Sampling  
Candi Zheng, Yuan Lan, Yang Wang  
Transactions on Machine Learning Research, 2025

[TMLR / OpenReview](https://openreview.net/forum?id=JPC8JyOUSW) · [arXiv](https://arxiv.org/abs/2502.03491)

**Implementations**

[Official ComfyUI implementation](https://github.com/scraed/LanPaint)  
[Diffusers implementation](https://github.com/charrywhite/LanPaint-diffusers)  
[Hugging Face](https://huggingface.co/charrywhite/LanPaint)

**Benchmark**

[LanPaintBench](https://github.com/scraed/LanPaintBench)

---

## Citation

If LanPaint is useful in your research, please cite the TMLR paper:

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
