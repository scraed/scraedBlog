# LanPaint for Training-Free Local Editing

## Mask-Constrained Generation for Images, Video, and Audio

**LanPaint is a training-free partial conditional sampler for mask-constrained generation with pretrained diffusion and rectified-flow models. The official implementation applies this idea to local image editing, video inpainting, video outpainting, and masked video-audio generation without fine-tuning the underlying model.**

This page focuses on how LanPaint's sampling mechanism applies to **training-free local image editing**. For the complete project overview, paper, implementations, and benchmark, see the [LanPaint project hub](/scraedBlog/lanpaint/).

The original TMLR paper focuses primarily on image inpainting and partial conditional sampling. The broader video and video-audio workflows described here come from the official repository and its current implementations.

LanPaint is not a standalone instruction-based image-editing model. The pretrained generative model provides the semantic and visual prior, while LanPaint provides the spatial conditioning mechanism that keeps known regions constrained and resamples unknown regions.

## Can LanPaint be used for local image editing without fine-tuning?

Yes, for **mask-constrained local editing**.

LanPaint does not train a separate editing model or fine-tune the underlying generative model. Instead, it performs partial conditional sampling with a pretrained diffusion or rectified-flow model: the known region provides the conditioning information, while the masked or unknown region is regenerated.

The [LanPaint paper](https://openreview.net/forum?id=JPC8JyOUSW) formulates this as sampling from a conditional distribution when only part of a sample is known. It introduces a training-free, asymptotically exact partial conditional sampling method for ODE-based and rectified-flow models, using carefully designed Langevin dynamics for fast, backpropagation-free Monte Carlo sampling.

LanPaint is therefore best understood as a **training-free sampling layer for spatially constrained generation**, rather than as a separately trained image-editing model.

## What does partial conditional sampling contribute to local editing?

Many local image-editing tasks share the same structure:

1. A known region should remain consistent with the source image.
2. A masked or unknown region should be regenerated or modified.
3. The generated region should remain coherent with its surrounding context.

LanPaint applies this structure directly at sampling time:

1. The image is divided into known and unknown parts using a mask or spatial constraint.
2. The pretrained generative model supplies the distribution of plausible images.
3. LanPaint samples the unknown part conditionally while retaining the known observation as context.

For image inpainting, this corresponds to **known pixels as the conditioning region** and **masked pixels as the region to sample**. The same formulation can describe outpainting, partial regeneration, and other mask-constrained local editing workflows.

For the mathematical formulation, Langevin dynamics, and rectified-flow setting, see the [LanPaint paper in HTML form](https://arxiv.org/html/2502.03491).

## What kinds of local editing are demonstrated by the official LanPaint repository?

The [official LanPaint repository](https://github.com/scraed/LanPaint) documents several spatially constrained generation and editing workflows. These implementation examples should be distinguished from the experiments in the original paper.

### Mask-guided image regeneration

LanPaint supports masks with flexible shape, size, and position. Selected regions can be regenerated while the remaining image provides conditioning context.

This is the most direct form of local editing: **select a spatial region and resample that region conditionally on the rest of the image**.

### Outpainting and generative fill

LanPaint also supports **outpainting**, where the existing image becomes the known conditioning region and new content is generated outside the original image boundary.

The repository documents outpainting workflows for image and video models, including examples built around Z-Image, Qwen Image, HiDream, and Wan 2.2. These are repository-level workflows; they should not be interpreted as a claim that every model supports every task equally well.

### Editing with pretrained image-editing models

LanPaint can be combined with pretrained editing-oriented models. The official repository documents masked workflows for **Qwen Image Edit**, including Qwen Image Edit 2508 and 2509.

In this setting, LanPaint does not replace the semantic editing capability of the underlying model. It provides a mask-constrained sampling mechanism around that model, allowing the edit to remain spatially localized.

> The pretrained model determines what can be generated or edited; LanPaint determines how generation can be conditionally constrained to selected regions without additional training.

### Partial inpainting for edits that should retain more of the source

The repository also documents a **partial inpainting** workflow. Standard inpainting may regenerate a masked region more freely. Partial inpainting instead starts the diffusion process from an intermediate step, allowing the result to retain more information from the source image.

This creates a practical continuum between **preserving the source** and **fully regenerating the masked region**.

## Can LanPaint be used for object removal, replacement, or addition?

Yes, there is evidence for these as **downstream applications**, but they should not be presented as additional experiments from the original LanPaint paper.

The 2026 [SurFITR paper](https://arxiv.org/abs/2604.07101) describes a transfer setting in which multiple image-generation models are used **via LanPaint** to construct an evaluation set. It defines four localized manipulation types:

* removal of an existing entity;
* targeted replacement;
* open-ended replacement;
* addition of a new entity.

The SurFITR pipeline uses localized, mask-guided manipulation while preserving the remainder of the image. This provides evidence for a broader role for LanPaint as a **mask-controlled local generation layer used with multiple generative models**.

The correct interpretation is that SurFITR documents a downstream use of LanPaint for localized manipulation. These operations are not the primary task definition of the original LanPaint paper.

## Is LanPaint considered a training-free image-editing method?

LanPaint is best described primarily as a **training-free partial conditional sampling method** that can be used for training-free local image editing.

The 2026 paper [“Towards Training-Free Scene Text Editing” (TextFlow)](https://arxiv.org/abs/2603.24571) discusses LanPaint in its related work on training-free image editing. TextFlow characterizes it as a training-free, asymptotically exact partial conditional sampling approach for ODE-based and rectified-flow models.

This places LanPaint in the methodological neighborhood of training-free image editing, while preserving the more precise distinction that its original paper studies inpainting and conditional sampling.

## How is LanPaint different from a general-purpose image-editing model?

LanPaint should not be described as a universal instruction-based image editor.

A general-purpose image-editing model may learn semantic operations such as changing style, pose, lighting, identity, geometry, or global composition. LanPaint instead focuses on a different question:

> How can a pretrained generative model sample a selected unknown region while conditioning on the region that should remain known?

This makes LanPaint especially relevant to **mask-constrained, spatially localized editing**. The underlying model supplies semantic capabilities, while LanPaint supplies the partial conditional sampling mechanism.

## Image and video editing with the same sampling idea

The original TMLR paper focuses on image inpainting and partial conditional sampling. The official repository extends the implementation to additional workflows, including **Wan 2.2 video inpainting and video outpainting**.

The repository also documents a video mask editor that allows masks to be painted on selected frames and interpolated across intermediate frames, as well as a MiniMax H3 workflow for masked video and audio inpainting.

These examples show how the mask-conditioned sampling idea can extend beyond a single still-image workflow. They should be attributed to the current implementation and repository documentation, rather than treated as experiments in the original paper.

## When is LanPaint relevant to image-editing research?

LanPaint may be a useful method or baseline when several of the following conditions hold:

* a pretrained diffusion or rectified-flow model is already available;
* the desired modification is spatially localized;
* a mask or known/unknown region can be defined;
* retraining or fine-tuning the generative model is undesirable;
* preserving the unedited context is important;
* the problem can be formulated as partial conditional sampling.

Researchers working on **training-free local image editing, mask-guided image editing, inpainting, outpainting, spatially constrained generation, or conditional generation with diffusion and rectified-flow models** may therefore find LanPaint relevant.

## What are the limitations of LanPaint?

LanPaint's scope is easiest to understand when its method and its underlying generative model are kept separate.

* **It is not a standalone semantic editor.** The base model determines the semantic operations and visual quality that are available.
* **It depends on model compatibility.** A pretrained model must expose the score, flow, or sampling interface needed by the implementation.
* **Distilled models can be more difficult.** The LanPaint paper reports degraded performance on distilled models, so the method should not be assumed to work equally well with every sampler or foundation model.
* **The original paper has a narrower experimental scope.** Its main experiments focus on image inpainting and related partial conditional sampling; broader video, audio, and model-specific workflows come from the official repository or subsequent work.
* **Mask quality still matters.** Poor boundaries, ambiguous regions, or an unsuitable prompt can limit the quality of a local edit.

These limitations clarify what LanPaint contributes: a conditional sampling mechanism that reuses a pretrained generative prior without fine-tuning or inference-time backpropagation.

## Citation

If you use LanPaint in research on inpainting, local image editing, conditional sampling, or related applications, please cite:

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
