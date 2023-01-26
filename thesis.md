---
documentclass: book
lang: en-US
papersize: a4

toc-title: Contents
toc-depth: 3

list-tables: true
list-figures: true
link-citations: true
linkReferences: true
nameInLink: true
colorlinks: true
links-as-notes: true

figPrefix: [Fig, Figs]
eqnPrefix: [Eq, Eqns]
tblPrefix: [Tbl, Tbls]
lstPrefix: [Lst, Lsts]
secPrefix: [Sec, Secs]

bibliography: [bibliography.bib]
csl: acm-sig-proceedings-long-author-list.csl

title: |
    AI-Assisted Creative Expression: a Case for Automatic Lineart Colorization
author: Yliess Hati
date: \today
rights: © 2023 by Yliess Hati is licensed under CC BY 4.0
keywords: [keyword]

acronyms:
    ai:
        short: AI
        long: Artificial Intelligence
    nn:
        short: NN
        long: Neural Network
    ann:
        short: ANN
        long: Artificial Neural Network
    dl:
        short: DL
        long: Deep Learning
    gpu:
        short: GPU
        long: Graphical Processing Unit
    tpu:
        short: TPU
        long: Tensor Processing Unit
    npu:
        short: NPU
        long: Neural Processing Unit
    gd:
        short: GD
        long: Gradient Descent
    vae:
        short: VAE
        long: Variational Autoencoder
    gan:
        short: GAN
        long: Generative Adversarial Network
    ddm:
        short: DDM
        long: Denoising Diffusion Model
    llm:
        short: LLM
        long: Large Language Model
---

\newpage{}

## Abstract
\newpage{}

## Introduction

Humans possess the ability to perceive and understand the world allowing us to accomplish a wide range of complex tasks through the combination of visual recognition, scene understanding, and communication. The ability to quickly and accurately extract information from a single image is a testament to the complexity and sophistication of the human brain and is often taken for granted. One of the +ai field's ultimate goals is to empower computers with such human-like abilities, one of them being creativity, being able to produce something original and worthwhile [@mumford_2012]. 

Computational creativity is the field at the intersection of +ai, cognitive psychology, philosophy, and art, which aims at understanding, simulating, replicating, or in some cases enhancing human creativity. One definition of computational creativity [@newell_1959] is the ability to produce something that is novel and useful, demands that we reject common beliefs, results from intense motivation and persistence, or comes from clarifying a vague problem. Top-down approaches to this definition use a mix of explicit formulations of recipes and randomness such as procedural generation. On the opposite, bottom-up approaches use [+ann]{.plural} to learn patterns and heuristics from large datasets to enable non-linear generation.

We, as a species, are currently witnessing the beginning of a new era where the gap between machines and humans is starting to blur. Current breakthroughs in the field of +ai, more specifically in +dl, are giving computers the ability to perceive and understand our world, but also to interact with our environment using natural interactions such as speech and natural language. [+ann]{.plural}, once mocked by the +ai community [@lecun_2019], are now trainable using +gd [@rumelhart_1986] thanks to the massive availability of data and the processing power of modern hardware accelerators such as [+gpu]{.plural}, [+tpu]{.plural}, and [+npu]{.plural}.

[+nn]{.plural}, those trainable general function approximators, gave rise to the field of generative [+nn]{.plurals}. Specialized +dl architectures such as [+vae]{.plural} [@kingma_2013], [+gan]{.plural} [@goodfellow_2014], [+ddm]{.plural} [@ho_2020], and [+llm]{.plural} [@vaswani_2017; @brown_2020] are used to generate artifacts such as text, audio, images, and videos of unprecedented quality and complexity.

This thesis aims at exploring how one could train and use generative +nn to create +ai-powered tools capable of enhancing human creative expression.

### Motivations

> - A case for Lineart Colorization

### Problem Statement

> - Black & White Lineart VS Gray Scale
> - Incomplete Information Challenge fo Computer Vision
> - Natural Artisitic Control Back to the User

### Contributions

> - Recipe for curating datasets for the task of automatic colorization
> - 3 Models exploring different aspect of the topic:
>     - PaintsTorch: High Quality, User-Guided, Fast Realtime Feedback
>     - StencilTorch: Human-Machine Collaboration, Human-in-the-Loop
>     - StableTorch: Variance and Iterative Exploration
> - A reflexion on Current Generative AI Ethical and Societal Impact in our Society

### Concerns

> - Raise awareness about
>   - Deepfakes
>   - Model Fabulations
>   - Ownership & Copyright Ambiguities
>   - Biases & Discrimination
> - About this work
>   - Images used only for Educational and Research Purposes
>   - Only describe recipes for reproducibility
>   - Dataset and Weights are not Distributed (Only Code)

### Outline

> - Plain Language Expanded TOC

\newpage{}

## Background
### History of Artificial Intelligence
### Neural Networks
### Autoencoders
### Variational Autoencoders
### Generative Adversarial Networks
### Denoising Diffusion Models
\newpage{}

## Contrib I (Find Catchy Explicit Name)
### State of the Art
### Method
### Setup
### Results
### Summary
\newpage{}

## Contrib II (Find Catchy Explicit Name)
### State of the Art
### Method
### Setup
### Results
### Summary
\newpage{}

## Contrib III (Find Catchy Explicit Name)
### State of the Art
### Method
### Setup
### Results
### Summary
\newpage{}

## Contrib IV (Find Catchy Explicit Name)
### State of the Art
### Method
### Setup
### Results
### Summary
\newpage{}

## Ethical and Societal Impact
\newpage{}

## Conclusion
\newpage{}

## References