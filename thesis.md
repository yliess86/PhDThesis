---
documentclass: book
lang: en-US
papersize: a4

toc-title: Contents
toc-depth: 3
numbersections: true

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
    cv:
        short: CV
        long: Computer Vision
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
    rl:
        short: RL
        long: Reinforcement Learning
    mlp:
        short: MLP
        long: Multi-Layer Perceptron
---

\newpage{}

## Abstract {-}
\newpage{}

## Aknowledgements {-}
\newpage{}

# Context
\newpage{}

## Introduction {#ch:introduction}

Humans possess the ability to perceive and understand the world allowing us to accomplish a wide range of complex tasks through the combination of visual recognition, scene understanding, and communication. The ability to quickly and accurately extract information from a single image is a testament to the complexity and sophistication of the human brain and is often taken for granted. One of the +ai field's ultimate goals is to empower computers with such human-like abilities, one of them being creativity, being able to produce something original and worthwhile [@mumford_2012]. 

Computational creativity is the field at the intersection of +ai, cognitive psychology, philosophy, and art, which aims at understanding, simulating, replicating, or in some cases enhancing human creativity. One definition of computational creativity [@newell_1959] is the ability to produce something that is novel and useful, demands that we reject common beliefs, results from intense motivation and persistence, or comes from clarifying a vague problem. Top-down approaches to this definition use a mix of explicit formulations of recipes and randomness such as procedural generation. On the opposite, bottom-up approaches use [+ann]{.plural} to learn patterns and heuristics from large datasets to enable non-linear generation.

We, as a species, are currently witnessing the beginning of a new era where the gap between machines and humans is starting to blur. Current breakthroughs in the field of +ai, more specifically in +dl, are giving computers the ability to perceive and understand our world, but also to interact with our environment using natural interactions such as speech and natural language. [+ann]{.plural}, once mocked by the +ai community [@lecun_2019], are now trainable using +gd [@rumelhart_1986] thanks to the massive availability of data and the processing power of modern hardware accelerators such as [+gpu]{.plural}, [+tpu]{.plural}, and [+npu]{.plural}.

[+nn]{.plural}, those trainable general function approximators, gave rise to the field of generative [+nn]{.plurals}. Specialized +dl architectures such as [+vae]{.plural} [@kingma_2013], [+gan]{.plural} [@goodfellow_2014], [+ddm]{.plural} [@ho_2020], and [+llm]{.plural} [@vaswani_2017; @brown_2020] are used to generate artifacts such as text, audio, images, and videos of unprecedented quality and complexity.

This dissertation aims at exploring how one could train and use generative +nn to create +ai-powered tools capable of enhancing human creative expression. The task of automatic lineart colorization act as the example case used to illustrate this process throughout the entire thesis. 

### Motivations

Lineart colorization is an essential aspect of the work of artists, illustrators, and animators. The task of manually coloring lineart can be time-consuming, repetitive, and exhausting, particularly in the animation industry, where every frame of an animated product must be colored and shaded. This process is typically done using image editing software such as Photoshop [@photoshop], Clip Studio PAINT [@clipstudiopaint], and PaintMan [@paintman]. Automating the colorization process can greatly improve the workflow of these creative professionals and has the potential to lower the barrier for newcomers and amateurs. Such a system was integrated into Clip Studio PAINT [@clipstudiopaint], demonstrating the growing significance of automatic colorization in the field.

The most common digital illustration process can be broken down into four distinct stages: sketching, inking, coloring, and post-processing. As demonstrated by the work of Kandinsky [@kandinsky_1977], the colorization process can greatly impact the overall meaning of a piece of art through the introduction of various color schemes, shading, and textures. These elements of the coloring process present significant challenges for the +cv task of automatic lineart colorization, particularly in comparison to its grayscale counterpart [@furusawa_2O17; @hensman_2017; @zhang_richard_2017]. Without the added semantic information provided by textures and shadows, inferring materials and 3D shapes from black and white linearts is difficult. They can only be deduced from silhouettes.

### Problem Statement

One major challenge of automatic lineart colorization is the availability of qualitative public datasets. Illustrations do not always come with their corresponding lineart. The few datasets available for the task are lacking consistency in the quality of the illustrations, gathering images from different types, mediums and styles. For those reasons, online scrapping and synthetic lineart extraction is the method of choice for many of the contributions in the field [@ci_2018; @zhang_richard_2017].

Previous works in automatic lineart colorization are based on the +gan [@goodfellow_2014] architecture. They can generate unperfect but high-quality illustrations in a quasi realtime setting. They achieve user control and guidance via different means, color hints [@frans_2017; @liu_2017; @sangkloy_2016; @paintschainer_2017; @ci_2018], style transfer [@zhang_ji_2017], tagging [@kim_2019], and more recently natural language [@ho_2020]. One common pattern in these methods is the use of a feature extractor such as Illustration2Vec [@saito_2015] allowing to compensate for the lack of semantic descriptors by injecting its feature vector into the models.

### Contributions

This work focuses on the use of color hints in the form of user strokes as it fits the natural digital artist workflow and does not involve learning and mastering a new skill. While previous works offers improving quality compared to classical +cv techniques, they are still subject to noisy training data, artifacts, a lack of variety, and a lack of fidelity in the user intent. In this dissertation we explore the importance of a clean, qualitative and consistent dataset. We investigate how to better capture the user intent via natural artistic controls and how to reflect them into the generated model artifact while preserving or improving its quality. We also look at how the creative process can be transformed into a dynamic iterative workflow where the user collaborates with the machine to refine and carry out variations of his artwork.

Here is a brief enumeration of this thesis's contributions:

- We present a recipe for curating datasets for the task of automatic lineart colorization [@hati_2019; @hati_2022]
- We introduce three generative models:
    - PaintsTorch [@hati_2019], a double GAN generator that improved generation quality compared to previous work while allowing realtime interaction with the user.
    - StencilTorch [@hati_2022], an upgrade upon PaintsTorch, shifting the colorization problem to in-painting allowing for human collaboration to emerge as a natural workflow where the input of a first pass becomes the potential input for a second.
    - StablePaint, an exploration of +ddm for bringing more variety into the generated outputs allowing for variation exploration and conserving the iterative workflow introduced by StencilTorch for the cost of inference speed.
- We offer an advised reflection on current generative +ai ethical and societal impact.

### Concerns

Recent advances in generative +ai for text, image, audio, and video synthesis are raising important ethical and societal concerns, especially because of its availability and ease of use. Models such as Stable Diffusion [@rombach_2021] and more recently Chat-GPT [@openai_2023] are disturbing our common beliefs and relation with copyright, creativity, the distribution of fake information and so on.

One of the main issues with generative AI is the potential for model fabulation. Generative models can create entirely new, synthetic data that is indistinguishable from real data. This can lead to the dissemination of false information and the manipulation of public opinion. Additionally, there are ambiguities surrounding the ownership and copyright of the generated content, as it is unclear who holds the rights to the generated images and videos. Training data is often obtained via online scrapping and thus copyright ownership is not propagated. This is especially true for commercial applications.

Another important concern is the potential for biases and discrimination. These models are trained on large amounts of data, and if the data is not diverse or representative enough, the model may perpetuate or even amplify existing biases. The Microsoft Tay Twitter bot [@wolf_2017] scandal is an outcome of such a phenomenon. This initially innocent chatbot has been easily turned into a racist bot perpetuating hate speech. The task was made easier because of the inherently biased dataset it was trained on.

In this work, we are committed to addressing and raising awareness for these concerns. The illustrations used for training our models and for our experiments are only used for educational and research purposes. We only provide recipes for reproducibility and do not distribute the dataset nor the weights resulting from model training, only the code. We hope this will not ensure that our work is used ethically and responsibly but limit its potential misuse.

### Outline

The first part of this thesis (chapters [1](#ch:introduction)-[3](#ch:methodology)) provides context to the recent advances in generative +ai and introduces the +cv task of user-guided automatic lineart colorization, its challenges, and our contributions to the field. It then provides additional background, from +dl first principles to current architectures used in modern generative +nn, and introduces the methodology used throughout the entire document. This part should be accessible to the majority, experts and non-experts, and serve as an introduction to the field.

The second part (chapters [4](#ch:contrib-1)-[7](#ch:contrib-4)) presents our contributions, some of which have previously been presented in [@hati_2019; @hati_2022]. It introduces into detail our recipe for sourcing and curating consistent and qualitative datasets for automatic lineart colorization, PaintsTorch [@hati_2019] our first double generator +gan conditioned on user strokes, StencilTorch [@hati_2022] our in-painting reformulation introducing the use of masks to allow the emergence of iterative workflow and collaboration with the machine, and finally StablePaint, an exploration of the use of +ddm models for variations qualitative exploration.

The third and final part (chapters [7](#ch:ethdical-and-societal-impact)-[8](#ch:conclusion)) offers a detailed reflection on this thesis's contributions and more generally about the field of generative +ai ethical and societal impact, identifies the remaining challenges and discusses future work.

The code base for the experiments and contributions is publicly available on GitHub at [https://github.com/yliess86](https://github.com/yliess86).

\newpage{}

## Background {#ch:background}

This chapter introduces the reader to the field of [+dl]{.full} from first principles to the current architectures used in modern generative +ai. The first section (section [1](#sec:history)) presents a brief history of the +ai field to ground this technical dissertation into its historical context. The following sections (sections [2](#sec:core)-[4](#sec:attention)) are discussing the first principles of modern +dl from the early Perceptron to more modern frameworks such as [+llm]{.full .plural}.

Additional snippets of code are included to make this chapter more insightful and valuable for newcomers.

### A Brief History of Artificial Intelligence {#sec:history}

The history of the field of +ai is not a simple linear and straightforward story. The field had its success and failures. The term [+ai]{.full} has first been introduced in 1956 by John Mc Carthy and Marvin Lee Minsky at a workshop sponsored by Dartmouth College, gathering about twenty researchers and intellectuals such as the renowned Claude Shanon. The field was supposed to solve all the modern world's problems in a short period.

However, the reality has been far less rosy. Over the years, AI has gone through several “winters”, periods of inactivity and disillusion where funding was cut and research interest dropped. But with the advent of Big Data and the rise of [+dl]{.full}, +ai is once again in the spotlight. The following sections provide a brief overview of the history of AI, from its early days to the current state of the field.

#### The Early Years

The term +ai was first used at the 1956 Dartmouth Workshop, where John McCarthy proposed the idea of creating a machine that could learn from its mistakes and improve its performance over time. This was a revolutionary idea at the time, and the work done at Dartmouth attracted a great deal of attention and funding.

Much of the early research focused on symbolic AI, which uses symbols and logical operations to represent and manipulate data. This approach was based on the early work of Alan Turing and the development of data-driven languages such as the Functional Language LISP from MIT.

One significant contribution of this period was the Perceptron by Frank Rosenblat, a simplified biomedical model of a single neuron. This neuron fires when the weighted sum of its input is above a predefined threshold. The weights are tuned iteratively and manually given supervised data, inputs with corresponding labels, until good enough classification accuracy is met.

#### The First AI Winter

The Perceptron was an early example of a connectionist approach, which uses a network of artificial neurons to process data. The Perceptron was met with much enthusiasm but was eventually criticized by Marvin L. Minsky and Seymour Papert, who argued that it could not solve the simple XOR problem.

The criticisms, as well as other issues, led to a period of disillusion in the field of +ai, known as the "First AI Winter". It was a time when +ai research lost its momentum and funding was not abundant anymore. This period lasted from the late 1970s to the early 1980s.

#### Expert Systems and Symbolic AI

The eighties saw a resurgence of interest in +ai. Expert systems were the new hot +ai topic. It uses hierarchical and specialized ensembles of symbolic reasoning models to solve complex problems. Symbolic +ai continued to prosper as the dominant approach until the mid-nineties.

During this period, +ai was developped as logic-based systems, search-based systems such as depth-first-search or genetic algorithms requiring complex engineering and domain specific knwoledge from experts to work. It was also the time of the first cognitive architectures inspired by advances in the field of neuroscience such as SOAR and ACT-R attempting at simulating the the human cognitive process.

Others, enven through they were not much and where often rejected from +ai conferences at the time, where still working and believed on the connectionnist approach. It was the case for Kunihiko Fukushima responsible for the Neocognitron, and works on Hopfield Networks and the +mlp. Rumbelhart et al. also presented one of the first learning rule for training such complex +nn.

#### The Second AI Winter

Unfortunetly, this periode was also marked by a lack of progress because of the resource limitations of the time. Those algorithms required to much power and data to work. They were not sufficient to make AI truly successful.

The lack of progress in the 1980s led to the "Second AI Winter", which lasted from the mid-1990s to the early 2000s. AI research was largely abandoned during this period, and funding and enthusiasm dwindled.

##### The Era of Data

The rapid adoption of the internet, the search engines and social networks of the five giants (GAFAM) led to an abundance in data, what we call Big Data. This phenomemon and the processing power of that period driven by the highly specialized and parallel computing architectures, the [+gpu]{.full .plural}, are responsible of the resurgence of the interest for +ai. This allowed researchers to develop more powerful algorithms and model architectures, symbolic and statistical methods, such as the Support Vector Machines (SVMs) from Vapnik.

But the real breakthrough came with the development of [+dl]{.full}, which uses hierarchical deep [+nn]{.full .plural}, with many layers to process and extract non-linear patterns from data. +dl is now the dominant approach in +ai and has achieved remarkable progress in a wide range of applications, from speech and image recognition to natural language processing.

#### The Modern Deep Learning Success

<!-- Deep learning has revolutionized the field of AI and has achieved unprecedented successes in a wide range of tasks, from computer vision to natural language processing. The success of deep learning can be attributed to several factors.

First, deep learning architectures are able to learn complex features from large amounts of data. This is due in large part to the use of convolutional neural networks (CNNs), which are able to detect and recognize patterns in images. Furthermore, recurrent neural networks (RNNs) can be used to process sequences of data, such as text or audio.

Second, the use of GPUs has allowed researchers to train deep learning models with large datasets in a fraction of the time that it would take with traditional CPUs. This has enabled researchers to develop more powerful models and has allowed deep learning to be used in real-time applications, such as autonomous driving.

Finally, the development of open-source frameworks, such as TensorFlow and PyTorch, has enabled the development of deep learning models to become more accessible to researchers. This has further accelerated the development of deep learning and has enabled the field to reach new heights of success. -->

### Core Principles {#sec:core}
#### Perceptron
#### Multi-Laye Perceptron
#### Convolutional Neural Network
### Generative Architectures {#sec:generative}
#### Autoencoders
#### Variational Autoencoders
#### Generative Adversarial Networks
#### Denoising Diffusion Models
### Attention is all you Need {#sec:attention}
#### Multihead Self-Attention
#### Large Language Models
\newpage{}

## Methodology {#ch:methodology}
### Implementation
### Objective Evaluation
### Subjective Evaluation
### Reproducibility
\newpage{}

# Core
\newpage{}

## Contrib I (Find Catchy Explicit Name) {#ch:contrib-1}
### State of the Art
### Method
### Setup
### Results
### Summary
\newpage{}

## Contrib II (Find Catchy Explicit Name) {#ch:contrib-2}
### State of the Art
### Method
### Setup
### Results
### Summary
\newpage{}

## Contrib III (Find Catchy Explicit Name) {#ch:contrib-3}
### State of the Art
### Method
### Setup
### Results
### Summary
\newpage{}

## Contrib IV (Find Catchy Explicit Name)  {#ch:contrib-4}
### State of the Art
### Method
### Setup
### Results
### Summary
\newpage{}

# Reflection
\newpage{}

## Ethical and Societal Impact {#ch:ethical-and-societal-impact}
\newpage{}

## Conclusion {#ch:conclusion}
\newpage{}

## References {-}