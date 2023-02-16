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
    ml:
        short: ML
        long: Machine Learning
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
    sgd:
        short: SGD
        long: Stochastic Gradient Descent
    ae:
        short: AE
        long: Autoencoder
    vae:
        short: VAE
        long: Variational Autoencoder
    gan:
        short: GAN
        long: Generative Adversarial Network
    wgan:
        short: WGAN
        long: Wasserstein Generative Adversarial Network
    ddm:
        short: DDM
        long: Denoising Diffusion Model
    ddpm:
        short: DDPM
        long: Denoising Diffusion Probabilistic Model
    ddim:
        short: DDIM
        long: Denoising Diffusion Implicit Model
    llm:
        short: LLM
        long: Large Language Model
    rl:
        short: RL
        long: Reinforcement Learning
    mlp:
        short: MLP
        long: Multi-Layer Perceptron
    svm:
        short: SVM
        long: Support Vector Machine
    rnn:
        short: RNN
        long: Recurrent Neural Network
    lstm:
        short: LSTM
        long: Long Short-Term Memory
    cnn:
        short: CNN
        long: Convolutional Neural Network
    actr:
        short: ACT-R
        long: Adaptive Control of Thought—Rational
    rlhf:
        short: RLHF
        long: Reinforcement Learning from Human Feedback
    mse:
        short: MSE
        long: Mean Squared Error
    dag:
        short: DAG
        long: Directed Acyclic Graph
    ad:
        short: AD
        long: Automatic Differentiation
    ast:
        short: AST
        long: Abstract Syntax Tree
    relu:
        short: ReLU
        long: Rectified Linear Unit
    mnist:
        short: MNIST
        long: Modified National Institute of Standards and Technology
    umap:
        short: UMAP
        long: Uniform Manifold Approximation and Projection
    vi:
        short: VI
        long: Variational Inference
    kld:
        short: KL-Divergence
        long: Kullback-Leibler Divergence
---

\newpage{}
## List of Abbreviations {-}
::: {#acronyms}
:::

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

[+nn]{.plural}, those trainable general function approximators, gave rise to the field of generative [+nn]{.plural}. Specialized +dl architectures such as [+vae]{.plural} [@kingma_2013], [+gan]{.plural} [@goodfellow_2014], [+ddm]{.plural} [@ho_2020], and [+llm]{.plural} [@vaswani_2017; @brown_2020] are used to generate artifacts such as text, audio, images, and videos of unprecedented quality and complexity.

This dissertation aims at exploring how one could train and use generative +nn to create +ai-powered tools capable of enhancing human creative expression. The task of automatic lineart colorization act as the example case used to illustrate this process throughout the entire thesis. 

![Common illustration process. From left to right: sketching, inking, coloring, and pros-processing. Credits: Taira Akitsu](./figures/motivations_steps.svg){#fig:steps}

### Motivations

Lineart colorization is an essential aspect of the work of artists, illustrators, and animators. The task of manually coloring lineart can be time-consuming, repetitive, and exhausting, particularly in the animation industry, where every frame of an animated product must be colored and shaded. This process is typically done using image editing software such as Photoshop [@photoshop], Clip Studio PAINT [@clipstudiopaint], and PaintMan [@paintman]. Automating the colorization process can greatly improve the workflow of these creative professionals and has the potential to lower the barrier for newcomers and amateurs. Such a system was integrated into Clip Studio PAINT [@clipstudiopaint], demonstrating the growing significance of automatic colorization in the field.

The most common digital illustration process can be broken down into four distinct stages: sketching, inking, coloring, and post-processing (see @fig:steps). As demonstrated by the work of Kandinsky [@kandinsky_1977], the colorization process can greatly impact the overall meaning of a piece of art through the introduction of various color schemes, shading, and textures. These elements of the coloring process present significant challenges for the +cv task of automatic lineart colorization, particularly in comparison to its grayscale counterpart [@furusawa_2O17; @hensman_2017; @zhang_richard_2017]. Without the added semantic information provided by textures and shadows, inferring materials and 3D shapes from black and white linearts is difficult. They can only be deduced from silhouettes.

### Problem Statement

One major challenge of automatic lineart colorization is the availability of qualitative public datasets. Illustrations do not always come with their corresponding lineart. The few datasets available for the task are lacking consistency in the quality of the illustrations, gathering images from different types, mediums and styles. For those reasons, online scrapping and synthetic lineart extraction is the method of choice for many of the contributions in the field [@ci_2018; @zhang_richard_2017].

Previous works in automatic lineart colorization are based on the +gan [@goodfellow_2014] architecture. They can generate unperfect but high-quality illustrations in a quasi realtime setting. They achieve user control and guidance via different means, color hints [@frans_2017; @liu_2017; @sangkloy_2016; @paintschainer_2017; @ci_2018], style transfer [@zhang_ji_2017], tagging [@kim_2019], and more recently natural language [@ho_2020]. One common pattern in these methods is the use of a feature extractor such as Illustration2Vec [@saito_2015] allowing to compensate for the lack of semantic descriptors by injecting its feature vector into the models.

### Contributions

This work focuses on the use of color hints in the form of user strokes as it fits the natural digital artist workflow and does not involve learning and mastering a new skill. While previous works offers improving quality compared to classical +cv techniques, they are still subject to noisy training data, artifacts, a lack of variety, and a lack of fidelity in the user intent. In this dissertation we explore the importance of a clean, qualitative and consistent dataset. We investigate how to better capture the user intent via natural artistic controls and how to reflect them into the generated model artifact while preserving or improving its quality. We also look at how the creative process can be transformed into a dynamic iterative workflow where the user collaborates with the machine to refine and carry out variations of his artwork.

Here is a brief enumeration of this thesis's contributions:

- We present a recipe for curating datasets for the task of automatic lineart colorization [@hati_2019; @hati_2023]
- We introduce three generative models:
    - PaintsTorch [@hati_2019], a double GAN generator that improved generation quality compared to previous work while allowing realtime interaction with the user.
    - StencilTorch [@hati_2023], an upgrade upon PaintsTorch, shifting the colorization problem to in-painting allowing for human collaboration to emerge as a natural workflow where the input of a first pass becomes the potential input for a second.
    - StablePaint, an exploration of +ddm for bringing more variety into the generated outputs allowing for variation exploration and conserving the iterative workflow introduced by StencilTorch for the cost of inference speed.
- We offer an advised reflection on current generative +ai ethical and societal impact.

### Concerns

Recent advances in generative +ai for text, image, audio, and video synthesis are raising important ethical and societal concerns, especially because of its availability and ease of use. Models such as Stable Diffusion [@rombach_2021] and more recently Chat-GPT [@openai_2023] are disturbing our common beliefs and relation with copyright, creativity, the distribution of fake information and so on.

One of the main issues with generative AI is the potential for model fabulation. Generative models can create entirely new, synthetic data that is indistinguishable from real data. This can lead to the dissemination of false information and the manipulation of public opinion. Additionally, there are ambiguities surrounding the ownership and copyright of the generated content, as it is unclear who holds the rights to the generated images and videos. Training data is often obtained via online scrapping and thus copyright ownership is not propagated. This is especially true for commercial applications.

Another important concern is the potential for biases and discrimination. These models are trained on large amounts of data, and if the data is not diverse or representative enough, the model may perpetuate or even amplify existing biases. The Microsoft Tay Twitter bot [@wolf_2017] scandal is an outcome of such a phenomenon. This initially innocent chatbot has been easily turned into a racist bot perpetuating hate speech. The task was made easier because of the inherently biased dataset it was trained on.

In this work, we are committed to addressing and raising awareness for these concerns. The illustrations used for training our models and for our experiments are only used for educational and research purposes. We only provide recipes for reproducibility and do not distribute the dataset nor the weights resulting from model training, only the code. We hope this will not ensure that our work is used ethically and responsibly but limit its potential misuse.

### Outline

The first part of this thesis (chapters [1](#ch:introduction)-[3](#ch:methodology)) provides context to the recent advances in generative +ai and introduces the +cv task of user-guided automatic lineart colorization, its challenges, and our contributions to the field. It then provides additional background, from +dl first principles to current architectures used in modern generative +nn, and introduces the methodology used throughout the entire document. This part should be accessible to the majority, experts and non-experts, and serve as an introduction to the field.

The second part (chapters [4](#ch:contrib-1)-[7](#ch:contrib-4)) presents our contributions, some of which have previously been presented in [@hati_2019; @hati_2023]. It introduces into detail our recipe for sourcing and curating consistent and qualitative datasets for automatic lineart colorization, PaintsTorch [@hati_2019] our first double generator +gan conditioned on user strokes, StencilTorch [@hati_2023] our in-painting reformulation introducing the use of masks to allow the emergence of iterative workflow and collaboration with the machine, and finally StablePaint, an exploration of the use of +ddm models for variations qualitative exploration.

The third and final part (chapters [7](#ch:ethdical-and-societal-impact)-[8](#ch:conclusion)) offers a detailed reflection on this thesis's contributions and more generally about the field of generative +ai ethical and societal impact, identifies the remaining challenges and discusses future work.

The code base for the experiments and contributions is publicly available on GitHub at [https://github.com/yliess86](https://github.com/yliess86).

\newpage{}

## Background {#ch:background}

This chapter introduces the reader to the field of [+dl]{.full} from first principles to the current architectures used in modern generative +ai. The first section (section [1](#sec:history)) presents a brief history of +ai to ground this technical dissertation into its historical context. The following sections (sections [2](#sec:core)-[4](#sec:attention)) are discussing the first principles of modern +dl from the early Perceptron to more modern frameworks such as [+llm]{.full .plural}.

```python {#lst:snippet}
# This is a code snippet
print("Hello World!")
```

Additional code snippets (see @lst:snippet) are included to make this chapter more insightful and valuable for newcomers.

![A brief timeline of the History of [+ai]{.full}.](./figures/boai_timeline.svg){#fig:timeline}

### A Brief History of Artificial Intelligence {#sec:history}

The history of the field of +ai is not a simple linear and straightforward story. The field had its success and failures. The term [+ai]{.full} has first been introduced in 1956 by John Mc Carthy and Marvin Lee Minsky at a workshop sponsored by Dartmouth College [@dartmouth_2006], gathering about twenty researchers and intellectuals such as the renowned Claude Shannon (see @fig:dartmouth). The field's main questions were supposed to be solved in a short period.

However, the reality has been far less rosy. Over the years, AI has gone through several “winters”, periods of inactivity and disillusion where funding was cut and research interest dropped (see @fig:timeline). But with the advent of Big Data and the rise of [+dl]{.full}, +ai is once again in the spotlight. The following sections provide a brief overview of the history of AI, from its early days to the current state of the field. For a more in-depth look at the history of modern +ai, +dl, we recommend "Quand la machine apprend" from Yann LeCun @lecun_2019.

![Photography of seven of the Dartmouth workshop participants. From left to right: John McCarthy, Marvin Lee Minsky, Nathaniel Rochester, Claude Elwood Shannon, Ray Solomonoff, Trenchard More, and Oliver Gordon Selfridge. Credit: Margaret Minksy](./figures/boai_dartmouth.png){#fig:dartmouth}

#### The Early Years

The term [+ai]{.full} was first used at the 1956 Dartmouth Workshop [@dartmouth_2006], where John McCarthy proposed the idea of creating a machine that could learn from its mistakes and improve its performance over time. The twenty researchers and intellectuals present worked on topics such as the automatic computer, the use of natural language by machines, neuron nets ([+nn]{.full}), randomness and creativity, and many more. This was a revolutionary idea at the time, and the work done at Dartmouth attracted a great deal of attention and funding.

Much of the early research focused on symbolic +ai, which uses symbols and logical operations to represent and manipulate data. Logic programming, production rules, semantic nets and frames, knowledge-based systems, symbolic mathematics, automatons, automated provers, ontologies and other paradigms were at the core of symbolic +ai [@russell_2016]. This approach was based on the early work of Alan Turing and the development of functional languages such as the LISP by McCarthy and al. at MIT [@mccarthy_1978].

One significant contribution of this period was the Perceptron by Frank Rosenblatt [@rosenblatt_1958], a simplified biomimical model of a single neuron. This artificial neuron fires when the weighted sum of its input is above a predefined threshold. The weights, scalars attributed to the connection edges of the neuron's inputs, are tuned iteratively and manually given supervised data, inputs with corresponding labels, until good enough classification accuracy is met.

#### The First AI Winter

The Perceptron was an early example of a connectionist approach, which uses a network of artificial neurons to process data. The Perceptron was met with much enthusiasm but was eventually criticized by Marvin L. Minsky and Seymour Papert [@minsky_1969], who argued that it could not solve a simple XOR problem. The criticisms, as well as other issues, led to a period of disillusion in the field of +ai, known as the "First AI Winter". It was a time when +ai research lost its momentum and funding was not abundant anymore. This period lasted from 1973 to 1980.

#### Expert Systems and Symbolic AI

The eighties saw a resurgence of interest in +ai. Expert systems [@jackson_1998] were the new hot +ai topic. They are made of hierarchical and specialized ensembles of symbolic reasoning models and are used to solve complex problems. Symbolic +ai continued to prosper as the dominant approach until the mid-nineties.

During this period, +ai was developed as logic-based systems, search-based systems using depth-first-search, and genetic algorithms, requiring complex engineering and domain-specific knowledge from experts to work. It was also the time of the first cognitive architectures [@lieto_2021] inspired by advances in the field of neuroscience such as SOAR [@larid_2019] and +actr [@john_1992] attempting at simulating the human cognitive process for solving and task automation.

Although the connectionist approaches were not well received by the community at the time, some individuals are known for significant contributions that later would form the basis for modern +nn architectures. It was the case for Kunihiko Fukushima and his NeoCognitron [@fukushima_1980], or David E. Rumelhart et al. who introduced the most used learning procedure for training [+mlp]{.full .plural}, the backpropagation [@rumelhart_1986].

#### The Second AI Winter

Unfortunately, this period was also marked by a lack of progress because of the resource limitations of the time. Those algorithms required too much power, data, and investments to work. They were not sufficient to make AI truly successful. The lack of progress in the eighties led to the "Second AI Winter". AI research was largely abandoned during this period. Funding and enthusiasm dwindled. This winter lasted from 1988 to early 2000.

##### The Indomitable Researchers

The second AI winter limited research for +nn. However, some indomitable individuals continued their work. During this period, Vladimir Vapnik et al. developed the +svm [@cortes_1995], a robust non-probabilistic binary linear classifier. The method has the advantage to generalize well even with small datasets. Sepp Hochreiter et al. introduced the +lstm for [+rnn]{.plural} [@hochreiter_1997], a complex recurrent cell using gates to route the information flow and simulate long and short-term memory buffers. In 1989, Yann LeCun provided the first practical and industrial demonstration of backpropagation at Bell Labs with a +cnn to read handwritten digits [@lecun_1989; @lecun_1998] later used by the American postal services to sort letters.

![A brief timeline of the [+dl]{.full} Revolution.](./figures/boai_revolution.svg){#fig:revolution}

#### The Deep Learning Revolution

The next significant evolutionary step [+dl]{.full}, those deep hierarchical +nn, descendants of the connectionist movement, occurred in the early twenty-first century (see @fig:revolution). Computers were now faster and [+gpu]{.plural} were developed for high compute parallelization. Data was starting to be abundant thanks to the internet and the rapid rise of search engines and social networks. It is the era of Big Data. +nn were competing with +svm. In 2009 Fei-Fei Li and her group launched ImageNet [@deng_2009], a dataset assembling billions of labeled images.

By 2011, the speed of [+gpu]{.plural} had increased significantly, making it possible to train [+cnn]{.plural} without layer-by-layer pre-training. The rest of the story includes a succession of deep +nn architectures including, AlexNet [@krizhevsky_2012], one of the first award-winning deep +cnn, ResNet [@he_2016], introducing residual connections, the [+gan]{.full .plural} [@goodfellow_2014], a high fidelity and high-resolution generative framework, attention mechanisms with the rise of the Transformer "Attention is all you Need" architecture [@vaswani_2017] present in almost all modern +dl contributions, and more recently the [+ddm]{.full} [@ho_2020], the spiritual autoregressive successor of the +gan.

![A brief timeline of the [+dl]{.full} Milestones.](./figures/boai_milestones.svg){#fig:milestones}

#### Deep Learning Milestones

+dl is responsible for many +ai milestones in the past decade (see @fig:milestones). These milestones have been essential in advancing the field and enabling its applications within various sectors. One of the first notable milestones was AlphaGo from DeepMind in 2016 [@silver_2016], where an +ai system was able to beat the Korean world champion Lee Se Dol in the game of Go. AlphaGo is an illustration of the compression and pattern recognition capabilities of deep +nn in combination with efficient search algorithms.

In 2019, AlphStar [@vinyals_2019] from DeepMind also was able to compete and defeat grandmasters in StarCraft the real-time strategy game of Blizzard. This demonstrated the capability of Deep Learning algorithms to achieve beyond human-level performance in real-time and long-term planification.  In 2020, AlphaFold [@senior_2020] improved the Protein Folding competition by quite a margin, showing that +dl could be used to help solve complex problems that have implications for medical research and drug discovery. In 2021 a follow-up model, AlphaFold 2 [@jumper_2021], was presented as an impressive successor of AlphaFold, showcasing further advances in this field.

In 2021, Stable Diffusion [@rombach_2021] from Stability AI was released. This Latent +ddm conditioned on text prompts allows to generate images of unprecedented quality and met unprecedented public reach. Finally, Chat-GPT [@openai_2023] was released in 2023 as a chatbot based on GPT3 [@brown_2020] and fine-tuned using +rlhf for natural question-answering interaction publicly available as a web demo. However, these last two milestones are also responsible for ethical and societal concerns about copyright, creativity, and more. This highlights both the potential of Deep Learning algorithms but also the need for further research around their implications.

### Core Principles {#sec:core}

This section introduces the technical background necessary to understand this thesis dissertation. It introduces [+nn]{.full .plural} from first principles. A more detailed and complete introduction to the field can be found in "the Deep Learning book" by Ian Goodfellow et al [@goodfellow_2016] or in "Dive into Deep Learning" by Aston Zhang et al. [@aston_zhang_2021].

#### Supervised Learning

In +ml, problems are often formulated as data-driven learning tasks, where a computer is used to find a mapping $f: X \rightarrow Y$ from input space $X$ to output space $Y$. For example, $X$ could represent data about an e-mail and $Y$ the probability of this e-mail being spam. In practice, manually defining all the characteristics of a function $f$ that would satisfy this task is considered unpractical. It would require one to manually describe all potential rules defining spam. In +ml, the supervised framework offers a practical solution consisting of acquiring label data pairs, $(x, y) \in X \times Y$ for the current problem (see @fig:dataflow). In our case, this would require gathering a dataset of e-mails and asking humans to label those as spam or not.

**Objective Function**: Let us consider such a training dataset containing n independent pairs $\{(x_1, y_1), \dots, (x_n, y_n)\}$ sampled from the data distribution $D$, $(x_i, y_i) \sim D$. In +ml, we seek for learning a mapping $f: X \rightarrow Y$ by searching the space of the candidates function class $\mathcal{F}$. Defining a scalar objective function $L(\hat{y}, y)$ measuring the distance from true label $y$ and our prediction $f(x_i) = \hat{y}_i$ given $f \in \mathcal{F}$, the ultimate objective is to find the function $f^* \in F$ that best satisfy the following minimization problem (see @eq:f_star_objective):

$$
f^* = arg \; \underset{f \in \mathcal{F}}{min} \; E_{(x, y) \sim D} L(\hat{y}, y)
$$ {#eq:f_star_objective}

The function $f^*$ must minimize the expected loss $L$ over the entire data distribution $D$. Once such a function is learned one can use it to perform inference and map any element from the input space $X$ to the output space $Y$.

However, this minimization problem is intractable as it is impossible to represent the entire distribution $D$. Fortunately, as every pair $(x_i, y_i)$ is independently sampled and identically distributed, the objective can be approximated by sampling and minimizing the loss over the training dataset (see @eq:f_star_objective_approx):

$$
f^* \approx arg \; \underset{f \in \mathcal{F}}{min} \; \frac{1}{n} \sum_{i=1}^{n} L(\hat{y}_i, y_i)
$$ {#eq:f_star_objective_approx}

**Regularization**: While simplifying the problem allows us to perform loss minimization, this approximation comes at a cost. This optimization problem can have multiple solutions, a set of functions $\{f_1, \dots, f_m\} \in F$ performing well on the given training set, but would behave differently outside of the training data and outside of the data distribution. Those functions would not necessarily be able to generalize. To mitigate those concerns, we can introduce a regularization term $R$ into the objective function (see @eq:f_star_objective_regul), a scalar function that is independent of the data distribution and represent a preference on certain function class.

$$
f^* \approx arg \; \underset{f \in \mathcal{F}}{min} \; \frac{1}{n} \sum_{i=1}^{n} L(\hat{y}_i, y_i) + R(f)
$$ {#eq:f_star_objective_regul}

![Supervised learning data flow. The dataset ${(x_i, y_i)} \in D$ is used to train the model $f \in \mathcal{F}$ to minimize an objective function with two terms, a data dependant loss $L$, and a regularization $R$ measuring the system complexity.](./figures/core_nn_dataflow.svg){#fig:dataflow}

In the following, we investigate two examples where supervised learning is first applied to a [+nn]{.full} regression problem, and then a +nn classification problem. The examples highlight the objective functions composed by the loss and the regularization term for regression and classification respectively.

**Regression Problem:** Let us consider the distribution $D$ represented by the $sin$ function in the $[-3 \pi; 3 \pi]$ range (see @fig:regression). We sample $50$ pairs $(x_i, y_i)$ with $X \in [-3 \pi; 3 \pi]$  and $Y \in [-1; 1]$. Our objective is to learn a regressor $f_\theta$, a three layers +nn parametrized by its weights $\{w_0, W_1, w_2\} = \theta$. $w_0$ contains $(1 \times 16) + 1$ weights, $W_1$, $(16 \times 16) + 1$, and $w_2$, $(16 \times 1) + 1$. In this case, the function space is limited to the three layers +nn family with $291$ parameters $\mathcal{F}$.

![[+nn]{.full} regression example. The model $f_\theta$ is fit on the training set $(X, Y) \in D$ representing the $sin$ function in the range $[-3 \pi; 3 \pi]$.](./figures/core_nn_regression.svg){#fig:regression}

To achieve this goal using supervised learning, we can optimize the following objective function (see @eq:reg_sin_objective): 

$$
f^* = arg \; \underset{\theta}{min} \; \frac{1}{n} \sum_{i=1}^{n} (f_\theta(x_i) - y_i)^2 + \lambda ||\theta||_2^2
$$ {#eq:reg_sin_objective}

where the loss is the +mse $||.||_2^2$ between the ground-truth $y_i$ and the prediction $\hat{y_i} = f_\theta(x_i)$, and the weighted regularization term $\lambda ||\theta||_2^2$ to penalize the model for having large weights and converge to a simpler solution. A python code snippet for the objective function and the model is provided below (see @lst:regression):

```python {#lst:regression}
from torch.nn import (Linear, Sequential, Tanh)

# Loss and Regularization
L = lambda y_, y = (y_ - y).pow(2)
R = lambda f: sum(w.pow(2).sum() for w in f.parameters())

# Neural Network model
f = Sequential(
    Linear(1, 16), Tanh(),
    Linear(16, 16), Tanh(),
    Linear(16, 1),
)

# Objective function
C = (1 / n) * L(f(X), Y).sum() + lam *  R(f)
```

**Classification Problem:** Let us consider the distribution $D$ representing the 2d positions of two clusters ${0, 1} \in K$ of moons (see @fig:classification). We sample $250$ moon $(x_i, y_i)$ with $X \in [-1; 1]$  and $Y \in [-1; 1]$. Our objective is to learn a classifier $f_\theta$, a three layers +nn parametrized by its weights $\{w_0, W_1, w_2\} = \theta$. $w_0$ contains $(1 \times 32) + 1$ weights, $W_1$, $(32 \times 32) + 1$, and $w_2$, $(32 \times 1) + 1$. In this case, the function space is limited to the three layers +nn family with $1,091$ parameters $\mathcal{F}$.

![[+nn]{.full} classification example. The model $f_\theta$ is trained to classify moons based on their positions. The decision boundary is shown.](./figures/core_nn_classification.svg){#fig:classification}

To achieve this goal using supervised learning, we can optimize an objective function similar to the regression problem (see @eq:reg_sin_objective) using the cross-entropy as the loss function (see @eq:cross_entropy), measuring the classification discordance.

$$
\mathcal{L} (\hat{y}, y) = \sum_{k=1}^{K} y_k \; log \; \hat{y}_k
$$ {#eq:cross_entropy}

A python code snippet for the loss function and the model is provided below (see @lst:classification):

```python {#lst:classification}
from torch.nn import (Linear, Sequential, Tanh)
from torch.nn.functional import cross_entropy

# Loss
L = lambda y_, y = cross_entropy(y_, y, reduce=False)

# Neural Network model
f = Sequential(
    Linear(1, 32), Tanh(),
    Linear(32, 32), Tanh(),
    Linear(32, 1),
)
```

#### Optimization {#sec:optimization}

In +ml, supervised problems can be reduced to an optimization problem where the computer has to find a set of parameters, weights $\theta$, for a given function class $\mathcal{F}$ by optimizing an objective function $\theta^* = arg \; min_\theta \mathcal{C(\theta)}$ made out of two components, a data-dependant loss $L$ and a regularization $R$.

**Random Search:** One way to find such a function $f_\theta$ that satisfies this objective is to estimate the objective function for a set of random parameter initializations and take the one that minimizes $C$ the most. This $\theta$ setting can then be refined by applying random perturbations to the parameters and repeating the operation (see @lst:random_search). This is possible due to the fact that we can computer $C(\theta)$ for any value of $\theta$ taking the average loss for a given dataset. However, such an approach to optimization is unpractical. +nn often comes with millions or billions of parameters $\theta$ making random-search intractable.

```python {#lst:random_search}
import copy
import numpy as np


for step in range(steps):
    fs, os = [f] + [copy.deepcopy(f) for f in range(n_copy)], []
    for f_ in fs:
        # Apply weight perturbation
        for w in f_.parameters():
            w.normal_(0.0, 1.0 / step)
        # Estimate the objective function
        os.append(C(f_(X), Y))
    
    # Retrieve the winner
    f = fs[np.argmax(os)]
```

**First Order Derivation:** A more efficient approach is to make the objective function $C$ and the model $f_\theta$ differentiable. This constraint allows us to compute the gradient of the cost $C$ with respect to the model's parameters $\theta$. The value $\nabla_\theta C$ can be obtained using backpropagation (discussed in the next sub-section @sec:backpropagation). This vector of first order derivatives indicates the direction from which we need to move the weights $\theta$ away. By taking small iterative steps toward the negative direction of the gradients, we can improve $\theta$. This algorithm is called +gd. In practice, due to the very large size of the datasets ($14,197,122$ images for ImageNet [@deng_2009]), the objective gradient is approximated using a small subset of the training data for each step referred to as a minibatch. This approximation of the +gd is called +sgd (see @lst:sgd).

```python {#lst:sgd}
for step in range(1_000):
    # Retrieve the next minibatch
    x, y = next_minibatch(X, Y)

    # Compute the objective function and the gradients
    C = L(f(x), y) + lam * R(f)
    C.backward()

    # Update the weights and reset the gradients
    for w in f.parameters():
        w -= eps * w.grad
    f.zero_grad(set_to_none=True)
```

One critical aspect of the +sgd algorithm is the hyperparameter $\epsilon$, the learning rate. It controls the size of the step we take toward the negative gradients. If it is too height or too low, the optimization may not converge toward an acceptable local minimum. A toy example is provided in @fig:toysgd where different learning rates are used to find the minimum of the square function $y = x^2$.

![Toy example where different learning rates $\epsilon$ are used to find the minimum of the square function $y = x^2$ using the [+gd]{.full} algorithm starting from $x = -1$. Some learning rate setup result in situations where the optimization does not converge to the solution. A learning rate $\epsilon = 2$ diverges toward infinity, $\epsilon = 1$ is stuck and bounces between two positions $-1$ and $1$. However, a small learning rate $\epsilon = 0.1 < 1$ converges towards the minimum $y = 0$. This example illustrates the impact of the hyperparameter $\epsilon$ on +gd.](./figures/core_nn_sgd.svg){#fig:toysgd}

**First Order Derivation with Momentum:** The +dl literature contains abundant work on first order optimizer variants aiming for faster convergence such as +sgd with Momentum [@qian_1999],  Adagrad [@duchi_2011], RMSProp [@hinton_lecture6a], Adam [@kingma_2014], and its correction AdamW [@loshchilov_2017]. A toy example is shown @fig:sgd_moments.

The Momentum update [@qian_1999] introduces the use of a momentum inspired by physics' first principles to favor small and consistent gradient directions. In this particular case, the momentum is represented by a variable $v$ updated to store an exponential decaying sum of the previous gradients $v := \alpha v + \nabla_\theta C(\theta)$. The weights are then updated using negative $v$ as the gradient direction instead of $\nabla_\theta C(\theta)$.

Other optimizers also make use of the second moment of the gradients. Adagrad [@duchi_2011] uses another variable $r$ to store the second moment $r := r + \nabla_\theta C(\theta) \odot \nabla_\theta C(\theta)$ and modulate the update rule toward the negative direction $\frac{1}{\delta + \sqrt{r}} \odot \nabla_\theta C(\theta)$ where $\delta$ is a small value to avoid division by zero. Similarly, RMSProp [@hinton_lecture6a] maintains a running mean of the second moment $r := \rho r + (1 - \rho) \nabla_\theta C(\theta) \odot \nabla_\theta C(\theta)$.

Finally Adam [@kingma_2014], and its correction AdamW [@loshchilov_2017], are applying both Momentum and RMSProp estimating the first and second moment to make parameters with large gradients take small steps and parameters with low gradients take larger ones. This has the advantage to allow for bigger learning rates and faster convergence at the cost of triple the amount of parameters to store during training. A simple implementation of Adam is shown below (see @lst:adam):

<!-- - Adagrad, RMSProp, AdamW
- Adam: Big Gradient = Small Steps, Small Gradient == Big Steps -->

```python {#lst:adam}
# Adam state (parameters, gradients first and second moments)
params = list(f.parameters())
d_means = [w.clone().zeros_() for w in params]
d_vars  = [w.clone().zeros_() for w in params]

for step in range(1_000):
    # Retrieve the next minibatch
    x, y = next_minibatch(X, Y)

    # Compute the objective function and the gradients
    C = L(f(x), y) + lam * R(f)
    C.backward()

    data = zip(params, d_means, d_vars)
    for w_idx, (w, d_m, d_v) in enumerate(data):
        # Update the moments (mean and uncentered variance)
        d_m = beta1 * d_m + (1 - beta1) * w.grad
        d_v = beta2 * d_v + (1 - beta2) * (w.grad ** 2)

        # Compute bias correction
        corr_m = d_m / (1.0 - beta1 ** step)
        corr_v = d_v / (1.0 - beta2 ** step)

        # Update weight and reset the gradient
        w -= eps * (corr_m / (corr_v.sqrt() + 1e-8))
        w.grad = None
```

![This toy example illustrates the impact of the optimizer choice during objective minimization with first order methods. +sgd, Momentum, Adagrad, RMSProp and Adam are tasked to find the minimum of a 1-dimensional mixture of Gaussians given the same starting point $x = 1$ and the same learning rate $\epsilon = 0.5$. In this particular setup, Moments and Adagrad find the solution, RMSProp explodes, and +sgd and Adam are stuck into a local minimum.](./figures/core_nn_sgd_moments.svg){#fig:sgd_moments}

**Cross-Validation and HyperParameter Search:** As illustrated by the toy examples (see [@fig:toysgd; @fig:sgd_moments]), the training of +nn using +sgd is highly dependent on the initial setting of hyperparameters. One could ask if there is a rule for choosing such parameters. Unfortunately, this is not the case. The field is highly empirical and driven by exploration using the scientific method.

One common approach is to set up metrics to evaluate the performance of the model during the optimization process. It is a good practice to divide the dataset into validation folds that are different from the training data to evaluate the generalization capabilities of the model. This practice is referred to as $k$-fold cross-validation and is most of the time in +dl, because of the large datasets, reduced to a single fold, called the validation set. By defining such a process, +nn can be compared in a controlled manner and the hyperparameter space can be searched. Hyperparameter search is so important that it is a subfield of its own. The broad +dl literature however contains many examples of initial parameters and architectures that can be used to bootstrap this search.

#### Backpropagation {#sec:backpropagation}

In the previous sub-section (see @sec:optimization), we saw how to learn parametrized functions $f_\theta$ given a training dataset. By evaluating the gradients of the objective function with respect to the model's parameters, it is possible to obtain a good enough mapping $f_\theta: X \rightarrow Y$. In this sub-section, we discuss backpropagation, the recursive algorithm used to efficiently compute those gradients exploiting the chain rule $\frac{\partial z}{\partial x} = \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial x}$ with $z$ dependant on $y$ and $y$ on $x$.

![Illustration of reverse mode [+ad]{.full}. This [+dag]{.full} shows the forward pass in green and backward in red. The gradient of an activation is computed by multiplying the local gradient of a node by its output gradient computed in the previous step when following backward differentiation $\frac{\partial C}{\partial x} = \frac{\partial z}{\partial x} \cdot \frac{\partial C}{\partial z}$ where $\frac{\partial z}{\partial x}$ is the location derivative and $\frac{\partial C}{\partial z}$ the output one.](./figures/core_nn_dag.svg){#fig:dag}

**Automatic Differentiation:** In mathematics, +ad describes the set of techniques used to evaluate the derivative of a function and exploits the fact that any complex computation can be transformed into a sequence of elementary operations and functions with known symbolic derivatives. By applying the chain rule recursively to this sequence of operations, one can automatically compute the derivatives with precision at the cost of storage.

We distinguish two modes of operation for +ad, forward mode differentiation, and reverse mode differentiation. In forward mode, the derivatives are computed after applying each elementary operation and function in order using the chain rule. It requires storing the gradients along the way and carrying them until the last computation. This mode is preferred when the size of the outputs exceeds the size of the inputs. This is generally not the case for +nn where the input, an image for example, is larger than the output, a scalar for the objective function. On the opposite, reverse mode differentiation traverses the sequence of operations from end to start using the chain rule and requires storing the output of the operations instead. This method is preferred when the size of the inputs exceeds the outputs. This mode thus has to happen in two passes, a forward pass where one computes the output of every operation in the order, and a backward pass, where the sequence of operations is traversed in backward order to compute the derivatives.

**Computation Graph:** A +nn can be defined as a succession of linear transformations followed by non-linear activations (discussed in the next section [@sec:nn]). Those elementary operations are differentiable and when thinking of the data flow can be viewed as a computation +dag to which backpropagation, reverse mode differentiation, can be applied.

In modern +dl frameworks [@pytorch; @tensorflow], the +ad is centered on the implementation of a Graph object with Nodes. Both entities possess a `forward()` and a `backward()` function. The forward pass calls the `forward()` function of each node of the graph by traversing it in order while saving the node output for differentiation. The backward pass traverses the graph recursively in backward order calling the `backward()` function responsible for computing the local gradient of the node operation and multiplying it by its output gradient following the chain rule. Nodes are in most frameworks referred to as Layers, the elementary building block of the +nn operation chain.

**Toy Implementation:** Here is a simple implementation of such a computation graph for backpropagation and +ad engines adapted from Micrograd by Andrej Karpathy [@karpathy_micrograd]. The Node class is responsible for storing the value, the chained gradient, and additional information to trace the graph for the backward pass.

```python
from dataclasses import dataclass

@dataclass
class Node:
    value: float
    grad: float = 0.0
    _backward = lambda: None
    _children: set[Node] = {}
    _op = ""
```

The Node can then be populated with elementary operations (`__add__`, `__mul__`) and functions (`tanh`).

```python
import numpy as np


class Node:
    ...
    def __add__(self, other: Node) -> Node:
        out = Node(self.value + other.value, {self, other}, "+")

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        
        return out

    def __mul__(self, other: Node) -> Node:
        out = Node(self.value * other.value, {self, other}, "*")

        def _backward() -> None:
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad
        out._backward = _backward
        
        return out

    def tanh(self) -> Node:
        act = np.tanh(self.value)
        out = Node(act, {self}, "tanh")

        def _backward() -> None:
            self.grad += (1.0 - act ** 2) * out.grad
        out._backward = _backward
        
        return out
```

Every elementary transformation needs to be differentiable and implements its own backward function using the chain rule. The chained gradient stored in the node is the multiplication of the local gradient with its output gradient computed when the parent node is encountered during the backward pass. The Node object needs to be extended with support for other elementary operations (e.g. `__pow__`, `__neg__`) and functions (e.g. `sigmoid`, `relu`) to be useful for +dl.

We add the ability for a Node to compute its backward pass by first tracing all the current +dag operations recursively. The gradients can then be computed by initializing the first node (the last in the graph) gradient to $1$. The backward call on the graph iteratively traverses the graph from end to start and applies the inner backward functions to compute the chain gradients along the way while storing them in their respective Node object.

```python
class Node:
    ...
    def backward(self) -> None:
        trace, visited = [], {}
        def trace_graph(node: Node) -> None:
            if node not in visited:
                visited.add(node)
                for child in node._children:
                    trace_graph(child)
                trace.append(node)
        trace_graph(self)

        self.grad = 1.0
        for node in trace[::-1]:
            node._backward()
```

The simple [+ad]{.full} engine is now ready to perform forward and backward passes. The gradients stored in the node can then be used for [+sgd]{.full} to update the weights of a [+nn]{.full} for example.

```python
w1, w2 = Node(0.1), Node(0.2)  # Weights
a,  b  = Node(1.0), Node(0.0)  # Inputs

z = (w1 * a + w2 * b).tanh()   # Eager forward pass
z.backward()                   # Backward pass
```

Fortunatly open-source implementations of such engines are already available and extensively used by the +dl community. They have the adantage to work at the Tensor level, not at the Scalar level like Micrograd, and offer support for accelerated hardware such as [+gpu]{.full .plural}, [+tpu]{.full .plural}, and [+npu]{.full .plural}. In this dissertation, most examples are using the PyTorch [@pytorch] framework, a Python Tensor library written in C++ and equipped with a powerful eager mode reverse +ad engine.  

**Eager or Graph Execution:** Modern +dl frameworks such as PyTorch [@pytorch] and Tensorflow [@tensorflow] now propose two execution modes. An eager mode, where the graph is built dynamically and operations are applied immediately, and a graph mode where the computational graph has to be defined beforehand. Both modes come with advantages and inconveniences. Eager mode is useful for iterative development and provides an intuitive interface similar to imperative programming, it is easier to debug and offers natural control flows as well as hardware acceleration support. On the other side, graph mode allows for more efficient execution. The graph can be optimized by applying operations similar to the ones used in programming language [+ast]{.plural}. Graph edges can be merged into a single fused operation, and execution can be optimized for parallelization. It is often the preferred way for deployment where the execution time and memory are at stake.

### Neural Networks {#sec:nn}

In the previous section, we described the general setup for +ml, where one has to fit a model from a given function family $f \in \mathcal{F}$ on a given dataset $(X, Y) \in D$ optimized using +sdg and backpropagation. This section begins discussing a particular class of parameterized function $f_\theta$ called [+nn]{.full .plural}. 

#### Perceptron {#sec:perceptron}

The Perceptron, introduced by Frank Rosenblatt in 1958 [@rosenblatt_1958], is the building block of [+nn]{.full .plural}. It was introduced as a simplified model of the human neuron, containing three parts: dendrites handling incoming signals from other neurons, a soma with a nucleus responsible for signal aggregation, and an axone responsible for the transmission of the processed signal to other neurons. When the signal aggregation in the soma reaches a predefined threshold, the neuron activates. This phenomenon is called an action potential. Although this is not an accurate representation of the modern neuroscience state of knowledge, this simplified model was believed to be accurate at the time.

![Diagram of a Perceptron with three inputs $\{x_1; x_2; x_3\}$. The perceptron computes an activated weighted sum of its inputs $y = \sigma(\sum_{i=1}^{3} w_i \cdot x_i)$ where $\sigma$, the activation function is a threshold function.](./figures/core_nn_perceptron.svg){#fig:perceptron}

Similarly, the Perceptron computes a weighted sum of its inputs and activates if a certain threshold is reached (see @fig:perceptron). The Perceptron is parametrized by the weights representing the importance attributed to the incoming inputs and are part of the parameters $\theta$ that are trained on a given dataset. It can be viewed as a learned linear regressor followed by a non-linear activation, historically a threshold function, a function $\sigma$ that activates $\sigma(x) = 1$ when $x > 0.5$ and $\sigma(x) = 0$ otherwise (see @lst:perceptron).

```python {#lst:perceptron}
def perceptron(self, x: Tensor, W: Tensor) -> Tensor:
    return (x * self.W.T) > 0.5
```

The objective of a perceptron is to learn a hyperplane, a plane with $n - 1$ dimensions where $n$ is the number of inputs, that can perform binary classification, separate two classes. However, as mentioned by Marvin L. Minsky and al. in their controversial book Perceptrons [@minsky_1969], a hyperplane regressor cannot solve a simple XOR problem (see @fig:xor).

![Illustration of the Perceptron's decision hyperplane when trained to solve the AND problem on the left, the OR problem in the middle, and the XOR problem on the right. The first two problems are linearly sperable, thus adapted for a Perceptron. However, a single perceptron cannot solve the XOR problem as it is not linearly separable.](./figures/core_nn_xor.svg){#fig:xor}

#### Multi-Layer Perceptron {#sec:mlp}

The real value of the Perceptron comes when assembled into a hierarchical and layer-wise architecture, a [+nn]{.full}. By repeating matrix multiplications (linear transformations) and non-linearities the network is able to handle non-linear problems and act as a universal function approximator [@hornik_1989]. This arrangement of layered perceptrons is called a [+mlp]{.full} (see @fig:mlp).

![Diagram of a 3-layer [+mlp]{.full}. When using the matrix formulation, this arrangement of neurons can be summarized into a single expression $y = \sigma(\sigma(x \cdot W_1^T) \cdot W_2^T) \cdot W_3^T$.](./figures/core_nn_mlp.svg){#fig:mlp width=90%}

A +mlp with Identity as its activation function is useless as its chain of linear transformations can be collapsed into a single one. Since the advent of the Perceptron, the literature has moved away from using threshold functions as activations. Common activation functions are the sigmoid $\sigma(x) = \frac{1}{1 + e^{-x}}$, tanh $tanh(x) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}$, +relu $ReLU(x) = max(x, 0)$ functions and variants presenting additional properties such as infinite continuity, gradient smoothness, and more (see @fig:activations).

![Activation functions. Sigmoid $\sigma(x) = \frac{1}{1 + e^{-x}}$ acts as a filter $y \in [0; 1]$, tanh $tanh(x) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}$ acts as a normalization compressor $y \in [-1; 1]$, +relu $ReLU(x) = max(x, 0)$ folds all negatives down to zero $y \in [0; +\infty]$.](./figures/core_nn_activations.svg){#fig:activations}

**MNIST Classifier:** A classic toy example showing the capabilities of [+mlp]{.plural} is the handwritten digit classification challenge on the +mnist dataset [@mnist]. +mnist contains $60,000$ training and $10,000$ test examples. It has been written by high school students and gather $28 \times 28$ centered black and white handwritten digits from $0$ to $9$ (see @fig:mnist).

![First $27$ handwritten digits from the [+mnist]{.full} dataset. The digits are stored as $28 \times 28$ centered black and white images.](./figures/core_nn_mnist.svg){#fig:mnist}

Training a +mlp on such a challenge is simple and effective. With little training, parameters (according to the +dl standards), and no hyperparameter tweaking, a vanilla 3-layer +nn with ReLU activations can achieve $97.5%$ accuracy on the test set. The inputs however need to be transformed before ingestion by the model as [+mlp]{.plural} are constrained to $1$-dimensional input vectors. The following demonstrates how to implement such a model and train it on +mnist.

```python
from torch.utils.data import (Subset, DataLoader)
from torchvision.datasets.mnist import MNIST
from torchvision.transforms.functional import to_tensor

# Load MNIST images as Tensors and Normalize [0; 1]
T = lambda x: to_tensor(x).float().flatten()
dataset = MNIST("dataset", train=True,  transform=T.ToTensor())
testset = MNIST("dataset", train=False, transform=T.ToTensor())

# Split dataset in Train and Validation Splits
n, split = len(dataset), int(np.floor(0.8 * len(dataset)))
train_idxs = np.random.choice(range(n), size=split, replace=False)
valid_idxs = [idx for idx in range(n) if idx not in train_idxs]
trainset = Subset(dataset, indices=train_idxs)
validset = Subset(dataset, indices=valid_idxs)

# Mini Batch Loaders (Shuffle Order for Training)
trainloader = DataLoader(trainset, batch_size=1_024, shuffle=True )
validloader = DataLoader(validset, batch_size=1_024, shuffle=False)
testloader  = DataLoader(testset,  batch_size=1_024, shuffle=False)
```

The first step consists in loading the +mnist dataset and applying preprocessing to the data for preparing the ingestion by the model. The images need to be transformed into a normalized tensor and flatten to form a $1$-dimensional vector. The datasets are split into a training set, a validation set, and a test set. A mini-batch loader is then used to wrap the dataset and load multiple input and output pairs at the same time.

```python
from torch.nn import (Linear, Module, ReLU, Sequential)
from torch.optim import (AdamW, Optimizer)

# Model and Optimizer
model = Sequential(
    Linear(28 * 28, 128), ReLU(),
    Linear(    128, 128), ReLU(),
    Linear(    128,  10),
)
optim = AdamW(model.parameters(), lr=1e-2)
```

Then, the model is defined as a sequence of three linear layers (linear transformations with a bias for the intercept) and ReLU activations except for the last one responsible for outputting the logits, used for computing the loss, here the cross entropy for multi-class classification. The enhanced +sgd optimizer, Adam, is then initialized with the model's weight and a learning rate $\epsilon$. AdamW is a variant of Adam with a corrected weight decay term for regularization.

```python
from torch import Tensor
from torch.nn.functional import cross_entropy

# Perform one Step and estimate Metrics
def step(
    model: Module,
    optim: Optimizer,
    imgs: Tensor,
    labels: Tensor,
    split: str,
) -> Tuple[float, float]:
    logits = model(imgs)                  # Prediction
    loss = cross_entropy(logits, labels)  # Mean Loss
    n_correct = logits.argmax(dim=-1)     # Correct Predictions

    # Train if split is "train"
    if split == "train":
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)

    return loss.item(), n_correct.item()
```

The `step` function is responsible for performing one training step when the given split is set to `"train"` and computes the metrics used for monitoring. In our case, we monitor the average loss and the accuracy of the model. For a more complete evaluation, other metrics such as the F-$1$ score, the perplexity, the recall, and a confusion matrix can be evaluated. They are here omitted for the sake of illustration and simplicity. 

```python
# Train for 10 epochs
for epoch in range(10):
    # Training
    model.train()
    loss, acc = 0, 0
    for imgs, labels in trainloader:
        metrics = step(model, optim, imgs, label, "train")
        loss += metrics[0] / len(trainloader)
        acc  += metrics[1] / len(trainloader.dataset)
    print(f"[Train] Epoch {epoch}, loss: {loss:.2e}, acc: {acc * 100:.2f}%")

    # Validation
    model.eval()
    with torch.inference_mode():
        loss, acc = 0, 0
        for imgs, labels in validloader:
            metrics = step(model, optim, imgs, label, "valid")
            loss += metrics[0] / len(validloader)
            acc  += metrics[1] / len(validloader.dataset)
    print(f"[Valid] Epoch {epoch}, loss: {loss:.2e}, acc: {acc * 100:.2f}%")

# Test
model.eval()
with torch.inference_mode():
    loss, acc = 0, 0
    for imgs, labels in testloader:
        metrics = step(model, optim, imgs, label, "test")
        loss += metrics[0] / len(testloader)
        acc  += metrics[1] / len(testloader.dataset)
print(f"[Test] loss: {loss:.2e}, acc: {acc * 100:.2f}%")
```

Finally, the model is trained for $10$ epochs, the number of times the entire dataset is looped through. This number was arbitrarily chosen to correspond with the loss saturation when the model does not improve much. A training loop is divided into a few steps, a training phase where one continuously performs a training step followed by a validation step to monitor generalization, and when stopped, a test phase to monitor model generalization without bias. This last step prevents trying to overfit the validation set specifically and should be performed at the very end. An example of training history is shown in @fig:mnist_history. In this example, the model reaches $97.5%$ accuracy. By spending time tweaking the hyperparameters (the model's weights, the learning rate, the number of epochs, ...), the model can be improved further.

![Training history of a 3-layer [+mlp]{.full} with $128$ neurons in every layer on the +mnist dataset. The average loss (cross-entropy) on the left, and the accuracy on the right are displayed for the training, validation, and test splits.](./figures/core_nn_mnist_history.svg){#fig:mnist_history}

#### Convolutional Neural Network {#sec:cnn}

While [+mlp]{.plural} can be viewed as universal function approximators, they scale poorly with respect to high dimensional inputs such as images, videos, sound representations such as a spectrogram, volumetric data, and long sequences. For example, if we consider a small RGB image of size $256 \times 256 \times 3$, the input of a +mlp would be a 1-dimensional vector of size $196,608$. The input layer of a +mlp with $64$ neurons would already mean that the network contains more than $12,582,912$ parameters. For this reason, researchers have created specialized [+nn]{.plural} with biases in their architecture inspired by cognitive and biophysical mechanisms. [+cnn]{.full .plural} (ConvNets) are such a +nn specialized in handling spatially correlated data such as images.

![Illustration of a single $3 \times 3 \times 3$ filter convolution in the middle applied to a $8 \times 8 \times 3$ input tensor on the left. The result is a $6 \times 6 \times 1$ activation map on the right. The filter receptive field is drawn in dashed lines. This convolution is applied in valid model, no passing was applied to the input resulting in a lower resolution output tensor.](./figures/core_nn_convolution.svg){#fig:convolution}

**Convolution:** The core component of a ConvNet is the convolution operation. A +CNN operates by convolving (rolling) a set of parametrized filters on the input. If we reconsider our $W_1 \times H_1 \times D_1 = 256 \times 256 \times 3$, convolving a single filter of size $F_W \times F_H \times D_1 = 3 \times 3 \times 3$ would require sliding the filter across the entire input image tensor and computing the dot product of the overlapping tensor chunk and the filter. This operation results in what is called an activation map, or feature map. The filter can be convolved in different configurations. The stride $S$ defines the hop size when rolling the filter over the input, and the padding $P$ defines the additional border added to the input tensor in order to parkour the input border ($252$ unique positions for the filter in the $256$ image, $256$ positions with a padding of $1$ on each side of the input). A +cnn convolves multiple parametrized filters $K$ in a single convolution operation. Given a convolution setting, the operation requires $F_H \times F_W \times D_1 \times D_2$ parameters and outputs a feature map tensor of size $W_2 = (W_1 - F_W + 2P_W) / S + 1$, $H_2 = (H_1 - F_H + 2P_H) / S + 1$, and $D_2 = K$ (see @fig:convolution). The different filters are responsible for looking for the activation of different patterns in the input. The Convolution layer introduces the notion of weight sharing enabled by the sliding filter (neurons) and reduces computation by a large margin in comparison to a standard +mlp layer.

**Pooling:** It is common to follow convolution layers by pooling layers to reduce the dimensionality when growing the ConvNet deeper. The pooling layer reduces its input by applying a reduction operation. The reduction operation can be taking the `max`, `min`, or `average`, of a rolling window. This operation does not involve any additional parameter and is applied channel-wise. If we consider a max-pooling operation with a $2 \times 2$ kernel and a stride of $2$, the output becomes half the size of the input. It also has the benefit of making the +cnn more robust to scale and translation. It is sometimes more strategic to make use of stride instead of adding pooling layers. It has the same benefit of reducing the feature map size while avoiding an additional operation.

![Illustration of a small [+cnn]{.full .plural} containing a convolution (conv) layer, a max-pooling (maxpool), and another convolution followed by another max-pooling. The last feature map is then flattened into a $1$-dimensional vector and used as the input for the [+mlp]{.full} classifier.](./figures/core_nn_convnet.svg){#fig:convnet}

**ConvNet:** Finally, a +cnn is assembled by stacking multiple convolution layers and pooling layers. When the feature maps are small enough, the final feature map is flattened and passed to an additional +mlp in charge of the classification or regression. This combination of a parametric convolutional feature extractor and a +mlp is what we call a ConvNet.

![Visualization of VGG16's four first activation maps (feature maps). The input image is left and the activations are shown in order of the layers top to bottom and left to right. Credit [https://images.all4ed.org/](https://images.all4ed.org/)](./figures/core_nn_vgg_activations.svg){#fig:vgg_activations}

**Feature Maps:** The feature maps learned by a +cnn are hierarchical. In the first layers, the learned filters are focusing on simple features such as lines, diagonals, and arcs, and act as edge detectors. The deeper the layers are, the more complex the features are because they are resulting from a succession of combinations from previous activations (see @fig:vgg_activations). 

**Finetuning:** Training [+cnn]{.plural} on bigger and more diverse datasets allows learning more general filters increasing the likelihood that the network will perform on out-of-domain data. In practice, [+cnn]{.plural} are not often trained from scratch. Such a process requires the use of expensive dedicated hardware and hours of training. However, thanks to the open-source mindset of the +dl field, big actors often share the weights of such models referred to as pretrained models, or foundation models [@foundation_2021].

Foundation models can be further refined through smaller training on smaller and specialized datasets containing few good-quality examples. This process is called finetuning and is less expensive and time-consuming than full training. One method for finetuning consists in removing the classification head of a pre-trained model such as VGG16 [@simonyan_2014] and replacing it with a new one adapted to the number of classes required for the task. The pretrained weights are then frozen (not updated during training), and the new weights are trained following a standard supervised-learning procedure (see @lst:finetune).

```python {#lst:finetune}
from torchvision.models import (vgg16, VGG16_Weights)

# Import pretrained VGG16 model
model = vgg16(weights=VGG16_Weights.DEFAULT)

#Freeze pretrained features weights
for param in model.features.parameters():
    param.requires_grad = False

# Replace the classifier head
model.classifier = Sequential(
    Linear(512 * 7 * 7, 512), ReLU(),
    Linear(512,         512), ReLU(),
    Linear(512, num_classes),
)
```

**MNIST Classifier:** Let us reconsider the +mnist toy classification example and replace the +mlp with a +cnn. The model is divided in two sections, the feature extractor made out of two convolutional and max-pooling layers with $5 \times 5$ filters, the middle layer responsible for flattening the feature maps down to a $1$-dimensional vector fed to the classifier head, a $3$-layer +mlp similar to the first one. The training procedure is left unchanged, the number of parameters is approximately similar, a little less for the +cnn, and the number of epochs is the same. The input is however not flattened as the +cnn consumes a full image tensor.

```python
from collections import OrderedDict
from torch.nn import (Conv2d, Flatten, Linear, MaxPool2d, ReLU, Sequential)
from torchvision.transforms.functional import to_tensor

# Load MNIST images as Tensors and Normalize [0; 1]
T = lambda x: to_tensor(x).float()
...

# Model
model = Sequential(OrderedDict(
    features=Sequential(
        Conv2d(1,  6, 5), ReLU(), MaxPool2d(2),
        Conv2d(6, 16, 5), ReLU(), MaxPool2d(2),
    ),
    flatten=Flatten(),
    classifier=Sequential(
        Linear(256, 128), ReLU(),
        Linear(128,  64), ReLU(),
        Linear( 64,  10),
    ),
))
...
```

The +cnn is able to achieve a $99%$ accuracy on the test set early during training (epoch $5$). The +cnn is a more robust, specialized, and thus more efficient architecture for handling images. The training history can be observed in @fig:mnist_convnet_history. 

![Training history of a [+cnn]{.full} made out of a sequence of two convolutions followed by max-pooling and a 3-layer [+mlp]{.full} classifier on the +mnist dataset. The average loss (cross-entropy) on the left, and the accuracy on the right are displayed for the training, validation, and test splits.](./figures/core_nn_mnist_convnet_history.svg){#fig:mnist_convnet_history}

### Generative Architectures {#sec:generative}

In this section, we extend our [+dl]{.full} architecture toolbox with generative +ai architectures such as the [+ae]{.full} (see @sec:ae), the [+vae]{.full} (see @sec:vae), the [+gan]{.full} (see @sec:gan), and the [+ddm]{.full} (see @sec:ddm) with a strong focus on image generation. Similarly to the previous sections, the [+mnist]{.full} dataset is used for illustrative purposes.

This section does not only discuss the technical details of those architectures but also compares them on three criteria, generation inference speed, generation variance, and generation quality and complexity.

#### Autoencoders {#sec:ae}

[+ae]{.full .plural} are part of a family of feedforward [+nn]{.plural} for which the input tensor is the same as the output tensor. They encode (compress), the input into a low-dimensional code in a latent space, and then decode (reconstruct) the original input from this compressed representation (see @fig:gai_autoencoder). An +ae is built using two network parts, an encoder $E$, +nn that reduces the input dimension, a decoder $D$ that recovers the input $x$ from the reduced tensor $z$, and a reconstruction objective. This architecture can be viewed as a dimensionality reduction technique but can be used as a generative model. By feeding the decoder $D$ with arbitrary latent codes $z$, one can generate unseen data points $\hat{x}$ similar to the training distribution by interpolation. Additional training objectives can be used to disentangle the latent representation so that the data points are organized mindfully in the latent space, semantically for example.

![[+ae architecture. The encoder compresses the input into a latent code that is reconstructed using the decoder.]{.full} architecture.](./figures/core_gai_autoencoder.svg){#fig:gai_autoencoder}

**Properties:** Compared to a traditional compression method, [+ae]{.plural} are tied to their training data. They are trained to learn data-specific features useful for in-domain compression not for out-of-domain. An +ae trained on +mnist cannot be used for compressing photos of faces. Such architecture cannot be considered a lossless compression algorithm. The reconstruction is most of the time degraded. One strong advantage of using an +ae is that they do not require complex data preparation. They are part of the unsupervised training family, where labeled data is not needed for training, and in this case, self-supervised learning where the target output is built synthetically from the input.

**MNIST Digit Image Generation:** Let us consider +mnist and train a small [+ae]{.full} to compress handwritten digits to $32$ latent codes. Our +ae is made out of a small +mlp encoder and decoder both with two inner layers with a hidden dimension of $128$.

```python
from collections import OrderedDict
from torch.nn import (Linear, ReLU, Sequential, Sigmoid)

# Model definition
model = Sequential(OrderedDict(
    encoder=Sequential(
        Linear(28 * 28, 256), ReLU(),
        Linear(    256,   2),
    ),
    decoder=Sequential(
        Linear(  2,     256), ReLU(),
        Linear(256, 28 * 28), Sigmoid(),
    ),
))
```

The model is trained on $10$ epochs with no hyperparameter tuning using the `binar_cross_entropy` objective function as the dataset contains black and white images normalized in the $[0; 1]$ range.

```python
from torch.nn.functional import binary_cross_entropy

# Compute loss
loss = binary_cross_entropy(model(x), x)
```

![Training history of a small $2$-layer [+ae]{.full}. The binary cross entropy loss is shown on the left, a training sample in the middle, and its corresponding reconstruction on the right.](./figures/core_gai_autoencoder_history.svg){#fig:gai_autoencoder_history}

The result of the training can be observed in @fig:gai_autoencoder_history. Despite little degradation, our model can reconstruct the handwritten digits from their latent code. The degradation is minimized by the fact that we are dealing with a toy dataset. The phenomenon can be observed by reducing the number of parameters of the network or the size of the latent space. To reconstruct the images, we first need to get a latent code, either by encoding an existing image, or by randomly initializing a latent vector in a reasonable range, and providing it to the decoder as shown below.

```python
# Generate sample given latent-code
x_ = model.decoder(z)
```

![Trained [+ae]{.full} $2$-dimensional latent space visualization. The data points represent the encoded latent code of images from the +mnist dataset and are colored based on their corresponding label (digit). The latent space is not organized in a way that allows us to visually separate these classes.](./figures/core_gai_autoencoder_latent.svg){#fig:gai_autoencoder_latent width=60%}

The $2$-dimensional latent space can be observed in @fig:gai_autoencoder_latent. Our latent space is not organized in a way that we can visually distinguish between the digit classes. This clearly demonstrates a lack of structural organization preventing the +ae from being used as a generator by sampling its latent space.

![Trained [+ae]{.full} $2$-dimensional latent space sampling visualization. The decoder is used for generation by sampling the latent space in a grid pattern.](./figures/core_gai_autoencoder_latent_sampling.svg){#fig:gai_autoencoder_latent_sampling}


#### Variational Autoencoders {#sec:vae}

Due to a lack of latent space regularization as shown in the previous sub-section, +ae cannot be used without any hacking to generate, or produce unseen samples. A vanilla +ae does not encode any structure on the latent space. It is trained only for reconstruction and is thus subject to high overfitting resulting in a meaningless structural organization of the latent codes. The [+vae]{.full} architecture [@kingma_2013] is one answer to this issue. It can be viewed as a special +ae hacked by adding a regularization objective enabling generation by exploring the learned and structured latent space (see @fig:gai_vae).

![[+vae architecture. The encoder compresses the input and regresses the latent distribution parameters $\mu$ and $\rho$ from which a latent code is sampled using the reparametrization trick with a surrogate parameter $\zeta$ sampled from the standard Gaussian distribution and then decoded to recover the input using the decoder.]{.full} architecture.](./figures/core_gai_vae.svg){#fig:gai_vae}

**Regularization:** [+vae]{.plural} are topologically similar to +ae. They possess an encoder to compress the input into a latent code, and a decoder to reconstruct the signal from it. However, instead of encoding the input as a single point, it encodes it as a distribution in the latent space. In practice, the distribution used is chosen to be close to a normal distribution. The encoder is changed to output the parameters of this distribution, the mean $\mu$, and the variance $\sigma^2$. $\sigma^2$ is often replaced by a proxy $\rho = log(\sigma^2)$ to enforce positivity and stability. The new inference scheme is changed for $\hat{x} = D(z)$, where the latent code $z \sim \mathcal{N}(E(x)_\mu, exp(E(x)_\rho))$.

**Probabilistic Formulation:** Let us consider the +vae as a probabilistic model. $x$, our data, is generated from the latent variable $z$ that cannot be observed. In this framework, the generation steps are the following: $z$ is sampled from the prior distribution $p(z)$, and $x$ is sampled from the conditional likelihood $x \sim p(x | z)$. In this setting, the probabilistic decoder is $p(x | z)$, and the probabilistic encoder is $p(z | x)$. The Bayes theorem allows expressing a natural relation between the prior $p(z)$, the likelihood $p(x | z)$, and the posterior $p(z|x)$:

$$
p(z | x) = \frac{p(x | z)p(z)}{p(x)} = \frac{p(x | z)p(z)}{\int p(x | u) p(u) du}
$$ {#eq:vae_bayes}

A standard Gaussian distribution is often assumed for the prior $p(z)$, and a parametric Gaussian for the likelihood $p(x | z)$ with its mean being defined by a deterministic function $f \in F$ and a positive constant $c \cdot I$ for the covariance. In this setting:

$$
\begin{aligned}
p(z)     &\sim \mathcal{N}(0, I) \\
p(x | z) &\sim \mathcal{N}(f(z), cI), \; f \in F, \; c > 0
\end{aligned}
$$ {#eq:vae_gaussian}

These equations (see [@eq:vae_bayes; @eq:vae_gaussian]) define a classical Bayesian Inference problem. This problem is however intractable because of the denominator's integral $\int p(x | u) p(u) du$ and thus requires the use of approximation techniques. 

**Variational Inference**: In statistics, +vi is one of the techniques used to approximate complex distributions. It consists in setting a parametrized distribution family, in our case Gaussians with its mean and covariance, and searching for the best approximation of the target distribution in this family. To search for the best candidate, we use the +kld between the approximation and the target and minimize it with [+gd]{.full}.

Let us approximate the posterior $p(z|x)$ using +vi with a Gaussian distribution $q_x(z)$ with a mean $g(x) \in G$ and covariance $h(x) \in H$ where $q_x(z) \sim \mathcal{N}(g(x), h(x))$. we can now look for the optimal $g^*$ and $h^*$ minimizing the +kld between the target and the approximation:

$$
\begin{aligned}
(g^*, h^*) &= \underset{(g, h) \in G \times H}{arg \; min} KL(q_x(z) || p(z | x)) \\
           &= \underset{(g, h) \in G \times H}{arg \; min} (E_{z \sim q_x} \; log \; q_x(z) - E_{z \sim q_x} \; log \; \frac{p(x | z) p(z)}{p(x)}) \\
           &= \underset{(g, h) \in G \times H}{arg \; min} (E_z \; log \; q_x(z) - E_z \; log \; p(z) - E_z \; log \; p(x | z) + E_z \; log \; p(x)) \\
           &= \underset{(g, h) \in G \times H}{arg \; min} (E_z [log \; p(x | z) - KL(q_x(z) || p(z)]) \\
           &= \underset{(g, h) \in G \times H}{arg \; min} (E_z \; log \; p(x | z) - KL(q_x(z) || p(z)) \\
           &= \underset{(g, h) \in G \times H}{arg \; min} (E_z [-\frac{||x - f(z)||^2}{2c}] - KL(q_x(z) || p(z)) \\
\end{aligned}
$$  {#eq:vae_objective}

This rewrite of the objective equations demonstrates a natural tradeoff between the data confidence $E_z [-\frac{||x - f(z)||^2}{2c}]$ and the prior confidence $KL(q_x(z) || p(z))$. The first term describes a reconstruction loss where the decoder parametrized by the function $f \in F$ has to recover the input $x$ from the latent code $z$, and the second term a regularization objective between $q_x(z)$ and the prior $p(z)$ which is gaussian. We can view the constant $c$ as a strength parameter that can adjust how we favor the regularization.

**Reparametrization Trick:** The +vae architecture is trained to find the parameters of the functions $f$, $g$, and $h$ by minimizing the +vi objective (see @eq:vae_objective). The encoder is charged to output two vectors, one for representing $g(x)$ the mean, in the case of a Gaussian distribution $\mu$, and the other representing the variance of the distribution $h(x)$, $\rho = log(\sigma^2)$. The latent code $z$ is then sampled from the Gaussian distribution $z \sim \mathcal{N}(\mu, \sigma)$ and finally decoded to reconstruct the original input $x$.

There is however a catch. The sampling process is stochastic and thus not differentiable. And we know that a +nn needs to be differentiable to be optimized using +sgd. To solve this problem, Kingma et al. [@kingma_2013] propose to use what they call a reparametrization trick. It consists in sampling a surrogate standard Gaussian distribution $\zeta \sim \mathcal{N}(0, I)$ and scaling it by the output of the learned encoder $\mu$ and $\sigma^2$. This the process becomes:

$$
\begin{aligned}
E(x)    &= (\mu, \rho) \\
\hat{x} &= D(\mu + \zeta \; exp(\rho)), \; \zeta \sim N(O, I)
\end{aligned}
$$ {#eq:vae_reparametrization}

Performing the latent sampling using the reparametrization trick (see @eq:vae_reparametrization) conserves the gradient flow. The +vae can thus be trained to learn a structured latent space that can be used to interpolate latent codes and decode them into samples similar to the training distribution.

**MNIST Digit Image Generation:** Let us reconsider our toy example with training a +vae on the +mnist handwritten digit dataset. There is almost nothing change in comparison to the +ae.

We implement a `GaussianDistribution` class helper for handling the parametrized Gaussian distribution taking the output parameters $(\mu, \rho)$ from the encoder in its constructor from which the mean $\mu$ and variance $\sigma^2$ are derived. The class is augmented with utility functions such as `sample` to sample the distribution using the reparametrization trick, and a `kld` function to compute the average +kld regularization loss when compared with a standard Gaussian distribution target prior.

```python
from torch import Tensor

import torch

# Parametrized Gaussian Distribution
class GaussianDistribution:
    def __init__(self, params: Tensor) -> None:
        self.mu, self.rho = params.chunk(2, dim=-1)
        self.std = torch.exp(0.5 * self.rho)
        self.var = torch.exp(self.rho)

    # Reparametrization Trick
    def sample(self) -> Tensor:
        return self.mu + self.std * torch.randn_like(self.mu)

    # Average KL-Divergence with Gaussian Prior
    def kld(self) -> Tensor:
        kld = 0.5 * (self.mu.pow(2) + self.var - 1.0 - self.rho)
        return torch.mean(torch.sum(kld, dim=1), dim=0)
```

The network is the same as for the +ae except in the encoder output which in our case has to double its output size to encode both the mean $\mu$ and the $\rho = log(\sigma^2)$ of the latent distribution.

```python
from collections import OrderedDict
from torch.nn import (Linear, ReLU, Sequential, Sigmoid)

# Model definition (encoder out: mu and rho)
model = Sequential(OrderedDict(
    encoder=Sequential(
        Linear(28 * 28,  256), ReLU(),
        Linear(    256, 2 * 2),
    ),
    decoder=Sequential(
        Linear(  2,     256), ReLU(),
        Linear(256, 28 * 28), Sigmoid(),
    ),
))
```

The encoder can then be used to compress the input into the latent distribution parameters from which the latent code can be sampled and decoded using the decoder. The loss is a combination of the reconstruction term using the `binary_cross_entropy`, and the regularization term using the `kld` computed from the posterior distribution helper. The +kld term is weighted by the `kld_weight` and is set to $0$ at the beginning of training and slowly increased toward a defined weight set to $1e-4$ in this case.

```python
from torch.nn.functional import binary_cross_entropy

# Compute output
posterior = GaussianDistribution(model.encoder(x))
x_ = model.decoder(posterior.sample())

# Compute loss
loss_reco = binary_cross_entropy(x_, x)
loss_kld = p.kld()
loss = loss_reco + kld_weight * loss_kld
```

The model is trained to minimize the objective functions. Satisfying both the reconstruction loss and the +kld regularization is harder than the task of the vanilla +ae. Intuitively this has to result in a structured latent space at the cost of a small reconstruction quality degradation. The training history and reconstruction capabilities are shown in @fig:gai_vae_history.

![Training history of a small [+vae]{.full}. The combination of the binary cross entropy loss and the +kld regularization is shown on the left, a training sample in the middle, and its corresponding reconstruction on the right.](./figures/core_gai_vae_history.svg){#fig:gai_vae_history}

As expected, the latent space (see @fig:gai_vae_latent) presents structural organization. The specific digit classes are visible and separable.

![Trained [+vae]{.full} $2$-dimensional latent space visualization. The data points represent the encoded latent code of images from the +mnist dataset and are colored based on their corresponding label (digit). The latent space is organized in a way that allows us to visually separate these classes.](./figures/core_gai_vae_latent.svg){#fig:gai_vae_latent width=60%}

The structure of the trained +vae latent space allows the generation of new data points by sampling latent codes and decoding them using the trained decoder (see @fig:gai_vae_latent_sampling).

![Trained [+vae]{.full} $2$-dimensional latent space sampling visualization. The decoder is used for generation by sampling the latent space in a grid pattern.](./figures/core_gai_vae_latent_sampling.svg){#fig:gai_vae_latent_sampling}

#### Generative Adversarial Networks {#sec:gan}

While [+vae]{.plurals} allow learning structured latent spaces ready for sampling and generation, they suffer from quality degradation. The [+gan]{.full} architecture, introduced by Ian Goodfellow et al. [@goodfellow_2014], is one answer to this problem. [+gan]{.plural} can generate high-quality images in real-time settings.

**Vanilla GAN:** In its vanilla formulation, a +gan consists of a generator $G$ trained to produce images $G(z)$ similar to the training distribution given a latent code $z$ which is then fed to a discriminator $D$ trained to differentiate fake images (generated images) from true images $x$. The two networks are jointly trained in an end-to-end fashion to optimize a Min-Max objective (see @eq:gan_minmax) that can be split into two objectives, one for the generator, and another for the discriminator (see @eq:gan_minmax_split).

$$
\underset{D}{min} \; \underset{G}{max} \; E_x \; log \; D(x) + E_z \; log(1 - D(G(z))) \\
$$ {#eq:gan_minmax}

$$
\begin{aligned}
\frac{1}{n} \sum_{i=1}^{n} \; &log \; D(x_i) + log(1 - D(G(z_i))) \\
\frac{1}{n} \sum_{i=1}^{n} \; &log \; D(G(z_i))
\end{aligned}
$$ {#eq:gan_minmax_split}


**Vanishing Gradients and Mode Collapse**: In practice, the vanilla formulation is however unstable. The generator training often saturates if it cannot keep up with the discriminator training which is in most cases easier to satisfy. It suffers from vanishing gradients where the loss signal becomes too small and gradients do not propagate to layers resulting in the early stop of the generator training. It is also subject to mode collapse where the generator finds a simple solution fooling the discriminator and failing at generating diverse enough outputs. Solutions such as the hinge loss [@lim_2017], the Wasserstein distance [@arjovsky_2017], gradient penalty [@ishaan_2017], the use of batch [@ioffe_2015] and spectral [@miyato_2018] normalization can be found in the literature.

**Wasserstein GAN:** The most famous improvement of the vanilla +gan is the +wgan [@arjovsky_2017]. It both resolves the mode collapse and limits the vanishing gradients issues. The last activation of the discriminator is swapped from a sigmoid to a linear activation. This small change turns the discriminator into a critic network in charge of scoring the quality (fidelity to the original distribution) of input images. The new simplified objectives are shown in @eq:gan_wasserstein and are often combined with gradient clipping to satisfy a Lipschitz constraint.

$$
\begin{aligned}
\frac{1}{n} \sum_{i=1}^{n} \; & D(x_i) - D(G(z_i)) \\
\frac{1}{n} \sum_{i=1}^{n} \; & D(G(z_i))
\end{aligned}
$$ {#eq:gan_wasserstein}

**Gradient Penalty:** In their contribution, Gulrajani et al. [@gulrajani_2017] propose to replace gradient clipping with gradient penalty to enforce a constraint on the critic such that its gradients with respect to the inputs are unit vectors. The critic loss is thus augmented with an additional term (see @eq:gan_gp) where $\hat{x}$ is sampled from a linear interpolation between real and fake samples to satisfy the critic's Lipschitz constraint.

$$
\lambda \; E_{\hat{x}} \; (||\nabla_{\hat{x}}D(\hat{x})||_2 - 1)^2
$$ {#eq:gan_gp}

**MNIST Digit Image Generation:** Let us reconsider the +mnist dataset and train a +gan to enable qualitative generation by providing a latent code to the generator. We first need to define the generator $G$ model and the critic network $C$. In this case, we keep the same architecture for both. They are $3$-layer [+nn]{.plural} with a sigmoid activation for the generator and a linear activation for the critic. For the sake of the example and comparability with the previous generative architecture, we choose a latent dimension of $2$.

```python
from torch.nn import (Linear, ReLU, Sequential, Sigmoid)

# Generator model
generator = Sequential(
    Linear(  2,      256), ReLU(),
    Linear(256,      256), ReLU(),
    Linear(256,  28 * 28), Sigmoid(),
)

# Critic model
critic = Sequential(
    Linear(28 * 28, 256), ReLU(),
    Linear(    256, 256), ReLU(),
    Linear(    256, 1),
).cuda()
```

Even though the formulation of the Minmax objective is common, both networks require their own optimizer in practice. We thus define the optimizer for the generator and the critic. Their hyperparameters can be different and tweaked to further optimize the training. For the sake of simplicity, we keep the default parameters in this toy example.

```python
from torch.optim import AdamW

# Optimizers (one per model)
g_optim = AdamW(generator.parameters(), lr=1e-3)
c_optim = AdamW(critic.parameters(), lr=1e-3)
```

Latent variables are sampled from the standard normal distribution $z \sim \mathcal{N}(O, I)$. The `gen_latent_code` is a helper that generates such latent codes given a mini-batch size as +dl frameworks work on the batch level.

```python
# Generate latent code
gen_latent_code = lambda B: torch.randn((B, 2))
```

One generator training step consists in sampling a latent code $z$, using it to generate a handwritten digit fake image $G(z)$ and computing the average generator hinge/Wasserstein loss $ReLU(1 - C(G(z)))$. This loss is then used to backpropagate gradients with respect to the generator's parameters. The generator's weights are finally updated using the Adam optimizer policy.

```python
from torch import Tensor
from torch.nn import Module

def generator_step(generator: Module, optim: AdamW, real: Tensor) -> None:
    fake = generator(gen_latent_code(real.shape[0]))
    loss = torch.relu(1.0 - critic(fake)).mean(dim=0)
    loss.backward()
    optim.step()
    optim.zero_grad(set_to_none=True)
```

For stability purposes and to satisfy the $1$-Lipschitz constraint of the critic network we implement a `grad_penalty` helper for computing the gradient penalty term. It consists of sampling interpolations between fake and real samples $t = \alpha x_{real} + (1 - \alpha) x_{fake}$ with $\alpha \sim \mathcal{U}(0, 1)$, scoring them using the critic network $C$. We then compute the gradients with respect to the critic's parameters and optimize them to be close to $1$, $\; E_{\hat{x}} \; (||\nabla_{\hat{x}}D(\hat{x})||_2 - 1)^2$.

```python
from torch import Tensor
from torch.autograd import grad
from torch.nn import Module

# Compute Gradient Penatly for Critic
def grad_penalty(critic: Module, real: Tensor, fake: Tensor) -> Tensor:
    alpha = torch.rand((real.shape[0], 1), requires_grad=True)
    t = alpha * real - (1 - alpha) * fake
    mixed = critic(t)
    grads = grad(mixed, t, torch.ones_like(mixed), True, True, True)[0]
    grads = grads.view(grads.shape[0], -1)
    grads = grads.norm(2, dim=1)
    return torch.mean((grads - 1).pow(2))
```

One critic training step consists in sampling a latent code $z$, using it to generate a fake image $G(z)$ and computing the three average critic's loss terms $ReLU(1 + C(G(z)))$, $ReLU(1 - C(x)), and $\lambda \nabla_{gp}$ with $\lambda = 10$ to minimize. The losses are then backward and the resulting gradients are used to update the critic's weights using the Adam policy.

```python
from torch import Tensor
from torch.nn import Module

def critic_step(critic: Module, optim: AdamW, real: Tensor) -> None:
    with torch.no_grad():
        fake = generator(gen_latent_code(real.shape[0]))
    loss_fake = torch.relu(1.0 + critic(fake)).mean(dim=0)
    loss_real = torch.relu(1.0 - critic(real)).mean(dim=0)
    loss_gp = grad_penalty(critic, real.detach(), fake.detach())
    loss = loss_fake + loss_real + 10 * loss_gp
    loss.backward()
    optim.step()
    optim.zero_grad(set_to_none=True)
```

The overall training loop consists in alternating between generator training steps and critic training steps until convergence. It is common to train the critic for more cycles, especially at the beginning of training where the training signal for the generator is not constructive enough. In this toy example, we optimize the critic five times more than the generator.

```python
# Training Loop
for epoch in range(100):
    for real in loader:
        # Train the Generator
        generator_step(generator, g_optim, real)
        
        # Train the Critic
        for _ in range(5):
            critic_step(critic, c_optim, real)
```

The training history monitoring the generator's and critic's loss terms are shown in @fig:gai_gan_history. +gan are in practice hard to train and monitoring each term independently allows us to diagnose mode collapse and vanishing gradients when the gradient distributions are checked.

![Training history of a small [+gan]{.full}. The objective terms are plotted separately and follow a hinge loss and Wasserstein objective. The loss for the generator is shown in orange and the losses for the critic in blue.](./figures/core_gai_gan_history.svg){#fig:gai_gan_history}

Overall, the +gan architecture is highly efficient at generating new samples giving latent codes. The quality of the generation is superior to those of +vae at the cost of a small loss in variation (see @fig:gai_gan_latent_sampling).

![Trained [+gan]{.full} $2$-dimensional latent space sampling visualization. The generator is used for generation by sampling the latent space in a grid pattern.](./figures/core_gai_gan_latent_sampling.svg){#fig:gai_gan_latent_sampling}

#### Denoising Diffusion Models {#sec:ddm}

[+gan]{.full} architectures have been the framework of choice when approaching image generation tasks for real-time or near real-time applications with perceptual quality needs. However, a more recent proposal named [+ddm]{.full} [@ho_2020] is currently challenging this position. They allow for better quality generation and free user-guided controls such as in-paint, out-painting, super-resolution, and more at the cost of inference time, at least concerning its vanilla formulation.

The +ddm architecture is inspired by non-equilibrium thermodynamics and consists of a Markov chain on diffusion steps slowly adding Gaussian noise to the data. The task is then to learn the reverse operation to reconstruct the original data from noise. In this particular framework, the latent code is as big as the input tensor and reversed using a fixed procedure. 

**Forward Diffusion:** Let us consider a data point sampled from real data distribution $x_0 \sim q(x)$. The forward diffusion process consists of adding small and successive Gaussian noise to the initial data sample during $T$ steps. The chained noisy transformations produce $x_1, \dots, x_T$ samples (see @eq:ddm_froward). The step size is given by a variance $\beta$-scheduler ${\beta \in [0; 1]}_{t=1}^{T}$.

$$
\begin{aligned}
q(x_t | x_{t - 1}) = \mathcal{N}(x_t; \sqrt{1 -  \beta_t} x_{t - 1}, \beta_t I) \\
q(x_{1:T} | x_0) = \prod_{t = 1}^{T} q(x_t | x_{t - 1})
\end{aligned}
$$ {#eq:ddm_froward}

By adding noise to the input data in small enough steps, when $T \rightarrow +\infty$, $x_T$ converge towards an isotropic Gaussian distribution. One feature of the forward diffusion process is that $x$ can be computed in a closed form making use of the reparametrization trick introduced previously (see @sec:vae). Defining $\alpha_t = 1 - \beta_t$ and $\bar{\alpha_t} = \prod_{i=1}^{t} \alpha_i$,

$$
\begin{aligned}
x_t &= \sqrt{\alpha_t} x_{t - 1} + \sqrt{1 - \alpha_t} \zeta_{t-1} \\
    &= \sqrt{\alpha_t \alpha_{t - 1}} x_{t - 2} + \sqrt{1 - \alpha_t \alpha_{t - 1}} \zeta_{t-2} \\
    &= \dots \\
    &= \sqrt{\bar{\alpha_t}} x_{0} + \sqrt{1 - \bar{\alpha_t}} \zeta_0
\end{aligned}
$$ {#eq:ddm_froward_closed}

where $\zeta_t \sim \mathcal{N}(0, I)$ and $\zeta_{t-2}$ results from merging two Gaussian distributions, $\mathcal{N}(0, \sigma_1^2 I) + \mathcal{N}(0, \sigma_2^2 I) = \mathcal{N}(0, (\sigma_1^2 + \sigma_2^2) I)$.

To summarize, $x_t$ can be sampled as follow:

$$
x_t \sim q(x_t | x_0) = \mathcal{N}(x_T; \sqrt{\bar{\alpha_t}} x_0, (1 - \bar{\alpha}_t) I)
$$

**Beta Schedule:** $\beta_t$, the variance parameter can be fixed to a constant or chosen using a schedule over $T$ timesteps. In the original paper [@ho_2020] and follow-up contribution [@nichol_2021], the authors propose a linear ($\beta_1=1e^{-4}$, $\beta_T=0.02$), a quadratic, and a cosine schedule. Their experiments show that the cosine schedule results are better.

**Backward Diffusion:** The backward diffusion process, also called reverse diffusion, consists in learning the reverse mapping $q(x_{t-1} | x_t)$. By taking into account that with enough steps $T \rightarrow +\infty$, the latent variable $x_T$ follows an isotropic Gaussian distribution, $x_T$ can be sampled from $\mathcal{0, I}$ and $x_0$ reconstructed by successively applying this process resulting in $q(x_0)$, a novel data sample from the training data distribution.

The reverse transformation $q(x_{t-1} | x_t)$ is however intractable. It would require sampling the entire data distribution. Similarly to the +vae, $q(x_{t-1} | x_t)$ is approximated using a parametrized model $p_\theta$, in our case a +nn. For small enough steps, $p_\theta$ can be chosen to be a Gaussian distribution whose mean $\mu_\theta$ and variance $\Sigma_\theta$ need to be parameterized.

$$
\begin{aligned}
p_\theta(x_{t - 1} | x_t) = \mathcal{N}(x_{t - 1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t)) \\
p_\theta(x_{0:T}) = p_\theta(x_T) \prod_{t = 1}^{T} p_\theta(x_{t - 1} | x_t)
\end{aligned}
$$ {#eq:ddm_backward_param}

We can then optimize the negative log-likelihood of the training data. After a series of arrangements and simplifications, see the original paper [@ho_2020] for full derivation, the objective can be written as follow:

$$
\begin{aligned}
log \; p(x) &\geq L_0 - L_T - \sum_{t=2}^{T} L_{t-1} \\
L_0 &= E_{q(x_1 | x_0)} log \; p_\theta(x_0 | x_1) \\
L_T &= D_{KL}(q(x_T | x_0) || p(x_T)) \\
L_t &= E_{q(x_t | x_0)} \; D_{KL}(q(x_{t- 1 } | x_t, x_0) || p_\theta(x_{t - 1} | x_t))
\end{aligned}
$$ {#eq:ddm_backward_elbo}

where $L_0$ can be seen as a reconstruction term, $L_T$ as a similarity between $x_T$'s distribution and the standard Gaussian prior, and $L_t$ the difference between the target noise step and its estimation. It can be demonstrated that $q(x_{t - 1} | x_t)$ can be made tractable by conditioning it on $x_0$, $q(x_{t - 1} | x_t, x_0)$. In this setting:

$$
\begin{aligned}
q(x_{t - 1} | x_t, x_0) &= \mathcal{N}(x_{t - 1}; \tilde{\mu}(x_t, x_0), \tilde{\beta_t} I) \\
\tilde{\beta_t} &= [(1 - \bar{\alpha}_{t - 1}) \beta_t] / (1 - \bar{\alpha}_t) \\
\tilde{\mu}(x_t, x_0) &= [(\sqrt{\bar{\alpha}_{t - 1}} \beta_t) x_0 + (\sqrt{\bar{\alpha}_t} (1 - \bar{\alpha}_{t - 1})) x_t] / (1 - \bar{\alpha}_t)
\end{aligned}
$$ {#eq:ddm_backward_conditionned}

**Denoising Diffusion Implicit Model:** 
...

**MNIST Digit Image Generation:**
...

### Attention Machanism {#sec:attention}
#### Multihead Self-Attention {#sec:mha}
#### Large Language Models {#sec:llm}
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