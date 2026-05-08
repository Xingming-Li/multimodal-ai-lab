# Vision Model Playground: From MLPs to CNNs to Vision Transformers

## Goal

Build, train, and compare simple image classifiers using:

1. A **basic MLP** (fully connected)
2. A **CNN** (with convolutional layers)
3. A **Vision Transformer (ViT)**

Through this, I hope to understand:

- How each model processes visual information differently
- Why CNNs outperform MLPs on image tasks
- How ViTs use attention to replace convolution
- Key trade-offs (accuracy, efficiency, inductive bias, generalization)

## Model Overview

**Model** | **Key Idea** | **Strength** | **Weakness**
-- | - | - | -
**MLP** | Flatten the image, learn pixel correlations directly | Simple to implement | Ignores spatial structure