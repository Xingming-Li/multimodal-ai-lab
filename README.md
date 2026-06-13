# Multimodal AI Lab

This repo collects my learning and research themes on computer vision and multimodal language models. Each folder is one focused experiment, project, or course:

- **`dl4cv_stanford/`** — Course materials and assignments for Stanford's CS231n: Deep Learning for Computer Vision.

- **`vision_model_playground/`** — Building, training, and comparing image classifiers from MLPs to CNNs to Vision Transformers, to understand how each processes visual information and their trade-offs. Includes:
  - `PIL2Tensor.ipynb` — a short notebook on image-to-tensor preprocessing.
  - `ImageClassificationMLP.ipynb`, `ImageClassificationCNN.ipynb`, and `ViT/vision_transformer.ipynb` — the model progression.
  - `cnn_deployment/` — a Flask app that serves the trained CNN for image-upload classification.
  - `vla_demo/` — a self-contained Vision–Language–Action build-up: a vision-only CNN baseline vs. a fused CNN+text model in a synthetic driving environment.

- **`image_text_alignment/`** — Using OpenAI **CLIP** (`clip-vit-base-patch32` via Hugging Face Transformers) for zero-shot image classification and image–text similarity, applied to medical imaging (brain and spine MRI).

## Setup

The projects are mostly self-contained Jupyter notebooks. Common dependencies include `torch`, `torchvision`, `transformers`, `numpy`, `pillow`, and `matplotlib`. The deployable CNN has a pinned dependency list in [`vision_model_playground/cnn_deployment/requirements.txt`](vision_model_playground/cnn_deployment/requirements.txt).

To run the CNN deployment app:

```bash
cd vision_model_playground/cnn_deployment
pip install -r requirements.txt
python app.py
```

## Data

The CLIP experiments expect MRI data under `image_text_alignment/data/` (e.g. `brain_tumor_dataset/`, `spine_mri/`), with a few sample images in `data/samples/` used by the notebook.
