---
library_name: segmentation-models-pytorch
license: other
pipeline_tag: image-segmentation
tags:
- model_hub_mixin
- pytorch_model_hub_mixin
- segmentation-models-pytorch
- semantic-segmentation
- pytorch
- segformer
languages:
- python
---
# Segformer Model Card

Table of Contents:
- [Load trained model](#load-trained-model)
- [Model init parameters](#model-init-parameters)
- [Model metrics](#model-metrics)
- [Dataset](#dataset)

## Load trained model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qubvel/segmentation_models.pytorch/blob/main/examples/segformer_inference_pretrained.ipynb)

1. Install requirements.

```bash
pip install -U segmentation_models_pytorch albumentations
```

2. Run inference.

```python
import torch
import requests
import numpy as np
import albumentations as A
import segmentation_models_pytorch as smp

from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load pretrained model and preprocessing function
checkpoint = "smp-hub/segformer-b0-512x512-ade-160k"
model = smp.from_pretrained(checkpoint).eval().to(device)
preprocessing = A.Compose.from_pretrained(checkpoint)

# Load image
url = "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Preprocess image
np_image = np.array(image)
normalized_image = preprocessing(image=np_image)["image"]
input_tensor = torch.as_tensor(normalized_image)
input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
input_tensor = input_tensor.to(device)

# Perform inference
with torch.no_grad():
    output_mask = model(input_tensor)

# Postprocess mask
mask = torch.nn.functional.interpolate(
    output_mask, size=(image.height, image.width), mode="bilinear", align_corners=False
)
mask = mask.argmax(1).cpu().numpy()  # argmax over predicted classes (channels dim)
```

## Model init parameters
```python
model_init_params = {
    "encoder_name": "mit_b0",
    "encoder_depth": 5,
    "encoder_weights": None,
    "decoder_segmentation_channels": 256,
    "in_channels": 3,
    "classes": 150,
    "activation": None,
    "aux_params": None
}
```

## Dataset
Dataset name: [ADE20K](https://ade20k.csail.mit.edu/)

## More Information
- Library: https://github.com/qubvel/segmentation_models.pytorch
- Docs: https://smp.readthedocs.io/en/latest/
- License: https://github.com/NVlabs/SegFormer/blob/master/LICENSE

This model has been pushed to the Hub using the [PytorchModelHubMixin](https://huggingface.co/docs/huggingface_hub/package_reference/mixins#huggingface_hub.PyTorchModelHubMixin)