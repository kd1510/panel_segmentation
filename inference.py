import os
from PIL import Image
import numpy as np

from transformers import AutoModelForSemanticSegmentation
from transformers import AutoImageProcessor

import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np


# We need to load the trained checkpoint here
model_checkpoint = os.environ['MODEL_CHKPT']
model = AutoModelForSemanticSegmentation.from_pretrained(model_checkpoint)

# Load test image
test_image = os.environ['TEST_IMAGEPATH']
test_image = Image.open(test_image, mode="r").convert("RGB")

# Run inference
processor_checkpoint = "nvidia/mit-b0"
image_processor = AutoImageProcessor.from_pretrained(processor_checkpoint, reduce_labels=True)
encoding = image_processor(test_image, return_tensors="pt")
pixel_values = encoding.pixel_values.to('cuda')
outputs = model(pixel_values=pixel_values)
logits = outputs.logits.cpu()
upsampled_logits = nn.functional.interpolate(
    logits,
    size=test_image.size[::-1],
    mode="bilinear",
    align_corners=False,
)
pred_seg = upsampled_logits.argmax(dim=1)[0]

# View result
colormap = np.zeros((151, 3))
colormap[122] = [0, 133, 255] # make colormap all white except for index 123
color_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)
palette = np.array(colormap)
for label, color in enumerate(palette):
    color_seg[pred_seg == label, :] = color
color_seg = color_seg[..., ::-1]  # convert to BGR
img = np.array(test_image) * 0.5 + color_seg * 0.5  # plot the image with the segmentation map
img = img.astype(np.uint8)

plt.figure(figsize=(15, 10))
plt.imshow(img)
plt.show()