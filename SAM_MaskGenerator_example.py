import numpy as np
import time
import matplotlib.pyplot as plt
import cv2
import sys
import io
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# params
model_size = "large" # small, medium, large
device = "cuda:0" # cuda:0, cpu
image_name = "cartagena.png"

# load image
image = cv2.imread('image_examples/' + image_name)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

sys.path.append("..")

# download model checkpoint (https://github.com/facebookresearch/segment-anything#model-checkpoints)
if(model_size == "small"):
    sam_checkpoint = "model_checkpoint/sam_vit_b_01ec64.pth"
    model_type = "vit_b"

if(model_size == "medium"):
    sam_checkpoint = "model_checkpoint/sam_vit_l_0b3195.pth"
    model_type = "vit_l"

if(model_size == "large"):
    sam_checkpoint = "model_checkpoint/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

# load model and send to device
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

#load mask generator
mask_generator = SamAutomaticMaskGenerator(sam)

# get time
start = time.time()

masks = mask_generator.generate(image)

# get time
end = time.time()
print('Elapsed time = ' + str((end - start)*1000) + ' ms')

#plot image with all masks overlayed
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

plt.figure(figsize=(10,10))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show() 

#plot first mask
# plt.figure(figsize=(10,10))
# plt.imshow(image)
# plt.imshow(masks[0]['segmentation'], alpha=0.5)
# plt.axis('off')
# plt.show()

# plot all masks
# plt.figure(figsize=(10,10))
# plt.imshow(image)
# for mask in masks:
#     plt.imshow(mask['segmentation'], alpha=0.5)
# plt.axis('off')
# plt.show()

