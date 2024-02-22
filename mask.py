from pycocotools.coco import COCO
import os 
from PIL import Image
import numpy as np 
from matplotlib import pyplot as plt 
import cv2 

masks_dir = "../data/masks"
annotations_dir = "../data/annotations.json"
img_dir = "../data"

train_images = "../../dataset/train_images"
train_masks = "../../dataset/train_masks"

#check paths exist
if(os.path.exists(masks_dir) == False):
    print("masks_Dir does not exist")
    exit()
    
if(os.path.exists(annotations_dir) == False):
    print("annotations_Dir does not exist")
    exit()

coco = COCO(annotations_dir)

#makes all crop masks the same color
def recolor(pixel):
    if(pixel != 0):
        return 0
    return 255
vector_recolor = np.vectorize(recolor)

#saves images to a directory
for image_id in range(len(coco.imgs)):
    
    img = coco.imgs[image_id]
    
    file_name = img['file_name']
    print(file_name)
    image = np.array(cv2.imread(os.path.join(img_dir, file_name)))
    cat_ids = coco.getCatIds()
    anns_ids = coco.getAnnIds(imgIds = img['id'], catIds = cat_ids, iscrowd = None)
    anns = coco.loadAnns(anns_ids)
    
    file_name = file_name[7:]
    
    #if no annotations save as blank image
    if(len(anns) == 0):
        image = np.zeros((image.shape[:2]), dtype=np.uint8)
        cv2.imwrite(os.path.join(masks_dir, file_name), image)
        continue
        
    mask = coco.annToMask(anns[0])
    for i in range(len(anns)):
        mask += coco.annToMask(anns[i])
    mask = vector_recolor(mask)
    
    #saves to the mask_dir
    cv2.imwrite(os.path.join(masks_dir, file_name), mask)
    
    cv2.imwrite(os.path.join(train_images, file_name), image)
    cv2.imwrite(os.path.join(train_masks, file_name), mask)
    


