# Automatic Tooth Segmentation and Classification in Dental Panoramic X-Ray Images


# Abstract

In dentistry, radiological examinations aid specialists in diagnosing various dental conditions. However, interpretations can vary, leading to differences in diagnoses. This paper proposes a deep learning method for instance segmentation of teeth in panoramic X-ray images to improve diagnostic accuracy. Performance evaluation is conducted using a challenging dataset comprising 1500 images with high variation and 10 categories of buccal images.
Introduction

X-ray images are essential tools in dentistry for diagnosing dental diseases. Panoramic X-rays provide a comprehensive view of the entire mouth, aiding in the diagnosis of various dental conditions. This paper focuses on automatic tooth segmentation and classification in panoramic X-ray images, leveraging deep learning techniques for improved accuracy.

# Literature Survey

Panoramic radiography, also known as panoramic X-ray, captures the entire mouth in a single image, facilitating comprehensive dental examinations. Instance segmentation, a challenging task in computer vision, aims to label each pixel in an image with both class and instance information. Previous research has explored various segmentation methods, including threshold-based, cluster-based, and boundary-based approaches.

# Background (Description of Dataset)
The dataset consists of 1500 challenging panoramic X-ray images, categorized into 10 types of buccal images. Each category contains images with different dental characteristics, such as teeth with or without restoration, dental implants, and missing teeth.
Methodology

Mask R-CNN model is employed for instance segmentation of panoramic X-ray images. The model consists of two stages: proposal generation and object classification. The backbone of the network, ResNet101, is used to extract features, followed by a feature pyramid network (FPN) for region proposal generation. The final model is trained on a subset of annotated images and fine-tuned for tooth segmentation.

# Data Pre-processing
Data pre-processing involves splitting images into individual teeth for training the Mask R-CNN model. Annotated tooth images are used for training, validation, and testing, with pre-trained weights initialized using the MSCOCO dataset.
Results and Discussions

The model achieves promising results in segmenting teeth in panoramic X-ray images. Evaluation metrics such as VOC-Style mAP @ IoU=0.5 demonstrate high accuracy, with the model correctly identifying tooth regions in most cases.

# How to run
## Dental Segmentation Master.ipynb: Automatic Dental Image Segmentation

This project utilizes Mask R-CNN to automatically segment teeth in dental images. 

### Instructions:

**1. Setting Up:**

* Clone this repository or download the `.ipynb` file.
* Make sure you have Python 3 and the following libraries installed:
    * tensorflow, keras, imgaug, coco, mrcnn
    * You can install them using `pip install tensorflow keras imgaug coco mrcnn`
* Download the pre-trained weights file `mask_rcnn_tooth.h5` and place it in the same directory as the `.ipynb` file.

**2. Running the Code:**

* Open the `.ipynb` file in a Jupyter Notebook environment.
* Run the code cells in order.
* Change the paths in the code to point to your own data directories if needed.

**3. Code Breakdown:**

* The code first loads the datasets and configures the model.
* It then trains the model in three stages:
    * Stage 1: Train the heads layer.
    * Stage 2: Fine-tune layers from ResNet stage 4 and up.
    * Stage 3: Fine-tune all layers.
* Finally, it performs inference on some validation images and evaluates the model's performance.

**4. Additional Notes:**

* This code is intended for educational purposes and may require further adaptation for specific use cases.
* You can adjust the hyperparameters and experiment with different training configurations.
* For large datasets, it's recommended to use a GPU for faster training.


**Please note:**

* This readme does not include instructions on how to mount Google Drive in Colab, as this may not be relevant for everyone.

# Conclusion
Automatic tooth segmentation in dental X-ray images is a crucial step in improving diagnostic accuracy. The proposed deep learning approach shows promising results in segmenting teeth and can be further extended to detect missing teeth and dental artifacts. Future work aims to enhance the segmentation of mouth components and automate medical report generation.
