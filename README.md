# vit-ucec-detection
Compare the performance of vision transformer with ResNet for detection of Uterine corpus endometrial carcinoma


## Usage:

### Process SVS Slides and Train Vision Transformer
Note: UCEC directory with SVS files must be present

``python vit_pipeline.py``


### Train ResNet CNN model
### Run VIT Pipeline to process svs slides and save pngs before training resnet
``python resnet_pipeline.py``

### Preprocess : Slide UID to Slide name mapping using selenium
slide_uids.sjon is generated using colab: https://colab.research.google.com/drive/1hTa1f8V6SRzCaHQcT5BXbVoGHLFh8nMD#scrollTo=bvlrYBeiTdFC

``python ./label_preprocessing/map_slide_name_to_type.py``


### Output directory: 
out_model