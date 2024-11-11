# MetaFormer and CNN Hybrid Model for Polyp Image Segmentation
Authors : [Hyunnam Lee](mailto:hyunnamlee@gmail), [Joohan Yoo](mailto:unchinto@semyung.ac.kr)

This is the official implementation.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/metaformer-and-cnn-hybrid-model-for-polyp/medical-image-segmentation-on-cvc-clinicdb)](https://paperswithcode.com/sota/medical-image-segmentation-on-cvc-clinicdb?p=metaformer-and-cnn-hybrid-model-for-polyp)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/metaformer-and-cnn-hybrid-model-for-polyp/medical-image-segmentation-on-kvasir-seg)](https://paperswithcode.com/sota/medical-image-segmentation-on-kvasir-seg?p=metaformer-and-cnn-hybrid-model-for-polyp)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/metaformer-and-cnn-hybrid-model-for-polyp/medical-image-segmentation-on-cvc-colondb)](https://paperswithcode.com/sota/medical-image-segmentation-on-cvc-colondb?p=metaformer-and-cnn-hybrid-model-for-polyp)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/metaformer-and-cnn-hybrid-model-for-polyp/medical-image-segmentation-on-etis)](https://paperswithcode.com/sota/medical-image-segmentation-on-etis?p=metaformer-and-cnn-hybrid-model-for-polyp)




## Introduction
Transformer-based methods have become dominant in the medical image research field since the Vision Transformer achieved superior performance. Although transformer-based approaches have resolved long-range dependency problems inherent in Convolutional Neural Network (CNN) methods, they struggle to capture local detail information. Recent research focuses on the robust combination of local detail and semantic information. To address this problem, we propose a novel transformer-CNN hybrid network named RAPUNet. The proposed approach employs MetaFormer as the transformer backbone and introduces a custom convolutional block, RAPU (Residual and Atrous Convolution in Parallel Unit), to enhance local features and alleviate the combination problem of local and global features. We evaluate the segmentation performance of RAPUNet on popular benchmarking datasets for polyp segmentation, including Kvasir-SEG, CVC-ClinicDB, CVC-ColonDB, EndoScene-CVC300, and ETIS-LaribPolypDB. Experimental results show that our model achieves competitive performance in terms of mean Dice and mean IoU. Particularly, RAPUNet outperforms state-of-the-art methods on the CVC-ClinicDB dataset. 
## RAPUNet Architecture
<img src="imgs/RAPUNet.jpg" width="700">

## RAPU Component
![RAPU component](imgs/RAPU2.png)

## Running the project
### Implementation Environments
Ubuntu 20.04, Python 3.8.10, Tensorflow 2.13.0 keras_cv_attention_models 1.3.9

### Data-Sets

The datasets used in this study are publicly available at: 
- Kvasir-SEG: [here](https://datasets.simula.no/kvasir-seg/). 
- CVC-ClinicDB: [here](https://polyp.grand-challenge.org/CVCClinicDB/). 
- ETIS-LaribpolypDB: [here](https://drive.google.com/drive/folders/10QXjxBJqCf7PAXqbDvoceWmZ-qF07tFi?usp=share_link). 
- CVC-ColonDB: [here](https://drive.google.com/drive/folders/1-gZUo1dgsdcWxSdXV9OAPmtGEbwZMfDY?usp=share_link).

You can also download Train/Test datasets seperated by Pranet
- [Google Drive Link (327.2MB)](https://drive.google.com/file/d/1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao/view?usp=sharing). It contains five sub-datsets: CVC-300 (60 test samples), CVC-ClinicDB (62 test samples), CVC-ColonDB (380 test samples), ETIS-LaribPolypDB (196 test samples), Kvasir (100 test samples).
    
- [Google Drive Link (399.5MB)](https://drive.google.com/file/d/1YiGHLw4iTvKdvbT6MgwO9zcCv8zJ_Bnb/view?usp=sharing). It contains two sub-datasets: Kvasir-SEG (900 train samples) and CVC-ClinicDB (550 train samples).

### Training
```pyhon train.py```

### Test with pretrained model
```pyhon test.py```

### Generate predict images
```pyhon predict_img.py```

## Result
### Performance Comparison
<img src="imgs/Clinic.jpeg" width="400">
<img src="imgs/Kvasir.jpeg" width="400">
<img src="imgs/ETIS.jpeg" width="400">
<img src="imgs/CVCColon.jpeg" width="400">
<img src="imgs/unseen.jpeg" width="500">

### Qualitative Results
<img src="imgs/results_comparison.png" width="700">

## Pretrained Model
 - Trained on Kvasir-SEG: [Naver Link(504.2M)](http://naver.me/5GpcKEUa)
 - Trained on CVC-ClinincDB: [Naver Link(504.2M)](http://naver.me/GJTZxzOl)
 - Trained on Kvasir-SEG and CVC-Clinic: [Naver Link(504.2M)](http://naver.me/502B3D8U)
 - Predict Image: [Naver Link(25M)](http://naver.me/FafeIgI8)
   
## Citation
```
@ARTICLE{lee2024RAPUNet,
  author={Lee, Hyunnam and Yoo, Juhan},
  journal={IEEE Access}, 
  title={MetaFormer and CNN Hybrid Model for Polyp Image Segmentation}, 
  year={2024},
  volume={12},
  number={},
  pages={133694-133702},
  keywords={Image segmentation;Transformers;Convolutional neural networks;Feature extraction;Image resolution;Training;Biomedical image processing;Convolutional neural network;image segmentation;medical image processing;MetaFormer;polyp segmentation;vision transformer},
  doi={10.1109/ACCESS.2024.3461754}}
```
## License

The source code is free for research and education use only. Any comercial use should get formal permission first.
