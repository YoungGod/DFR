# DFR
Project: Unsupervised Anomaly Detection and Segmentation

Paper: Unsupervised anomaly segmentation via deep feature reconstruction  | **[Neurocomputing]**[`pdf`](https://www.sciencedirect.com/science/article/pii/S0925231220317951)[`code`](https://github.com/YoungGod/DFR) | **arxive preprint**[`pdf`](https://arxiv.org/abs/2012.07122)

Introduction: Automatic detecting anomalous regions in images of objects or textures without priors of the anomalies is challenging, especially when the anomalies appear in very small areas of the images, making difficult-to-detect visual variations, such as defects on manufacturing products.
	This paper proposes an effective unsupervised anomaly segmentation approach that can detect and segment out the anomalies in small and confined regions of images. Concretely, we develop a multi-scale regional feature generator which can generate multiple spatial context-aware representations from pre-trained deep convolutional networks for every subregion of an image. 
	The regional representations not only describe the local characteristics of corresponding regions but also encode their multiple spatial context information, making them discriminative and very beneficial for anomaly detection.
	Leveraging these descriptive regional features, we then design a deep yet efficient convolutional autoencoder and detect anomalous regions within images via fast feature reconstruction.
	Our method is simple yet effective and efficient. It advances the state-of-the-art performances on several benchmark datasets and shows great potential for real applications.
	
# Qualitative results
![image](https://github.com/YoungGod/DFR/tree/master/figs/seg-quality-l12.jpg)

# Citation
If you find something useful, wellcome to cite our paper:
```
@article{YANG2022108874,
title = {Learning Deep Feature Correspondence for Unsupervised Anomaly Detection and Segmentation},
journal = {Pattern Recognition},
pages = {108874},
year = {2022},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2022.108874},
url = {https://www.sciencedirect.com/science/article/pii/S0031320322003557},
author = {Jie Yang and Yong Shi and Zhiquan Qi},
}
```
```
@article{DFR2020,
    title = "Unsupervised anomaly segmentation via deep feature reconstruction",
    journal = "Neurocomputing",
    year = "2020",
    issn = "0925-2312",
    doi = "https://doi.org/10.1016/j.neucom.2020.11.018",
    url = "http://www.sciencedirect.com/science/article/pii/S0925231220317951",
    author = "Yong Shi and Jie Yang and Zhiquan Qi",
}
```

```
@misc{yang2020dfr,
      title={DFR: Deep Feature Reconstruction for Unsupervised Anomaly Segmentation}, 
      author={Jie Yang and Yong Shi and Zhiquan Qi},
      year={2020},
      eprint={2012.07122},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
