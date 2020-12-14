# DFR
Project: Unsupervised Anomaly Segmentation via Deep Feature Reconstruction

Paper: Unsupervised anomaly segmentation via deep feature reconstruction  | **[Neurocomputing]**[`pdf`](https://www.sciencedirect.com/science/article/pii/S0925231220317951)['code'](https://github.com/YoungGod/DFR) | **arxive preprint**['pdf'](https://arxiv.org/user/)

Introduction: Automatic detecting anomalous regions in images of objects or textures without priors of the anomalies is challenging, especially when the anomalies appear in very small areas of the images, making difficult-to-detect visual variations, such as defects on manufacturing products.
	This paper proposes an effective unsupervised anomaly segmentation approach that can detect and segment out the anomalies in small and confined regions of images. Concretely, we develop a multi-scale regional feature generator which can generate multiple spatial context-aware representations from pre-trained deep convolutional networks for every subregion of an image. 
	The regional representations not only describe the local characteristics of corresponding regions but also encode their multiple spatial context information, making them discriminative and very beneficial for anomaly detection.
	Leveraging these descriptive regional features, we then design a deep yet efficient convolutional autoencoder and detect anomalous regions within images via fast feature reconstruction.
	Our method is simple yet effective and efficient. It advances the state-of-the-art performances on several benchmark datasets and shows great potential for real applications.
# Qualitative results
![image](https://github.com/YoungGod/DFR/tree/master/figs/seg-quality-l12.jpg)
