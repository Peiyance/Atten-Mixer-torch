# Attention Matters: Session-based Recommendation withMulti-Prior Attention Mixture Network

![python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)
![cuda 11.2](https://img.shields.io/badge/cuda-11.2-green.svg)

## About
- This is the code for WSDM 2023 paper [Efficiently Leveraging Multi-level User Intent for Session-based Recommendation via Atten-Mixer Network](https://arxiv.org/pdf/2206.12781.pdf).

## Abstract
- Session-based recommendation (SBR) aims to predict the user’s next action based on short and dynamic sessions. Recently, there has been an increasing interest in utilizing various elaborately designed graph neural networks (GNNs) to capture the pair-wise relationships among items, seemingly suggesting the design of more complicated models is the panacea for improving the empirical performance. However, these models achieve relatively marginal improvements with exponential growth in model complexity. In this paper, we dissect the classical GNN-based SBR models and empirically find that some sophisticated GNN propagations are redundant, given the readout module plays a significant role in GNN-based models. Based on this observation, we intuitively propose to remove the GNN propagation part, while the readout module will take on more responsibility in the model reasoning process. To this end, we propose the Multi-Level Attention Mixture Network (Atten-Mixer), which leverages both concept-view and instance-view readouts to achieve multi-level reasoning over item transitions. As simply enumerating all possible high-level concepts is infeasible for large real-world recommender systems, we further incorporate SBR-related inductive biases, i.e., local invariance and inherent priority to prune the search space. Experiments on three benchmarks demonstrate the effectiveness and efficiency of our proposal. We also have already launched the proposed techniques to a large-scale e-commercial online service since April 2021, with significant improvements of top-tier business metrics demonstrated in the online experiments on live traffic.

## Table of Contents  
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Model Training and Testing](#model-training-and-testing)
- [Atten-Mixer Enhancement Study](#atten-mixer-enhancement-study)

## Requirements
- Python 3.7.9
- Other common packages listed in `environment.yaml`  
- Install required environment: `conda env create -f environment.yaml`  

## Dataset
Three widely used dataset are adopted:  

- [DIGINETICA](http://cikm2016.cs.iupui.edu/cikm-cup): It is a transaction dataset that is obtained from CIKM Cup 2016 Challange.  
- [GOWALLA](https://snap.stanford.edu/data/loc-gowalla.html): It is a dataset that contains users’ check-in infor- mation for point-of-interest recommendation.
- [Last.fm](http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html): It is a music-artist dataset that is used for music interest recommendation.

We have provided the preprocessed version in the folder `datasets/`.

## Model Training and Testing
- To train our model on DIGINETICA, after changing the dataset path in `src/dataset.py`, run from the root of the project:
```
cd src

python main_area_semantic.py --dataset diginetica
```
- To train our model on GOWALLA, after changing the dataset path in `src/dataset.py`, run from the root of the project:
```
cd src

python main_area_semantic.py --dataset gowalla
```
- To train our model on Last.fm, after changing the dataset path in `src/dataset.py`, run from the root of the project:
```
cd src

python main_area_semantic.py --dataset lastfm
```
To test our model on each dataset, we add following parameters after the above corresponding commands:
```
--tran_flag False --PATH ../checkpoint/saved_model_name
```
For example, to test our model on DIGINETICA:
```
cd src

python main_area_semantic.py --dataset diginetica --tran_flag False --PATH ../checkpoint/model.pt
```


## Atten-Mixer Enhancement Study 
- We provide our implemented two Atten-Mixer enhanced model: GNN with Atten-Mixer (GNN-AM) and SGNN-HN with Atten-Mixer (SGNN-HN-AM) for users to run and test:
    - [GNN-AM/model.py]
        - Example usuage of how to plug in our idea into SR-GNN
        -  [Line 15-61] 

    - [SGNN-HN-AM/model_star.py]
        - Example usuage of how to plug in our idea into SGNN-HN
        -  [Line 88-130] 
- **Notes:** 
To run the above models on each dataset: 
    - To train our model on DIGINETICA, after changing the dataset path in `main.py`, run from the root of the project:
```
cd GNN-AM/SGNN-HN-AM

python main.py --dataset diginetica
```
- To train our model on GOWALLA, after changing the dataset path in `main.py`, run from the root of the project:
```
cd GNN-AM/SGNN-HN-AM

python main.py --dataset gowalla
```
- To train our model on Last.fm, after changing the dataset path in `main.py`, run from the root of the project:
```
cd GNN-AM/SGNN-HN-AM

python main.py --dataset lastfm
```
The test command is also adding the following parameters after the above corresponding commands:
```
--tran_flag False --PATH ../checkpoint/saved_model_name
```
For example, to test our model on DIGINETICA:
```
cd GNN-AM/SGNN-HN-AM

python main.py --dataset diginetica --tran_flag False --PATH ../checkpoint/model.pt
```

## Citation

```
@inproceedings{zhang2023efficiently,
  title={Efficiently leveraging multi-level user intent for session-based recommendation via atten-mixer network},
  author={Zhang, Peiyan and Guo, Jiayan and Li, Chaozhuo and Xie, Yueqi and Kim, Jae Boum and Zhang, Yan and Xie, Xing and Wang, Haohan and Kim, Sunghun},
  booktitle={Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining},
  pages={168--176},
  year={2023}
}
```
