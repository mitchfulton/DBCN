# Deformable Bayesian Convolutional Networks (DBCNs)

In small medical datasets it is of paramount importance to generalize well to unseen domains. However, this can be difficult as many medical datasets are small and come from a single domain. DBCNs promote domain shift robustness by combining deformable convolutions (DCNs) and Bayesian convolutions (BCNs). They retain the benefits of both components, with the flexibility and increased accuracy of DCNs and the quick robust learning of BCNs. Both the 2D and 3D DBCNs use Bayesian CNNs from Kumar Shridhar's repository [here](https://github.com/kumar-shridhar/PyTorch-BayesianCNN). The 3D DCNs are from Xinyi Ying's repository [here](https://github.com/XinyiYing/D3Dnet) and 2D DCNs are from Matthew Howe's repository [here](https://github.com/MatthewHowe/DCNv2). See these repos for more detailed install instructions than the ones provided here.

## Installation
To install DBCNs the DCNs must first be compiled. To do this go to the DCN folder in either the 2D or 3D folder and run `make.sh`, then optionally test with `python testcpu.py` or `python testcuda.py`.

## Citation
If you use DBCNs for your research please cite:
```
    @inproceedings{fulton2021deformable,
    title={Deformable Bayesian Convolutional Networks for Disease-Robust Cardiac MRI Segmentation},
    author={Fulton, Mitchell J and Heckman, Christoffer R and Rentschler, Mark E},
    booktitle={International Workshop on Statistical Atlases and Computational Models of the Heart},
    pages={296--305},
    year={2021},
    organization={Springer}
    }
```
