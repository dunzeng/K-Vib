# K-Vib
Code for paper - [Enhanced Federated Optimization: Adaptive Unbiased Client Sampling with Reduced Variance](https://openreview.net/forum?id=Gb4HBGG9re)


## Dataset

Our experiment setting of FEMNIST follows [Optimal Client Sampling](https://github.com/SamuelHorvath/FL-optimal-client-sampling), please download the modified dataset following their instructions. You can download from [google driver](https://drive.google.com/drive/folders/1bpdCOB9zgO_X7n4vATm46FozAlTRRvYO?usp=sharing) as well. 

They are expected to be located in datasets directory.

## Run

synthetic task

> python main_synthetic.py -com_round 500 -sample_ratio 0.1 -num_clients 100 -batch_size 64 -epochs 1 -lr 0.02 -dseed 4399 -seed [run_seed] -data synthetic -sampler [uniform/kvib] -reg 0.1

femnist task

> python main_femnist.py -com_round 500 -k [num_of_client_per_round] -dataset v2 -batch_size 20 -epochs 3 -lr [learning_rate] -freq 10 -sampler [uniform/kvib] -reg 0.1 -seed [run_seed]

## Citation

```
@article{
zeng2025enhanced,
title={Enhanced Federated Optimization: Adaptive Unbiased Client Sampling with Reduced Variance},
author={Dun Zeng and Zenglin Xu and Yu Pan and Xu Luo and Qifan Wang and Xiaoying Tang},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=Gb4HBGG9re},
note={}
}
```