# [KDD 2025] Temporal Restoration and Spatial Rewiring for Source-Free Multivariate Time Series Domain Adaptation
#### *by: Peiliang Gong, Yucheng Wang, Min Wu, Zhenghua Chen, Xiaoli Li, Daoqiang Zhang* <br/> 

## Accepted in the [31st SIGKDD Conference on Knowledge Discovery and Data Mining - Research Track](https://kdd2025.kdd.org).

#### Requirements:
einops==0.7.0  
matplotlib==3.5.0  
pandas==2.2.1  
scipy==1.12.0  
seaborn==0.13.2  
torch==2.1.0  
torchaudio==2.1.0  
torchdata==0.7.1  
torchmetrics==1.2.1  
torchvision==0.16.0  
wandb==0.16.5

#### Datasets:
We used three public datasets in this study. 
- UCIHAR
- SSC
- WISDM

#### Training procedure:
- Here, we provide a demo for running the experiments.  
To train a model using the following script file:
```
python trainers/train.py --run_description demo --da_method TERSE --dataset HAR --backbone TemporalSpatialNN_new --num_runs 3
```

#### Results:
At the end of all runs, the overall average and standard deviation results will be saved in the `save_dir` directory.

## Citation
If you found this work useful for you, please consider citing it.

```
@inproceedings{terse,
  author = {Gong, Peiliang and Wang, Yucheng and Wu, Min and Chen, Zhenghua and Li, Xiaoli and Zhang, Daoqiang},
  title = {Temporal Restoration and Spatial Rewiring for Source-Free Multivariate Time Series Domain Adaptation},
  booktitle={31st SIGKDD Conference on Knowledge Discovery and Data Mining - Research Track},
  year = {2025}
}
``
