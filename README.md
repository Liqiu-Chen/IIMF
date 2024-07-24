# Integrating Interaction into Modality shared Fearues with ViT
The official implementation for the ACM MM 2024 paper [Simplifying Cross-modal Interaction via Modality-Shared Features for RGBT Tracking].

## Environment Installation
```
conda create -n iimf python=3.8
conda activate iimf
bash install.sh
```

## Project Paths Setup
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ../ --save_dir ./output
```

The data_dir should be the place where datasets are. For example, if you want to train on LasHeR and LasHeR is in the ../data, then --data_dir ../data. 
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

It should be noted that you should modify the file lib/test/evaluation/local.py when testing the model. The lasher_test is not always set well. You will need to set the lasher_test by yourself like 
```
settings.lasher_path = '/username/lasher/testingset'
```

## Data Preparation
Put the tracking datasets in the path you set. It should look like:

```
-- lasher
    |-- trainingset
    |-- testingset
    |-- trainingsetList.txt
    |-- testingsetList.txt
    ...
```

## Training
Download [ImageNet or SOT](https://pan.baidu.com/s/1U42J6b3g1htma0OvmXRQCw?pwd=at5b) pre-trained weights and put them under `$PROJECT_ROOT$/pretrained_models`. The pre-trained weights are provided by the work TBSI and we thank the contribution by them.

```
python tracking/train.py --script tbsi_track --config vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k --save_dir ./output/vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k --mode multiple --nproc_per_node 4
python tracking/train.py --script tbsi_track --config vitb_256_tbsi_32x1_1e4_lasher_15ep_sot --save_dir ./output/vitb_256_tbsi_32x1_1e4_lasher_15ep_sot --mode multiple --nproc_per_node 4
```

Replace `--config` with the desired model config under `experiments/tbsi_track`. 
Four GPUs available make nproc_per_node 4 and only one GPU available makes the mode single with nproc_per_node not needed.

## Evaluation
Put the checkpoint into `$PROJECT_ROOT$/output/config_name/...` or modify the checkpoint path in the testing code.

```
CUDA_VISIBLE_DEVICES=3 python tracking/test.py tbsi_track vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k --dataset_name lasher_test --threads 6 --num_gpus 1
CUDA_VISIBLE_DEVICES=3 python tracking/test.py tbsi_track vitb_256_tbsi_32x1_1e4_lasher_15ep_sot --dataset_name lasher_test --threads 6 --num_gpus 1

python tracking/analysis_results.py --tracker_name tbsi_track --tracker_param vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k --dataset_name lasher_test
python tracking/analysis_results.py --tracker_name tbsi_track --tracker_param vitb_256_tbsi_32x1_1e4_lasher_15ep_sot --dataset_name lasher_test
```

### Results on LasHeR testing set

Model | Backbone | Pretraining | Precision | NormPrec | Success | Checkpoint | Raw Result

IIMF | ViT-Base | SOT | 72.35 | 68.39 | 58.07 | [download](https://pan.baidu.com/s/12m_cmMvbTbMnd-3ih3YQxA?pwd=Chen) | [download](https://pan.baidu.com/s/12m_cmMvbTbMnd-3ih3YQxA?pwd=Chen)

It should be noted that our work is tested and trained on 4 V100 GPUs, and different GPUs might lead to different test results. The raw result and checkpoint that we provide are all achieved on V100 GPUs.

## Acknowledgments
Our project is developed on [OSTrack](https://github.com/botaoye/OSTrack) and [TBSI](https://github.com/RyanHTR/TBSI). Thanks for their contributions which help us to quickly implement our ideas.

## Citation
If our work is useful for your research, please consider citing our work as follows:
```
@inproceedings{IIMF,
  title={Simplifying Cross-modal Interaction via Modality-Shared Features for RGBT Tracking},
  author={Chen, Liqiu and Huang, Yuqing and Li, Hengyu and Zhou, Zikun and He, Zhenyu},
  booktitle={Proceedings of the 32th ACM International Conference on Multimedia},
  year={2024}
}
```










