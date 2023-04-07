# CAMS

This is a PyTorch implementation of [CAMS](https://cams-hoi.github.io). 

### Environment

Install PyTorch and most other packages we use are listed in [environment.yml](https://github.com/cams-hoi/cams-hoi.github.io/environment.yml). We use the implementation of MANO hand from [manotorch](https://github.com/lixiny/manotorch). 

### Data Preparation

#### MANO assets

This is used for synthesizing and evaluation. Please follow these steps:
1. Download the [mano_assets.zip](https://drive.google.com/file/d/1QfTv8lThfptlz22sC5bkVDoCD88-zfqy/view?usp=share_link)
1. Place it under the folder `data/` and unzip it, then it should be like `data/mano_assets`

#### HOI4D CAMS version

This is the dataset we use.

##### How to use

1. Download the meta file, eg. [pliers_meta.torch](https://drive.google.com/file/d/13unEc7dxC4ouX63m6h_rNiTQ_hm_qxWk/view?usp=share_link)
1. Place it under the folder `data/meta` and then it should be like `data/meta/pliers_meta.torch`
1. Optional: We also release several other categories including [scissors_meta.torch](https://drive.google.com/file/d/1daVbJDj3TfZpMlWLR50yGFL26ew0CH-y/view?usp=share_link) and [bucket_meta.torch](https://drive.google.com/file/d/1G1eTjnmTpI32noMJdboFDwFS1NxaQoMX/view?usp=share_link). You may edit the `data` attribute according to [experiments/pliers/config.yml](https://github.com/cams-hoi/cams-hoi.github.io/blob/master/experiments/pliers/config.yaml) to run our code on new category.

##### Optional: Details about generating ground truth CAMS Embedding

If you open the meta file, eg. [pliers_meta.torch](https://drive.google.com/file/d/13unEc7dxC4ouX63m6h_rNiTQ_hm_qxWk/view?usp=share_link), you will find that every manipulation sequence consists of two keys: `data` and `cams`. Under the key `data`, you will find ground truth data we copied from [HOI4D](https://github.com/leolyliu/HOI4D-Instructions), and the key `cams` reserves our generated ground truth CAMS Embedding. 

The following script is a demo about how we generate our own CAMS Embedding from mocap data. You can modify and generate your CAMS Embedding from other mocap data (which guarantees that you can find the right contact pairs by simply calculating the sdf between object and hand, otherwise you may need some predefined policies to get the right contact).
```
cd data/preparation
python -u gen_cams_meta_pliers.py 
```

### Training Demo: Pliers

After finishing data preparation, you can use the following command to start training.
```
sh experiments/pliers/train.sh [1] [2] [3]
# [1] = GPU IDs you use, eg. 0, 1
# [2] = number of GPUs you use, eg. 2
# [3] = port
```

### Synthesis and Evaluate Demo: Pliers

After training, you will get some outputs in `experiments/pliers/tmp`, use the following command to start synthesizing.
```
sh synthesizer/run.sh [1] [2]
# [1] = aforementioned output path, eg. experiments/pliers/tmp/val/
# [2] = meta data path, eg. data/meta/pliers_meta.torch
```

You have finished generation of new manipulation after synthesizing, the results are in `experiments/pliers/synth`. You can also step forward and run evaluation metrics using the following command.

```
sh eval/run.sh [1] [2]
# [1] = final results path, eg. experiments/pliers/synth
# [2] = name of the file saving evaluation result, eg. eval.txt
```
