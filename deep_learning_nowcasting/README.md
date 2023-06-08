Nowcasting Deep Learning Model for TAASRAD19 dataset
-----

To activate the nowcasting virtualenv:
```
source nowcasting/bin/activate
```
## To change to 480x480 files:
```
cd /home/ubuntu/data/TAASRAD19/
ln -s hdf_archives_480x480 hdf_archives
```

## To change to downsampled (240x240) files:
```
cd /home/ubuntu/data/TAASRAD19/
ln -s hdf_archives_240x240 hdf_archives
```

Deep Learning Nowcasting Model for TAASRAD19 dataset.
The model code is a based on the original release from: https://github.com/sxjscience/HKO-7

The precomputed dataset sequences in HDF5 format for training/test can be downloaded from here:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3591404.svg)](https://doi.org/10.5281/zenodo.3591404)

The data directory (`data_dir` argument) used for both training ([train.py](train.py))
and prediction ([predict.py](predict.py)) scripts must respect the following structure:

```
DATA_DIR/   [data_dir]
  |
  +-- hdf_archives/ 
  |     |
  |     +-- 20100601.hdf
  |     +-- ...
  |     +-- all_data.hdf
  +-- hdf_metadata.csv
  +-- mask.png
```

#### TRAIN / VALIDATE MODEL
Training the model using the included configuration requires either one GPU with 16GB RAM or two GPU with 8GB RAM.
To train and validate the model on the years 2010 to 2016 with two GPUs run:
```
python train.py \
    --data_dir  /home/ubuntu/data/TAASRAD19 \
    --save_dir  /home/ubuntu/gitrepo/CS230_Nowcasting/deep_learning_nowcasting/trainOutput/baselineOutputwNewMask \
    --cfg  configurations/trajgru_55_55_33_1_64_1_192_1_192_13_13_9_b4.yml \
    --ctx  gpu \
    --date_start 2010-06-01 \
    --date_end   2012-06-01

python train.py \
    --data_dir  /home/ubuntu/data/TAASRAD19 \
    --save_dir  /home/ubuntu/gitrepo/CS230_Nowcasting/deep_learning_nowcasting/trainOutput/convGRU_3x \
    --cfg  configurations/trajConvgru_55_55_33_1_64_1_192_1_192_13_13_9_b4.yml \
    --ctx  gpu \
    --date_start 2010-06-01 \
    --date_end   2012-06-01

python train.py \
    --data_dir  /home/ubuntu/data/TAASRAD19 \
    --save_dir  /home/ubuntu/gitrepo/CS230_Nowcasting/deep_learning_nowcasting/trainOutput/TG_CG_CG_v2Mask \
    --cfg  configurations/TG_TG_TG.yml \
    --ctx  gpu \
    --date_start 2010-06-01 \
    --date_end   2012-06-01

  python train.py \
    --data_dir  /home/ubuntu/data/TAASRAD19 \
    --save_dir  /home/ubuntu/gitrepo/CS230_Nowcasting/deep_learning_nowcasting/trainOutput/TG_CL_TG_v2Mask_2 \
    --cfg  configurations/trajConvlstm_55_55_33_1_64_1_192_1_192_13_13_9_b4.yml \
    --ctx  gpu \
    --date_start 2010-06-01 \
    --date_end   2012-06-01
  

```

Use `python train.py --help` to see all options

#### GENERATE PREDICTIONS
To generate predictions using the pretrained model weights on GPU: (needs batch_size 2 to work on EC2 instance)
```
python predict.py \
    --model_cfg  pretrained_model/cfg0.yml \
    --model_dir  pretrained_model \
    --model_iter 99999 \
    --save_dir  /home/ubuntu/data/modelOut/newMask \
    --data_dir  /home/ubuntu/data/TAASRAD19 \
    --date_start 2016-11-01 \
    --date_end   2016-11-03 \
    --ctx gpu \
    --batch_size 4

    python predict.py \
    --model_cfg  pretrained_model/cfg0.yml \
    --model_dir  pretrained_model \
    --model_iter 99999 \
    --save_dir  /home/ubuntu/data/modelOut/baselineOutput_v1Mask_iter99999\
    --data_dir  /home/ubuntu/data/TAASRAD19 \
    --date_start 2017-03-01 \
    --date_end   2017-03-05 \
    --ctx gpu \
    --batch_size 4
    
    python predict.py \
    --model_cfg  trainOutput/baselineOutput_v1Mask/cfg0.yml \
    --model_dir  trainOutput/baselineOutput_v1Mask \
    --model_iter 9 \
    --save_dir  /home/ubuntu/data/modelOut/baselineOutput_v1Mask\
    --data_dir  /home/ubuntu/data/TAASRAD19 \
    --date_start 2017-03-01 \
    --date_end   2017-03-05 \
    --ctx gpu \
    --batch_size 4

    python predict.py \
    --model_cfg  trainOutput/TGTGTG_v2Mask/cfg0.yml \
    --model_dir  trainOutput/TGTGTG_v2Mask \
    --model_iter 0 \
    --save_dir  /home/ubuntu/data/modelOut/TGTGTG_v2Mask_Iter0 \
    --data_dir  /home/ubuntu/data/TAASRAD19 \
    --date_start 2017-03-01 \
    --date_end   2017-03-05 \
    --ctx gpu \
    --batch_size 4

    python predict.py \
    --model_cfg  trainOutput/convGRU_3x/cfg0.yml \
    --model_dir  trainOutput/convGRU_3x \
    --model_iter 9 \
    --save_dir  /home/ubuntu/data/modelOut/convGRU_3x_v2Mask_Iter9 \
    --data_dir  /home/ubuntu/data/TAASRAD19 \
    --date_start 2017-03-01 \
    --date_end   2017-03-05 \
    --ctx gpu \
    --batch_size 4

    python predict.py \
    --model_cfg  trainOutput/TGCLTG_v2Mask/cfg0.yml \
    --model_dir  trainOutput/TGCLTG_v2Mask \
    --model_iter 9 \
    --save_dir  /home/ubuntu/data/modelOut/TGCLTG_v2Mask_Iter9 \
    --data_dir  /home/ubuntu/data/TAASRAD19 \
    --date_start 2017-03-04 \
    --date_end   2017-03-05 \
    --ctx gpu \
    --batch_size 4

    python predict.py \
    --model_cfg  trainOutput/TGCLTG_v2Mask_2/cfg0.yml \
    --model_dir  trainOutput/TGCLTG_v2Mask_2 \
    --model_iter 48 \
    --save_dir  /home/ubuntu/data/modelOut/TGCLTG_v2Mask_Iter48 \
    --data_dir  /home/ubuntu/data/TAASRAD19 \
    --date_start 2017-03-04 \
    --date_end   2017-03-05 \
    --ctx gpu \
    --batch_size 4
```

Use `python predict.py --help` to see all options

Predictions are saved in as numpy array in npz format. 
For each TAASRAD19 sequence 3 files are generated in `save_dir`:
input (`in`), ground truth (`gt`) and prediction (`pred`) sequences.

For example for the following TAASRAD19 sequence:
```
start_datetime    2017-01-12 14:20:00
end_datetime      2017-01-12 23:55:00
run_length                        116
```

The generated output is:
```
-201701121440_in_92.npz
|____________|__|__| 
       |      |   |
  start time  |   |
              |   |
            type  |
                  |
         nr. of subsequences
           (5 frames each)

-201701121440_pred_92.npz
|____________|____|__| 
       |       |    |
  start time   |    |
               |    |
             type   |
                    |
           nr. of subsequences
            (20 frames each)

-201701121440_gt_92.npz
|____________|__|__| 
       |      |   |
  start time  |   |
              |   |
            type  |
                  |
         nr. of subsequences
          (20 frames each)
```