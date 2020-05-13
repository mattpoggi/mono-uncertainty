# On the uncertainty of <br> self-supervised monocular depth estimation

Demo code of "On the uncertainty of self-supervised monocular depth estimation", [Matteo Poggi](https://vision.disi.unibo.it/~mpoggi/), [Filippo Aleotti](https://filippoaleotti.github.io/website/), [Fabio Tosi](https://vision.disi.unibo.it/~ftosi/) and [Stefano Mattoccia](https://vision.disi.unibo.it/~smatt/), CVPR 2020.

[[Paper]](https://mattpoggi.github.io/assets/papers/poggi2020cvpr.pdf) - [[Poster]](https://mattpoggi.github.io/assets/papers/poggi2020cvpr_poster.pdf) - [[Youtube Video]](https://www.youtube.com/watch?v=bxVPXqf4zt4)

<p align="center"> 
<img src=https://mattpoggi.github.io/assets/img/uncertainty/poggi2020cvpr.gif>
</p>

## Citation
```shell
@inproceedings{Poggi_CVPR_2020,
  title     = {On the uncertainty of self-supervised monocular depth estimation},
  author    = {Poggi, Matteo and
               Aleotti, Filippo and
               Tosi, Fabio and
               Mattoccia, Stefano},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2020}
}
```   

## Contents

1. [Abstract](#abstract)
2. [Usage](#usage)
3. [Contacts](#contacts)
4. [Acknowledgements](#acknowledgements)

## Abstract

Self-supervised paradigms for monocular depth estimation are very appealing since they do not require ground truth annotations at all. Despite the astonishing results yielded by such methodologies, learning to reason about the uncertainty of the estimated depth maps is of paramount importance for practical applications, yet uncharted in the literature. Purposely, we explore for the first time how to estimate the uncertainty for this task and how this affects depth accuracy, proposing a novel peculiar technique specifically designed for self-supervised approaches. On the standard KITTI dataset, we exhaustively assess the performance of each method with different self-supervised paradigms. Such evaluation highlights that our proposal i) always improves depth accuracy significantly and ii) yields state-of-the-art results concerning uncertainty estimation when training on sequences and competitive results uniquely deploying stereo pairs. 

## Usage

### Requirements

* `PyTorch 0.4` 
* `python packages` such as opencv, PIL, numpy, matplotlib
* `Monodepth2` framework (https://github.com/nianticlabs/monodepth2)

### Getting started

Clone Monodepth2 repository and set it up using

```shell
sh prepare_monodepth2_engine.sh kitti_data
```
with `kitti_data` being the datapath to KITTI (if it does not exist, the script will create it and download KITTI archives there). 
Get pre-trained networks using

```
sh download_weights.sh M
```
Actually, weights for monocular experiment (M) are available.

### Run inference

Launch variants of the following command (see `batch_generate.sh` for a complete list)

```shell
python generate_maps.py --data_path kitti_data \
                        --load_weights_folder weights/M/Monodepth2-Post/models/weights_19/ \
                        --post_process \
                        --eval_split eigen_benchmark \
                        --eval_mono
```

Extended options (in addition to Monodepth2 arguments):
* `--bootstraps N`: loads N models from different trainings
* `--snapshots N`: loads N models from the same training
* `--dropout`: enables dropout inference
* `--repr`: enables repr inference
* `--log`: enables log-likelihood estimation (for Log and Self variants)
* `--no_eval`: saves results with custom scale factor (see below), for visualization purpose only
* `--custom_scale`: custom scale factor
* `--qual`: save qualitative maps for visualization

Results are saved in `--output_dir/raw` and are ready for evaluation. Qualitatives are save in `--output_dir/qual`.

### Run evaluation

Launch the following command

```shell
python evaluate.py --ext_disp_to_eval experiments/Post/raw/ \
                   --eval_mono \
                   --max_depth 80 \
                   --eval_split eigen_benchmark \
                   --eval_uncert
```

Optional arguments:
* `--eval_uncert`: evaluates estimated uncertainty

### Results

Results for evaluating `Post` depth and uncertainty maps:

```
   abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 |
&   0.088  &   0.508  &   3.842  &   0.134  &   0.917  &   0.983  &   0.995  \\

   abs_rel |          |     rmse |          |       a1 |          |
      AUSE |     AURG |     AUSE |     AURG |     AUSE |     AURG |
&   0.044  &   0.012  &   2.864  &   0.412  &   0.056  &   0.022  \\
```
Minor changes can occur with different versions of the python packages (not greater than 0.01)

## Contacts
m [dot] poggi [at] unibo [dot] it

## Acknowledgements

Thanks to Niantic and Clément Godard for sharing Monodepth2 code
