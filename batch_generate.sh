#
# MIT License
#
# Copyright (c) 2020 Matteo Poggi m.poggi@unibo.it
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

if [ "$#" -eq  "0" ]
then
    echo Usage: $0 kitti_datapath
    exit
fi

########
# Post example
python generate_maps.py --data_path $1 --load_weights_folder weights/M/Monodepth2-Post/models/weights_19/ --post_process --eval_split eigen_benchmark --eval_mono --output_dir experiments/Post

######## Empirical methods
# Drop example
python generate_maps.py --data_path $1 --load_weights_folder weights/M/Monodepth2-Drop/models/weights_19/ --dropout --eval_split eigen_benchmark --eval_mono --output_dir experiments/Drop

# Boot example
python generate_maps.py --data_path $1 --load_weights_folder weights/M/Monodepth2-Boot/models/ --bootstraps 8 --eval_split eigen_benchmark --eval_mono --output_dir experiments/Boot

# Snap example
python generate_maps.py --data_path $1 --load_weights_folder weights/M/Monodepth2-Snap/models/ --snapshots 8 --eval_split eigen_benchmark --eval_mono --output_dir experiments/Snap

######## Predictive methods
# Repr example
python generate_maps.py --data_path $1 --load_weights_folder weights/M/Monodepth2-Repr/models/weights_19/ --repr --eval_split eigen_benchmark --eval_mono --output_dir experiments/Repr

# Log example
python generate_maps.py --data_path $1 --load_weights_folder weights/M/Monodepth2-Log/models/weights_19/ --log --eval_split eigen_benchmark --eval_mono --output_dir experiments/Log

# Self example
python generate_maps.py --data_path $1 --load_weights_folder weights/M/Monodepth2-Self/models/weights_19/ --log --eval_split eigen_benchmark --eval_mono --output_dir experiments/Self

######## Bayesian methods
# Boot+Log example
python generate_maps.py --data_path $1 --load_weights_folder weights/M/Monodepth2-Boot+Log/models/ --bootstraps 8 --log --eval_split eigen_benchmark --eval_mono --output_dir experiments/Boot+Log

# Snap+Log example
python generate_maps.py --data_path $1 --load_weights_folder weights/M/Monodepth2-Snap+Log/models/ --snapshots 8 --log --eval_split eigen_benchmark --eval_mono --output_dir experiments/Snap+Self

# Boot+Self example
python generate_maps.py --data_path $1 --load_weights_folder weights/M/Monodepth2-Boot+Self/models/ --bootstraps 8 --log --eval_split eigen_benchmark --eval_mono --output_dir experiments/Boot+Self

# Snap+Self example
python generate_maps.py --data_path $1 --load_weights_folder weights/M/Monodepth2-Snap+Self/models/ --snapshots 8 --log --eval_split eigen_benchmark --eval_mono --output_dir experiments/Snap+Self
