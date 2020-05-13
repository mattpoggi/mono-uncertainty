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
# Lr example
python generate_maps.py --data_path $1 --load_weights_folder weights/M/Monodepth2-Lr/models/weights_19/ --repr --eval_split eigen_benchmark --eval_mono --output_dir experiments/Lr

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
