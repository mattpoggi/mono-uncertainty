if [ "$#" -eq  "0" ]
then
    echo Usage: prepare_monodepth2_engine.sh kitti_datapath
    exit
fi

# clone monodepth2 repository
git clone https://github.com/nianticlabs/monodepth2
touch monodepth2/__init__.py

# small fix to kitti dataloader to work from root directory
sed -i 's/ kitti_utils/ ..kitti_utils/g' monodepth2/datasets/kitti_dataset.py

# change __init__ file in monodepth2/network to exclude depth network
rm monodepth2/networks/__init__.py
echo from .resnet_encoder import ResnetEncoder  >> monodepth2/networks/__init__.py
echo from .pose_decoder import PoseDecoder  >> monodepth2/networks/__init__.py
echo from .pose_cnn import PoseCNN >> monodepth2/networks/__init__.py 

# create groundtruth files
cd monodepth2
if [ -d $1/2011_09_26 ]
then
    echo Found KITTI dataset at $1
else
    echo Download KITTI dataset...
    mkdir $1
    wget -i splits/kitti_archives_to_download.txt -P $1
fi
python export_gt_depth.py --data_path $1 --split eigen
#python export_gt_depth.py --data_path $1 --split eigen_benchmark
cd ..

# ready to go!
