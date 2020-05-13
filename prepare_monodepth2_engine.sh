# clone monodepth2 repository
git clone https://github.com/nianticlabs/monodepth2
touch monodepth2/__init__.py

# small fix to kitti dataloader to work from root directory
sed -i 's/ kitti_utils/ ..kitti_utils/g' monodepth2/datasets/kitti_dataset.py
sed -i 's/MonodepthOptions/MonodepthOptions(object)/g' monodepth2/options.py

# change __init__ file in monodepth2/network to exclude depth network
rm monodepth2/networks/__init__.py
echo from .resnet_encoder import ResnetEncoder  >> monodepth2/networks/__init__.py
echo from .pose_decoder import PoseDecoder  >> monodepth2/networks/__init__.py
echo from .pose_cnn import PoseCNN >> monodepth2/networks/__init__.py 

# ready to go!
