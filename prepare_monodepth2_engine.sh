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
