from __future__ import absolute_import, division, print_function

import os
import argparse
from monodepth2.options import MonodepthOptions

# Extended set of options
class UncertaintyOptions(MonodepthOptions):

    def __init__(self):
    
        super(UncertaintyOptions, self).__init__()
        
        self.parser.add_argument("--custom_scale", type=float, default=100., help="custom scale factor for depth maps")
        
        self.parser.add_argument("--eval_uncert", help="if set enables uncertainty evaluation", action="store_true")
        self.parser.add_argument("--log", help="if set, adds the variance output to monodepth2 according to log-likelihood maximization technique", action="store_true")
        self.parser.add_argument("--lr", help="if set, adds the LR output to monodepth2", action="store_true")

        self.parser.add_argument("--dropout", help="if set enables dropout inference", action="store_true")

        self.parser.add_argument("--bootstraps", type=int, default=1, help="if > 1, loads multiple checkpoints from different trainings to build a bootstrapped ensamble")
        self.parser.add_argument("--snapshots", type=int, default=1, help="if > 1, loads the last N checkpoints to build a snapshots ensemble")

        self.parser.add_argument("--output_dir", type=str, default="output", help="output directory for predicted depth and uncertainty maps")
        self.parser.add_argument("--qual", help="if set save colored depth and uncertainty maps", action="store_true")

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
