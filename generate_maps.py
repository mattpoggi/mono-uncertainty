from __future__ import absolute_import, division, print_function
import warnings

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

import monodepth2
import monodepth2.kitti_utils as kitti_utils
from monodepth2.layers import *
from monodepth2.utils import *
from extended_options import *
import monodepth2.datasets as datasets
import monodepth2.networks as legacy
import networks
import progressbar
import matplotlib.pyplot as plt

import sys

splits_dir = os.path.join(os.path.dirname(__file__), "monodepth2/splits")

def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def get_mono_ratio(disp, gt):
    """Returns the median scaling factor
    """
    mask = gt>0
    return np.median(gt[mask]) / np.median(cv2.resize(1/disp, (gt.shape[1], gt.shape[0]))[mask])

def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
    opt.batch_size = 1

    assert sum((opt.eval_mono, opt.eval_stereo, opt.no_eval)) == 1, "Please choose mono or stereo evaluation by setting either --eval_mono, --eval_stereo, --custom_run"
    assert sum((opt.log, opt.repr)) < 2, "Please select only one between LR and LOG by setting --repr or --log"
    assert opt.bootstraps == 1 or opt.snapshots == 1, "Please set only one of --bootstraps or --snapshots to be major than 1"
    
    # get the number of networks
    nets = max(opt.bootstraps,opt.snapshots)
    do_uncert = (opt.log or opt.repr or opt.dropout or opt.post_process or opt.bootstraps > 1 or opt.snapshots > 1)

    print("-> Beginning inference...")

    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
    assert os.path.isdir(opt.load_weights_folder), "Cannot find a folder at {}".format(opt.load_weights_folder)

    print("-> Loading weights from {}".format(opt.load_weights_folder))

    filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))

    if opt.bootstraps > 1:
    
        # prepare multiple checkpoint paths from different trainings
        encoder_path = [os.path.join(opt.load_weights_folder, "boot_%d"%i, "weights_19", "encoder.pth") for i in range(1,opt.bootstraps+1)]
        decoder_path = [os.path.join(opt.load_weights_folder, "boot_%d"%i, "weights_19", "depth.pth") for i in range(1,opt.bootstraps+1)]
        encoder_dict = [torch.load(encoder_path[i]) for i in range(opt.bootstraps)]
        height = encoder_dict[0]['height']
        width = encoder_dict[0]['width']

    elif opt.snapshots > 1:
    
        # prepare multiple checkpoint paths from the same training
        encoder_path = [os.path.join(opt.load_weights_folder, "weights_%d"%i, "encoder.pth") for i in range(opt.num_epochs-opt.snapshots,opt.num_epochs)]
        decoder_path = [os.path.join(opt.load_weights_folder, "weights_%d"%i, "depth.pth") for i in range(opt.num_epochs-opt.snapshots,opt.num_epochs)]
        encoder_dict = [torch.load(encoder_path[i]) for i in range(opt.snapshots)]
        height = encoder_dict[0]['height']
        width = encoder_dict[0]['width']

    else:

        # prepare just a single path
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
        encoder_dict = torch.load(encoder_path)
        height = encoder_dict['height']
        width = encoder_dict['width']

    img_ext = '.png' if opt.png else '.jpg'
    dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                           height, width, 
                                           [0], 4, is_train=False, img_ext=img_ext)
    dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)

    if nets > 1:

        # load multiple encoders and decoders 
       	encoder = [legacy.ResnetEncoder(opt.num_layers, False) for i in range(nets)]
        depth_decoder = [networks.DepthUncertaintyDecoder(encoder[i].num_ch_enc, num_output_channels=1, uncert=(opt.log or opt.repr), dropout=opt.dropout) for i in range(nets)]

        model_dict = [encoder[i].state_dict() for i in range(nets)]
        for i in range(nets):
            encoder[i].load_state_dict({k: v for k, v in encoder_dict[i].items() if k in model_dict[i]})
            depth_decoder[i].load_state_dict(torch.load(decoder_path[i]))
            encoder[i].cuda()
            encoder[i].eval()
            depth_decoder[i].cuda()
            depth_decoder[i].eval()

    else:
    
        # load a single encoder and decoder
       	encoder = legacy.ResnetEncoder(opt.num_layers, False)
        depth_decoder = networks.DepthUncertaintyDecoder(encoder.num_ch_enc, num_output_channels=1, uncert=(opt.log or opt.repr), dropout=opt.dropout)
        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))
        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

    # accumulators for depth and uncertainties
    pred_disps = []
    pred_uncerts = []

    print("-> Computing predictions with size {}x{}".format(width, height))
    with torch.no_grad():
        bar = progressbar.ProgressBar(max_value=len(dataloader))
        for i, data in enumerate(dataloader):
            
            input_color = data[("color", 0, 0)].cuda()

            # updating progress bar
            bar.update(i)
            if opt.post_process:

                # post-processed results require each image to have two forward passes
                input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)
            if nets > 1:

                # infer multiple predictions from multiple networks
                disps_distribution = []
                uncerts_distribution = []
                for i in range(nets):
                    output =  depth_decoder[i](encoder[i](input_color))
                    disps_distribution.append( torch.unsqueeze(output[("disp", 0)],0) )
                    if opt.log:
                        uncerts_distribution.append( torch.unsqueeze(output[("uncert", 0)],0) )

                disps_distribution = torch.cat(disps_distribution, 0)
                if opt.log:
                
                    # bayesian uncertainty
                    pred_uncert = torch.var(disps_distribution, dim=0, keepdim=False) +  torch.sum(torch.cat(uncerts_distribution, 0), dim=0, keepdim=False)
                else:
                
                    # uncertainty as variance of the predictions 
                    pred_uncert = torch.var(disps_distribution, dim=0, keepdim=False)
                pred_uncert = pred_uncert.cpu()[0].numpy()
                output = torch.mean(disps_distribution, dim=0, keepdim=False)
                pred_disp, _ = disp_to_depth(output, opt.min_depth, opt.max_depth)                
            elif opt.dropout:
            
                # infer multiple predictions from multiple networks with dropout
                disps_distribution = []
                uncerts = []
                
                # we infer 8 predictions as the number of bootstraps and snaphots
                for i in range(8):
                    output = depth_decoder(encoder(input_color))
                    disps_distribution.append( torch.unsqueeze(output[("disp", 0)],0) )
                disps_distribution = torch.cat(disps_distribution, 0)

                # uncertainty as variance of the predictions
                pred_uncert = torch.var(disps_distribution, dim=0, keepdim=False).cpu()[0].numpy()

                # depth as mean of the predictions                
                output = torch.mean(disps_distribution, dim=0, keepdim=False)
                pred_disp, _ = disp_to_depth(output, opt.min_depth, opt.max_depth)
            else:
                output = depth_decoder(encoder(input_color))
                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                if opt.log:
                
                    # log-likelihood maximization
                    pred_uncert = torch.exp(output[("uncert", 0)]).cpu()[:, 0].numpy()
                elif opt.repr:
                
                    # learned reprojection
                    pred_uncert = (output[("uncert", 0)]).cpu()[:, 0].numpy()

            pred_disp = pred_disp.cpu()[:, 0].numpy()
            if opt.post_process:
            
                # applying Monodepthv1 post-processing to improve depth and get uncertainty
                N = pred_disp.shape[0] // 2
                pred_uncert = np.abs(pred_disp[:N] - pred_disp[N:, :, ::-1])		
                pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])
                pred_uncerts.append(pred_uncert)

            pred_disps.append(pred_disp)

            # uncertainty normalization
            if opt.log or opt.repr or opt.dropout or nets > 1:	
                pred_uncert = (pred_uncert - np.min(pred_uncert)) / (np.max(pred_uncert) - np.min(pred_uncert))
                pred_uncerts.append(pred_uncert)
    pred_disps = np.concatenate(pred_disps)
    if do_uncert:		
        pred_uncerts = np.concatenate(pred_uncerts)

    # saving 16 bit depth and uncertainties
    print("-> Saving 16 bit maps")
    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]

    if not os.path.exists(os.path.join(opt.output_dir, "raw", "disp")):
        os.makedirs(os.path.join(opt.output_dir, "raw", "disp"))

    if not os.path.exists(os.path.join(opt.output_dir, "raw", "uncert")):
        os.makedirs(os.path.join(opt.output_dir, "raw", "uncert"))
        
    if opt.qual: 
        if not os.path.exists(os.path.join(opt.output_dir, "qual", "disp")):
            os.makedirs(os.path.join(opt.output_dir, "qual", "disp"))
        if do_uncert:
            if not os.path.exists(os.path.join(opt.output_dir, "qual", "uncert")):
                os.makedirs(os.path.join(opt.output_dir, "qual", "uncert"))

    bar = progressbar.ProgressBar(max_value=len(pred_disps))
    for i in range(len(pred_disps)):
        bar.update(i)
        if opt.eval_stereo:
        
            # save images scaling with KITTI baseline
            cv2.imwrite(os.path.join(opt.output_dir, "raw", "disp", '%06d_10.png'%i), (pred_disps[i]*(dataset.K[0][0]*gt_depths[i].shape[1])*256./10).astype(np.uint16))

        elif opt.eval_mono:
        
            # save images scaling with ground truth median
            ratio = get_mono_ratio(pred_disps[i], gt_depths[i])
            cv2.imwrite(os.path.join(opt.output_dir, "raw", "disp", '%06d_10.png'%i), (pred_disps[i]*(dataset.K[0][0]*gt_depths[i].shape[1])*256./ratio/10.).astype(np.uint16))
        else:
        
            # save images scaling with custom factor
            cv2.imwrite(os.path.join(opt.output_dir, "raw", "disp", '%06d_10.png'%i), (pred_disps[i]*(opt.custom_scale)*256./10).astype(np.uint16))

        if do_uncert:
        
            # save uncertainties
            cv2.imwrite(os.path.join(opt.output_dir, "raw", "uncert", '%06d_10.png'%i), (pred_uncerts[i]*(256*256-1)).astype(np.uint16))

        if opt.qual:
        
            # save colored depth maps
            plt.imsave(os.path.join(opt.output_dir, "qual", "disp", '%06d_10.png'%i), pred_disps[i], cmap='magma')
            if do_uncert:
            
                # save colored uncertainty maps
                plt.imsave(os.path.join(opt.output_dir, "qual", "uncert", '%06d_10.png'%i), pred_uncerts[i], cmap='hot')

    # see you next time! 
    print("\n-> Done!")

if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning)
    options = UncertaintyOptions()
    evaluate(options.parse())