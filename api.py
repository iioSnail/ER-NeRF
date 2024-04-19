# -*- coding: UTF-8 -*-

import torch
import argparse

from nerf_triplane.provider import NeRFDataset
from nerf_triplane.utils import *
from nerf_triplane.network import NeRFNetwork

import os
import argparse
import numpy as np
import pandas as pd
from data_utils.deepspeech_features.deepspeech_store import get_deepspeech_model_file
from data_utils.deepspeech_features.deepspeech_features import conv_audios_to_deepspeech

# torch.autograd.set_detect_anomaly(True)
# Close tf32 features. Fix low numerical accuracy on rtx30xx gpu.
try:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
except AttributeError as e:
    print('Info. This pytorch version is not support with tf32.')


class AudioOption(object):

    def __init__(self):
        self.deepspeech = '~/.tensorflow/models/deepspeech-0_1_0-b90017e8.pb'


class AudioExtractor(object):

    def __init__(self):
        args = AudioOption()

        deepspeech_pb_path = os.path.expanduser(args.deepspeech)
        if not os.path.exists(deepspeech_pb_path):
            deepspeech_pb_path = get_deepspeech_model_file()

        self.deepspeech_pb_path = deepspeech_pb_path

    def extract_features(self,
                         in_audios,
                         out_files,
                         deepspeech_pb_path
                         ):
        num_frames_info = [None] * len(in_audios)

        for i, in_audio in enumerate(in_audios):
            if not out_files[i]:
                file_stem, _ = os.path.splitext(in_audio)
                out_files[i] = file_stem + ".npy"
                # print(out_files[i])
        conv_audios_to_deepspeech(
            audios=in_audios,
            out_files=out_files,
            num_frames_info=num_frames_info,
            deepspeech_pb_path=deepspeech_pb_path)

    def extract(self, filename):
        output = "/root/output/" + str(int(time.time())) + ".npy"

        in_audio = os.path.expanduser(filename)
        if not os.path.exists(in_audio):
            raise Exception("Input file/directory doesn't exist: {}".format(in_audio))

        self.extract_features(
            in_audios=[in_audio],
            out_files=[output],
            deepspeech_pb_path=self.deepspeech_pb_path)

        return output


class Option(object):

    def __init__(self):
        self.workspace = 'trial_zhf_torso/'
        self.aud = '/root/demo2.npy'
        self.path = 'data/zhf/'

        self.H = 450
        self.O = True
        self.W = 450
        self.amb_aud_loss = 1
        self.amb_dim = 2
        self.amb_eye_loss = 1
        self.asr = False
        self.asr_model = 'deepspeech'
        self.asr_play = False
        self.asr_save_feats = False
        self.asr_wav = ''
        self.att = 2
        self.bg_img = ''
        self.bound = 1
        self.ckpt = 'latest'
        self.color_space = 'srgb'
        self.cuda_ray = True
        self.data_range = [0, -1]
        self.density_thresh = 10
        self.density_thresh_torso = 0.01
        self.dt_gamma = 0.00390625
        self.emb = False
        self.exp_eye = True
        self.fbg = False
        self.finetune_lips = False
        self.fix_eye = -1
        self.fovy = 21.24
        self.fp16 = True
        self.fps = 50
        self.gui = False
        self.head_ckpt = ''
        self.ind_dim = 4
        self.ind_dim_torso = 8
        self.ind_num = 10000
        self.init_lips = False
        self.iters = 200000
        self.l = 10
        self.lambda_amb = 0.0001
        self.lr = 0.01
        self.lr_net = 0.001
        self.m = 50
        self.max_ray_batch = 4096
        self.max_spp = 1
        self.max_steps = 16
        self.min_near = 0.05
        self.num_rays = 65536
        self.num_steps = 16
        self.offset = [0, 0, 0]
        self.part = False
        self.part2 = False
        self.patch_size = 1
        self.preload = 0
        self.r = 10
        self.radius = 3.35
        self.scale = 4
        self.seed = 0
        self.smooth_eye = False
        self.smooth_lips = False
        self.smooth_path = False
        self.smooth_path_window = 7
        self.test = True
        self.test_train = True
        self.torso = True
        self.torso_shrink = 0.8
        self.train_camera = False
        self.unc_loss = 1
        self.update_extra_interval = 16
        self.upsample_steps = 0
        self.warmup_step = 10000


class ER_NeRF(object):

    def __init__(self, save_path="/root/output/"):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        print("Device:", device)

        self.save_path = save_path

        opt = Option()
        model = NeRFNetwork(opt)

        # manually load state dict for head
        if opt.torso and opt.head_ckpt != '':

            model_dict = torch.load(opt.head_ckpt, map_location='cpu')['model']

            missing_keys, unexpected_keys = model.load_state_dict(model_dict, strict=False)

            if len(missing_keys) > 0:
                print(f"[WARN] missing keys: {missing_keys}")
            if len(unexpected_keys) > 0:
                print(f"[WARN] unexpected keys: {unexpected_keys}")

                # freeze these keys
            for k, v in model.named_parameters():
                if k in model_dict:
                    print(f'[INFO] freeze {k}, {v.shape}')
                    v.requires_grad = False

        criterion = torch.nn.MSELoss(reduction='none')

        metrics = [PSNRMeter(), LPIPSMeter(device=device), LMDMeter(backend='fan')]

        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16,
                          metrics=metrics, use_checkpoint=opt.ckpt)

        self.audio_extractor = AudioExtractor()

        self.trainer = trainer
        self.model = model
        self.opt = opt

    def gene_test_loader(self, audio_npy):
        # temp fix: for update_extra_states
        self.opt.aud = audio_npy
        test_set = NeRFDataset(self.opt, device=self.device, type='train')
        # a manual fix to test on the training dataset
        test_set.training = False
        test_set.num_rays = -1

        test_loader = test_set.dataloader()
        self.model.aud_features = test_loader._data.auds
        self.model.eye_areas = test_loader._data.eye_area

        return test_loader

    def inference(self, audio_filename):
        start = time.time()
        audio_npy = self.audio_extractor.extract(audio_filename)
        test_loader = self.gene_test_loader(audio_npy)

        filename = str(int(time.time())) + ".mp4"
        self.trainer.test(test_loader, save_path=self.save_path, name=filename, inference=True)

        print("Duration:", time.time() - start)
        return os.path.join(self.save_path, filename)
