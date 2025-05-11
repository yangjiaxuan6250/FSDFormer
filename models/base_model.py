# GPL License
# Copyright (C) 2022, UESTC
# All Rights Reserved 
#
# @Time    : 2022
# @Author  : Xiao Wu
# @reference:
import torch
import torch.nn as nn
import imageio
import kornia
import os
from UDL.Basis.dist_utils import reduce_mean
from UDL.Basis.python_sub_class import ModelDispatcher, TaskDispatcher
from common.metrics import PSNR_ycbcr, sub_mean, add_mean, SSIM, quantize

class DerainModel(ModelDispatcher, name='derain'):
    _models = dict()
    def __init__(self, model=None, criterion=None):
        super(DerainModel, self).__init__()

        self.model = model
        self.criterion = criterion
        self.set_metrics()
        self.loss_l1 = nn.L1Loss()

    def __init_subclass__(cls, name='', **kwargs):

        # print(name, cls)
        if name != '':
            cls._models[name] = cls
            cls._name = name
        else:
            cls._models[cls.__name__] = cls
            cls._name = cls.__name__
            # warnings.warn(f'Creating a subclass of MetaModel {cls.__name__} with no name.')

    def train_step(self, data, *args, **kwargs):
        log_vars = {}
        samples, gt = data['O'].cuda(), data['B'].cuda()

        samples = sub_mean(samples)

        gt_y = sub_mean(gt)
        # gt_y = kornia.geometry.transform.build_pyramid(gt_y, 3)

        outputs = self.model(samples)

        m_loss = self.criterion.losses['m_loss'](outputs[1], gt_y) + self.criterion.losses['m_loss'](outputs[2], gt_y)
        l1_loss = self.criterion.losses['l1_loss'](outputs[0], gt_y)

        f_loss = self.criterion.losses['f_loss'](outputs[0], gt_y)
        edge_loss = self.criterion.losses['edge_loss'](outputs[0], gt_y)
        # final_loss = l1_loss +  0.01*f_loss
        final_loss = l1_loss + 0.01*f_loss + 0.05*edge_loss + 0.1*m_loss

        # loss_fft = self.criterion.losses['f_loss'](outputs[0], gt_y[0]) + self.criterion.losses['f_loss'](outputs[1], gt_y[1])+self.criterion.losses['f_loss'](outputs[2], gt_y[2])
        # loss_char = self.criterion.losses['l1_loss'](outputs[0], gt_y[0]) + self.criterion.losses['l1_loss'](outputs[1],
        #                                                                                                   gt_y[1]) + \
        #            self.criterion.losses['l1_loss'](outputs[2], gt_y[2])
        # loss_edge = self.criterion.losses['edge_loss'](outputs[0], gt_y[0]) + self.criterion.losses['edge_loss'](outputs[1], gt_y[1])+self.criterion.losses['edge_loss'](outputs[2], gt_y[2])
        # final_loss = loss_char + 0.01 * loss_fft + 0.05 * loss_edge

        loss = {'loss': final_loss}
        pred = add_mean(outputs[0])

        log_vars.update(ssim=reduce_mean(torch.mean(self.ssim(pred / 255.0, gt))))
        log_vars.update(psnr=reduce_mean(self.psnr(pred, gt * 255.0, 4, 255.0)))
        log_vars.update(**loss)

        return {'loss': loss['loss'], 'log_vars': log_vars}

    def val_step(self, batch, *args, **kwargs):
        metrics = {}

        O, B = batch['O'].cuda(), batch['B'].cuda()
        samples = sub_mean(O)
        # pred = self.model(samples)[0]
        pred = self.model.module.forward_chop(samples)[0]
        pred = quantize(add_mean(pred), 255)
        normalized = pred[0]
        tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()

        imageio.imwrite(os.path.join(kwargs['save_dir'], ''.join([batch['filename'][0], '.png'])),
                        tensor_cpu.numpy())
        # print(os.path.join(kwargs['save_dir'], ''.join([batch['filename'][0], '.png'])))

        with torch.no_grad():
            metrics.update(ssim=reduce_mean(torch.mean(self.ssim(pred / 255.0, B))))
            metrics.update(psnr=reduce_mean(self.psnr(pred, B * 255.0, 4, 255.0)))
        return {'log_vars': metrics}

    def set_metrics(self, rgb_range=255.):

        self.psnr = PSNR_ycbcr()
        self.ssim = SSIM(size_average=False, data_range=rgb_range)
