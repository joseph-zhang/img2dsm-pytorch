import os
import sys
import time
import torch
import numpy as np
import torchvision.utils as vutils

PYLIBS_DIR = None
if PYLIBS_DIR is not None:
    sys.path.insert(1, PYLIBS_DIR)

import datagen
from models import Pix2PixModel
from tensorboardX import SummaryWriter
from options.train_options import TrainOptions

class AverageMeter(object):
	def __init__(self):
		self.reset()


	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def main():
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    opt = TrainOptions().parse()
    model = Pix2PixModel(opt)
    model.setup(opt)

    gpu_nums = torch.cuda.device_count()
    print("{} GPUs!".format(gpu_nums))
    batch_size = opt.batch_size * gpu_nums

    train_loader = datagen.getTrainingData(opt, batch_size)
    writer = SummaryWriter(log_dir=opt.checkpoints_dir)

    train_iteration = 0
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        batch_time = AverageMeter()
        iter_data_time = time.time()
        for i, sample_batched in enumerate(train_loader):
            train_iteration = (epoch-1) * len(train_loader) + i

            image, depth, src_image = sample_batched['image'], sample_batched['depth'], sample_batched['src']
            model.set_input(image, depth)
            model.optimize_parameters()

            batch_time.update(time.time() - iter_data_time)
            iter_data_time = time.time()

            losses = model.get_current_losses()
            visual_ret = model.get_current_visuals()
            if train_iteration % opt.display_freq == 0:
                writer.add_scalar('train/G_GAN', losses['G_GAN'], train_iteration)
                writer.add_scalar('train/G_L1', losses['G_L1'], train_iteration)
                writer.add_scalar('train/D_real', losses['D_real'], train_iteration)
                writer.add_scalar('train/D_fake', losses['D_fake'], train_iteration)
                writer.add_image('train/Img', vutils.make_grid(src_image), train_iteration)
                writer.add_image('train/Prediction', vutils.make_grid(visual_ret['fake_B']), train_iteration)
                writer.add_image('train/Depth_GT', vutils.make_grid(depth), train_iteration)


            print('Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'\
                  'G_L1 {g_loss:.4f}\t'\
                  'G_GAN {g_gan:.4f}'
                  .format(epoch, i, len(train_loader), batch_time=batch_time, g_loss=losses['G_L1'], g_gan=losses['G_GAN']))

        if epoch % opt.save_epoch_freq == 0:
            print('saving the latest model (epoch %d)' % (epoch))
            model.save_networks(epoch)

        model.update_learning_rate()

if __name__ == '__main__':
    main()
