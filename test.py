import os
import sys
import time
import torch
import numpy as np
import matplotlib.image as mpimg
import torchvision.utils as vutils

PYLIBS_DIR = None
if PYLIBS_DIR is not None:
    sys.path.insert(1, PYLIBS_DIR)

import datagen
import tifffile
from tqdm import tqdm
from models import Pix2PixModel
from options.test_options import TestOptions

def metrics_preprocess(pred, gt):
    """
    This function do some preprocessing before metrics calculation.
    - check zero value to avoid numerical problems;
    - do scale matching;

    Note that the input `pred` and `gt` are both 4D npArraires,
    return the corresponding image pair.
    """

    # squeeze the first and last idx (which is one in test processing)
    pred = np.squeeze(pred)
    gt = np.squeeze(gt)

    # get mask array
    mask = gt > 1e-6

    # scale matching
    #scalor = np.median(gt[mask]) / np.median(pred[mask])
    #pred[mask] *= scalor

    # avoid numerical problems
    pred[pred < 1e-6] = 1e-6
    gt[gt < 1e-6] = 1e-6

    nanMask = np.isnan(gt)
    gt[nanMask] = 0
    pred[nanMask] = 0

    return pred, gt


def compute_metrics(pred, gt):
    """
    This function computes the metrics value on a pair of (pred, gt).
    Note that the input `pred` and `gt` are both npArraires.
    Return a list of result float-values which correspond to MAE, MSE, RMSE and a1, a2, a3, ZNCC respectively.
    """

    # test image pre-processing
    pred, gt = metrics_preprocess(pred, gt)

    # compute MSE and RMSE
    mse = ((gt - pred) ** 2).mean()
    rmse = np.sqrt(mse)

    # compute MAE
    mae = np.mean(np.abs(gt - pred))

    # compute ap accuracy
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    # compute ZNCC
    pred_mean = np.mean(pred)
    dsm_mean = np.mean(gt)
    pred_std = np.std(pred, ddof=1)
    dsm_std = np.std(gt, ddof=1)
    zncc = (((pred - pred_mean) * (gt - dsm_mean)) / (pred_std * dsm_std)).mean()

    return [mae, mse, rmse, a1, a2, a3, zncc]


def save_test(handle, result_log):
    """
    This function save the test metrics in a given file.
    -----
    handle: text file handle
    result_log: the metrics results, a 2D list
    -----
    """
    mae = np.array(result_log[0]).mean()
    mse = np.array(result_log[1]).mean()
    rmse = np.array(result_log[2]).mean()
    a1 = np.array(result_log[3]).mean()
    a2 = np.array(result_log[4]).mean()
    a3 = np.array(result_log[5]).mean()
    zncc = np.array(result_log[6]).mean()

    # write test result to test file by using handle
    handle.write("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}\n" \
            .format('MAE', 'MSE', 'RMSE', 'a1', 'a2', 'a3', 'ZNCC'))

    handle.write("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}"\
            .format(mae, mse, rmse, a1, a2, a3, zncc))


def save_diff(opt, global_step, result):
    """
    This function save the source, pred, gt and diff images as a test case result.
    -----
    global_step: global_step of testing process, the number of current iteration
    result: current iteration result, a dict of result images
    -----
    """
    mpimg.imsave(os.path.join(opt.results_dir, "src_test_{}.png".format(global_step)), result["src"])
    mpimg.imsave(os.path.join(opt.results_dir, "gt_test_{}.png".format(global_step)), result["gt"], cmap="jet")
    mpimg.imsave(os.path.join(opt.results_dir, "pred_test_{}.png".format(global_step)), result["pred"], cmap="jet")
    mpimg.imsave(os.path.join(opt.results_dir, "diff_test_{}.png".format(global_step)), result["diff"], cmap="jet")
    tifffile.imsave(os.path.join(opt.results_dir, "gt_test_{}_sc.tif".format(global_step)), result["gt"])
    tifffile.imsave(os.path.join(opt.results_dir, "pred_test_{}_sc.tif".format(global_step)), result["pred"])
    print("saved diff images (iter={})\n".format(global_step))


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    model = Pix2PixModel(opt)
    model.setup(opt)
    model.eval()

    test_loader = datagen.getTestingData(opt, 1)
    if not os.path.exists(opt.results_dir):
        os.makedirs(opt.results_dir)

    result_log = [[] for i in range(7)]
    step = 0

    for sample_batched in tqdm(test_loader):
        image, depth, src_image = sample_batched['image'], sample_batched['depth'], sample_batched['src']
        model.set_input(image, depth)
        model.test()
        visual_ret = model.get_current_visuals()
        output_np = np.squeeze(visual_ret['fake_B'].cpu().detach().numpy())
        height_np = np.squeeze(visual_ret['real_B'].cpu().detach().numpy())
        src_np = np.squeeze(src_image.numpy()).astype(np.uint)
        test_result = compute_metrics(output_np, height_np)

        for it, item in enumerate(test_result):
            result_log[it].append(item)

        step = step + 1
        if step % 30 == 0:
            result = {"src": src_np,
                      "gt": height_np,
                      "pred": output_np,
                      "diff": np.abs(output_np - height_np)}
            save_diff(opt, step, result)

    f = open(opt.results_dir+"evalog.txt", 'w')
    f.write("Done testing -- epoch limit reached")
    f.write(" after %d iteration\n\n" % (step))
    save_test(f, result_log)
    f.close()

