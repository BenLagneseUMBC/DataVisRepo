import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import random
from math import ceil

model_paths = ['models/' + x for x in os.listdir('models/') if 'pth' in x]  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')

test_img_folder = '../Real-ESRGAN/datasets/ln_covers/*'

models = [arch.RRDBNet(3, 3, 64, 23, gc=32) for i in range(len(model_paths))]

picks = ['../Real-ESRGAN/dataset_sample/Orig/' + path for path in os.listdir('../Real-ESRGAN/dataset_sample/Orig/')]
lr_picks = ['../Real-ESRGAN/dataset_sample/LR/' + path for path in os.listdir('../Real-ESRGAN/dataset_sample/LR/')]

for model_path, model in zip(model_paths, models):
    # if 'product' not in model_path:
    #     continue
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    print('Model path {:s}. \nTesting...'.format(model_path))

    idx = 0
    if not os.path.exists(f'results/ln_covers/{os.path.basename(model_path)}'):
        os.mkdir(f'results/ln_covers/{os.path.basename(model_path)}')

    with open(f'results/ln_covers/{os.path.basename(model_path)}/results.txt', 'w') as metrics:
        running_ssim = 0
        running_psnr = 0
        running_mse = 0
        for path, lr_path in zip(picks, lr_picks):
            idx += 1
            base = osp.splitext(osp.basename(path))[0]
            print(idx, base)
            # read images
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img_LR = cv2.imread(lr_path, cv2.IMREAD_COLOR)
            img_LR = img_LR * 1.0 / 255
            img_LR = torch.from_numpy(np.transpose(img_LR[:, :, [2, 1, 0]], (2, 0, 1))).float()
            img_LR = img_LR.unsqueeze(0)
            img_LR = img_LR.to(device)

            with torch.no_grad():
                output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)
            
            cv2.imwrite('results/ln_covers/{:s}/{:s}.png'.format(os.path.basename(model_path), base), output)
            diffs = ((img.shape[0] - output.shape[0])//2, img.shape[0]-int(ceil((img.shape[0] - output.shape[0])/2)),
                     (img.shape[1] - output.shape[1])//2, img.shape[1]-int(ceil((img.shape[1] - output.shape[1])/2)))
            crop = img[diffs[0]:diffs[1], diffs[2]:diffs[3]]

            run_ssim, _ = ssim(crop, output, full=True, multichannel=True, channel_axis=2)
            run_psnr = psnr(crop, output)
            run_mse = mse(crop, output)
            metrics.write(path + ' | SSIM: ' + str(run_ssim) + ' | PSNR: ' + str(run_psnr) + '\n')
            running_ssim += run_ssim
            running_psnr += run_psnr
            running_mse += run_mse
        metrics.write(model_path + ' | Average SSIM: ' + str(running_ssim/len(picks)) + ' | PSNR: ' + str(running_psnr/len(picks)) + '\n\n\n')

if not os.path.exists(f'results/ln_covers/bicubic'):
        os.mkdir(f'results/ln_covers/bicubic')

idx = 0
with open(f'results/ln_covers/bicubic/results.txt', 'w') as metrics:
    running_ssim = 0
    running_psnr = 0
    running_mse = 0
    for path in picks:
        idx += 1
        base = osp.splitext(osp.basename(path))[0]
        print(idx, base)
        # read images
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img_LR = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4), interpolation=cv2.INTER_CUBIC)
        output = cv2.resize(img_LR, (img_LR.shape[1] * 4, img_LR.shape[0] * 4))
        
        cv2.imwrite('results/ln_covers/bicubic/{:s}.png'.format(base), output)
        diffs = ((img.shape[0] - output.shape[0])//2, img.shape[0]-int(ceil((img.shape[0] - output.shape[0])/2)),
                    (img.shape[1] - output.shape[1])//2, img.shape[1]-int(ceil((img.shape[1] - output.shape[1])/2)))
        crop = img[diffs[0]:diffs[1], diffs[2]:diffs[3]]

        run_ssim, _ = ssim(crop, output, full=True, multichannel=True, channel_axis=2)
        run_psnr = psnr(crop, output)
        run_mse = mse(crop, output)
        metrics.write(path + ' | SSIM: ' + str(run_ssim) + ' | PSNR: ' + str(run_psnr) + '\n')
        running_ssim += run_ssim
        running_psnr += run_psnr
        running_mse += run_mse
    metrics.write('bicubic' + ' | Average SSIM: ' + str(running_ssim/len(picks)) + ' | PSNR: ' + str(running_psnr/len(picks)) + '\n\n\n')