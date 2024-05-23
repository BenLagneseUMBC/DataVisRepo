import torch
from collections import OrderedDict

net_PSNR_path = './models/psnr.pth'
net_ESRGAN_path = './models/esrgan.pth'

net_PSNR = torch.load(net_PSNR_path)
net_ESRGAN = torch.load(net_ESRGAN_path)

def func0(psnr, esrgan):
    alpha = 0.5
    return (1 - alpha) * psnr + alpha * esrgan

# Not working!
def func1(psnr, esrgan):
    old_max = min(torch.max(psnr), torch.max(esrgan))
    sq = torch.mul(psnr, esrgan)
    return torch.div(sq, old_max)

def func2(psnr, esrgan):
    alpha = .8 if torch.sum(psnr) > psnr.nelement() else .2
    return (1 - alpha) * psnr + alpha * esrgan

def func3(psnr, esrgan):
    alpha = .8 if torch.sum(esrgan) > torch.sum(psnr) else .2
    return (1 - alpha) * psnr + alpha * esrgan

paths = ('models/' + x for x in ('interp_5.pth','interp_product.pth', 'interp_2_8_psn.pth', 'interp_2_8_cmp.pth'))
funcs = (func0, func1, func2, func3)

for path, func in zip(paths, funcs):
    net_interp = OrderedDict()
    for k, v_PSNR in net_PSNR.items():
        v_ESRGAN = net_ESRGAN[k]
        net_interp[k] = func(v_PSNR, v_ESRGAN)

    torch.save(net_interp, path)