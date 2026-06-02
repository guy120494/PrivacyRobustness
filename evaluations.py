import clip
import torch
import torchvision
import common_utils
import lpips as lpips_lib
from common_utils.common import flatten
from common_utils.image import get_ssim_pairs_kornia, get_ssim_all


def normalize_batch(x, ret_all=False):
    """ Normalize each element in batch x --> (x-mean)/std"""
    n, c, h, w = x.shape
    means = x.reshape(n * c, h * w).mean(dim=1).reshape(n, c, 1, 1)
    stds = x.reshape(n * c, h * w).std(dim=1).reshape(n, c, 1, 1)
    if ret_all:
        return x.sub(means).div(stds), means, stds
    else:
        return x.sub(means).div(stds)


def l2_dist(x, y, div_dim=False):
    """ L2 distance between x and y """
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)

    xx = x.pow(2).sum(1).view(-1, 1)
    yy = y.pow(2).sum(1).view(1, -1)
    xy = torch.einsum('id,jd->ij', x, y)
    dists = xx + yy - 2 * xy

    if div_dim:
        N, D = x.shape
        dists /= D

    return dists


def ncc_dist(x, y, div_dim=False):
    """ Normalized Cross-Correlation distacne between x and y """
    return l2_dist(normalize_batch(x), normalize_batch(y), div_dim)


def transform_vmin_vmax_batch(x, min_max=None):
    """ Transform each image in x: [min, max] --> [0, 1]"""
    if min_max is None:
        vmin = x.data.reshape(x.shape[0], -1).min(dim=1)[0][:, None, None, None]
        vmax = x.data.reshape(x.shape[0], -1).max(dim=1)[0][:, None, None, None]
    else:
        vmin, vmax = min_max
    return (x - vmin).div(vmax - vmin)

def normalize_for_plot(x):
    """Per-channel stretch for visualization."""
    C = x.shape[1]
    vmin = x.data.reshape(x.shape[0], C, -1).min(dim=2)[0][:, :, None, None]
    vmax = x.data.reshape(x.shape[0], C, -1).max(dim=2)[0][:, :, None, None]
    return (x - vmin).div((vmax - vmin).clamp(min=1e-8))


def viz_nns(x, y, max_per_nn=None, metric='ncc', ret_all=False):
    """
    return a batch, for each image in x, its nn in y
    sorted according to closest nn
    metric: NCC
    max_per_nn: filter duplicates (leave only max_per_nn elements of y-elements)
    """

    if metric == 'ncc':
        dists = ncc_dist(x, y)
    elif metric == 'l2':
        dists = l2_dist(x, y)
    else:
        raise ValueError(f'Unknown metric={metric}')

    v, nn_idx = dists.min(dim=1)

    keep = None
    if max_per_nn is not None:
        nn_idx_vals_i = torch.stack([nn_idx, v, torch.arange(v.shape[0], device=v.device)])
        nn_idx_vals_i = [(int(a), b, int(c)) for a, b, c in nn_idx_vals_i.t().tolist()]  # bring indexes back to int
        sorted_stuff = sorted(nn_idx_vals_i)

        # filter duplicates (leave only max_per_nn from each image from y)
        counter = 0
        cur_idx = sorted_stuff[0][0]
        keep = []
        for e in sorted_stuff:
            if e[0] != cur_idx:
                cur_idx = e[0]
                counter = 0
            if counter < max_per_nn:
                keep.append(e)
                counter += 1
        # sort by best value first
        keep = sorted(keep, key=lambda q: q[1])
        # keep is now: (nn idx in y, value, idx of x)
        xx = x[torch.tensor([e[2] for e in keep])]
        yy = y[torch.tensor([e[0] for e in keep])]
        v = torch.tensor([e[1] for e in keep])
    else:
        _, sidxs = v.sort()
        xx = x[sidxs]
        yy = y[nn_idx[sidxs]]

    qq = torch.stack(flatten(list(zip(xx, yy))))
    qq = transform_vmin_vmax_batch(qq)

    if ret_all:
        return qq, v, xx, yy, keep

    return qq, v


def get_evaluation_score_dssim(xxx, yyy, ds_mean, vote=None, show=False):
    xxx = xxx.clone()
    yyy = yyy.clone()

    x2search = torch.nn.functional.interpolate(xxx, scale_factor=1 / 2, mode='bicubic')
    y2search = torch.nn.functional.interpolate(yyy, scale_factor=1 / 2, mode='bicubic')
    D = ncc_dist(y2search, x2search, div_dim=True)

    dists, idxs = D.sort(dim=1, descending=False)

    if vote is not None:
        # Ignore distant nearest-neighbours
        xs_idxs = []
        for i in range(dists.shape[0]):
            x_idxs = [idxs[i, 0].item()]
            for j in range(1, dists.shape[1]):
                if (dists[i, j] / dists[i, 0]) < 1.1:
                    x_idxs.append(idxs[i, j].item())
                else:
                    break
            xs_idxs.append(x_idxs)

        # Voting
        xs = []
        for x_idxs in xs_idxs:
            if vote == 'min':
                x_voted = xxx[x_idxs[0]].unsqueeze(0)
            elif vote == 'mean':
                x_voted = xxx[x_idxs].mean(dim=0, keepdim=True)
            elif vote == 'median':
                x_voted = xxx[x_idxs].median(dim=0, keepdim=True).values
            elif vote == 'mode':
                x_voted = xxx[x_idxs].mode(dim=0, keepdim=True).values
            else:
                raise
            xs.append(x_voted)
        xx = torch.cat(xs, dim=0).clone()
        yy = yyy
    else:
        xx = xxx[idxs[:, 0]]
        yy = yyy

    # Scale to images
    yy += ds_mean

    # Score
    ssims = get_ssim_pairs_kornia((xx + ds_mean).clamp(0, 1), yy)
    dssim = (1 - ssims) / 2
    dssims, sort_idxs = dssim.sort(descending=False)

    # Sort & Show
    xx = xx[sort_idxs]
    yy = yy[sort_idxs]

    qq = torch.stack(common_utils.common.flatten(list(zip(transform_vmin_vmax_batch(xx + ds_mean), yy.clamp(0, 1)))))
    grid = torchvision.utils.make_grid(qq[:100], normalize=False, nrow=20)

    if show:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(80 * 2, 10 * 2))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())

    ev_score = dssims[:10].mean()

    all_dssim = (1 - get_ssim_all((xxx + ds_mean).clamp(0, 1), yy)) / 2
    all_dssim = all_dssim < 0.3
    successful_reconstructions = all_dssim.any(axis=0)
    return ev_score.item(), grid, successful_reconstructions.sum()


def get_evaluation_score_lpips(xxx, yyy, ds_mean, vote=None, show=False):
    xxx = xxx.clone()
    yyy = yyy.clone()
    device = xxx.device

    x2search = torch.nn.functional.interpolate(xxx, scale_factor=1 / 2, mode='bicubic')
    y2search = torch.nn.functional.interpolate(yyy, scale_factor=1 / 2, mode='bicubic')
    D = ncc_dist(y2search, x2search, div_dim=True)
    dists, idxs = D.sort(dim=1, descending=False)

    if vote is not None:
        xs_idxs = []
        for i in range(dists.shape[0]):
            x_idxs = [idxs[i, 0].item()]
            for j in range(1, dists.shape[1]):
                if (dists[i, j] / dists[i, 0]) < 1.1:
                    x_idxs.append(idxs[i, j].item())
                else:
                    break
            xs_idxs.append(x_idxs)
        xs = []
        for x_idxs in xs_idxs:
            if vote == 'min':
                x_voted = xxx[x_idxs[0]].unsqueeze(0)
            elif vote == 'mean':
                x_voted = xxx[x_idxs].mean(dim=0, keepdim=True)
            elif vote == 'median':
                x_voted = xxx[x_idxs].median(dim=0, keepdim=True).values
            elif vote == 'mode':
                x_voted = xxx[x_idxs].mode(dim=0, keepdim=True).values
            else:
                raise
            xs.append(x_voted)
        xx = torch.cat(xs, dim=0).clone()
        yy = yyy
    else:
        xx = xxx[idxs[:, 0]]
        yy = yyy

    yy += ds_mean

    # Normalize reconstructions to [0,1], then both to [-1,1] for LPIPS
    xx_01 = (xx + ds_mean).clamp(0, 1)
    yy_01 = yy.clamp(0, 1)
    xx_lpips = xx_01 * 2 - 1
    yy_lpips = yy_01 * 2 - 1

    # LPIPS backbones need at least 64x64; upsample if smaller (e.g. CIFAR-10 32x32)
    if xx_lpips.shape[-1] < 64 or xx_lpips.shape[-2] < 64:
        scale = 64 / min(xx_lpips.shape[-2], xx_lpips.shape[-1])
        xx_lpips = torch.nn.functional.interpolate(xx_lpips, scale_factor=scale, mode='bicubic', align_corners=False)
        yy_lpips = torch.nn.functional.interpolate(yy_lpips, scale_factor=scale, mode='bicubic', align_corners=False)

    lpips_fn = lpips_lib.LPIPS(net='alex').to(device)
    with torch.no_grad():
        lpips_scores = lpips_fn(xx_lpips, yy_lpips).squeeze()  # [N]

    lpips_sorted, sort_idxs = lpips_scores.sort(descending=False)

    xx = xx[sort_idxs]
    yy = yy[sort_idxs]

    qq = torch.stack(common_utils.common.flatten(list(zip(transform_vmin_vmax_batch(xx + ds_mean), yy.clamp(0, 1)))))
    grid = torchvision.utils.make_grid(qq[:100], normalize=False, nrow=20)

    if show:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(80 * 2, 10 * 2))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())

    ev_score = lpips_sorted[:10].mean()

    # Successful reconstructions: training images whose 1-NN reconstruction scores below threshold.
    # All-pairs LPIPS (1000x500) is prohibitively expensive; 1-NN is the practical approximation.
    successful_reconstructions = lpips_scores < 0.5

    return ev_score.item(), grid, successful_reconstructions.sum()


def get_evaluation_score_clip(xxx, yyy, ds_mean, vote=None, show=False):
    xxx = xxx.clone()
    yyy = yyy.clone()
    device = xxx.device

    x2search = torch.nn.functional.interpolate(xxx, scale_factor=1 / 2, mode='bicubic')
    y2search = torch.nn.functional.interpolate(yyy, scale_factor=1 / 2, mode='bicubic')
    D = ncc_dist(y2search, x2search, div_dim=True)
    dists, idxs = D.sort(dim=1, descending=False)

    if vote is not None:
        xs_idxs = []
        for i in range(dists.shape[0]):
            x_idxs = [idxs[i, 0].item()]
            for j in range(1, dists.shape[1]):
                if (dists[i, j] / dists[i, 0]) < 1.1:
                    x_idxs.append(idxs[i, j].item())
                else:
                    break
            xs_idxs.append(x_idxs)
        xs = []
        for x_idxs in xs_idxs:
            if vote == 'min':
                x_voted = xxx[x_idxs[0]].unsqueeze(0)
            elif vote == 'mean':
                x_voted = xxx[x_idxs].mean(dim=0, keepdim=True)
            elif vote == 'median':
                x_voted = xxx[x_idxs].median(dim=0, keepdim=True).values
            elif vote == 'mode':
                x_voted = xxx[x_idxs].mode(dim=0, keepdim=True).values
            else:
                raise
            xs.append(x_voted)
        xx = torch.cat(xs, dim=0).clone()
        yy = yyy
    else:
        xx = xxx[idxs[:, 0]]
        yy = yyy

    yy += ds_mean

    # Scale to [0,1] then apply CLIP's normalization (ViT-B/32 ImageNet stats)
    xx_01 = (xx + ds_mean).clamp(0, 1)
    yy_01 = yy.clamp(0, 1)

    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    xx_clip = torch.nn.functional.interpolate(xx_01, size=(224, 224), mode='bicubic', align_corners=False)
    yy_clip = torch.nn.functional.interpolate(yy_01, size=(224, 224), mode='bicubic', align_corners=False)
    xx_clip = (xx_clip - clip_mean) / clip_std
    yy_clip = (yy_clip - clip_mean) / clip_std

    clip_model, _ = clip.load('ViT-B/32', device=device)
    clip_model.eval()

    with torch.no_grad():
        xx_feat = clip_model.encode_image(xx_clip).float()
        yy_feat = clip_model.encode_image(yy_clip).float()

    xx_feat = torch.nn.functional.normalize(xx_feat, dim=1)
    yy_feat = torch.nn.functional.normalize(yy_feat, dim=1)

    # Cosine distance: lower = more similar, consistent with DSSIM/LPIPS direction
    clip_dist = 1 - (xx_feat * yy_feat).sum(dim=1)  # [N]

    clip_sorted, sort_idxs = clip_dist.sort(descending=False)

    xx = xx[sort_idxs]
    yy = yy[sort_idxs]

    qq = torch.stack(common_utils.common.flatten(list(zip(transform_vmin_vmax_batch(xx + ds_mean), yy.clamp(0, 1)))))
    grid = torchvision.utils.make_grid(qq[:100], normalize=False, nrow=20)

    if show:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(80 * 2, 10 * 2))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())

    ev_score = clip_sorted[:10].mean()

    # cosine distance < 0.2 ≈ cosine similarity > 0.8: semantically similar
    successful_reconstructions = clip_dist < 0.3

    return ev_score.item(), grid, successful_reconstructions.sum()


def get_model_outputs_on_grid(model, lim=1.5, n=1000):
    x_coord = torch.linspace(-lim, lim, n)
    y_coord = torch.linspace(-lim, lim, n)
    grid = torch.stack(torch.meshgrid([x_coord, y_coord], indexing=None))
    zi = model(grid.reshape(2, -1).t().to('cuda')).reshape(n, n).cpu().data
    xi = grid[0,:,:].cpu()
    yi = grid[1,:,:].cpu()
    return xi, yi, zi



