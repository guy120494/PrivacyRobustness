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
        k = min(5, idxs.shape[1])
        N_train = yyy.shape[0]
        xx_candidates = xxx[idxs[:, :k]]  # [N_train, k, C, H, W]
        yy = yyy

    # Scale to images
    yy += ds_mean

    # Score: for each training image pick the best-scoring reconstruction among top-k NCC candidates
    if vote is not None:
        ssims = get_ssim_pairs_kornia((xx + ds_mean).clamp(0, 1), yy)
        dssim = (1 - ssims) / 2
    else:
        C, H, W = xxx.shape[1:]
        xx_flat = xx_candidates.reshape(N_train * k, C, H, W)
        yy_rep = yy.unsqueeze(1).expand(-1, k, -1, -1, -1).reshape(N_train * k, C, H, W)
        ssims_all = get_ssim_pairs_kornia((xx_flat + ds_mean).clamp(0, 1), yy_rep)
        best_ssim, best_k_idx = ssims_all.view(N_train, k).max(dim=1)
        xx = xx_candidates[torch.arange(N_train, device=xxx.device), best_k_idx]
        dssim = (1 - best_ssim) / 2
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
        k = min(5, idxs.shape[1])
        N_train = yyy.shape[0]
        xx_candidates = xxx[idxs[:, :k]]  # [N_train, k, C, H, W]
        yy = yyy

    yy += ds_mean

    lpips_fn = lpips_lib.LPIPS(net='alex').to(device)

    def _to_lpips(t):
        t = t * 2 - 1
        if t.shape[-1] < 64 or t.shape[-2] < 64:
            scale = 64 / min(t.shape[-2], t.shape[-1])
            t = torch.nn.functional.interpolate(t, scale_factor=scale, mode='bicubic', align_corners=False)
        return t

    if vote is not None:
        with torch.no_grad():
            lpips_scores = lpips_fn(_to_lpips((xx + ds_mean).clamp(0, 1)),
                                    _to_lpips(yy.clamp(0, 1))).squeeze()
    else:
        C, H, W = xxx.shape[1:]
        xx_flat = xx_candidates.reshape(N_train * k, C, H, W)
        yy_rep = yy.unsqueeze(1).expand(-1, k, -1, -1, -1).reshape(N_train * k, C, H, W)
        with torch.no_grad():
            scores_flat = lpips_fn(_to_lpips((xx_flat + ds_mean).clamp(0, 1)),
                                   _to_lpips(yy_rep.clamp(0, 1))).squeeze()
        best_lpips, best_k_idx = scores_flat.view(N_train, k).min(dim=1)
        xx = xx_candidates[torch.arange(N_train, device=device), best_k_idx]
        lpips_scores = best_lpips

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
        k = min(5, idxs.shape[1])
        N_train = yyy.shape[0]
        xx_candidates = xxx[idxs[:, :k]]  # [N_train, k, C, H, W]
        yy = yyy

    yy += ds_mean

    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    def _to_clip(t):
        t = torch.nn.functional.interpolate(t, size=(224, 224), mode='bicubic', align_corners=False)
        return (t - clip_mean) / clip_std

    clip_model, _ = clip.load('ViT-B/32', device=device)
    clip_model.eval()

    def _encode(imgs, batch_size=256):
        feats = []
        for i in range(0, imgs.shape[0], batch_size):
            with torch.no_grad():
                feats.append(clip_model.encode_image(imgs[i:i + batch_size]).float())
        return torch.nn.functional.normalize(torch.cat(feats, dim=0), dim=1)

    if vote is not None:
        xx_feat = _encode(_to_clip((xx + ds_mean).clamp(0, 1)))
        yy_feat = _encode(_to_clip(yy.clamp(0, 1)))
        clip_dist = 1 - (xx_feat * yy_feat).sum(dim=1)
    else:
        C, H, W = xxx.shape[1:]
        xx_flat = xx_candidates.reshape(N_train * k, C, H, W)
        yy_rep = yy.unsqueeze(1).expand(-1, k, -1, -1, -1).reshape(N_train * k, C, H, W)
        xx_feat = _encode(_to_clip((xx_flat + ds_mean).clamp(0, 1)))
        yy_feat = _encode(_to_clip(yy_rep.clamp(0, 1)))
        dist_flat = 1 - (xx_feat * yy_feat).sum(dim=1)
        best_dist, best_k_idx = dist_flat.view(N_train, k).min(dim=1)
        xx = xx_candidates[torch.arange(N_train, device=device), best_k_idx]
        clip_dist = best_dist

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



