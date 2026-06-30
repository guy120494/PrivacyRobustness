import numpy as np
import torch
import re
from matplotlib import pyplot as plt

THRESHOLDS = {'dssim': 0.3, 'lpips': 0.5, 'clip': 0.7}
METRIC_LABELS = {'dssim': 'DSSIM < 0.30', 'lpips': 'LPIPS < 0.50', 'clip': 'CLIP sim > 0.70'}
HIGHER_IS_BETTER = {'clip'}

def _covered_counts(best: np.ndarray, thresholds: np.ndarray, higher_is_better: bool) -> np.ndarray:
    sorted_best = np.sort(best)
    if higher_is_better:
        return len(sorted_best) - np.searchsorted(sorted_best, thresholds, side='left')
    return np.searchsorted(sorted_best, thresholds, side='right')


def _best_per_train(matrix: torch.Tensor, higher_is_better: bool) -> np.ndarray:
    fn = matrix.max if higher_is_better else matrix.min
    return fn(dim=1).values.cpu().numpy()


def plot_coverage_curve(entries: list[tuple[torch.Tensor, str]], metric_name: str,
                        n_thresholds: int = 1000, ax=None):
    """
    Plot coverage curves for one or more score matrices on the same axes.

    entries: list of (matrix, name) where name is a path or string from which
             a 'radius_X' tag is extracted for the legend label.
    For each entry and each threshold t in [0, 1], plots the number of training
    samples that have at least one candidate with metric score < t.

    Returns (fig, ax); fig is None if an existing ax was passed in.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    start = 0.1
    end = 0.4
    thresholds = np.linspace(start, end, n_thresholds)
    N_train = 0
    higher = metric_name.lower() in HIGHER_IS_BETTER

    for matrix, name in entries:
        best = _best_per_train(matrix, higher)
        N_train = max(N_train, len(best))
        counts = _covered_counts(best, thresholds, higher)

        match = re.search(r'radius_[\d.]+', str(name))
        label = match.group().replace('_', ' ') if match else str(name)

        ax.plot(thresholds, counts, linewidth=2, label=label)

    thresh_val = THRESHOLDS[metric_name.lower()]
    ax.axvline(thresh_val, color='red', linestyle='--', alpha=0.7,
               label=f'default threshold ({thresh_val})')
    direction = 'above' if higher else 'below'
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel(f'# training samples with >=1 candidate {direction} threshold', fontsize=12)
    ax.set_title(f'Coverage curve — {metric_name.upper()}', fontsize=13)
    ax.set_xlim(start, end)
    ax.set_ylim(0, N_train)
    if len(entries) > 1:
        ax.legend()
    ax.grid(True, alpha=0.3)

    if fig is not None:
        fig.tight_layout()

    return fig, ax


def plot_coverage_difference(entries: list[tuple[torch.Tensor, str]], metric_name: str,
                             n_thresholds: int = 1000):
    """
    Plot absolute and percentage coverage differences relative to the first entry (baseline).

    Creates a figure with two side-by-side subplots:
      Left:  counts_i(t) - counts_0(t)  (absolute difference)
      Right: (counts_i(t) - counts_0(t)) / counts_0(t) * 100  (% difference)

    Returns (fig, (ax_abs, ax_pct)).
    """
    fig, (ax_abs, ax_pct) = plt.subplots(1, 2, figsize=(14, 5))

    start = 0.1
    end = 0.4
    thresholds = np.linspace(start, end, n_thresholds)

    higher = metric_name.lower() in HIGHER_IS_BETTER
    baseline_matrix, baseline_name = entries[0]
    baseline_best = _best_per_train(baseline_matrix, higher)
    baseline_counts = _covered_counts(baseline_best, thresholds, higher).astype(float)
    baseline_match = re.search(r'radius_[\d.]+', str(baseline_name))
    baseline_label = baseline_match.group().replace('_', ' ') if baseline_match else str(baseline_name)

    for matrix, name in entries[1:]:
        best = _best_per_train(matrix, higher)
        counts = _covered_counts(best, thresholds, higher).astype(float)
        diff = counts - baseline_counts

        with np.errstate(invalid='ignore', divide='ignore'):
            pct = np.where(baseline_counts > 0, diff / baseline_counts * 100, np.nan)

        match = re.search(r'radius_[\d.]+', str(name))
        label = match.group().replace('_', ' ') if match else str(name)
        line_label = f'{label} - {baseline_label}'

        ax_abs.plot(thresholds, diff, linewidth=2, label=line_label)
        ax_pct.plot(thresholds, pct, linewidth=2, label=line_label)

    for ax, ylabel, title in [
        (ax_abs, f'Coverage difference vs {baseline_label}', f'Absolute difference — {metric_name.upper()}'),
        (ax_pct, f'Coverage % difference vs {baseline_label}', f'Percentage difference — {metric_name.upper()}'),
    ]:
        ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.set_xlim(start, end)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, (ax_abs, ax_pct)


if __name__ == '__main__':
    matrix_paths = [r"C:\Users\admin\Downloads\eval_lpips_matrix_all_radius_0.pth",
                    r"C:\Users\admin\Downloads\eval_lpips_matrix_all_radius_0.1.pth",
                    r"C:\Users\admin\Downloads\eval_lpips_matrix_all_radius_0.5.pth",
                   r"C:\Users\admin\Downloads\eval_lpips_matrix_all_radius_1.pth"]

    # matrix_paths = [r"C:\Users\admin\Downloads\eval_lpips_matrix_all_radius_0.pth",
    #                 r"C:\Users\admin\Downloads\eval_lpips_matrix_all_radius_1.pth"]

    metric_match = re.search(r'eval_(dssim|lpips|clip)_', matrix_paths[0])
    if metric_match is None:
        raise ValueError(f"Could not deduce metric name from path: {matrix_paths[0]}")
    metric_name = metric_match.group(1)

    entities = []
    for p in matrix_paths:
        entities.append((torch.load(p), p))
    f1, _ = plot_coverage_curve(entities, metric_name)
    f2, _ = plot_coverage_difference(entities, metric_name)
    plt.show()