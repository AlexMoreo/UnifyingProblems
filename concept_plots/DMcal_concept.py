import numpy as np
import matplotlib.pyplot as plt
from util import impose_monotonicity, smooth

np.random.seed(42)
n_samples = 1000
nbins = 20
param = 0.3
figsize = (4, 4)
file_ext = '.pdf'

bins = np.linspace(0,1,nbins+1)
bin_centers = (bins[:-1] + bins[1:]) / 2
width = bins[1] - bins[0]

# simulating distributions
def normal_trim(loc, scale, size):
    points = np.random.normal(loc=loc, scale=scale, size=size)
    points = points[np.logical_and(points>=0, points<=1)]
    return points

mu_neg_1, std_neg_1 = 0, 0.1
mu_neg_2, std_neg_2 = 0.4, 0.2
mu_pos, std_pos = 0.8, 0.1

neg_posteriors_1 = normal_trim(loc=mu_neg_1, scale=std_neg_1, size=n_samples)
neg_posteriors_2 = normal_trim(loc=mu_neg_2, scale=std_neg_2, size=n_samples)
neg_posteriors = np.concatenate([neg_posteriors_1,neg_posteriors_2])
pos_posteriors = normal_trim(loc=mu_pos, scale=std_pos, size=n_samples)

test_neg_posteriors_1 = normal_trim(loc=mu_neg_1, scale=std_neg_1, size=n_samples)
test_neg_posteriors_2 = normal_trim(loc=mu_neg_2, scale=std_neg_2-0.01, size=n_samples)
test_neg_posteriors = np.concatenate([test_neg_posteriors_1,test_neg_posteriors_2])
test_pos_posteriors = normal_trim(loc=mu_pos+0.02, scale=std_pos, size=n_samples)

# histograms (actually, binned density plots)
neg_hist, _ = np.histogram(neg_posteriors, bins=bins, density=True)
pos_hist, _ = np.histogram(pos_posteriors, bins=bins, density=True)

mixture_neg = (1 - param) * neg_hist
mixture_pos = param * pos_hist

test_neg_hist, _ = np.histogram(test_neg_posteriors, bins=bins, density=True)
test_pos_hist, _ = np.histogram(test_pos_posteriors, bins=bins, density=True)

test_hist_mix = (1-param)*test_neg_hist + param*test_pos_hist

# plot 0: bins proportions with calibration map
total_per_bin = mixture_neg + mixture_pos
normalized_neg = np.divide(mixture_neg, total_per_bin, out=np.zeros_like(mixture_neg), where=total_per_bin!=0)
normalized_pos = np.divide(mixture_pos, total_per_bin, out=np.zeros_like(mixture_pos), where=total_per_bin!=0)

plt.figure(figsize=figsize)
plt.bar(bin_centers, normalized_pos, width=width, label='Positive proportion', color='tab:orange', edgecolor='black', alpha=0.6)
plt.bar(bin_centers, normalized_neg, width=width, bottom=normalized_pos, label='Negative proportion', color='tab:blue', edgecolor='black', alpha=0.6)

# interpolation line (the effective calibration map)
separation_points = normalized_pos
interpolation_color = '#B22222'
plt.plot(bin_centers, separation_points, linewidth=3, marker='o', markersize=6, color='black')
plt.plot(bin_centers, separation_points, linewidth=1.5, label='Calibration map', marker='o', markersize=5, color=interpolation_color)


plt.xlabel('Posterior probability')
plt.ylabel('Class proportion per bin')
plt.title('Raw Calibration Map')
plt.xlim(0, 1.0)
plt.ylim(0, 1.0)
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('./calib_map'+file_ext)

# plot 0: bins proportions with calibration map after monotonicity+smoothing
normalized_pos = impose_monotonicity(normalized_pos)
normalized_pos = smooth(normalized_pos)
normalized_neg = 1-normalized_pos

plt.figure(figsize=figsize)
plt.bar(bin_centers, normalized_pos, width=width, label='Positive proportion', color='tab:orange', edgecolor='black', alpha=0.6)
plt.bar(bin_centers, normalized_neg, width=width, bottom=normalized_pos, label='Negative proportion', color='tab:blue', edgecolor='black', alpha=0.6)

# interpolation line (the effective calibration map)
separation_points = normalized_pos
interpolation_color = '#B22222'
plt.plot(bin_centers, separation_points, linewidth=3, marker='o', markersize=6, color='black')
plt.plot(bin_centers, separation_points, linewidth=1.5, label='Calibration map', marker='o', markersize=5, color=interpolation_color)

plt.xlabel('Posterior probability')
plt.ylabel('Class proportion per bin')
plt.title('Corrected Calibration Map')
plt.xlim(0, 1.0)
plt.ylim(0, 1.0)
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('./calib_map_corrected'+file_ext)


# plot 1: mixture model
plt.figure(figsize=figsize)
plt.bar(bin_centers, mixture_pos, width=width, label='Positive examples', color='tab:orange', edgecolor='black', alpha=0.6)
plt.bar(bin_centers, mixture_neg, width=width, bottom=mixture_pos, label='Negative examples', color='tab:blue', edgecolor='black', alpha=0.6)

plt.xlabel('Posterior probability')
plt.ylabel('Density')
plt.title(f'Mixture Model')
plt.xlim(0, 1.0)
# plt.ylim(0, 1.0)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('./mixture'+file_ext)

# plot 2: test density
plt.figure(figsize=figsize)
plt.bar(bin_centers, test_hist_mix, width=width, label='Test examples', color='tab:green', edgecolor='black', alpha=0.6)

plt.xlabel('Posterior probability')
plt.ylabel('Density')
plt.title(f'Test Distribution')
plt.xlim(0, 1.0)
# plt.ylim(0, 1.0)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('./test'+file_ext)

# plot 3: class-conditional densities
plt.figure(figsize=figsize)
plt.hist(neg_posteriors, bins=bins, alpha=0.6, label='Negative examples', color='tab:blue', edgecolor='black', density=True)
plt.hist(pos_posteriors, bins=bins, alpha=0.6, label='Positive examples', color='tab:orange', edgecolor='black', density=True)

plt.xlabel('Posterior probability')
plt.ylabel('Density')
plt.title('Class-conditional Distributions')
plt.xlim(0, 1.0)
# plt.ylim(0, 1.0)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('./class_conditional'+file_ext)
