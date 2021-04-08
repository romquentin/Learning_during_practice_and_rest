"""Measure online and offline learning on the ASRT dataset
(Dataset 2 in the corresponding manuscript)"""

# Authors: Romain Quentin <rom.quentin@gmail.com>
# License: BSD (3-clause)

import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from scipy.stats import ttest_1samp
from scipy.stats import binned_statistic
from base import anova_onoff
import warnings
from scipy.stats import linregress

plt.style.use('seaborn-whitegrid')
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['xtick.major.pad'] = 2
plt.rcParams['ytick.major.pad'] = 2
plt.rcParams['xtick.minor.pad'] = 2
plt.rcParams['ytick.minor.pad'] = 2
plt.rcParams['axes.labelpad'] = 0.1
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['text.color'] = '0'
plt.rcParams['xtick.color'] = '0'
plt.rcParams['ytick.color'] = '0'
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['font.sans-serif'] = 'Arial'
colors = ['#0173b2', '#de8f05', '#029e73']
colors_barplot = [['#cc78bc', '#949494'], ['#cc78bc', '#949494'],
                  ['#cc78bc', '#949494']]
# Indicate p_values on the figure
with_p_values = False

path_data = './data'

# Define blocks (and columns for anova)
blocks = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10',
          'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20',
          'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 'b28', 'b29', 'b30',
          'b31', 'b32', 'b33', 'b34', 'b35', 'b36', 'b37', 'b38', 'b39', 'b40',
          'b41', 'b42', 'b43', 'b44', 'b45']
columns = blocks.copy()
columns.insert(0, 'sub')
# Read behavioral file with all participants
behav = pd.read_spss(op.join(path_data, 'Implicit_ASRT_data.sav'))
subjects = np.unique(behav.Subject)

# Drop first 7 trials of each block (5 trials + 2 that start the first triplet)
behav_clean = behav[behav.TT != 'X']
# Drop incorrect trial (incorrect response)
behav_clean = behav_clean[behav_clean.firstACC == 1]
# Plot all reaction time per block
behav_clean.groupby('Block').mean().finalRT.plot()
plt.savefig(op.join(path_data, 'behavior_plots', 'mean_rt_per_block.png'))
plt.close()

bins = 5  # There are 10 sequences (of 8 trials) per block
# Average within blocks
all_RT_block = list()  # all trials type
high_RT_block = list()  # high-triplets trials
low_RT_block = list()  # low-triplets trials
pat_RT_block = list()  # pattern trials
ran_RT_block = list()  # random trials
ranhigh_RT_block = list()  # random-high trials
# Average within bins
all_RT_bin = list()
high_RT_bin = list()
low_RT_bin = list()
pat_RT_bin = list()
ran_RT_bin = list()
ranhigh_RT_bin = list()
for subject in tqdm(subjects):
    behav_sub = behav[behav.Subject == subject]
    # Drop incorrect trial (incorrect response)
    behav_sub_clean = behav_sub[behav_sub.firstACC == 1]
    # Drop repetitions or trill
    behav_sub_clean = behav_sub_clean[np.array(behav_sub_clean.TT == 'H') | np.array(behav_sub_clean.TT == 'L')]
    # Calculate mean reaction and std on all trials for each participant
    meanRT = behav_sub_clean.finalRT.mean()
    stdRT = behav_sub_clean.finalRT.std()
    # Average within blocks
    all_block = list()
    high_block = list()
    low_block = list()
    pat_block = list()
    ran_block = list()
    ranhigh_block = list()
    # Average within bins
    all_bin = list()
    high_bin = list()
    low_bin = list()
    pat_bin = list()
    ran_bin = list()
    ranhigh_bin = list()
    for block in np.unique(behav.Block):
        behav_sub_block = behav_sub[behav_sub.Block == block]
        # Drop first 5 practice trials of each block
        behav_sub_block = behav_sub_block[behav_sub_block.TrialType != 'Prac']
        if len(behav_sub_block) != 80:
            warnings.warn("Block without exactly 80 trials")
        # Drop incorrect trial (incorrect response)
        behav_sub_block_clean = behav_sub_block[behav_sub_block.firstACC == 1]
        # Drop repetitions or trill
        behav_sub_block_clean = behav_sub_block_clean[np.array(behav_sub_block_clean.TT == 'H') | np.array(behav_sub_block_clean.TT == 'L')]
        # Drop extreme value
        behav_sub_block_clean = behav_sub_block_clean[behav_sub_block_clean.finalRT <= meanRT+3*stdRT]
        behav_sub_block_clean = behav_sub_block_clean[behav_sub_block_clean.finalRT >= meanRT-3*stdRT]
        all_block.append(behav_sub_block_clean.finalRT.mean())
        high_block.append(behav_sub_block_clean[behav_sub_block_clean.TT == 'H'].finalRT.mean())
        low_block.append(behav_sub_block_clean[behav_sub_block_clean.TT == 'L'].finalRT.mean())
        pat_block.append(behav_sub_block_clean[behav_sub_block_clean.TrialType == 'P'].finalRT.mean())
        ran_block.append(behav_sub_block_clean[behav_sub_block_clean.TrialType == 'R'].finalRT.mean())
        ranhigh_block.append(behav_sub_block_clean[np.array(behav_sub_block_clean.TT == 'H') & np.array(behav_sub_block_clean.TrialType == 'R')].finalRT.mean())
        binnumber = binned_statistic(np.arange(len(behav_sub_block)),
                                     np.array(behav_sub_block.finalRT),
                                     bins=bins)[2]
        all = list()
        high = list()
        low = list()
        pat = list()
        ran = list()
        ranhigh = list()
        for bin in np.arange(bins):
            bin_values = behav_sub_block[binnumber == bin + 1]
            # Drop incorrect trial (incorrect response)
            bin_values = bin_values[bin_values.firstACC == 1]
            # Drop repetitions or trill
            bin_values = bin_values[np.array(bin_values.TT == 'H') | np.array(bin_values.TT == 'L')]
            # Drop extreme value
            bin_values = bin_values[bin_values.finalRT <= meanRT+3*stdRT]
            bin_values = bin_values[bin_values.finalRT >= meanRT-3*stdRT]
            # Append mean RT in each bin
            all.append(bin_values.finalRT.mean())
            high.append(bin_values[bin_values.TT == 'H'].finalRT.mean())
            low.append(bin_values[bin_values.TT == 'L'].finalRT.mean())
            pat.append(bin_values[bin_values.TrialType == 'P'].finalRT.mean())
            ran.append(bin_values[bin_values.TrialType == 'R'].finalRT.mean())
            ranhigh.append(bin_values[np.array(bin_values.TT == 'H') & np.array(bin_values.TrialType == 'R')].finalRT.mean())
        # Append RT for each block
        all_bin.append(all)
        high_bin.append(high)
        low_bin.append(low)
        pat_bin.append(pat)
        ran_bin.append(ran)
        ranhigh_bin.append(ranhigh)
    # Append RT for each participant
    # Average within blocks
    all_RT_block.append(all_block)
    high_RT_block.append(high_block)
    low_RT_block.append(low_block)
    pat_RT_block.append(pat_block)
    ran_RT_block.append(ran_block)
    ranhigh_RT_block.append(ranhigh_block)
    # Average within bins
    all_RT_bin.append(all_bin)
    high_RT_bin.append(high_bin)
    low_RT_bin.append(low_bin)
    pat_RT_bin.append(pat_bin)
    ran_RT_bin.append(ran_bin)
    ranhigh_RT_bin.append(ranhigh_bin)
# Average within blocks
all_RT_block = np.array(all_RT_block)
high_RT_block = np.array(high_RT_block)
low_RT_block = np.array(low_RT_block)
pat_RT_block = np.array(pat_RT_block)
ran_RT_block = np.array(ran_RT_block)
ranhigh_RT_block = np.array(ranhigh_RT_block)
# Average within bins
all_RT_bin = np.array(all_RT_bin)
high_RT_bin = np.array(high_RT_bin)
low_RT_bin = np.array(low_RT_bin)
pat_RT_bin = np.array(pat_RT_bin)
ran_RT_bin = np.array(ran_RT_bin)
ranhigh_RT_bin = np.array(ranhigh_RT_bin)

# Learning by triplet difference
# Average within blocks
highvslow_block = high_RT_block - low_RT_block
patvslow_block = pat_RT_block - low_RT_block
patvsran_block = pat_RT_block - ran_RT_block
patvsranhigh_block = pat_RT_block - ranhigh_RT_block
ranhighvslow_block = ranhigh_RT_block - low_RT_block
# Average within bins
highvslow_bin = high_RT_bin - low_RT_bin
patvslow_bin = pat_RT_bin - low_RT_bin
patvsran_bin = pat_RT_bin - ran_RT_bin
patvsranhigh_bin = pat_RT_bin - ranhigh_RT_bin
ranhighvslow_bin = ranhigh_RT_bin - low_RT_bin

# Plot block reaction time averaged accross participant
for name, blocked, ylims, color in zip(['all', 'highvslow', 'patvsranhigh', 'ranhighvslow'],
                                       [all_RT_block, -highvslow_block, -patvsranhigh_block, -ranhighvslow_block],
                                       [[340, 430], [-10, 30], [-10, 10], [-10, 30]],
                                       [colors[0], colors[1], colors[2], colors[1]]):
    plt.figure(figsize=(4.8, 1.8), dpi=200)
    mean = np.array(np.nanmean(blocked, 0))
    std = np.array(np.std(blocked, 0))
    plt.plot(np.arange(45) + 1, mean, color=color)
    plt.fill_between(np.arange(45) + 1, mean - (std/np.sqrt(len(subjects))),
                     mean + (std/np.sqrt(len(subjects))), color=color,
                     alpha=0.2)
    plt.xticks(np.arange(1, 46, 3))
    plt.ylim(ylims)
    plt.xlabel('Blocks')
    if name == 'all':
        plt.ylabel('Average reaction time \nper block (ms)')
    else:
        plt.ylabel('Average of reaction time \ndifference per block (ms)')
    plt.tight_layout()
    plt.savefig(op.join(path_data, 'behavior_plots', '%s_blocked_rt_per_block_sess.png' % name))
    plt.close()

# Plot binned learning
params = zip(['all', 'highvslow', 'patvsranhigh', 'ranhighvslow'],
             [all_RT_bin, -highvslow_bin, -patvsranhigh_bin, -ranhighvslow_bin],
             [[340, 430], [-10, 40], [-10, 10], [-10, 40]],
             [colors[0], colors[1], colors[2], colors[1]],
             [(4.7, 1.8), (4.7, 1.8), (4.7, 1.8), (4.7, 1.8)])
for name, binned, ylims, color, figsize in params:
    a = 0
    b = bins
    ticks = list()
    plt.figure(figsize=(4.7, 1.8), dpi=400)
    for ii in np.arange(45):
        mean = np.nanmean(binned[:, ii, :], 0)
        std = np.nanstd(binned[:, ii, :], 0)
        plt.plot(np.arange(a, b), mean, color=color, lw=0.8)
        plt.fill_between(np.arange(a, b), mean - (std/np.sqrt(len(subjects))),
                         mean + (std/np.sqrt(len(subjects))), color=color,
                         alpha=0.1)
        ticks.append((a+b)/2 - 1)
        a = 1 + b
        b = a + bins
    plt.xticks(ticks[::5], np.hstack([1, np.arange(0, 46, 5)[1:]]) )
    # plot average per block
    mean = np.array(np.nanmean(binned, (0, 2)))
    plt.plot(ticks, mean, color='black', linewidth=0.4)
    plt.xlabel('Blocks')
    plt.ylim(ylims)
    if name == 'all':
        plt.ylabel('Reaction time (ms)')
    else:
        plt.ylabel('Reaction time difference (ms)')
    plt.tight_layout()
    plt.savefig(op.join(path_data, 'behavior_plots', '%s_binned_trials_rt_per_block.png' % name))
    plt.close()

# Plot online, offline and both learning
params = zip(['all', 'highvslow', 'patvsranhigh', 'ranhighvslow'],
             [all_RT_bin, -highvslow_bin, -patvsranhigh_bin, -ranhighvslow_bin],
             [all_RT_block, -highvslow_block, -patvsranhigh_block, -ranhighvslow_block],
             [colors_barplot[0], colors_barplot[1], colors_barplot[2], colors_barplot[1]],
             ['General skill learning (ms)', 'Statistical learning (ms)', 'Deterministic rule Learning (ms)', 'Statistical learning (ms)'],
             [(-80, 80), (-60, 60), (-50, 50), (-60, 60)],
             [(-200, 200), (-60, 60), (-50, 50), (-60, 60)])
for name, bin_average, block_average, color_barplot, ylabel, ylim_curve, ylim_violin in params:
    off_ = bin_average[:, 1:, 0] - bin_average[:, :-1, -1]
    on_ = bin_average[:, :, -1] - bin_average[:, :, 0]
    both_ = block_average[:, 1:] - block_average[:, :-1]
    # Plot online and offline curve
    plt.figure(figsize=(2.1, 1.95), dpi=400)
    times = np.arange(45) + 1
    # plot online
    mean, std = np.nanmean(on_, 0), np.nanstd(on_, 0)
    err = std/np.sqrt(len(subjects))
    plt.errorbar(times, mean, yerr=err, fmt='o', color=color_barplot[0],
                 ms=0.6, lw=0.3)
    # Fit linear regression
    regress_params = linregress(times, mean)
    fit = regress_params[1] + regress_params[0] * times
    plt.plot(times, fit, lw=1.3, color=color_barplot[0], label='Online')
    # plot offline
    times = np.arange(44) + 1.5
    mean, std = np.nanmean(off_, 0), np.nanstd(off_, 0)
    err = std/np.sqrt(len(subjects))
    plt.errorbar(times, mean, yerr=err, fmt='o', color=color_barplot[1],
                 ms=0.6, lw=0.3)
    # Fit linear regression
    regress_params = linregress(times, mean)
    fit = regress_params[1] + regress_params[0] * times
    plt.plot(times, fit, lw=1.3, color=color_barplot[1], label='Offline')
    plt.axhline(0, ls='--', color='k', linewidth=0.5)
    plt.legend()
    plt.gca().legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                     ncol=2, mode="expand", borderaxespad=0.)
    plt.xticks(np.hstack([1, np.arange(0, 46, 5)[1:]]))
    plt.xlabel('Blocks')
    plt.ylabel(ylabel)
    plt.ylim(ylim_curve)
    plt.tight_layout()
    plt.xticks([1, 10, 20, 30, 40])
    # Run anova
    stats = anova_onoff(on_, off_, subjects, columns)
    plt.savefig(op.join(path_data, 'behavior_plots', '%s_offonline_curve.png' % name))
    plt.close()

    # Plot swarmplot for all blocks
    plt.figure(figsize=(1.4, 1.67), dpi=400)
    mean_on_ = np.nanmean(on_, 1)
    mean_off_ = np.nanmean(off_, 1)
    mean_both_ = np.nanmean(both_, 1)
    t_on_, p_on_ = ttest_1samp(mean_on_, 0)
    t_off_, p_off_ = ttest_1samp(mean_off_, 0)
    t_both_, p_both_ = ttest_1samp(mean_both_, 0)
    to_plot_np = np.array([mean_on_, mean_off_]).T
    to_plot_df = pd.DataFrame(data=to_plot_np, index=subjects, columns=['online', 'offline'])
    # Reverse data to show learning positive if general skill
    if name == 'all':
        to_plot_df = -to_plot_df
    ax = sns.violinplot(data=to_plot_df, palette=color_barplot, inner='quartile', linewidth=0.5, bw=.4)
    ax.axhline(0, ls='--', color='k', linewidth=0.5)
    ax = sns.swarmplot(data=to_plot_df, palette=['1', '1'], size=0.6, alpha=1)
    plt.ylim(ylim_violin)
    plt.ylabel(ylabel)
    if with_p_values:
        for (text, xpos) in zip([[t_on_, p_on_],
                                 [t_off_, p_off_]], [0, 1]):
            if text[1] < 0.05:
                ax.text(xpos, 10, 't=%.1f, p<.05' % (text[0]), fontsize=7)
            else:
                ax.text(xpos, 10, 't=%.1f, p=%.2f' % (text[0], text[1]), fontsize=7)
            # ax.text(xpos, 10, 't=%.3f, p=%.3f' % (text[0], text[1]), fontsize=9)
    plt.tight_layout()
    plt.savefig(op.join(path_data, 'behavior_plots', '%s_average.png' % name))
    plt.close()
    # Plot swarmplot for first block
    if name == 'all' or 'highvslow':
        plt.figure(figsize=(1.4, 1.67), dpi=400)
        mean_on_ = on_[:, 0]
        mean_off_ = off_[:, 0]
        t_on_, p_on_ = ttest_1samp(mean_on_, 0, nan_policy='omit')
        t_off_, p_off_ = ttest_1samp(mean_off_, 0, nan_policy='omit')
        to_plot_np = np.array([mean_on_, mean_off_]).T
        to_plot_df = pd.DataFrame(data=to_plot_np, index=subjects, columns=['online', 'offline'])
        # Reverse data to show learning positive if general skill
        if name == 'all':
            to_plot_df = -to_plot_df
        ax = sns.violinplot(data=to_plot_df, palette=color_barplot, inner='quartile', linewidth=0.5, bw=.4)
        ax.axhline(0, ls='--', color='k', linewidth=0.5)
        ax = sns.swarmplot(data=to_plot_df, palette=['1', '1'], size=0.6, alpha=1)
        plt.ylabel(ylabel)
        plt.ylim(ylim_violin)
        if with_p_values:
            for (text, xpos) in zip([[t_on_, p_on_],
                                     [t_off_, p_off_]], [0, 1]):
                if text[1] < 0.05:
                    ax.text(xpos, 10, 't=%.1f, p<.05' % (text[0]), fontsize=7)
                else:
                    ax.text(xpos, 10, 't=%.1f, p=%.2f' % (text[0], text[1]), fontsize=7)
        plt.tight_layout()
        plt.savefig(op.join(path_data, 'behavior_plots', 'block1_%s_average.png' % name))
        plt.close()
