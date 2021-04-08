"""Measure online and offline learning on the long ASRT dataset
(Dataset 3 in the corresponding manuscript)"""

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
          'b41', 'b42', 'b43', 'b44', 'b45', 'b46', 'b47', 'b48', 'b49', 'b50',
          'b51', 'b52', 'b53', 'b54', 'b55', 'b56', 'b57', 'b58', 'b59', 'b60',
          'b61', 'b62', 'b63', 'b64', 'b65', 'b66', 'b67', 'b68', 'b69', 'b70',
          'b71', 'b72', 'b73', 'b74', 'b75', 'b76', 'b77', 'b78', 'b79', 'b80',
          'b81', 'b82', 'b83', 'b84', 'b85', 'b86', 'b87', 'b88', 'b89', 'b90',
          'b91', 'b92', 'b93', 'b94', 'b95', 'b96', 'b97', 'b98', 'b99', 'b100',
          'b101', 'b102', 'b103', 'b104', 'b105', 'b106', 'b107', 'b108', 'b109', 'b110',
          'b111', 'b112', 'b113', 'b114', 'b115', 'b116', 'b117', 'b118', 'b119', 'b120',
          'b121', 'b122', 'b123', 'b124', 'b125', 'b126', 'b127', 'b128', 'b129', 'b130',
          'b131', 'b132', 'b133', 'b134', 'b135', 'b136', 'b137', 'b138', 'b139', 'b140',
          'b141', 'b142', 'b143', 'b144', 'b145', 'b146', 'b147', 'b148', 'b149', 'b150',
          'b151', 'b152', 'b153', 'b154', 'b155', 'b156', 'b157', 'b158', 'b159', 'b160',
          'b161', 'b162', 'b163', 'b164', 'b165', 'b166', 'b167', 'b168', 'b169', 'b170',
          'b171', 'b172', 'b173', 'b174', 'b175', 'b176', 'b177', 'b178', 'b179', 'b180',
          'b181', 'b182', 'b183', 'b184', 'b185', 'b186', 'b187', 'b188', 'b189', 'b190',
          'b191', 'b192', 'b193', 'b194', 'b195', 'b196', 'b197', 'b198', 'b199', 'b200']
columns = blocks.copy()
columns.insert(0, 'sub')
# Read behavioral file with all participants
behav = pd.read_spss(op.join(path_data, 'session1-8_spss_ketvaltozos.sav'))
subjects = np.unique(behav.Subject_old)

# Drop first 7 trials of each block (5 trials + 2 that start the first triplet)
behav_clean = behav[behav.TT_seq1 != 'X']
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
    behav_sub = behav[behav.Subject_old == subject]
    # Drop incorrect trial (incorrect response)
    behav_sub_clean = behav_sub[behav_sub.firstACC == 1]
    # Drop repetitions or trill
    behav_sub_clean = behav_sub_clean[np.array(behav_sub_clean.TT_seq1 == 'H') | np.array(behav_sub_clean.TT_seq1 == 'L')]
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
    for session in np.unique(behav.Session):
        behav_sub_sess = behav_sub[behav_sub.Session == session]
        for block in np.unique(behav.Block):
            behav_sub_block = behav_sub_sess[behav_sub.Block == block]
            # Drop first 5 practice trials of each block
            behav_sub_block = behav_sub_block[behav_sub_block.TrialType != 'Prac']
            if len(behav_sub_block) != 80:
                warnings.warn("Block without exactly 80 trials")
            # Drop incorrect trial (incorrect response)
            behav_sub_block_clean = behav_sub_block[behav_sub_block.firstACC == 1]
            # Drop repetitions or trill
            behav_sub_block_clean = behav_sub_block_clean[np.array(behav_sub_block_clean.TT_seq1 == 'H') | np.array(behav_sub_block_clean.TT_seq1 == 'L')]
            # Drop extreme value
            behav_sub_block_clean = behav_sub_block_clean[behav_sub_block_clean.finalRT <= meanRT+3*stdRT]
            behav_sub_block_clean = behav_sub_block_clean[behav_sub_block_clean.finalRT >= meanRT-3*stdRT]
            all_block.append(behav_sub_block_clean.finalRT.mean())
            high_block.append(behav_sub_block_clean[behav_sub_block_clean.TT_seq1 == 'H'].finalRT.mean())
            low_block.append(behav_sub_block_clean[behav_sub_block_clean.TT_seq1 == 'L'].finalRT.mean())
            pat_block.append(behav_sub_block_clean[behav_sub_block_clean.TrialType == 'P'].finalRT.mean())
            ran_block.append(behav_sub_block_clean[behav_sub_block_clean.TrialType == 'R'].finalRT.mean())
            ranhigh_block.append(behav_sub_block_clean[np.array(behav_sub_block_clean.TT_seq1 == 'H') & np.array(behav_sub_block_clean.TrialType == 'R')].finalRT.mean())

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
                bin_values = bin_values[np.array(bin_values.TT_seq1 == 'H') | np.array(bin_values.TT_seq1 == 'L')]
                # Drop extreme value
                bin_values = bin_values[bin_values.finalRT <= meanRT+2.5*stdRT]
                bin_values = bin_values[bin_values.finalRT >= meanRT-2.5*stdRT]
                # Append mean RT in each bin
                all.append(bin_values.finalRT.mean())
                high.append(bin_values[bin_values.TT_seq1 == 'H'].finalRT.mean())
                low.append(bin_values[bin_values.TT_seq1 == 'L'].finalRT.mean())
                pat.append(bin_values[bin_values.TrialType == 'P'].finalRT.mean())
                ran.append(bin_values[bin_values.TrialType == 'R'].finalRT.mean())
                ranhigh.append(bin_values[np.array(bin_values.TT_seq1 == 'H') & np.array(bin_values.TrialType == 'R')].finalRT.mean())
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

for name, blocked, ylims, color in zip(['all', 'highvslow', 'patvsranhigh', 'ranhighvslow'],
                                       [all_RT_block, -highvslow_block, -patvsranhigh_block, -ranhighvslow_block],
                                       [[250, 430], [-30, 60], [-30, 30], [-30, 60]],
                                       [colors[0], colors[1], colors[2], colors[1]]):
    mean = np.array(np.nanmean(blocked, 0))
    std = np.array(np.std(blocked, 0))
    plt.figure(figsize=(2.5, 1.8), dpi=200)
    plt.plot(np.arange(200) + 1, mean, color='black', lw=0.5)
    plt.fill_between(np.arange(200) + 1, mean - (std/np.sqrt(len(subjects))),
                     mean + (std/np.sqrt(len(subjects))), color=color,
                     alpha=0.2)
    plt.xticks(np.hstack([1, np.arange(25, 200, 25)]))
    plt.ylim(ylims)
    for x in [25.5, 50.5, 75.5, 100.5, 125.5, 150.5, 175.5]:
        plt.axvline(x=x, ls='-.', lw=0.5, color='gray')
    plt.gca().xaxis.grid(False)
    plt.xlabel('Blocks')
    if name == 'all':
        plt.ylabel('Average reaction time \nper block (ms)')
    else:
        plt.ylabel('Average of reaction time \ndifference per block (ms)')
    plt.tight_layout()
    plt.savefig(op.join(path_data, 'behavior_plots', '%s_blocked_rt_per_block_sess.png' % name))
    plt.close()


# Plot binned learning (all sessions together)
params = zip(['all', 'highvslow', 'patvsranhigh', 'ranhighvslow'],
             [all_RT_bin, -highvslow_bin, -patvsranhigh_bin, -ranhighvslow_bin],
             [[250, 430], [-30, 60], [-30, 30], [-30, 60]],
             [colors[0], colors[1], colors[2], colors[1]],
             [(1.9, 1.8), (1.9, 1.8), (1.9, 1.8), (1.9, 1.8)])
for name, binned, ylims, color, figsize in params:
    a = 0
    b = bins
    ticks = list()
    plt.figure(figsize=figsize, dpi=400)
    for ii in np.arange(200):
        mean = np.nanmean(binned[:, ii, :], 0)
        std = np.nanstd(binned[:, ii, :], 0)
        plt.plot(np.arange(a, b), mean, color=color, lw=0.8)
        plt.fill_between(np.arange(a, b), mean - (std/np.sqrt(len(subjects))),
                         mean + (std/np.sqrt(len(subjects))), color=color,
                         alpha=0.1)
        ticks.append((a+b)/2 - 1)
        a = 1 + b
        b = a + bins
    plt.xticks([ticks[50], ticks[100], ticks[150], ],
               np.arange(0, 200, 50)[1:])
    # plot average per block
    mean = np.array(np.nanmean(binned, (0, 2)))
    plt.plot(ticks, mean, color='black', linewidth=0.4)
    plt.xlabel('Blocks')
    plt.ylim(ylims)
    for x in [ticks[25], ticks[50], ticks[75], ticks[100], ticks[125], ticks[150], ticks[175]]:
        plt.axvline(x=x, ls='-', lw=0.5, color='0.4')
    plt.gca().xaxis.grid(False)

    if name == 'all':
        plt.ylabel('Reaction time (ms)')
    else:
        plt.ylabel('Reaction time difference (ms)')
    plt.tight_layout()
    plt.savefig(op.join(path_data, 'behavior_plots', '%s_binned_trials_rt_per_block.png' % name))
    plt.close()

# Plot binned learning (separately for day1, middle, and day 8)
params = zip(['all', 'highvslow', 'patvsranhigh', 'ranhighvslow'],
             [all_RT_bin, -highvslow_bin, -patvsranhigh_bin, -ranhighvslow_bin],
             [[250, 430], [-30, 60], [-30, 30], [-30, 60]],
             [colors[0], colors[1], colors[2], colors[1]])
for name, binned, ylims, color in params:
    # Figure day 1
    a = 0
    b = bins
    ticks = list()
    plt.figure(figsize=(1.9, 1.8), dpi=400)
    for ii in np.arange(0, 25):
        mean = np.nanmean(binned[:, ii, :], 0)
        std = np.nanstd(binned[:, ii, :], 0)
        plt.plot(np.arange(a, b), mean, color=color, lw=0.8)
        plt.fill_between(np.arange(a, b), mean - (std/np.sqrt(len(subjects))),
                         mean + (std/np.sqrt(len(subjects))), color=color,
                         alpha=0.1)
        ticks.append((a+b)/2 - 1)
        a = 1 + b
        b = a + bins
    plt.xticks([ticks[1], ticks[10], ticks[20]], [1, 10, 20])
    # plot average per block
    mean = np.array(np.nanmean(binned[:, np.arange(0, 25), :], (0, 2)))
    plt.plot(ticks, mean, color='black', linewidth=0.4)
    plt.xlabel('Blocks')
    plt.ylim(ylims)
    if name == 'all':
        plt.ylabel('Reaction time (ms)')
    else:
        plt.ylabel('Reaction time difference (ms)')
    plt.tight_layout()
    plt.savefig(op.join(path_data, 'behavior_plots', '%s_binned_trials_rt_per_block_day1.png' % name))
    plt.close()
    # Figure day 8
    a = 0
    b = bins
    ticks = list()
    plt.figure(figsize=(1.9, 1.8), dpi=400)
    for ii in np.arange(175, 200):
        mean = np.nanmean(binned[:, ii, :], 0)
        std = np.nanstd(binned[:, ii, :], 0)
        plt.plot(np.arange(a, b), mean, color=color, lw=0.8)
        plt.fill_between(np.arange(a, b), mean - (std/np.sqrt(len(subjects))),
                         mean + (std/np.sqrt(len(subjects))), color=color,
                         alpha=0.1)
        ticks.append((a+b)/2 - 1)
        a = 1 + b
        b = a + bins
    plt.xticks([ticks[1], ticks[10], ticks[20]], [176, 185, 195])
    # plot average per block
    mean = np.array(np.nanmean(binned[:, np.arange(175, 200), :], (0, 2)))
    plt.plot(ticks, mean, color='black', linewidth=0.4)
    plt.xlabel('Blocks')
    plt.ylim(ylims)
    if name == 'all':
        plt.ylabel('Reaction time (ms)')
    else:
        plt.ylabel('Reaction time difference (ms)')
    plt.tight_layout()
    plt.savefig(op.join(path_data, 'behavior_plots', '%s_binned_trials_rt_per_block_day8.png' % name))
    plt.close()


# Plot online, offline and both learning
params = zip(['all', 'highvslow', 'patvsranhigh', 'ranhighvslow'],
             [all_RT_bin, -highvslow_bin, -patvsranhigh_bin, -ranhighvslow_bin],
             [all_RT_block, -highvslow_block, -patvsranhigh_block, -ranhighvslow_block],
             [colors_barplot[0], colors_barplot[1], colors_barplot[2], colors_barplot[1]],
             ['General skill learning (ms)', 'Statistical learning (ms)', 'High-order Learning (ms)', 'Statistical learning (ms)'],
             [(-80, 80), (-60, 60), (-50, 50), (-60, 60)],
             [(-200, 200), (-60, 60), (-50, 50), (-60, 60)])
for name, bin_average, block_average, color_barplot, ylabel, ylim_curve, ylim_violin in params:
    off_ = bin_average[:, 1:, 0] - bin_average[:, :-1, -1]
    on_ = bin_average[:, :, -1] - bin_average[:, :, 0]
    both_ = block_average[:, 1:] - block_average[:, :-1]
    # Separate micro_off and long off
    off_wh = np.ones(off_.shape[1])  # 1 if between blocks
    off_wh.put([24, 49, 74, 99, 124, 149, 174], 0)  # 0 if between sessions
    off_wh = off_wh.astype(int)
    # Plot online and offline curve
    plt.figure(figsize=(2.1, 1.95), dpi=400)
    times = np.arange(200) + 1
    # plot online
    mean, std = np.nanmean(on_, 0), np.nanstd(on_, 0)
    err = std/np.sqrt(len(subjects))
    plt.errorbar(times, mean, yerr=err, fmt='o', color=color_barplot[0],
                 ms=0.3, lw=0.2)
    # Fit linear regression
    regress_params = linregress(times, mean)
    fit = regress_params[1] + regress_params[0] * times
    plt.plot(times, fit, lw=1.2, color=color_barplot[0], label='Online')
    # plot offline
    times = np.arange(199) + 2
    mean, std = np.nanmean(off_, 0), np.nanstd(off_, 0)
    err = std/np.sqrt(len(subjects))
    plt.errorbar(times, mean, yerr=err, fmt='o', color=color_barplot[1],
                 ms=0.3, lw=0.2)
    # Fit linear regression
    regress_params = linregress(times, mean)
    fit = regress_params[1] + regress_params[0] * times
    plt.plot(times, fit, lw=1.2, color=color_barplot[1], label='Offline')
    plt.axhline(0, ls='--', color='k', linewidth=0.5)
    plt.legend()
    plt.gca().legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                     ncol=2, mode="expand", borderaxespad=0.)
    plt.ylim(ylim_curve)
    plt.xticks(np.hstack([1, np.arange(10, 201, 10)]))
    plt.xlabel('Blocks')
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.xticks([50, 100, 150], np.arange(0, 200, 50)[1:])
    for x in [25, 50, 75, 100, 125, 150, 175]:
        plt.axvline(x=x, ls='-.', lw=0.5, color='gray')
    # Run anova
    stats = anova_onoff(on_, off_, subjects, columns)
    plt.savefig(op.join(path_data, 'behavior_plots', '%s_offonline_curve.png' % name))
    plt.close()

    # Plot swarmplot
    plt.figure(figsize=(1.4, 1.67), dpi=400)
    mean_on_ = np.nanmean(on_, 1)
    mean_off_block = np.nanmean(off_[:, np.where(off_wh)[0]], 1)
    mean_off_session = np.nanmean(off_[:, np.where(1 - off_wh)[0]], 1)
    mean_both_ = np.nanmean(both_, 1)
    t_on_, p_on_ = ttest_1samp(mean_on_, 0)
    t_off_block, p_off_block = ttest_1samp(mean_off_block, 0)
    t_off_session, p_off_session = ttest_1samp(mean_off_session, 0)
    t_both_, p_both_ = ttest_1samp(mean_both_, 0)
    to_plot_np = np.array([mean_on_, mean_off_block]).T
    to_plot_df = pd.DataFrame(data=to_plot_np, index=subjects, columns=['online', 'offline'])
    # Reverse data to show learning positive
    if name == 'all':
        to_plot_df = -to_plot_df
    ax = sns.violinplot(data=to_plot_df, palette=color_barplot, inner='quartile', linewidth=0.5, bw=.4)
    ax.axhline(0, ls='--', color='k', linewidth=0.5)
    ax = sns.swarmplot(data=to_plot_df, palette=['1', '1'], size=0.6, alpha=1)
    plt.ylim(ylim_violin)
    plt.ylabel(ylabel)
    if with_p_values:
        for (text, xpos) in zip([[t_on_, p_on_],
                                 [t_off_block, p_off_block],
                                 [t_off_session, p_off_session]], [0, 1, 2]):
            if text[1] < 0.05:
                ax.text(xpos, 10, 't=%.1f, p<.05' % (text[0]), fontsize=7)
            else:
                ax.text(xpos, 10, 't=%.1f, p=%.2f' % (text[0], text[1]), fontsize=7)
    plt.tight_layout()
    plt.savefig(op.join(path_data, 'behavior_plots', '%s_average.png' % name))
    plt.close()
    # Plot swarmplot for first blocks
    if name == 'all' or 'highvslow':
        plt.figure(figsize=(1.4, 1.67), dpi=400)
        mean_on_ = np.nanmean(on_[:, [0, 25, 50, 75, 100, 125, 150, 175]], 1)
        mean_off_ = np.nanmean(off_[:, [0, 25, 50, 75, 100, 125, 150, 175]], 1)
        t_on_, p_on_ = ttest_1samp(mean_on_, 0)
        t_off_, p_off_ = ttest_1samp(mean_off_, 0)
        to_plot_np = np.array([mean_on_, mean_off_]).T
        to_plot_df = pd.DataFrame(data=to_plot_np, index=subjects, columns=['online', 'offline'])
        # Reverse data to show learning positive
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
