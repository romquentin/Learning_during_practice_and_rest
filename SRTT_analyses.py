"""Measure online and offline learning on the SRTT dataset
(Dataset 1 in the corresponding manuscript)"""

# Authors: Romain Quentin <rom.quentin@gmail.com>
# License: BSD (3-clause)

import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pingouin as pg
from tqdm import tqdm
import seaborn as sns
from scipy.stats import ttest_1samp
from scipy.stats import binned_statistic

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

color = '#0173b2'
color_barplot = ['#cc78bc', '#949494']

path_data = './data'

with_p_values = False
# Define blocks (and columns for anova)
blocks = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10',
          'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20',
          'b21', 'b22', 'b23', 'b24', 'b25', 'b26']
columns = blocks.copy()
columns.insert(0, 'sub')
# Read behavioral file with all participants
behav = pd.read_spss(op.join(path_data, 'AF_csrt_raw.sav'))
subjects = np.unique(behav.Subject)

# Bin trials into each sequence and calculate mean RT
bins = 5  # There are 5 sequence (of 12 trials) per block
# Average within blocks
all_RT_block = list()
# Average within bins
all_RT_bin = list()
for subject in tqdm(subjects):
    behav_sub = behav[behav.Subject == subject]
    # Drop incorrect trial (incorrect response)
    behav_sub_clean = behav_sub[behav_sub.firstACC == 1]
    # Drop negative RT value (apparently one bug value)
    behav_sub_clean = behav_sub_clean[behav_sub_clean.finalRT > 0]
    # Calculate mean reaction and std on all trials for each participant
    meanRT = behav_sub_clean.finalRT.mean()
    stdRT = behav_sub_clean.finalRT.std()
    # Average within blocks
    all_block = list()
    # Average within bins
    all_bin = list()
    for session in np.unique(behav.Session):
        # Get data for each session
        behav_sub_sess = behav_sub[behav_sub.Session == session]
        for block in np.unique(behav.Block):
            behav_sub_block = behav_sub_sess[behav_sub_sess.Block == block]
            # Drop incorrect trial (incorrect response)
            behav_sub_block_clean = behav_sub_block[behav_sub_block.firstACC == 1]
            # Drop extreme values (> 3 std)
            behav_sub_block_clean = behav_sub_block_clean[behav_sub_block_clean.finalRT <= meanRT+3*stdRT]
            behav_sub_block_clean = behav_sub_block_clean[behav_sub_block_clean.finalRT >= meanRT-3*stdRT]
            all_block.append(behav_sub_block_clean.finalRT.mean())
            binnumber = binned_statistic(np.arange(len(behav_sub_block)),
                                         np.array(behav_sub_block.finalRT),
                                         bins=bins)[2]
            all = list()
            for bin in np.arange(bins):
                bin_values = behav_sub_block[binnumber == bin + 1]
                # Drop incorrect trial (incorrect response)
                bin_values = bin_values[bin_values.firstACC == 1]
                # Drop negative RT value (apparently one bug value)
                bin_values = bin_values[bin_values.finalRT > 0]
                # Drop extreme value
                bin_values = bin_values[bin_values.finalRT <= meanRT+3*stdRT]
                bin_values = bin_values[bin_values.finalRT >= meanRT-3*stdRT]
                # Append mean RT in each bin
                all.append(bin_values.finalRT.mean())
            # Append RT for each block
            all_bin.append(all)
    # Append RT for each participant
    all_RT_block.append(all_block)
    all_RT_bin.append(all_bin)
all_RT_block = np.array(all_RT_block)
all_RT_bin = np.array(all_RT_bin)
# Calculate online and offline learning
off_ = all_RT_bin[:, 1:, 0] - all_RT_bin[:, :-1, -1]
on_ = all_RT_bin[:, :, -1] - all_RT_bin[:, :, 0]
both_ = all_RT_bin[:, 1:, :].mean(2) - all_RT_bin[:, :-1, :].mean(2)


# Plot block reaction time averaged accross participant
mean = np.array(np.nanmean(all_RT_block, 0))
std = np.array(np.std(all_RT_block, 0))
plt.figure(figsize=(2.4, 1.8), dpi=200)
plt.plot(np.arange(len(blocks)) + 1, mean, color='black', linewidth=0.5)
plt.fill_between(np.arange(len(blocks)) + 1, mean - (std/np.sqrt(len(subjects))),
                 mean + (std/np.sqrt(len(subjects))), color='black',
                 alpha=0.2)
plt.xticks(np.arange(len(blocks)) + 1)
plt.ylim([270, 500])
plt.xticks([1, 5, 10, 15, 20, 25])
plt.yticks([300, 400, 500])
plt.xlabel('Blocks')
plt.ylabel('Average reaction time \nper block (ms)')
plt.tight_layout()
plt.savefig(op.join(path_data, 'behavior_plots', 'blocked_rt_per_block.png'))
plt.close()


# Plot binned reaction time averaged accross participant
a = 0
b = bins
ticks = list()
plt.figure(figsize=(3.7, 1.8), dpi=400)
plt.ylim([270, 500])
for ii in np.arange(len(blocks)):
    mean = np.nanmean(all_RT_bin[:, ii, :], 0)
    std = np.std(all_RT_bin[:, ii, :], 0)
    plt.plot(np.arange(a, b), mean, color=color, lw=0.5)
    plt.fill_between(np.arange(a, b), mean - (std/np.sqrt(len(subjects))),
                     mean + (std/np.sqrt(len(subjects))), color=color,
                     alpha=0.2)
    ticks.append((a+b)/2 - 1)
    a = 1 + b
    # a = b
    b = a + bins
plt.xticks([ticks[0], ticks[4], ticks[9], ticks[13], ticks[17], ticks[22]],
           [1, 5, 10, 1, 5, 10])
# plot average per block
mean = np.array(np.nanmean(all_RT_block, 0))
plt.plot(ticks, mean, color='black', linewidth=0.4)
plt.axvline(x=(ticks[12] + ticks[13])/2., ls='-', lw=0.5, color='0.4')
plt.yticks([300, 400, 500])
plt.xlabel('Blocks')
plt.ylabel('Reaction time (ms)')
plt.tight_layout()
plt.savefig(op.join(path_data, 'behavior_plots', 'binned_rt_per_block.png'))
plt.close()


# Plot online and offline learning per block
# Reverse data so that learning is positive values
times = np.arange(len(blocks)) + 1
# plot online
mean, std = np.nanmean(on_, 0), np.nanstd(on_, 0)
plt.plot(times, -mean, color='C1', label='Online')
plt.fill_between(times, -mean - (std/np.sqrt(len(subjects))),
                 -mean + (std/np.sqrt(len(subjects))), color='C1',
                 alpha=0.2)
# plot offline
times = np.arange(len(blocks)-1) + 2
mean, std = np.nanmean(off_, 0), np.nanstd(off_, 0)
plt.plot(times, -mean, color='C2', label='Offline')
plt.fill_between(times, -mean - (std/np.sqrt(len(subjects))),
                 -mean + (std/np.sqrt(len(subjects))), color='C2',
                 alpha=0.2)
plt.axhline(0, ls='--', color='k', linewidth=0.5)
plt.ylim(-80, 80)
plt.legend()
plt.savefig(op.join(path_data, 'behavior_plots', 'offonline_curve.png'))
plt.close()


# Plot swarmplot for online and offline learning (with removing of blocks with random sequences)
on_blocks = [True, True, True, True, True, False, True, True, True, True,
             True, False, True, True, True, True, True, True, False, True,
             True, True, True, True, False, True]
off_blocks = [True, True, True, True, False, False, True, True, True, True,
              False, False, False, True, True, True, True, False, False, True, True,
              True, True, False, False]
plt.figure(figsize=(1.4, 1.67), dpi=400)
mean_on_ = np.nanmean(on_[:, on_blocks], 1)
mean_off_ = np.nanmean(off_[:, off_blocks], 1)
mean_both_ = np.nanmean(both_[:, off_blocks], 1)
t_on_, p_on_ = ttest_1samp(mean_on_, 0)
t_off_, p_off_ = ttest_1samp(mean_off_, 0)
t_both_, p_both_ = ttest_1samp(mean_both_, 0)
to_plot_np = np.array([mean_on_, mean_off_]).T
to_plot_df = pd.DataFrame(data=to_plot_np, index=subjects, columns=['online', 'offline'])
# Reverse data so that learning is positive values
to_plot_df = -to_plot_df
ax = sns.violinplot(data=to_plot_df, palette=color_barplot, inner='quartile', linewidth=0.5, bw=.4)
ax.axhline(0, ls='--', color='k', linewidth=0.5)
ax = sns.swarmplot(data=to_plot_df, palette=['1', '1'], size=0.6, alpha=1)
plt.ylabel('General skill learning (ms)')
plt.ylim(-200, 200)
plt.yticks([-200, -100, 0, 100, 200])
if with_p_values:
    for (text, xpos) in zip([[t_on_, p_on_],
                             [t_off_, p_off_]], [0, 1]):
        # ax.text(xpos, 10, 't=%.2f, p=%.2f' % (text[0], text[1]), fontsize=8))
        if text[1] < 0.05:
            ax.text(xpos, 10, 't=%.1f, p<.05' % (text[0]), fontsize=7)
        else:
            ax.text(xpos, 10, 't=%.1f, p=%.2f' % (text[0], text[1]), fontsize=7)
plt.tight_layout()
plt.savefig(op.join(path_data, 'behavior_plots', 'allRT_average.png'))
plt.close()

# Plot swarmplot on only first block
mean_on_ = np.nanmean(on_[:, [0, 13]], 1)
mean_off_ = np.nanmean(off_[:, [0, 13]], 1)
t_on_, p_on_ = ttest_1samp(mean_on_, 0)
t_off_, p_off_ = ttest_1samp(mean_off_, 0)
to_plot_np = np.array([mean_on_, mean_off_]).T
to_plot_df = pd.DataFrame(data=to_plot_np, index=subjects, columns=['online', 'offline'])
# Reverse data to show learning positive
to_plot_df = -to_plot_df
plt.figure(figsize=(1.4, 1.67), dpi=400)
ax = sns.violinplot(data=to_plot_df, palette=color_barplot, inner='quartile', linewidth=0.5, bw=.4)
ax.axhline(0, ls='--', color='k', linewidth=0.5)
ax = sns.swarmplot(data=to_plot_df, palette=['1', '1'], size=0.6, alpha=1)
plt.ylabel('General skill learning (ms)')
plt.ylim(-200, 200)
plt.yticks([-200, -100, 0, 100, 200])
if with_p_values:
    for (text, xpos) in zip([[t_on_, p_on_],
                             [t_off_, p_off_]], [0, 1]):
        # ax.text(xpos, 10, 't=%.2f, p=%.2f' % (text[0], text[1]), fontsize=8))
        if text[1] < 0.05:
            ax.text(xpos, 10, 't=%.1f, p<.05' % (text[0]), fontsize=7)
        else:
            ax.text(xpos, 10, 't=%.1f, p=%.2f' % (text[0], text[1]), fontsize=7)
plt.tight_layout()
plt.savefig(op.join(path_data, 'behavior_plots', 'block1_average.png'))
plt.close()

# Format the data and run anova (need a long-format df)
blocks = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10',
          'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20',
          'b21', 'b22', 'b23', 'b24', 'b25', 'b26']
columns = blocks.copy()
columns.insert(0, 'sub')
all = pd.DataFrame(data=np.insert(all_RT_block,
                                  0,
                                  np.arange(len(subjects)),
                                  axis=1),
                   columns=columns)
all = pd.melt(all,
              id_vars=['sub'],
              value_vars=blocks,
              var_name='block',
              value_name='RT')
all.insert(1, 'Triplet', np.zeros(len(all)))
aov_gen = pg.rm_anova(data=all, dv='RT',
                      within=['block'], subject='sub')
