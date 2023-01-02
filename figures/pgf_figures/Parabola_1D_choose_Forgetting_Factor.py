import numpy as np
import matplotlib as mpl

# Use the pgf backend (must be set before pyplot imported)
mpl.use('pgf')
import matplotlib.pyplot as plt
import pickle
import torch
import glob
from src.objective_functions_1D import long_riley_function_linear, min_long_riley_function_linear
from utils.postprocessing_utils import initialize_plot, set_size
import pandas as pd
import seaborn as sns

c, params_plot = initialize_plot('Presentation')
plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
    "text.usetex": True,  # use inline math for ticks
    "pgf.rcfonts": False,  # don't setup fonts from rc parameters
})

results_path = '/Users/paulbrunzema/Documents/Studium/Master/Masterarbeit/02_Code/results/'
img_path = '/Users/paulbrunzema/Documents/Studium/Master/Masterarbeit/02_Code/imgs/1D_parabola/'

TIME_HORIZON = 300

TRAINING_POINTS = 15

name01 = '1D_parabola/B2P_choosing_epsilon_unconstrained/results_B2P_OU_1Dparabola_unconstrained_forgetting_factor_0_005_0_005_*'
name02 = '1D_parabola/B2P_choosing_epsilon_unconstrained/results_B2P_OU_1Dparabola_unconstrained_forgetting_factor_0_007_0_007_*'
name03 = '1D_parabola/B2P_choosing_epsilon_unconstrained/results_B2P_OU_1Dparabola_unconstrained_forgetting_factor_0_009_0_009_*'
name1 = '1D_parabola/B2P_choosing_epsilon_unconstrained/results_B2P_OU_1Dparabola_unconstrained_forgetting_factor_0_01_0_01_*'
name2 = '1D_parabola/B2P_choosing_epsilon_unconstrained/results_B2P_OU_1Dparabola_unconstrained_forgetting_factor_0_01291549665014884_0_01291549665014884*'
name3 = '1D_parabola/B2P_choosing_epsilon_unconstrained/results_B2P_OU_1Dparabola_unconstrained_forgetting_factor_0_016681005372000592_0_016681005372000592_*'
name4 = '1D_parabola/B2P_choosing_epsilon_unconstrained/results_B2P_OU_1Dparabola_unconstrained_forgetting_factor_0_021544346900318832_0_021544346900318832*'
name5 = '1D_parabola/B2P_choosing_epsilon_unconstrained/results_B2P_OU_1Dparabola_unconstrained_forgetting_factor_0_027825594022071243_0_027825594022071243*'
name6 = '1D_parabola/B2P_choosing_epsilon_unconstrained/results_B2P_OU_1Dparabola_unconstrained_forgetting_factor_0_03593813663804628_0_03593813663804628*'
name7 = '1D_parabola/B2P_choosing_epsilon_unconstrained/results_B2P_OU_1Dparabola_unconstrained_forgetting_factor_0_046415888336127774_0_046415888336127774*'
name8 = '1D_parabola/B2P_choosing_epsilon_unconstrained/results_B2P_OU_1Dparabola_unconstrained_forgetting_factor_0_05994842503189409_0_05994842503189409*'
name9 = '1D_parabola/B2P_choosing_epsilon_unconstrained/results_B2P_OU_1Dparabola_unconstrained_forgetting_factor_0_0774263682681127_0_0774263682681127*'
name10 = '1D_parabola/B2P_choosing_epsilon_unconstrained/results_B2P_OU_1Dparabola_unconstrained_forgetting_factor_0_1_0_1*'
name11 = '1D_parabola/B2P_choosing_epsilon_unconstrained/results_B2P_OU_1Dparabola_unconstrained_forgetting_factor_0_3_0_3*'
name12 = '1D_parabola/B2P_choosing_epsilon_unconstrained/results_B2P_OU_1Dparabola_unconstrained_forgetting_factor_0_5_0_5*'

names = [name01, name02, name03, name1, name2, name3, name4, name5, name6, name7, name8, name9, name10, name11, name12]

SIGMAS = [0.005, 0.007, 0.009, 0.01, 0.013, 0.0165, 0.0215, 0.028, 0.036, 0.0465, 0.060, 0.077, 0.1, 0.3, 0.5]

results1 = []
for name in names:
    inter_results = []
    files = glob.glob(results_path + name, recursive=True)
    for file in files:
        with open(file, 'rb') as f:
            result = pickle.load(f)
        inter_results.append(result)
    results1.append(inter_results)

name01 = '1D_parabola/B2P_choosing_epsilon_constrained/results_B2P_OU_1Dparabola_constrained_forgetting_factor_0_005_0_005_*'
name02 = '1D_parabola/B2P_choosing_epsilon_constrained/results_B2P_OU_1Dparabola_constrained_forgetting_factor_0_007_0_007_*'
name03 = '1D_parabola/B2P_choosing_epsilon_constrained/results_B2P_OU_1Dparabola_constrained_forgetting_factor_0_009_0_009_*'
name1 = '1D_parabola/B2P_choosing_epsilon_constrained/results_B2P_OU_1Dparabola_constrained_forgetting_factor_0_01_0_01_*'
name2 = '1D_parabola/B2P_choosing_epsilon_constrained/results_B2P_OU_1Dparabola_constrained_forgetting_factor_0_013_0_013*'
name3 = '1D_parabola/B2P_choosing_epsilon_constrained/results_B2P_OU_1Dparabola_constrained_forgetting_factor_0_0165_0_0165*'
name4 = '1D_parabola/B2P_choosing_epsilon_constrained/results_B2P_OU_1Dparabola_constrained_forgetting_factor_0_0215_0_0215*'
name5 = '1D_parabola/B2P_choosing_epsilon_constrained/results_B2P_OU_1Dparabola_constrained_forgetting_factor_0_028_0_028*'
name6 = '1D_parabola/B2P_choosing_epsilon_constrained/results_B2P_OU_1Dparabola_constrained_forgetting_factor_0_036_0_036*'
name7 = '1D_parabola/B2P_choosing_epsilon_constrained/results_B2P_OU_1Dparabola_constrained_forgetting_factor_0_0465_0_0465*'
name8 = '1D_parabola/B2P_choosing_epsilon_constrained/results_B2P_OU_1Dparabola_constrained_forgetting_factor_0_06_0_06*'
name9 = '1D_parabola/B2P_choosing_epsilon_constrained/results_B2P_OU_1Dparabola_constrained_forgetting_factor_0_077_0_077*'
name10 = '1D_parabola/B2P_choosing_epsilon_constrained/results_B2P_OU_1Dparabola_constrained_forgetting_factor_0_1_0_1*'
name11 = '1D_parabola/B2P_choosing_epsilon_constrained/results_B2P_OU_1Dparabola_constrained_forgetting_factor_0_3_0_3*'
name12 = '1D_parabola/B2P_choosing_epsilon_constrained/results_B2P_OU_1Dparabola_constrained_forgetting_factor_0_5_0_5*'

names = [name01, name02, name03, name1, name2, name3, name4, name5, name6, name7, name8, name9, name10, name11, name12]

SIGMAS2 = SIGMAS
results2 = []
for name in names:
    inter_results = []
    files = glob.glob(results_path + name, recursive=True)
    for file in files:
        with open(file, 'rb') as f:
            result = pickle.load(f)
        inter_results.append(result)
    results2.append(inter_results)

# start_train = result['start_training']
# min_trajectory = result['min_trajectory']

t_abtast = np.arange(TRAINING_POINTS + 1, TIME_HORIZON + 1, 1)
x_star = min_long_riley_function_linear(torch.empty(1), torch.from_numpy(t_abtast)).numpy()
fx_star = long_riley_function_linear(torch.from_numpy(x_star), torch.from_numpy(t_abtast)).numpy()

x, y = set_size(398, subplots=(1, 2), fraction=1.)
fig = plt.figure(1, figsize=(x, y * 3))
ax1 = plt.subplot2grid((3, 1), (0, 0))

max = 0

data = []
models1 = []
for i, result in enumerate(results1):
    trails = []
    for trail in result:
        chosen_query = trail['trajectory']
        if len(chosen_query.shape) > 1:
            chosen_query = chosen_query[:, 0]
        fx = long_riley_function_linear(torch.from_numpy(chosen_query), torch.from_numpy(t_abtast)).numpy()
        error_each_timestep = abs(fx - fx_star)
        cum_error = np.cumsum(error_each_timestep)
        trails.append(cum_error[-1])
        data.append([SIGMAS[i], cum_error[-1], False])

    trails = np.asarray(trails)
    models1.append(trails)

models2 = []
for i, result in enumerate(results2):
    trails = []
    for trail in result:
        chosen_query = trail['trajectory']
        if len(chosen_query.shape) > 1:
            chosen_query = chosen_query[:, 0]
        fx = long_riley_function_linear(torch.from_numpy(chosen_query), torch.from_numpy(t_abtast)).numpy()
        error_each_timestep = abs(fx - fx_star)
        cum_error = np.cumsum(error_each_timestep)
        trails.append(cum_error[-1])
        data.append([SIGMAS2[i], cum_error[-1], True])

    trails = np.asarray(trails)
    models2.append(trails)

data = pd.DataFrame(data, columns=['Sigma', 'CumRegret', 'Constrained'])
boxenplot = sns.boxenplot(x="Sigma", y="CumRegret", hue="Constrained",
                          data=data, linewidth=0.75,
                          palette=[c['bordeaux100'], c['violett100'], ], k_depth='full', showfliers=False)
boxenplot.legend(loc='upper center', title='Constrained', edgecolor='white', facecolor='white',
                 framealpha=1, )

for i in range(len(SIGMAS) - 1):
    plt.vlines(i + 0.5, 0, 800, linestyles='-', colors=c['schwarz75'], lw=0.5)
# plot mean trajectory
idx = data.index[data['Constrained']]
data_constrained = data.loc[idx]
data_mean_constrained = data_constrained.groupby('Sigma').mean()
means_constrained = data_mean_constrained[['CumRegret']]
print(np.argmin(means_constrained))
x = np.arange(len(means_constrained)) + 0.2
ax1.plot(x, means_constrained, color=c['violett100'], marker='o', linestyle='dashed',
         linewidth=0.85, markersize=4, markeredgecolor='white', markeredgewidth=0.75)

idx = data.index[~data['Constrained']]
data_unconstrained = data.loc[idx]
data_mean_unconstrained = data_unconstrained.groupby('Sigma').mean()
x = np.arange(len(means_constrained)) - 0.2
means_unconstrained = data_mean_unconstrained[['CumRegret']]
print(np.argmin(means_unconstrained))
ax1.plot(x, means_unconstrained, color=c['bordeaux100'], marker='o', linestyle='dashed',
         linewidth=0.85, markersize=4, markeredgecolor='white', markeredgewidth=0.75)

plt.xticks(rotation=45)
plt.ylabel('$R_T$')
plt.xlabel('$\\epsilon$')
plt.ylim([0, 800])
plt.xlim([-0.5, 14.5])

#### UI
name01 = '1D_parabola/UI_choosing_sigma_unconstrained/results_UI_1Dparabola_unconstrained_forgetting_factor_0_005_0_005_*'
name02 = '1D_parabola/UI_choosing_sigma_unconstrained/results_UI_1Dparabola_unconstrained_forgetting_factor_0_007_0_007_*'
name03 = '1D_parabola/UI_choosing_sigma_unconstrained/results_UI_1Dparabola_unconstrained_forgetting_factor_0_009_0_009_*'
name1 = '1D_parabola/UI_choosing_sigma_unconstrained/results_UI_1Dparabola_unconstrained_forgetting_factor_0_01_0_01_*'
name2 = '1D_parabola/UI_choosing_sigma_unconstrained/results_UI_1Dparabola_unconstrained_forgetting_factor_0_01291549665014884_0_01291549665014884*'
name3 = '1D_parabola/UI_choosing_sigma_unconstrained/results_UI_1Dparabola_unconstrained_forgetting_factor_0_016681005372000592_0_016681005372000592_*'
name4 = '1D_parabola/UI_choosing_sigma_unconstrained/results_UI_1Dparabola_unconstrained_forgetting_factor_0_021544346900318832_0_021544346900318832*'
name5 = '1D_parabola/UI_choosing_sigma_unconstrained/results_UI_1Dparabola_unconstrained_forgetting_factor_0_027825594022071243_0_027825594022071243*'
name6 = '1D_parabola/UI_choosing_sigma_unconstrained/results_UI_1Dparabola_unconstrained_forgetting_factor_0_03593813663804628_0_03593813663804628*'
name7 = '1D_parabola/UI_choosing_sigma_unconstrained/results_UI_1Dparabola_unconstrained_forgetting_factor_0_046415888336127774_0_046415888336127774*'
name8 = '1D_parabola/UI_choosing_sigma_unconstrained/results_UI_1Dparabola_unconstrained_forgetting_factor_0_05994842503189409_0_05994842503189409*'
name9 = '1D_parabola/UI_choosing_sigma_unconstrained/results_UI_1Dparabola_unconstrained_forgetting_factor_0_0774263682681127_0_0774263682681127*'
name10 = '1D_parabola/UI_choosing_sigma_unconstrained/results_UI_1Dparabola_unconstrained_forgetting_factor_0_1_0_1*'
name11 = '1D_parabola/UI_choosing_sigma_unconstrained/results_UI_1Dparabola_unconstrained_forgetting_factor_0_3_0_3*'
name12 = '1D_parabola/UI_choosing_sigma_unconstrained/results_UI_1Dparabola_unconstrained_forgetting_factor_0_5_0_5*'

names = [name01, name02, name03, name1, name2, name3, name4, name5, name6, name7, name8, name9, name10, name11, name12]

SIGMAS = [0.005, 0.007, 0.009, 0.01, 0.013, 0.0165, 0.0215, 0.028, 0.036, 0.0465, 0.060, 0.077, 0.1, 0.3, 0.5]

results1 = []
for name in names:
    inter_results = []
    files = glob.glob(results_path + name, recursive=True)
    for file in files:
        with open(file, 'rb') as f:
            result = pickle.load(f)
        inter_results.append(result)
    results1.append(inter_results)

name01 = '1D_parabola/UI_choosing_sigma_constrained/results_UI_1Dparabola_constrained_forgetting_factor_0_005_0_005_*'
name02 = '1D_parabola/UI_choosing_sigma_constrained/results_UI_1Dparabola_constrained_forgetting_factor_0_007_0_007_*'
name03 = '1D_parabola/UI_choosing_sigma_constrained/results_UI_1Dparabola_constrained_forgetting_factor_0_009_0_009_*'
name1 = '1D_parabola/UI_choosing_sigma_constrained/results_UI_1Dparabola_constrained_forgetting_factor_0_01_0_01_*'
name2 = '1D_parabola/UI_choosing_sigma_constrained/results_UI_1Dparabola_constrained_forgetting_factor_0_013_0_013*'
name3 = '1D_parabola/UI_choosing_sigma_constrained/results_UI_1Dparabola_constrained_forgetting_factor_0_0165_0_0165*'
name4 = '1D_parabola/UI_choosing_sigma_constrained/results_UI_1Dparabola_constrained_forgetting_factor_0_0215_0_0215*'
name5 = '1D_parabola/UI_choosing_sigma_constrained/results_UI_1Dparabola_constrained_forgetting_factor_0_028_0_028*'
name6 = '1D_parabola/UI_choosing_sigma_constrained/results_UI_1Dparabola_constrained_forgetting_factor_0_036_0_036*'
name7 = '1D_parabola/UI_choosing_sigma_constrained/results_UI_1Dparabola_constrained_forgetting_factor_0_0465_0_0465*'
name8 = '1D_parabola/UI_choosing_sigma_constrained/results_UI_1Dparabola_constrained_forgetting_factor_0_06_0_06*'
name9 = '1D_parabola/UI_choosing_sigma_constrained/results_UI_1Dparabola_constrained_forgetting_factor_0_077_0_077*'
name10 = '1D_parabola/UI_choosing_sigma_constrained/results_UI_1Dparabola_constrained_forgetting_factor_0_1_0_1*'
name11 = '1D_parabola/UI_choosing_sigma_constrained/results_UI_1Dparabola_constrained_forgetting_factor_0_3_0_3*'
name12 = '1D_parabola/UI_choosing_sigma_constrained/results_UI_1Dparabola_constrained_forgetting_factor_0_5_0_5*'

names = [name01, name02, name03, name1, name2, name3, name4, name5, name6, name7, name8, name9, name10, name11, name12]

SIGMAS2 = SIGMAS

results2 = []
for name in names:
    inter_results = []
    files = glob.glob(results_path + name, recursive=True)
    for file in files:
        with open(file, 'rb') as f:
            result = pickle.load(f)
        inter_results.append(result)
    results2.append(inter_results)

dataUI = []
models1 = []
for i, result in enumerate(results1):
    trails = []
    for trail in result:
        chosen_query = trail['trajectory']
        if len(chosen_query.shape) > 1:
            chosen_query = chosen_query[:, 0]
        fx = long_riley_function_linear(torch.from_numpy(chosen_query), torch.from_numpy(t_abtast)).numpy()
        error_each_timestep = abs(fx - fx_star)
        cum_error = np.cumsum(error_each_timestep)
        trails.append(cum_error[-1])
        dataUI.append([SIGMAS[i], cum_error[-1], False])

    trails = np.asarray(trails)
    models1.append(trails)

models2 = []
for i, result in enumerate(results2):
    trails = []
    for trail in result:
        chosen_query = trail['trajectory']
        if len(chosen_query.shape) > 1:
            chosen_query = chosen_query[:, 0]
        fx = long_riley_function_linear(torch.from_numpy(chosen_query), torch.from_numpy(t_abtast)).numpy()
        error_each_timestep = abs(fx - fx_star)
        cum_error = np.cumsum(error_each_timestep)
        trails.append(cum_error[-1])
        dataUI.append([SIGMAS2[i], cum_error[-1], True])

    trails = np.asarray(trails)
    models2.append(trails)

dataUI = pd.DataFrame(dataUI, columns=['Sigma', 'CumRegret', 'Constrained'])
ax2 = plt.subplot2grid((3, 1), (1, 0))
boxenplot = sns.boxenplot(x="Sigma", y="CumRegret", hue="Constrained",
                          data=dataUI, k_depth='full', linewidth=0.75, palette=[c['blau100'], c['gruen100'], ],
                          showfliers=False)
boxenplot.legend(loc='upper center', title='Constrained', edgecolor='white', facecolor='white',
                 framealpha=1, )

for i in range(len(SIGMAS) - 1):
    plt.vlines(i + 0.5, 0, 800, linestyles='-', colors=c['schwarz75'], lw=0.5)
# plot mean trajectory
idx = dataUI.index[dataUI['Constrained']]
data_constrainedUI = dataUI.loc[idx]
data_mean_constrainedUI = data_constrainedUI.groupby('Sigma').mean()
means_constrainedUI = data_mean_constrainedUI[['CumRegret']]
print(np.argmin(means_constrainedUI))
x = np.arange(len(means_constrainedUI)) + 0.2
ax2.plot(x, means_constrainedUI, color=c['gruen100'], marker='o', linestyle='dashed',
         linewidth=0.85, markersize=4, markeredgecolor='white', markeredgewidth=0.75)

idx = dataUI.index[~data['Constrained']]
data_unconstrainedUI = dataUI.loc[idx]
data_mean_unconstrainedUI = data_unconstrainedUI.groupby('Sigma').mean()
x = np.arange(len(means_constrainedUI)) - 0.2
means_unconstrainedUI = data_mean_unconstrainedUI[['CumRegret']]
print(np.argmin(means_unconstrainedUI))
ax2.plot(x, means_unconstrainedUI, color=c['blau100'], marker='o', linestyle='dashed',
         linewidth=0.85, markersize=4, markeredgecolor='white', markeredgewidth=0.75)

plt.xticks(rotation=45)
plt.ylabel('$R_T$')
plt.xlabel('$\\epsilon$')
plt.ylim([0, 800])
plt.xlim([-0.5, 14.5])

ax3 = plt.subplot2grid((3, 1), (2, 0))
x = np.arange(len(means_constrainedUI))
ax3.plot(x, means_unconstrained, color=c['bordeaux100'], marker='o', linestyle='-', markersize=4,
         markeredgecolor='k', markeredgewidth=0.75)
ax3.plot(x, means_constrained, color=c['violett100'], marker='o', linestyle='-', markersize=4,
         markeredgecolor='k', markeredgewidth=0.75)
ax3.plot(x, means_constrainedUI, color=c['gruen100'], marker='o', linestyle='-', markersize=4,
         markeredgecolor='k', markeredgewidth=0.75)
ax3.plot(x, means_unconstrainedUI, color=c['blau100'], marker='o', linestyle='-', markersize=4,
         markeredgecolor='k', markeredgewidth=0.75)
for i in range(len(SIGMAS) - 1):
    plt.vlines(i + 0.5, 0, 300, linestyles='-', colors=c['schwarz75'], lw=0.5)

ax1.text(0, 700, 'B2P forgetting', ha='left', va='center', backgroundcolor='w')
ax2.text(0, 700, 'UI forgetting', ha='left', va='center', backgroundcolor='w')

plt.xlim([-0.5, 14.5])
plt.xticks(x, SIGMAS)
plt.xticks(rotation=45)
plt.ylabel('$R_T$')
plt.xlabel('Forgetting Factor ($\epsilon$/$\hat{\sigma}_w^2$)')
plt.ylim([0, 300])

for ax in fig.get_axes():
    ax.label_outer()

fig.subplots_adjust(bottom=0.135, top=0.97, left=0.1, right=0.97, hspace=0.1)
# plt.show()
plt.savefig('../pgf_figures/Parabola1D_choose_forgetting_factor.pgf', format='pgf')
plt.close()
