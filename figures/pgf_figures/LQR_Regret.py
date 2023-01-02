import numpy as np
import matplotlib as mpl

# Use the pgf backend (must be set before pyplot imported)
mpl.use('pgf')
import matplotlib.pyplot as plt
import pickle
import torch
import glob
from src.objective_functions_LQR import lqr_objective_function_2D, get_params, get_opt_state_controller, \
    get_linearized_model, perform_simulation
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
img_path = '/Users/paulbrunzema/Documents/Studium/Master/Masterarbeit/02_Code/imgs/trajectories/'

TIME_HORIZON = 300
sample_time = 0.02
Q = np.eye(4) * 10
R = np.eye(1)
TRAINING_POINTS = 30
t_abtast = np.arange(TRAINING_POINTS + 1, TIME_HORIZON + 1, 1)
Ks = []
costs = []
for ti in t_abtast:
    test_params = get_params(ti)
    model = get_linearized_model(test_params, sample_time)
    K = get_opt_state_controller(model, Q, R)
    Ks.append(K)
    lqr_cost = perform_simulation(model, K, ti)
    costs.append(lqr_cost)

fx_star = np.asarray(costs)
scale = np.array([5, 0.25])

# initial optimal controller
inital_costs = []
test_params = get_params(0)
model = get_linearized_model(test_params, sample_time)
K_initial = get_opt_state_controller(model, Q, R)
for ti in t_abtast:
    test_params = get_params(ti)
    model = get_linearized_model(test_params, sample_time)
    lqr_cost = perform_simulation(model, K_initial, ti)
    inital_costs.append(lqr_cost)

inital_costs = np.asarray(inital_costs)

data = []
model_stay_trail = []

name1 = f'LQR_problem/B2P_unconstrained/results_B2P_OU_2DLQR_unconstrained_forgetting_factor_0_03_0_03_*'
name2 = f'LQR_problem/UI_unconstrained/results_UI_2DLQR_unconstrained_forgetting_factor_0_03_0_03_*'
name3 = f'LQR_problem/B2P_constrained/results_B2P_OU_2DLQR_constrained_forgetting_factor_0_03_0_03*'
name4 = f'LQR_problem/UI_constrained/results_UI_2DLQR_constrained_forgetting_factor_0_03_0_03_*'
#
name5 = f'LQR_problem/B2P_unconstrained/results_B2P_OU_2DLQR_unconstrained_sliding_window_forgetting_factor_0_03_0_03_*'
name6 = f'LQR_problem/UI_unconstrained/results_UI_2DLQR_unconstrained_binning_forgetting_factor_0_03_0_03_*'
name7 = f'LQR_problem/B2P_constrained/results_B2P_OU_2DLQR_constrained_sliding_window_forgetting_factor_0_03_0_03*'
name8 = f'LQR_problem/UI_constrained/results_UI_2DLQR_constrained_binning_forgetting_factor_0_03_0_03_*'

names = [name1, name5, name2, name6, name3, name7, name4, name8]

# names = [name3, name4]
LEGEND = ['TV-GP-UCB',
          'SW TV-GP-UCB',
          'UI-TVBO',
          'B UI-TVBO',
          'C-TV-GP-UCB',
          'SW C-TV-GP-UCB',
          'C-UI-TVBO',
          'B C-UI-TVBO', ]

COLORS = [c['bordeaux100'],
          c['magenta100'],
          c['blau100'],
          c['petrol100'],
          c['violett100'],
          c['orange100'],
          c['gruen100'],
          c['lila100'], ]

# dic_keys = list(c.keys())[::5]
# COLORS = [c[color] for color in dic_keys]

results = []
for name in names:
    inter_results = []
    files = glob.glob(results_path + name, recursive=True)
    for file in files:
        with open(file, 'rb') as f:
            result = pickle.load(f)
        inter_results.append(result)
    results.append([inter_results, 1])
# start_train = result['start_training']
# min_trajectory = result['min_trajectory']

# models = []
stay_trails = []
for i, (result, model_id) in enumerate(results):
    trails = []
    for j, trail in enumerate(result):
        chosen_query = trail['trajectory'] * scale

        # true regret
        regret_t = lqr_objective_function_2D(torch.from_numpy(chosen_query), torch.from_numpy(t_abtast)).numpy()
        cum_regret = np.cumsum(abs(regret_t.reshape(-1) - fx_star.reshape(-1)))
        # trails.append(cum_regret.reshape(-1))
        # data.append([t_abtast.reshape(-1), cum_regret.reshape(-1), LEGEND[i]])
        data.append([cum_regret.reshape(-1)[-1], model_id, 0, LEGEND[i]])

x, y = set_size(398, subplots=(1, 2), fraction=1.)
fig, ax = plt.subplots(1, figsize=(x, y * 1.3))

# means = []
# for model, stay_trail in zip(models[:-1], model_stay_trail):
#     stay_trail = np.asarray(stay_trail)
#     stay_trail = np.mean(stay_trail[:, -1], axis=0)
#     means.append(stay_trail)
#
for i in range(len(LEGEND) - 1):
    plt.vlines(i + 0.5, 0, 100, linestyles='-', colors=c['schwarz75'], lw=0.5)

# plt.hlines(np.asarray(means).mean(), -1, 20, linestyles='--', colors=c['schwarz75'])
data = pd.DataFrame(data, columns=['Cumregret', 'ModelID', 'Prior Mean', 'Name', ])
boxplots = sns.boxplot(x="Name", y="Cumregret",
                       data=data,
                       palette=COLORS,
                       showmeans=True,
                       linewidth=0.75,
                       fliersize=2,
                       meanprops={"marker": "o",
                                  "markerfacecolor": "white",
                                  "markeredgecolor": "black",
                                  "markersize": "4"}
                       )  # k_depth='full', hue_order=hue_order, split=True, inner="stick", bw=.5)

for color, boxplot in zip(COLORS, boxplots.artists):
    boxplot.set_facecolor(color)

ax.set_ylabel('$R_T$')
plt.xticks(rotation=45)
# ax.set_xlabel('Time $t$')
#
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_elements = []
initial_regret = np.cumsum(inital_costs - fx_star)[-1]
K_0 = '\mathbf{K}_0^*'
legend_elements.append(
    Line2D([0], [0], color=c['schwarz75'], lw=1, linestyle='--',
           label=f'Initial ${K_0}$\n $R_T = {initial_regret:.03f}$'))
for label, color in zip(LEGEND, COLORS):
    legend_elements.append(Patch(facecolor=color, edgecolor=None,
                                 label=label))

legend1 = plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
ax.add_artist(legend1)

plt.xticks([])
plt.xlabel('')
plt.ylim([0, 70])
fig.subplots_adjust(bottom=0.05, top=0.95, left=0.1, right=0.62)
# plt.show()

plt.savefig('../pgf_figures/LQR_Regret.pgf', format='pgf')
plt.close()

print('hallo')
