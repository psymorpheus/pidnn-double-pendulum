import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from dp_datagen import iter
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import PillowWriter

with open("dp_config.yaml", "r") as f:
    all_configs = yaml.safe_load(f)
    common_config = all_configs['COMMON'].copy()
    # Filling in models from model templates
    for instance in common_config['ALL_MODEL_CONFIGS']:
        template_name = instance[:instance.rfind('_')]
        training_points = int(instance[(instance.rfind('_')+1):])
        template_config = all_configs[template_name].copy()
        template_config['num_datadriven'] = training_points
        template_config['model_name'] = template_name.lower() + '_' + str(training_points)
        all_configs[template_name + '_' + str(training_points)] = template_config

# theta1 = float(input('Enter initial Theta 1: '))
# theta2 = float(input('Enter initial Theta 2: '))
# tsteps = int(input('Enter number of timesteps: '))
theta1 = 85 * np.pi/180
theta2 = 85 * np.pi/180
tsteps = 1000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

active_data_config_name = 'SIMULATION_80_90'
active_model_config_name = 'FF_800'
noise = 0.00

active_data_config = all_configs[active_data_config_name].copy()
active_data_config.update(common_config)
active_model_config = all_configs[active_model_config_name].copy()
active_model_config.update(active_data_config)
config = active_model_config
config['t_range'] = np.arange(start=0.0, stop = config['TIMESTEP']*tsteps, step = config['TIMESTEP'])
t = config['t_range']

solved_data = iter(theta1, theta2, t, config['g'], config['m1'], config['m2'], config['l1'], config['l2'])
simulator_output = np.hstack([
    np.reshape(t,(-1,1)),
    solved_data[:,[0]],
    solved_data[:,[2]]
])
print("Simulator output obtained!")

nn_input = np.hstack([t.reshape(-1,1), theta1 * np.ones((tsteps,1)), theta2 * np.ones((tsteps,1))])
model = torch.load(f'Models/Noise_{int(noise*100)}/{active_data_config_name.lower()}/{active_model_config_name.lower()}.pt')
model.eval()
nn_output = model(nn_input)
print("NN output obtained!")

# has time, sim theta1, sim theta2, nn theta1, nn theta2
history = np.hstack([simulator_output, nn_output.detach().numpy()])

def get_x1y1x2y2(t, the1, the2, L1=config['l1'], L2=config['l2']):
    return (L1*np.sin(the1),
            -L1*np.cos(the1),
            L1*np.sin(the1) + L2*np.sin(the2),
            -L1*np.cos(the1) - L2*np.cos(the2))

x1, y1, x2, y2 = get_x1y1x2y2(t, history[:,1], history[:,2])
xp1, yp1, xp2, yp2 = get_x1y1x2y2(t, history[:,3], history[:,4])
def animate(i):
    ln1.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
    ln2.set_data([0, xp1[i], xp2[i]], [0, yp1[i], yp2[i]])
    
fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.set_facecolor('k')
ax.get_xaxis().set_ticks([])    # enable this to hide x axis ticks
ax.get_yaxis().set_ticks([])    # enable this to hide y axis ticks
ln1, = plt.plot([], [], 'ro--', lw=3, markersize=8)
ln2, = plt.plot([], [], 'bo--', lw=3, markersize=8)
ax.set_ylim(-4,4)
ax.set_xlim(-4,4)
ani = animation.FuncAnimation(fig, animate, frames=1000, interval=50)
print("Starting animation")
ani.save('pen.gif',writer='pillow',fps=25)
print("Animation finished.")
