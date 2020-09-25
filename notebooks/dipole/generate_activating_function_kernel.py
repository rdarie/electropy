import matplotlib.pyplot as plt
from cycler import cycler
from itertools import product
#
import seaborn as sns
import pandas as pd
import numpy as np
import vg
import pdb
from scipy import constants
import dill as pickle
import os
##
from electropy.charge import Charge
from electropy.volume import potentialSolution

##############################################################################################
currentUnits = 1e-6 # uA, <A>=<C/sec>
sigma = 0.23 # <1/(ohm * m)>, grey matter resistivity from capogrosso
# current/conductivity = charge/epsilon_0
eps0 = constants.epsilon_0 # <F/m> = <sec/(ohm*m)>
chargeVal = currentUnits * eps0 / sigma # C
#
electrodeSize = [1e-3, 4e-3]
electrodePosition = [0, 0]
h = 2e-3
electrode_h = .5e-3
volume_dim = [25e-3, 40e-3, 10e-3]
charges = []

dX = np.arange(0, electrodeSize[0] * 1.01, electrode_h) - electrodeSize[0] / 2
dY = np.arange(0, electrodeSize[1] * 1.01, electrode_h) - electrodeSize[1] / 2
nElements = dX.size * dY.size
for x in dX:
    for y in dY:
        xPos = electrodePosition[0] + x
        yPos = electrodePosition[1] + y
        print('placing charge at {}, {}'.format(xPos, yPos))
        charges.append(Charge(
            [xPos, yPos, 0], chargeVal / nElements))
# [ch.position[0] for ch in charges]
data = potentialSolution(
    charges,
    x_range=[-volume_dim[0], volume_dim[0]],
    y_range=[-volume_dim[1], volume_dim[1]],
    z_range=[0, volume_dim[2]],
    h=h, verbose=1)

kernelPath = './activating_kernel.pickle'
if os.path.exists(kernelPath):
    os.remove(kernelPath)
with open(kernelPath, 'wb') as f:
    pickle.dump(data, f)

plotting = True
if plotting:
    z_plane = 5e-3
    ##
    xMask = slice(None)
    yMask = slice(None)
    zMask = np.abs(data.zi - z_plane) < 1e-9
    phiDF = pd.DataFrame(
        np.round(np.squeeze(data.phi[xMask, yMask, zMask]), decimals=6),
        index=np.round(data.xi * 1e3, decimals=2),
        columns=np.round(data.yi * 1e3, decimals=2))
    ax = sns.heatmap(phiDF.T, square=True)
    ax.set_title('Extracellular voltage')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    plt.show()

    directionVector = vg.normalize(np.array([1, 0, 0]))
    xMask = slice(None)
    yMask = np.abs(data.yi) < 1e-9 
    zMask = np.abs(data.zi - z_plane) < 1e-9
    af = data.getActivatingFunction(
        xMask, yMask, zMask,
        directionVector)
    #
    custom_cycler = (cycler(
        color=list(sns.color_palette("Blues_d", n_colors=af.shape[-1]))))
    plt.rc('axes', prop_cycle=custom_cycler)
    plt.plot(data.xi, np.squeeze(af))
    plt.title('Activating function in the direction {}'.format(directionVector))
    plt.xlabel('X (mm)')
    plt.ylabel('AF')
    plt.show()

    xMask = np.abs(data.xi) < 1e-9
    yMask = slice(None)
    zMask = np.abs(data.zi - z_plane) < 1e-9
    af = data.getActivatingFunction(
        xMask, yMask, zMask,
        directionVector)

    custom_cycler = (cycler(
        color=list(sns.color_palette("Blues_d", n_colors=1))))
    plt.rc('axes', prop_cycle=custom_cycler)
    plt.plot(data.yi, np.squeeze(af))
    plt.title('Activating function in the direction {}'.format(directionVector))
    plt.xlabel('Y (mm)')
    plt.ylabel('AF')
    plt.show()

    directionVector = vg.normalize(np.array([1, 1, 0]))
    xMask = slice(None)
    yMask = slice(None)
    zMask = np.abs(data.zi - z_plane) < 1e-9
    af = data.getActivatingFunction(
        xMask, yMask, zMask,
        directionVector)
    afDF = pd.DataFrame(
        np.squeeze(af),
        index=np.round(data.xi * 1e3, decimals=2),
        columns=np.round(data.yi * 1e3, decimals=2))
    ax = sns.heatmap(afDF.T, square=True)
    ax.set_title('Activating function in the direction {}'.format(directionVector))
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    plt.show()
