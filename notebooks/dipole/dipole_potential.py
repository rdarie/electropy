import matplotlib.pyplot as plt
from cycler import cycler

import seaborn as sns
import pandas as pd
import numpy as np
import vg

##
from electropy.charge import Charge
from electropy.volume import *

charge_val = 1e-7 # Coulombs
position = .5
volume_dim = [3, 1, .2]
z_plane = 0

# charges = [
#     Charge([0, -position, 0], -charge_val),
#     Charge([0, +position, 0], +charge_val)]
charges = [Charge([0, 0, 0], -charge_val)]
h = 0.1
data = potentialSolution(
    charges,
    x_range=[-volume_dim[0], volume_dim[0]],
    y_range=[-volume_dim[1], volume_dim[1]],
    z_range=[-volume_dim[2], volume_dim[2]], verbose=1)

##
phiDF = pd.DataFrame(
    data.phi[:, :, z_plane],
    index=np.round(data.xi, decimals=2),
    columns=np.round(data.yi, decimals=2))
sns.heatmap(phiDF)
plt.show()
directionVector = vg.normalize(np.array([1, 0, 0]))
xMask = slice(None)
yMask = (data.yi == 0)
zMask = data.zi > 0
af = np.inner(
    directionVector,
    np.inner(data.hessianPhi[xMask, yMask, zMask, :, :],
    directionVector))

custom_cycler = (cycler(
    color=list(sns.color_palette("Blues_d",
    n_colors = af.shape[-1]))))
plt.rc('axes', prop_cycle=custom_cycler)
plt.plot(data.xi, np.squeeze(af)); plt.show()

xMask = (data.xi == 0)
yMask = (data.yi == 0)
zMask = slice(None)
af = np.inner(
    directionVector,
    np.inner(data.hessianPhi[xMask, yMask, zMask, :, :],
    directionVector))

custom_cycler = (cycler(
    color=list(sns.color_palette("Blues_d",
    n_colors=1))))
plt.rc('axes', prop_cycle=custom_cycler)
plt.plot(data.zi, np.squeeze(af)); plt.show()