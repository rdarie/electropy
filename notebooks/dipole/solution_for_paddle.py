import matplotlib.pyplot as plt
from cycler import cycler
from itertools import product
#
import seaborn as sns
import pandas as pd
import numpy as np
import vg
import pdb
import os
import dill as pickle
from scipy import constants, ndimage
##
from electropy.charge import Charge
from electropy.volume import *

with open('./paddle_solution.pickle', 'rb') as f:
    paddleSolution = pickle.load(f)
    solnPerElectrode = paddleSolution['solnPerElectrode']
    activatingKernel = paddleSolution['activatingKernel']
dummySoln = solnPerElectrode[list(solnPerElectrode.keys())[0]]


def solnForWaveform(eesWvf):
    data = potentialSolution(
        [],
        x_range=[dummySoln.xi[0], dummySoln.xi[-1]],
        y_range=[dummySoln.yi[0], dummySoln.yi[-1]],
        z_range=[0, dummySoln.zi[-1]],
        h=dummySoln.h, verbose=1)
    for rowIdx in eesWvf.index:
        elecName = rowIdx[0]
        if eesWvf[rowIdx] != 0:
            data.hessianPhi += eesWvf[rowIdx] * solnPerElectrode[elecName].hessianPhi
            data.phi += eesWvf[rowIdx] * solnPerElectrode[elecName].phi
    return data

            
inputPath = 'E:\\Neural Recordings\\scratch\\202009231400-Peep\\default\\stim\\_emg_XS_export.h5'
with pd.HDFStore(inputPath, 'r') as store:
    # each trial has its own eesKey, get list of all
    allEESKeys = [
        i
        for i in store.keys()
        if ('stim' in i)]
    # allocate lists to hold data from each trial
    eesList = []
    for idx, eesKey in enumerate(sorted(allEESKeys)):
        # print(eesKey)
        # data for this trial is stored in a pd.dataframe
        stimData = pd.read_hdf(store, eesKey)
        # print((stimData.abs() > 0).any())
        # metadata is stored in a dictionary
        eesMetadata = store.get_storer(eesKey).attrs.metadata
        # extract column names from first trial
        if idx == 0:
            eesColumns = [cn[0] for cn in stimData.columns if cn[1] == 'amplitude']
            metadataColumns = sorted([k for k in eesMetadata.keys()])
            eesColIdx = [cn for cn in stimData.columns if cn[1] == 'amplitude']
            metaDataDF = pd.DataFrame(
                None, index=range(len(allEESKeys)),
                columns=metadataColumns)
        metaDataDF.loc[idx, :] = eesMetadata
        eesList.append(stimData.loc[:, eesColIdx])

trialIdx = 1
eesInfo = metaDataDF.loc[trialIdx, :]
ees = eesList[trialIdx]

eesWaveform = pd.Series(0, index=ees.columns)
eesWaveform.loc[('caudalX_e06', 'amplitude')] = -750
eesWaveform.loc[('caudalX_e02', 'amplitude')] = 750
data = solnForWaveform(eesWaveform)

xCoord = 0
yCoord = 1e-3
zCoord = 4e-3
## data = solnPerElectrode['caudalX_e01_a']
xMask = slice(None)
yMask = slice(None)
zMask = np.abs(data.zi - zCoord) < 1e-9
phiDF = pd.DataFrame(
    np.round(np.squeeze(data.phi[xMask, yMask, zMask]), decimals=6),
    index=np.round(data.xi * 1e3, decimals=2),
    columns=np.round(data.yi * 1e3, decimals=2)).sort_index(axis=1, ascending=False)
ax = sns.heatmap(phiDF.T, square=True)
ax.set_title('Extracellular voltage')
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
plt.show()

directionVector = vg.normalize(np.array([1, 0, 0]))
xMask = slice(None)
yMask = np.abs(data.yi - yCoord) < 1e-9 
zMask = np.abs(data.zi - zCoord) < 1e-9
af = data.getActivatingFunction(
    xMask, yMask, zMask,
    directionVector)
custom_cycler = (cycler(
    color=list(sns.color_palette("Blues_d", n_colors=af.shape[-1]))))
plt.rc('axes', prop_cycle=custom_cycler)
plt.plot(data.xi, np.squeeze(af))
plt.title('Activating function in the direction {}'.format(directionVector))
plt.xlabel('X (mm)')
plt.ylabel('AF')
plt.show()

xMask = np.abs(data.xi - xCoord) < 1e-9
yMask = slice(None)
zMask = np.abs(data.zi - zCoord) < 1e-9
af = data.getActivatingFunction(
    xMask, yMask, zMask,
    directionVector)
custom_cycler = (cycler(
    color=list(sns.color_palette("Blues_d", n_colors=1))))
plt.rc('axes', prop_cycle=custom_cycler)
plt.plot(data.yi, np.squeeze(af))
plt.title(
    'Activating function in the direction {}'
    .format(directionVector))
plt.xlabel('Y (mm)')
plt.ylabel('AF')
plt.show()

directionVector = vg.normalize(np.array([0, 1, 0]))
xMask = slice(None)
yMask = slice(None)
zMask = np.abs(data.zi - zCoord) < 1e-9
af = data.getActivatingFunction(
    xMask, yMask, zMask,
    directionVector)
afDF = pd.DataFrame(
    np.squeeze(af),
    index=np.round(data.xi * 1e3, decimals=2),
    columns=np.round(data.yi * 1e3, decimals=2)).sort_index(axis=1, ascending=False)
ax = sns.heatmap(afDF.T, square=True)
ax.set_title('Activating function in the direction {} at Z = {}'.format(directionVector, zCoord))
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
plt.show()