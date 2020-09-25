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


def mapToDF(arrayFilePath):
    arrayMap = pd.read_csv(
        arrayFilePath, sep='; ',
        skiprows=10, header=None, engine='python',
        names=['FE', 'electrode', 'position'])
    cmpDF = pd.DataFrame(
        np.nan, index=range(146),
        columns=[
            'xcoords', 'ycoords', 'zcoords', 'elecName',
            'elecID', 'label', 'bank', 'bankID', 'nevID']
        )
    bankLookup = {'A.1': 0, 'A.2': 1, 'A.3': 2}
    for rowIdx, row in arrayMap.iterrows():
        processor, port, FEslot, channel = row['FE'].split('.')
        bankName = '{}.{}'.format(port, FEslot)
        array, electrodeFull = row['electrode'].split('.')
        if '_' in electrodeFull:
            electrode, electrodeRep = electrodeFull.split('_')
        else:
            electrode = electrodeFull
        x, y, z = row['position'].split('.')
        nevIdx = int(channel) - 1 + bankLookup[bankName] * 32
        cmpDF.loc[nevIdx, 'elecID'] = int(electrode[1:])
        cmpDF.loc[nevIdx, 'nevID'] = nevIdx
        cmpDF.loc[nevIdx, 'elecName'] = array
        cmpDF.loc[nevIdx, 'xcoords'] = float(x)
        cmpDF.loc[nevIdx, 'ycoords'] = float(y)
        cmpDF.loc[nevIdx, 'zcoords'] = float(z)
        cmpDF.loc[nevIdx, 'label'] = row['electrode'].replace('.', '_')
        cmpDF.loc[nevIdx, 'bank'] = bankName
        cmpDF.loc[nevIdx, 'bankID'] = int(channel)
        cmpDF.loc[nevIdx, 'FE'] = row['FE']
    #
    cmpDF.dropna(inplace=True)
    cmpDF.reset_index(inplace=True, drop=True)
    # import pdb; pdb.set_trace()
    # xIdx = np.array(
    #     cmpDF['xcoords'].values - cmpDF['xcoords'].min(),
    #     dtype=np.int)
    # yIdx = np.array(
    #     cmpDF['ycoords'].values - cmpDF['ycoords'].min(),
    #     dtype=np.int)
    # cmpDF.loc[:, 'nevID'] += 1
    return cmpDF


mapPath = 'C:\\Users\\Radu\\Documents\\GitHub\\Data-Analysis\\dataAnalysis\\analysis-code\\isi_nano1caudal_xAyBzC_ortho_nano2rostral_xAyBzC_ortho.map'
mapDF = mapToDF(mapPath)
mapDF = mapDF.loc[mapDF['label'].str.endswith('_a'), :]
mapDF.loc[:, 'xcoords'] = mapDF['xcoords'] * 1e-4
mapDF.loc[:, 'ycoords'] = mapDF['ycoords'] * 1e-4
mapDF.loc[mapDF['elecName'].str.contains('rostral'), 'ycoords'] += 5e-2
mapDF.loc[:, 'ycoords'] -= (mapDF.loc[:, 'ycoords'].max() - mapDF.loc[:, 'ycoords'].min())/2
mapDF.loc[:, 'xcoords'] -= (mapDF.loc[:, 'xcoords'].max() - mapDF.loc[:, 'xcoords'].min())/2

with open('./activating_kernel.pickle', 'rb') as f:
    activatingKernel = pickle.load(f)

volume_dim = [
    np.round((mapDF.loc[:, 'xcoords'].max() - mapDF.loc[:, 'xcoords'].min()), decimals=3),
    np.round((mapDF.loc[:, 'ycoords'].max() - mapDF.loc[:, 'ycoords'].min()) / 2, decimals=3),
    10e-3]

solnPerElectrode = {}
for idx, row in mapDF.iterrows():
    data = potentialSolution(
        [],
        x_range=[-volume_dim[0], volume_dim[0]],
        y_range=[-volume_dim[1], volume_dim[1]],
        z_range=[0, volume_dim[2]],
        h=activatingKernel.h, verbose=1)
    eXi = np.argmin(np.abs(data.xi - row['xcoords']))
    eYi = np.argmin(np.abs(data.yi - row['ycoords']))
    eZi = np.argmin(np.abs(data.zi - row['zcoords']))
    inputVol = np.zeros(data.phi.shape)
    inputVol[eXi, eYi, eZi] = 1
    data.phi += ndimage.convolve(
        inputVol, activatingKernel.phi, mode='constant')
    for firstDer in ['dx', 'dy', 'dz']:
        data.DPhi[firstDer] += ndimage.convolve(
            inputVol, activatingKernel.DPhi[firstDer], mode='constant')
    for i in range(3):
        for j in range(3):
            kernel = activatingKernel.hessianPhi[:, :, :, i, j]
            data.hessianPhi[:, :, :, i, j] += ndimage.convolve(
                inputVol, kernel, mode='constant')
    solnPerElectrode[row['label'][:-2]] = data

kernelPath = './paddle_solution.pickle'
if os.path.exists(kernelPath):
    os.remove(kernelPath)
with open(kernelPath, 'wb') as f:
    pickle.dump(
        {
            'activatingKernel': activatingKernel,
            'solnPerElectrode': solnPerElectrode
        },
        f)

plotting = True
if plotting:
    plotSolution = solnPerElectrode['caudalX_e01']
    z_plane = 4e-3
    ##
    xMask = slice(None)
    yMask = slice(None)
    zMask = np.abs(plotSolution.zi - z_plane) < 1e-9
    phiDF = pd.DataFrame(
        np.round(np.squeeze(plotSolution.phi[xMask, yMask, zMask]), decimals=6),
        index=np.round(plotSolution.xi * 1e3, decimals=2),
        columns=np.round(plotSolution.yi * 1e3, decimals=2))
    ax = sns.heatmap(phiDF.T, square=True)
    ax.set_title('Extracellular voltage')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    plt.show()

    directionVector = vg.normalize(np.array([1, 0, 0]))
    xMask = slice(None)
    yMask = np.abs(plotSolution.yi - 1e-3) < 1e-9 
    zMask = np.abs(plotSolution.zi - z_plane) < 1e-9
    af = plotSolution.getActivatingFunction(
        xMask, yMask, zMask,
        directionVector)
    #
    custom_cycler = (cycler(
        color=list(sns.color_palette("Blues_d", n_colors=af.shape[-1]))))
    plt.rc('axes', prop_cycle=custom_cycler)
    plt.plot(plotSolution.xi, np.squeeze(af))
    plt.title('Activating function in the direction {}'.format(directionVector))
    plt.xlabel('X (mm)')
    plt.ylabel('AF')
    plt.show()

    xMask = np.abs(plotSolution.xi) < 1e-9
    yMask = slice(None)
    zMask = np.abs(plotSolution.zi - z_plane) < 1e-9
    af = plotSolution.getActivatingFunction(
        xMask, yMask, zMask,
        directionVector)

    custom_cycler = (cycler(
        color=list(sns.color_palette("Blues_d", n_colors=1))))
    plt.rc('axes', prop_cycle=custom_cycler)
    plt.plot(plotSolution.yi, np.squeeze(af))
    plt.title('Activating function in the direction {}'.format(directionVector))
    plt.xlabel('Y (mm)')
    plt.ylabel('AF')
    plt.show()

    directionVector = vg.normalize(np.array([1, 1, 0]))
    xMask = slice(None)
    yMask = slice(None)
    zMask = np.abs(plotSolution.zi - z_plane) < 1e-9
    af = plotSolution.getActivatingFunction(
        xMask, yMask, zMask,
        directionVector)
    afDF = pd.DataFrame(
        np.squeeze(af),
        index=np.round(plotSolution.xi * 1e3, decimals=2),
        columns=np.round(plotSolution.yi * 1e3, decimals=2))
    ax = sns.heatmap(afDF.T, square=True)
    ax.set_title('Activating function in the direction {}'.format(directionVector))
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    plt.show()
