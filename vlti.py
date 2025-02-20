#!/usr/bin/env python3
"""
vlti.py

collection of functions to make calculations regarding the Very Large Telescope
Interferometer (VLTI).

# gallactic center:

vlti.nTelescopes(['U1DL1', 'U2DL2', 'U3DL3', 'U4DL4'], (17.75, -29.01),
        17.75+np.linspace(-6,6,100), plot=1, DLconstraints={'max1':100})
vlti.nTelescopes(['U1DL5', 'U2DL2', 'U3DL3', 'U4DL4'], (17.75, -29.01),
        17.75+np.linspace(-6,6,100), plot=2)

vlti.nTelescopes(['A0DL1', 'G1DL5', 'J2DL4', 'K0DL4'], (17.75, -29.01),
        17.75+np.linspace(-6,6,100), plot=1, DLconstraints={'max1':100})

# -- effect of increasing DL3 and DL4 up to the end of the DL tunnel
C = ['A0DL6IP1', 'G1DL5IP3', 'J2DL4IP5', 'J3DL3IP7']
C = ['K0DL4IP1', 'G2DL2IP3', 'D0DL1IP5', 'J3DL3IP7']
C = ['B5DL1IP1', 'J6DL3IP3']
vlti._test_dl34_plus30=0; vlti.skyCoverage(C, fig=1);
vlti._test_dl34_plus30=1; vlti.skyCoverage(C, fig=2);

"""
__author__ = 'Antoine Merand'
__email__ = "amerand@eso.org"
__version__ = "1.1"
__date__ = 'Sat Apr  9 23:42:36 CLT 2016'

import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.optimize
import scipy.special

import itertools
#import simbad
from astroquery.simbad import Simbad
from astropy import log
log.setLevel('ERROR')

try:
    #from matplotlib.font_manager import _rebuild; _rebuild()
    #plt.rc('font', family='monofur', size=11, style='normal')
    plt.rc('font', size=11, style='normal')
except:
    pass

# -- P94
# config = {'small':  ['A1DL5','C1DL2','D0DL6','B2DL1'],
#           'medium': ['H0DL4','I1DL3','D0DL5','G1DL6'],
#           'large':  ['A1DL5','J3DL3','K0DL4','G1DL6'],}

# -- P96: AT1,2,3,4
config = {'small':       ['A0DL6','B2DL5','D0DL1','C1DL2'],
          'medium':      ['K0DL4','G2DL2','D0DL1','J3DL3'],
          'new medium':  ['A0', 'G1', 'D0', 'B5' ],
          'large':       ['A0DL6','G1DL5','J2DL4','J3DL3'],
          'astrometric': ['A0DL1','G1DL5','J2DL4','K0DL3'],
          'extended':    ['A0', 'J6', 'J2', 'B5']}

# --     station   U        V        E        N       A0
# -- some problems found by G. Duvert in the 'A0' terms (last col):
# @wvgvlti:Appl_data:VLTI:vltiParameter:telStations.nomCentre
# -- altitude: W=4.5397 fot ATs, W=13.044 for the UTs
layout_orientation = -18.984 # degrees
layout = {'A0':(-32.0010, -48.0130, -14.6416, -55.8116, 129.8495),
          'A1':(-32.0010, -64.0210, -9.4342, -70.9489, 150.8475),
          'B0':(-23.9910, -48.0190, -7.0653, -53.2116, 126.8355),
          'B1':(-23.9910, -64.0110, -1.8631, -68.3338, 142.8275),
          'B2':(-23.9910, -72.0110, 0.7394, -75.8987, 150.8275, ),
          'B3':(-23.9910, -80.0290, 3.3476, -83.4805, 158.8455),
          'B4':(-23.9910, -88.0130, 5.9449, -91.0303, 166.8295),
          'B5':(-23.9910, -96.0120, 8.5470, -98.5942, 174.8285),
          'C0':(-16.0020, -48.0130, 0.4872, -50.6071, 118.8405),
          'C1':(-16.0020, -64.0110, 5.6914, -65.7349, 134.8385),
          'C2':(-16.0020, -72.0190, 8.2964, -73.3074, 142.8465),
          'C3':(-16.0020, -80.0100, 10.8959, -80.8637, 150.8375),
          'D0':(0.0100, -48.0120, 15.6280, -45.3973, 97.8375),
          'D1':(0.0100, -80.0150, 26.0387, -75.6597, 134.8305),
          'D2':(0.0100, -96.0120, 31.2426, -90.7866, 150.8275),
          'E0':(16.0110, -48.0160, 30.7600, -40.1959, 81.8405),
          'G0':(32.0170, -48.0172, 45.8958, -34.9903, 65.8357),
          'G1':(32.0200, -112.0100, 66.7157, -95.5015, 129.8255),
          'G2':(31.9950, -24.0030, 38.0630, -12.2894, 73.0153),
          'H0':(64.0150, -48.0070, 76.1501, -24.5715, 58.1953),
          'I1':(72.0010, -87.9970, 96.7106, -59.7886, 111.1613),
          'I2':(80, -24, 83.456, 3.330, 90),# -- XY are correct, A0 is guessed!
          'J1':(88.0160, -71.9920, 106.6481, -39.4443, 111.1713),
          'J2':(88.0160, -96.0050, 114.4596, -62.1513, 135.1843),
          'J3':(88.0160, 7.9960, 80.6276, 36.1931, 124.4875),
          'J4':(88.0160, 23.9930, 75.4237, 51.3200, 140.4845),
          'J5':(88.0160, 47.9870, 67.6184, 74.0089, 164.4785),
          'J6':(88.0160, 71.9900, 59.8101, 96.7064, 188.4815),
          'K0':(96.0020, -48.0060, 106.3969, -14.1651, 90.1813),
          'L0':(104.0210, -47.9980, 113.9772, -11.5489, 103.1823),
          'M0':(112.0130, -48.0000, 121.5351, -8.9510, 111.1763),
          'U1':(-16.0000, -16.0000, -9.9249, -20.3346, 189.0572),
          'U2':(24.0000, 24.0000, 14.8873, 30.5019, 190.5572),
          'U3':(64.0000, 48.0000, 44.9044, 66.2087, 199.7447),
          'U4':(112.0000, 8.0000, 103.3058, 43.9989, 209.2302),
          #'U5':[88.016, -96-4.2*16,     0,        0,      270],
          #'N0':[188,     -48,     0,        0, 188],
          #'J7':[ 88,    -167,     0,        0, 200]
          }
if 'U5' in layout:
    layout['U5'][2] = np.cos(layout_orientation*np.pi/180)*layout['U5'][0]+ \
                      np.sin(layout_orientation*np.pi/180)*layout['U5'][1]

    layout['U5'][3] = -np.sin(layout_orientation*np.pi/180)*layout['U5'][0]+ \
                      np.cos(layout_orientation*np.pi/180)*layout['U5'][1]

    # -- check U5 OPL0 makes sense
    # lab = (50, -48)
    # dl_tunnel_v = -36
    # for k in ['U1', 'U2', 'U3', 'U4', 'U5']:
    #     L = np.abs(layout[k][1]-dl_tunnel_v)
    #     L += np.sqrt((layout[k][0]-lab[0])**2+(dl_tunnel_v-lab[1])**2)
    #     print(k, layout[k][-1]-L)

# -- in U,V
# -- from Pierre Kervella Thesis, fig 14
road1 = (
-87.63428571428571, -141.94871794871796,
-69.94285714285714, -145.64102564102564,
-53.89714285714285, -149.33333333333331,
-39.90857142857143, -153.43589743589743,
-24.685714285714287, -157.1282051282051,
-6.171428571428572, -159.17948717948718,
11.52, -160.82051282051282,
26.742857142857144, -158.35897435897436,
47.31428571428572, -156.7179487179487,
65.00571428571429, -149.74358974358975,
80.64, -138.25641025641025,
93.80571428571429, -123.8974358974359,
103.26857142857142, -107.48717948717949,
111.90857142857142, -91.07692307692308,
121.37142857142857, -74.66666666666666,
128.77714285714285, -60.30769230769231,
135.77142857142857, -48,
138.24, -38.97435897435897,
146.88, -27.076923076923077,
150.17142857142858, -8.615384615384615,
149.76, 5.333333333333333,
143.17714285714285, 21.333333333333332,
132.48, 32.41025641025641)
road2 = (23.04, -157.1282051282051,
20.57142857142857, -146.87179487179486,
12.754285714285714, -141.1282051282051,
3.2914285714285714, -137.84615384615384,
-6.994285714285714, -134.15384615384616,
-18.925714285714285, -127.17948717948717,
-29.21142857142857, -120.61538461538461,
-38.26285714285714, -114.05128205128204,
-51.42857142857143, -103.7948717948718,
-61.714285714285715, -94.35897435897435,
-72, -85.33333333333333,
-79.40571428571428, -75.07692307692308,
-83.10857142857142, -63.589743589743584,
-81.46285714285715, -50.87179487179487,
-77.76, -38.97435897435897,
-73.23428571428572, -29.94871794871795)

edge = (-86.81142857142856, -153.84615384615384,
-74.88, -157.1282051282051,
-62.12571428571429, -160.82051282051282,
-49.78285714285714, -164.9230769230769,
-36.205714285714286, -169.43589743589743,
-18.925714285714285, -176,
99.97714285714285, -180.9230769230769,
111.49714285714286, -170.25641025641025,
120.96, -161.64102564102564,
133.71428571428572, -149.74358974358975,
145.2342857142857, -137.84615384615384,
152.64, -128,
164.57142857142856, -109.12820512820512,
173.21142857142857, -97.23076923076923,
178.97142857142856, -82.46153846153845,
183.49714285714285, -67.6923076923077)
cb = (-68.70857142857143, -36.92307692307692,
-55.13142857142857, -41.02564102564102,
-60.89142857142857, -60.30769230769231,
-62.12571428571429, -64.41025641025641,
-60.89142857142857, -67.6923076923077,
-39.90857142857143, -89.84615384615384,
-50.19428571428571, -98.87179487179486,
-72.82285714285715, -76.3076923076923,
-75.70285714285714, -70.56410256410255,
-77.34857142857143, -64.41025641025641,
-76.52571428571429, -58.256410256410255,
-68.70857142857143, -36.92307692307692)


# @wvgvlti:Appl_data:VLTI:vltiParameter:optics.delayLineParams
# --            U      Vin      Vout
DL_UV = {'DL1':(58.92, -37.245, -37.005),
         'DL2':(58.92, -37.995, -37.755),
         'DL3':(45.08, -38.505, -38.745),
         'DL4':(45.08, -39.255, -39.495),
         'DL5':(58.92, -40.745, -40.505),
         'DL6':(58.92, -41.495, -41.255)}

# @wvgvlti:Appl_data:VLTI:vltiParameter:optics.LabInputCoord
# -- U coordinates of IP 1 through 8:
IP_U = {'IP1':52.32, 'IP2':52.56, 'IP3':52.80, 'IP4':53.04,
        'IP5':53.28, 'IP6':53.52, 'IP7':53.76, 'IP8':54.00}

# @wvgvlti:Appl_data:VLTI:vltiParameter:optics.SwitchYardParams
# -- OPL per input: direct, BC, DDL, BC+DDL
OPL = {'IP1': (2.105, 11.5625, 11.5298, 20.9873),
       'IP2': (1.385, 12.7625,  9.6098, 20.9873),
       'IP3': (2.825, 11.8025, 11.2898, 20.2673),
       'IP4': (2.105, 13.0025,  9.3698, 20.2673),
       'IP5': (3.545, 12.0425, 11.0498, 19.5473),
       'IP6': (2.825, 13.2425,  9.1298, 19.5473),
       'IP7': (4.265, 12.2825, 10.8098, 18.8273),
       'IP8': (3.545, 13.4825,  8.8898, 18.8273),}

# ---------- horizons ---------------
# -- seems to come from VLT-TRE-ESO-15000-2551
# -- https://pdm.eso.org/kronodoc/1100/Get/265745/VLT-TRE-ESO-15000-2551_1.PDF
# hfiles = os.listdir('VLTI_horizons')
# hfiles = filter(lambda x: '.horizon' in x, hfiles)
# horizon = {}
# horCorrection = 0
# for h in hfiles:
#     f = open(os.path.join('VLTI_horizons', h), 'r')
#     lines = f.read().split('\n')[:-3] # last 2 lines are weird
#     az = np.array([float(l.split()[0]) for l in lines])
#     el = np.array([float(l.split()[1]) for l in lines]) + horCorrection
#     horizon[h[:2]] = (az, el)
#     f.close()

# -- use custom computation hzlib.tabulateAll()
horizon = None
def loadHorizon(filename=None):
    global horizon
    if filename is None:
        #filename = 'newVltiHzn.txt'
        filename = 'newVltiHzn_obsDoors.txt'

        if not os.path.exists(filename):
            #filename = '/Users/amerand/Codes/PYTHON3/PLANNINGTOOLS/newVltiHzn.txt'
            filename = '/Users/amerand/Codes/PYTHON3/PLANNINGTOOLS/newVltiHzn_obsDoors.txt'

    f = open(filename, 'r')
    print('- reading VLTI horizons from '+filename)
    for l in f.readlines():
        if not l.startswith('#'):
            az.append(float(l.split()[0]))
            for i, k in enumerate(keys):
                horizon[k].append(float(l.split()[i+1]))
        elif l.startswith('# az'):
            keys = l.split('# az')[1].split()
            az = []
            horizon = {k:[] for k in keys}
        else:
            pass
    f.close()
    for k in keys:
        horizon[k] = (az, horizon[k])

loadHorizon()
for k in layout:
    if not k in horizon:
        horizon[k] = (np.arange(360), np.zeros(360))

a = layout_orientation*np.pi/180.

for s in ['N0', 'J7']:
    if s in layout.keys():
        layout[s][2] =  layout[s][0]*np.cos(a) + layout[s][1]*np.sin(a)
        layout[s][3] = -layout[s][0]*np.sin(a) + layout[s][1]*np.cos(a)
        horizon[s] = (np.linspace(0,360,100), np.zeros(100))
baseline = lambda t1,t2: np.array(layout[t2])[2:4]-np.array(layout[t1])[2:4]

def allB(listT):
    res = []
    for i,t1 in enumerate(listT[:-1]):
        for t2 in listT[i+1:]:
            res.append(np.sqrt(np.sum(baseline(t1, t2)**2)))
    return res


# ------ EVM horizon:
# -- elevation, az1, az2, elevation, az1, az2
evm = {'A0':  ( 48. , 164.,  210. ,   0. ,    0. ,    0. ),
    'A1':  ( 33. , 162.,  195. ,   0. ,    0. ,    0. ),
    'B0':  ( 52. , 150.,  198. ,   0. ,    0. ,    0. ),
    'B1':  ( 35. , 153.,  187. ,   0. ,    0. ,    0. ),
    'B2':  (  0. ,   0.,    0. ,   0. ,    0. ,    0. ),
    'B3':  (  0. ,   0.,    0. ,   0. ,    0. ,    0. ),
    'B4':  (  0. ,   0.,    0. ,   0. ,    0. ,    0. ),
    'B5':  (  0. ,   0.,    0. ,   0. ,    0. ,    0. ),
    'C0':  ( 53. , 136.,  185. ,   0. ,    0. ,    0. ),
    'C1':  ( 35. , 143.,  178. ,   0. ,    0. ,    0. ),
    'C2':  (  0. ,   0.,    0. ,   0. ,    0. ,    0. ),
    'C3':  (  0. ,   0.,    0. ,   0. ,    0. ,    0. ),
    'D0':  ( 48. , 111.,  156. ,   0. ,    0. ,    0. ),
    'D1':  (  0. ,   0.,    0. ,   0. ,    0. ,    0. ),
    'D2':  (  0. ,   0.,    0. ,   0. ,    0. ,    0. ),
    'E0':  ( 37. ,  97.,  134. ,   0. ,    0. ,    0. ),
    'G0':  (  0. ,   0.,    0. ,   0. ,    0. ,    0. ),
    'G1':  (  0. ,   0.,    0. ,   0. ,    0. ,    0. ),
    'G2':  ( 35. ,  63.,   97. ,  35. ,  134. ,  168. ),
    'H0':  (  0. ,   0.,    0. ,   0. ,    0. ,    0. ),
    'I1':  (  0. ,   0.,    0. ,   0. ,    0. ,    0. ),
    'J1':  (  0. ,   0.,    0. ,   0. ,    0. ,    0. ),
    'J2':  (  0. ,   0.,    0. ,   0. ,    0. ,    0. ),
    'J3':  ( 36. , 112.,  147. ,  68. ,  219. ,  282. ),
    'J4':  ( 50. ,  92.,  139. ,  59. ,  257. ,  311. ),
    'J5':  ( 68. ,  39.,  102. ,  36. ,  292. ,  327. ),
    'J6':  ( 50. ,   2.,   49. ,   0. ,    0. ,    0. ),
    'K0':  (  0. ,   0.,    0. ,   0. ,    0. ,    0. ),
    'L0':  (  0. ,   0.,    0. ,   0. ,    0. ,    0. ),
    'M0':  ( 30. , 145.,  175. ,   0. ,    0. ,    0. ),
    'U1':  (  0. ,   0.,    0. ,   0. ,    0. ,    0. ),
    'U2':  (  0. ,   0.,    0. ,   0. ,    0. ,    0. ),
    'U3':  (  0. ,   0.,    0. ,   0. ,    0. ,    0. ),
    'U4':  (  0. ,   0.,    0. ,   0. ,    0. ,    0. ),}

horizonEVM = {}
az = np.arange(361)
for k in evm.keys():
    el = np.zeros(len(az))
    el[(az>=evm[k][1])*(az<=evm[k][2])] = evm[k][0]
    el[(az>=evm[k][4])*(az<=evm[k][5])] = evm[k][3]
    el = el[np.argsort((360-az+360)%360)]
    _az = (360-az[np.argsort((360-az+360)%360)])%360
    horizonEVM[k] = (_az, el)

# -- from FITS header:
vlti_latitude = -24.62743941
vlti_longitude = -70.40498688
#_min_alt = 27 # when AT start to display vignetting
_min_alt = 35 # VLTI manual?

_test_dl34_plus30 = False

# def withCHARA():
#     f = open('telescopes.chara')
#     stations = ['S1', 'S2', 'E1', 'E2', 'W1', 'W2']
#     tmp = {}
#     for l in f.readlines():
#         if l.replace().strip()
#     f.close()

def computeOpl0(config, compressed=True, plot=False):
    """
    config = ['A0DL2IP1', 'D0DL1IP3', etc]

    can be checked in the ISS configuration pannel. As far as I can see,
    this function is accurate to less to about 20cm

    U,V = x, y

    tests:
    June 2017:
    vlti.computeOpl0(['U1DL5IP1', 'U2DL2IP3', 'U3DL3IP5', 'U4DL4IP7']) - np.array([192.749, 187.089, 198.6808, 208.927])

    # -- this one has a 24cm offset on J2DL4IP3, which is highly suspicious!
    vlti.computeOpl0(['A0DL6IP1', 'J2DL4IP3', 'J3DL3IP5']) - np.array([147.5965, 149.6835, 139.1645])
    """
    DL_length = 60 # -- approximation
    ad_hoc_offset = 11.1675 # -- to match ISS number

    # -- DL Tunnel building, just for drawing, affects OPL0, but not OPD
    U0 = layout['A0'][0] - 5
    V0 = DL_UV['DL1'][1] + 1.5
    U1 = layout['M0'][0] + 10
    V1 = DL_UV['DL6'][2] - 1.75 - 1.5

    # -- coordinate of 0-OPD as a function of IP
    y_ip = {'IP1':V0 + 5*0.24, 'IP2':V0 + 1*0.24,
            'IP3':V0 + 6*0.24, 'IP4':V0 + 2*0.24,
            'IP5':V0 + 7*0.24, 'IP6':V0 + 3*0.24,
            'IP7':V0 + 8*0.24, 'IP8':V0 + 4*0.24,
            }
    x_ip = IP_U['IP1'] - 3 # -- for drawing

    if plot:
        plt.close(0)
        f = plt.figure(0, figsize=(10.5,8))
        plt.clf()
        axs = [f.add_axes([0.05, 0.20, 0.50, 0.75]),
               f.add_axes([0.05, 0.00, 0.93, 0.20]),
               f.add_axes([0.60, 0.25, 0.40, 0.70])]
        f.suptitle(' - '.join(config))
        # -- stations
        for k in layout.keys():
            axs[0].plot(layout[k][0], layout[k][1], 'oy' if 'U' in k else '.k' )
        for ax0 in axs:
            ax0.set_aspect('equal')
        # -- DL position
        for k in DL_UV.keys():
            if k in ['DL3', 'DL4']:
                for ax0 in axs:
                    ax0.plot(DL_UV[k][0]+np.array([0,-DL_length]),
                         0.5*(DL_UV[k][1]+DL_UV[k][2])*np.array([1,1]), '-k',
                         alpha=0.4, linewidth=1.5)
                axs[2].text(DL_UV[k][0] + 5, 0.5*(DL_UV[k][1]+DL_UV[k][2]),
                            k, color='k', alpha=0.5, va='center')
            else:
                for ax0 in axs:
                    ax0.plot(DL_UV[k][0]+np.array([0,DL_length]),
                         0.5*(DL_UV[k][1]+DL_UV[k][2])*np.array([1,1]), '-k',
                         alpha=0.4, linewidth=1.5)
                axs[2].text(DL_UV[k][0] - 5, 0.5*(DL_UV[k][1]+DL_UV[k][2]),
                            k, color='k', alpha=0.5, va='center')

        for ax0 in axs:
            ax0.plot([U0,U1,U1,U0,U0],[V0,V0,V1,V1,V0], '-', color='0.5',
                    linewidth=3)
        axs[1].text(U0, V0+1, 'Delay lines tunnel', ha='left', va='bottom',
                        color='k', alpha=0.5)
        axs[2].text(IP_U['IP4'], -39, 'M16', ha='center', va='center',
                    color='k', alpha=0.4)
        axs[2].text(IP_U['IP4'], -34.5, 'Switch Yard', ha='center', va='center',
                    color='k', alpha=0.4)
        axs[2].text(IP_U['IP4'], -31, 'Beam\nCompressors', ha='center', va='center',
                    color='k', alpha=0.4)

        colors = ['r', 'g', 'b', 'm', 'y', 'c']
    res = []
    for i, c in enumerate(config):
        st = c[:2] # station
        dl = c[2:5] # delay line
        ip = c[5:8] # input channel
        d = -DL_length/2. if dl in ['DL3', 'DL4'] else DL_length/2.
        d /= 4
        # -- station -> M12 -> middle DL -> M16
        X = [layout[st][0], layout[st][0], DL_UV[dl][0] + d, DL_UV[dl][0] + d, IP_U[ip] ]
        Y = [layout[st][1], DL_UV[dl][1], DL_UV[dl][1], DL_UV[dl][2], DL_UV[dl][2] ]
        if not compressed:
            # -> switch yard -> 0-OPD
            X.extend([IP_U[ip], x_ip])
            Y.extend([y_ip[ip], y_ip[ip]])
        else:
            # -> beam compressor -> switchyard -> 0-OPD
            X.extend([IP_U[ip], IP_U[ip]-0.1, IP_U[ip]-0.1, x_ip])
            # -- +3 is approx but does not influence OPD calculations, but OPL0 yes
            Y.extend([y_ip['IP7']+3, y_ip['IP7']+3, y_ip[ip], y_ip[ip]])
        # -- compute OPL0
        opl0 = 0
        for j in range(len(X)-1):
            opl0 += np.sqrt((X[j+1]-X[j])**2 + (Y[j+1]-Y[j])**2)
        opl0 += ad_hoc_offset
        if 'U' in st:
            opl0 += 43.53 # to match ISS numbers

        res.append(opl0)
        if plot:
            print(c, 'OPL0=%7.3f'%(opl0))
            for ax0 in axs:
                ax0.plot(X, Y, '-', color=colors[i])
            # -- station
            axs[0].text(X[0], Y[0], st, color=colors[i],
                        ha='right', va='top', fontsize=12)

            # -- OPL0
            axs[2].text(X[-1]-0.2, Y[-1], '%.4fm'%opl0, color=colors[i],
                        ha='right', va='center', fontsize=9)

    if plot:
        # -- all plateform
        axs[0].set_xlim(U0-5, U1+5)
        axs[0].set_ylim(layout['G1'][1]-5, layout['J6'][1]+5)
        X0, Y0, a = -20, 50, np.deg2rad(-18.984 )
        axs[0].plot(X0 + np.array([-20*np.sin(a), 0, 10*np.cos(a)]),
                    Y0 + np.array([ 20*np.cos(a), 0, 10*np.sin(a)]), '-',
                    linewidth=2, color='0.8')
        axs[0].text(X0 -20*np.sin(a), Y0 + 20*np.cos(a), 'N',
                    rotation=-18, va='bottom', ha='center')
        axs[0].text(X0 + 10*np.cos(a), Y0 + 10*np.sin(a), 'E',
                    rotation=-18, va='center', ha='left')

        # -- DL tunnel
        axs[1].set_xlim(U0-2, U1+2)
        axs[1].set_ylim(V1-3, V0+6)
        # -- M16 and switchyard
        axs[2].set_xlim(x_ip-1.5, IP_U['IP8']+2)
        axs[2].set_ylim(-42, -30.5)
        # -- Zero-OPD line after the switchyard
        axs[2].plot([x_ip, x_ip], [y_ip['IP2']-0.2, y_ip['IP7']+0.2], '-',
                    color='k', linestyle='dashed')
        axs[2].text(x_ip, y_ip['IP7']+0.2, '0-OPD\nin VLTI Lab',
                    ha='center', va='bottom', fontsize=10)
    else:
        return np.array(res)

def projBaseline(T1, T2, target, lst, ip1=1, ip2=3, DL1=None, DL2=None,
                 min_altitude=_min_alt, max_OPD=110, max_airmass=None):
    """
    - T1, T2: telescopes stations (e.g. 'A0', 'U1', etc.)

    - target : [ra, dec] in decimal hour and decimal deg or name for SIMBAD

    - lst   : decimal hour (scalar or array)

    - ip1, ip2: input channels (optional)

    - DL1, DL2: delay lines (optional)

    min_altitude (default 20 degrees) sets the minimum altitude for
    observation. Checks also for shadowinf from the UTs.

    max_OPD (default 100m, from 11, to 111m) maximum OPD stroke of the delay lines.

    return a self explanatory dictionnary. units are m and degrees
    """
    if _test_dl34_plus30:
        # -- test the effect of extending DL3,4 for 30 OPL:
        if DL1 in [3,4]:
            max_OPD += 30
        if DL2 in [3,4]:
            max_OPD += 30
    if isinstance(target, str):
        # -- old simbad
        #s = simbad.query(target)[0]
        #radec = [s['RA.h'], s['DEC.d']]
        # --
        s = Simbad.query_object(target)
        ra = np.sum(np.float64(s['RA'][0].split())*np.array([1, 1/60., 1/3600.]))
        dec = np.float64(s['DEC'][0].split())*np.array([1, 1/60., 1/3600.])
        dec = np.sum(np.abs(dec))*np.sign(dec[0])
        radec = ra, dec

    else:
        radec = target

    # -- hour angle
    ha = (np.array(lst) - radec[0]) *360/24.0

    # -- alt-az
    dec = radec[1]

    tmp1 = np.sin(np.radians(dec)) * np.sin(np.radians(vlti_latitude)) +\
           np.cos(np.radians(dec)) * np.cos(np.radians(vlti_latitude)) *\
           np.cos(np.radians(ha))
    alt = np.arcsin(tmp1)*180/np.pi

    tmp1 = np.cos(np.radians(dec)) * np.sin(np.radians(ha))
    tmp2 = -np.cos(np.radians(vlti_latitude)) * np.sin(np.radians(dec)) + \
           np.sin(np.radians(vlti_latitude)) * np.cos(np.radians(dec)) * \
           np.cos(np.radians(ha));
    az = (360-np.arctan2(tmp1, tmp2)*180/np.pi)%360

    b = [layout[T1][2]-layout[T2][2],
         layout[T1][3]-layout[T2][3],
         0.0] # assumes you *DO NOT* combine ATs and UTs

    # -- projected baseline
    ch_ = np.cos(ha*np.pi/180.)
    sh_ = np.sin(ha*np.pi/180.)
    cl_ = np.cos(vlti_latitude*np.pi/180.)
    sl_ = np.sin(vlti_latitude*np.pi/180.)
    cd_ = np.cos(radec[1]*np.pi/180.)
    sd_ = np.sin(radec[1]*np.pi/180.)

    # -- (u,v) coordinates in m
    u = ch_ * b[0] -  sl_*sh_ * b[1] + cl_*sh_ * b[2]
    v = sd_*sh_ * b[0] + (sl_*sd_*ch_+cl_*cd_) * b[1] -\
        (cl_*sd_*ch_ - sl_*cd_) * b[2]

    # -- static OPL to telescope
    if not DL1 is None and not DL2 is None:
        config = (T1+'DL%dIP%d'%(DL1, ip1), T2+'DL%dIP%d'%(DL2, ip2))
        # -- assume compressed, since it is the case for UT and AT-STS
        opl1, opl2 = computeOpl0(config, plot=False, compressed=True)
    else:
        opl1 = layout[T1][4] + 0.12*(ip1-1)
        opl2 = layout[T2][4] + 0.12*(ip2-1)

    # add telescope OPL (M3-M11 distance): only matters for UT-AT combination
    # email SMo 15/10/2013:
    # if T1.startswith('U'):
    #     opl1 += 65.80 # UT
    # else:
    #     opl1 += 18.19 # AT
    # if T2.startswith('U'):
    #     opl2 += 65.80 # UT
    # else:
    #     opl2 += 18.19 # AT

    # optical delay and optical path delay, in m
    d = -b[0]*cd_*sh_ -\
        b[1]*(sl_*cd_*ch_ - cl_*sd_) +\
        b[2]*(cl_*cd_*ch_ + sl_*sd_)
    d = np.array(d)

    # -- final OPD to correct for
    opd = d + opl2 - opl1

    # -- airmass:
    czt = np.cos(np.pi*(90-alt)/180.)
    airmass = 1/czt*(1-.0012*(1/czt**2-1))
    airmass *= (alt>20.)
    airmass += (alt<=20)*2.9

    # -- parallactic angle
    if np.cos(np.radians(dec))!=0:
        parang = 180*np.arcsin(np.sin(np.radians(az))*
                               np.cos(np.radians(vlti_latitude))/\
                               np.cos(np.radians(dec)))/np.pi
    else:
        parang = 0.

    # -- observability
    observable = (alt>min_altitude)* \
                 (np.abs(opd) < max_OPD)*\
                 (alt>np.interp(az%360, horizon[T1][0], horizon[T1][1]))*\
                 (alt>np.interp(az%360, horizon[T2][0], horizon[T2][1]))

    if not max_airmass is None:
        observable = observable*(airmass<=max_airmass)

    if isinstance(alt, np.ndarray):
        observable = np.array(observable)

    res = {'u':u, 'v':v, 'opd':opd, 'alt':alt, 'az':az,
           'observable':observable, 'parang':parang,
           'lst':lst, 'ra':radec[0],'dec':radec[1],
           'B':np.sqrt(u**2+v**2), 'PA':np.arctan2(u, v)*180/np.pi,
           'airmass':airmass, 'baseline':T1+T2, 'opd':opd,
           'horizon': np.maximum(np.interp(az%360, horizon[T1][0],
                                           horizon[T1][1]),
                                 np.interp(az%360, horizon[T2][0],
                                           horizon[T2][1])),
           # -- baseline on the ground:
           'ground':np.array([layout[T2][0]-layout[T1][0],
                              layout[T2][1]-layout[T1][1],
                              (20. if 'U' in T2 else 0.)-(20. if 'U' in T1 else 0.)])}
    return res

def nTelescopes(T, target, lst, ip=None, DLconstraints=None,
                min_altitude=_min_alt, max_OPD=100, max_vcmPressure=None,
                max_airmass=None, flexible=False, plot=False, STS=True,
                lstCart=None):
    """
    Compute u, v, alt, az, airmass, tracks for a nT configurartion

    - T: list of telescopes stations OR stationDL. example: ['A0','K1']
      or ['A0DL1','K0DL4']. If delaylines are given, it will compute the optimum
      position for the dlines and VCM pressure.

    - target: Can be a string with the name of the target resolvable by simbad.
      Can also be a tuple of 2 decimal numbers for the coordinates (ra, dec), in
      decimal hours and decimal degrees

    - lst: a list (or ndarray) of local sidearal times.

    - [ip]: list of input channels. The order should be the same as telescopes.

    - min_altitude: for observability computation, in degrees (default=20deg)

    - max_OPD: for observability, in m (default=100m)

    - max_vcmPressure: enforce the VCM limitation (SLOW!!!)

    - flexible: if False, qperform some checks on the Array Configuration: no
      AT/UT; no more than 4 ATs west of the I track (DL1,2,4,5) and no more
      than 2 ATs east ofq the G track (DL3,4); no more than one AT per track
      (e.g. A0-A1 is not possible). *flexible=False will allow currently
      impossible configurations*

    - plot: True/False

    - DLconstraints sets min and max. For example: {'min1':35, 'max2':70}

    Returns: return a self explanatory dictionnary. units are m and degrees

    the key 'observable' is a table of booleans (same length as 'lst')
    indicating wether or not the object is observable.

    """
    tmp = []
    if ip is None:
        # -- assumes 1,3,5,7 for 4T
        ip = range(2*len(T))[1::2]

    if all(['DL' in x for x in T]):
        dl = [int(t.split('DL')[1]) for t in T]
        T = [t.split('DL')[0] for t in T]
        if len(set(dl))<len(dl):
            print('ERROR: some DL used more than once')
            return None
        if len(set(T))<len(T):
            print('ERROR: some STATIONS used more than once')
            return None
    else:
        dl=None

    if not isinstance(DLconstraints, dict):
        DLconstraints={}

    if not flexible:
        #-- check is we have at most 4 Tel p<30m and at most 2T p>30m
        W = list(filter(lambda t: layout[t][0]<=35, T))
        E = list(filter(lambda t: layout[t][0]>=35, T))
        assert len(W)<=4, 'Too many stations West of the laboratory'
        assert len(E)<=2, 'Too many stations East of the laboratory'

        #-- check the delay lines association
        if not dl is None:
            for k in range(len(T)):
                assert (layout[T[k]][0]<=35 and dl[k] in [1,2,5,6]) or \
                       (layout[T[k]][0]>=35 and dl[k] in [3,4]) , \
                    'wrong delay line / station association'

        #-- check that all telescopes are UTs if any UTs
        tracks = [t[0] for t in T]
        if 'U' in tracks:
            assert len(set(tracks))==1, 'UTs and ATs cannot be mixed, use flexible=False'
        else:
            if not any(['J' in t for t in T]):
                assert len(set(tracks))==len(T), \
                    'only one AT per track is allowed, use flexible=False'
            else: # special case for J stations
                _T = list(filter(lambda t: not t.startswith('J'), T))
                _tracks = [t[0] for t in _T]
                assert len(set(_tracks))==len(_T), \
                    'only one AT per track is allowed, use flexible=False'
                _T = list(filter(lambda t: t.startswith('J'), T))
                _s = list(filter(lambda t: t in ['J1', 'J2'], T))
                _n = list(filter(lambda t: t in ['J3', 'J4', 'J5', 'J6'], T))
                assert len(_s)<=1, 'only AT on J south track, use flexible=False'
                assert len(_n)<=1, 'only AT on J north track, use flexible=False'

    if isinstance(target, str):
        #s = simbad.query(target)
        #target_name=target
        #target = [s[0]['RA.h'], s[0]['DEC.d']]
        s = Simbad.query_object(target)
        try:
            ra = np.sum(np.float64(s['RA'][0].split())*np.array([1, 1/60., 1/3600.]))
            dec = np.float64(s['DEC'][0].split())*np.array([1, 1/60., 1/3600.])
            dec = np.sum(np.abs(dec))*np.sign(dec[0])
        except:
            ra, dec = 0,0
            print('WARNING: simbad query failed for', target)
            print(s)
        target_name = target
        target = ra, dec

    else:
        # assume [ra dec]
        if int(60*(target[0]-int(target[0])))<10:
            target_name = 'RA=%2d:0%1d' % (int(target[0]),
                                           int(60*(target[0]-int(target[0]))),)
        else:
            target_name = 'RA=%2d:%2d' % (int(target[0]),
                                           int(60*(target[0]-int(target[0]))),)
        if int(60*(abs(target[1])-int(abs(target[1]))))<10:
            target_name += ' DEC=%3d:0%1d' % (int(target[1]),
                                           int(60*(abs(target[1])-
                                           int(abs(target[1])))),)
        else:
            target_name += ' DEC=%3d:%2d' % (int(target[1]),
                                           int(60*(abs(target[1])-
                                           int(abs(target[1])))),)
    res = {}
    if not dl is None:
        res['config']=[T[k]+'DL'+str(dl[k])+'IP'+str(ip[k])
                       for k in range(len(ip))]
    else:
        res['config']=[T[k]+'IP'+str(ip[k]) for k in range(len(ip))]

    for i in range(len(T)+1):
        for j in range(len(T))[i+1:]:
            tmp = projBaseline(T[i],T[j],
                               target, lst, ip1=ip[i], ip2=ip[j],
                               DL1 = None if dl is None else dl[i],
                               DL2 = None if dl is None else dl[j],
                               min_altitude = min_altitude,
                               max_airmass = max_airmass,
                               max_OPD = max_OPD)
            if not 'lst' in res.keys():
                # init
                for k in ['lst', 'airmass', 'observable',
                          'ra', 'dec', 'alt', 'az', 'parang',
                          'horizon']:
                    res[k] = tmp[k]
                res['baseline'] = [tmp['baseline']]
                for k in ['B', 'PA', 'u', 'v', 'opd', 'ground']:
                    res[k] = {}
            else:
                # update
                res['observable'] = res['observable']*tmp['observable']
                res['baseline'].append(tmp['baseline'])
                res['horizon']=np.maximum(res['horizon'],
                                          tmp['horizon'])
            for k in ['B', 'PA', 'u', 'v', 'opd', 'ground']:
                res[k][tmp['baseline']] = tmp[k]

    # -- convert OPL into DL position in the lab:
    # -- @wvgvlti:Appl_data:VLTI:vltiParameter:optics:LabInputCoord
    x_M16 = 52. # where the center of the DL tunnel
    dx_dl = 6.92 # DL start to M16 -> real number!
    y_lab = -28 #
    # -- coordinate of 0-OPD in the lab
    y_ip = {'IP1':y_lab+5*0.24, 'IP2':y_lab+1*0.24,
            'IP3':y_lab+6*0.24, 'IP4':y_lab+2*0.24,
            'IP5':y_lab+7*0.24, 'IP6':y_lab+3*0.24,
            'IP7':y_lab+8*0.24, 'IP8':y_lab+4*0.24,}
    x_ip = x_M16 - 5

    wdl = .3 # width of the DL (drawing only)
    fy_M12 = lambda x: y_lab - 1.8*x # M12 y as fct of DL index
    fx_IP = lambda x: x_M16 + 1*x -4 # IP x as fct of IP index

    minDL_opl = 11. # smallest OPL for each cart

    #-- compute DL position and VCM position
    if not dl is None: # Delay lines are given, so compute VCM limitation
        res['vcm']={}
        res['dl']={}
        for d in dl:
            res['vcm'][d]=[]
            res['dl'][d]=[]
        #-- compute OPD dictionnary
        for k in range(len(res['lst'])):
            opdD = {}
            for i1,t1 in enumerate(T):
                for i2,t2 in enumerate(T):
                    maxO = max_OPD
                    if t1+t2 in res['opd'].keys():
                        opdD[(dl[i1],dl[i2])] = res['opd'][t1+t2][k]
            s = solveDLposition(dl, opdD, stations=T, STS=STS,
                                dlPosMin=minDL_opl/2.,
                                dlPosMax=(max_OPD+minDL_opl)/2.,
                                constraints=DLconstraints)
            for d in dl:
                if np.isnan(s[d]):
                    res['observable'][k] = False
                res['dl'][d].append(s[d])
                res['vcm'][d].append(s['vcm'+str(d)])

        for d in dl:
            res['dl'][d]=np.array(res['dl'][d])
            res['vcm'][d]=np.array(res['vcm'][d])

        if not max_vcmPressure is None:
            maxP = 0
            for d in dl:
                maxP = np.maximum(maxP,res['vcm'][d])
            res['observable'] = res['observable']*(maxP<max_vcmPressure)

        res['x_dl']={}
        res['y_dl']={}

        for i,d in enumerate(dl):
            if d==3 or d==4:
                res['x_dl'][d] = x_M16 - dx_dl - res['dl'][d]
            else:
                res['x_dl'][d] = x_M16 + dx_dl + res['dl'][d]
            res['y_dl'][d] = fy_M12(d)

    # -- plot observability
    if plot:
        colors = ['r','g','b','y','c','m']
        colors = [(1,0,0), (0,1,0), (0,0,1),
                  (1,0.5,0), (0,1,0.5), (0.5,0,1)]
        colors = [(0.3+c[0]*0.5,
                   0.3+c[1]*0.5,
                   0.3+c[2]*0.5) for c in colors]
        where = lambda key: [res[key][k] for k in
                             range(len(res[key]))
                             if res['observable'][k]]
        where2 = lambda key1, key2: [res[key1][key2][k] for k in
                                     range(len(res['lst']))
                                     if res['observable'][k]]
        if isinstance(plot, int):
            fig = plot
        else:
            fig = 0
        if not dl is None:
            plt.figure(fig, figsize=(12,7))
            plt.clf()
            plt.subplots_adjust(hspace=0.16, top=0.94, left=0.06,
                                wspace=0.15, right=0.98, bottom=0.01)
            ax1 = plt.subplot(221)
        else:
            plt.figure(fig, figsize=(7,5))
            plt.clf()
            plt.subplots_adjust(wspace=0.07, top=0.90, left=0.08,
                                right=0.98, bottom=0.12)
            ax1 = plt.subplot(111)

        plt.plot(res['lst'], res['lst']*0 + min_altitude,
                linestyle='dashed', color='k', linewidth=3, alpha=0.4)

        plt.plot(res['lst'], res['horizon'],
                         color=(0.1, 0.2, 0.4), alpha=0.3,
                         label='UT shadow', linewidth=2)
        plt.fill_between(res['lst'],res['horizon']*0.0,
                            res['horizon'], color=(0.1, 0.2, 0.4),
                            alpha=0.5)
        plt.plot(res['lst'], res['alt'], '+', alpha=0.9,
                color=(0.5, 0.1, 0.3), label='not obs.',
                markersize=8)
        plt.plot(where('lst'), where('alt'), 'o', alpha=0.5,
                 color=(0.1, 0.5, 0.3), label='observable',
                 markersize=8)
        plt.legend(prop={'size':9}, numpoints=1)
        plt.ylim(0,92)
        plt.xlabel('lst (h)')
        plt.ylabel('altitude (deg)')
        plt.suptitle(target_name+' '+'-'.join(res['config'])+
                    (' [STS]' if STS else ''))
        #plt.hlines(min_altitude, plt.xlim()[0], plt.xlim()[1],
        #              color='r', linewidth=1, linestyle='dotted')
        #-- make LST labels to be in 0..24
        # xl = plt.xticks()
        # xl = [str(int(x)%24) for x in xl[0]]
        # ax1.set_xticklabels(xl)
        plt.grid()
        #-- DL cart position
        if dl is None:
            return

        #-- OPL (2xDL position) for each DL:
        ax3 = plt.subplot(222, sharex=ax1)
        for i,d in enumerate(dl):
            plt.plot(where('lst'), 2*np.array(where2('dl', d)),
                        '-', label='DL'+str(d),
                        alpha=0.7, color=colors[i], linewidth=3)
            # -- plt ref DL
            wr = np.where(np.diff(where2('dl', d))==0)
            plt.plot(np.array(where('lst'))[wr], 2*np.array(where2('dl', d))[wr],
                        alpha=1, color=colors[i], linewidth=5,
                        linestyle='dashed')

        plt.legend(loc='upper center', prop={'size':9}, numpoints=1,
                   ncol=1)
        plt.xlabel('lst (h)')
        plt.ylabel('OPL (m)')
        plt.hlines([minDL_opl, max_OPD+minDL_opl], res['lst'].min(), res['lst'].max(),
            linestyle='dashed', color='k', linewidth=3, alpha=0.4)
        plt.grid()
        plt.ylim(0,130)

        # ========================
        # == TUNNEL ==============
        # ========================

        # markers:
        DLm_W = [(-0.03, 0.04), (-0.1, 0.04), (-0.12, 0.01), (-0.12, -0.01),
                 (-0.1, -0.04), (-0.1, -0.06), (0.1, -0.06), (0.1, -0.04),
                 (-0.03, -0.04), (-0.03, 0.04)]

        DLm_E = [(-d[0], d[1]) for d in DLm_W]

        ax2 = plt.subplot(212)
        plt.text(x_M16, y_lab+0.3, 'LAB', ha='center', va='bottom',
                 color='0.5')
        # -- delay lines rails and ticks:
        for d in [1,2,3,4,5,6]:
            s = (-1)**(d==3 or d==4)
            # -- range
            plt.plot([x_M16 + s*dx_dl, x_M16 + s*dx_dl + s*60],
                     [fy_M12(d)-wdl/2, fy_M12(d)-wdl/2], '-k',
                     alpha=0.5)
            plt.text(x_M16 + s*dx_dl + s*62, fy_M12(d)-wdl/2,
                     'DL'+str(d), va='center', color='0.5', size=9,
                     ha='right' if s==-1 else 'left')
            # -- 10m ticks
            plt.plot(x_M16 + s*dx_dl + s*np.linspace(0,60,7),
                     fy_M12(d)-wdl/2 + np.zeros(7), '|',
                     markersize=9, color='0.6')
            # -- 5m ticks
            plt.plot(x_M16 + s*dx_dl + s*np.linspace(0,50,6)+s*5,
                     fy_M12(d)-wdl/2 + np.zeros(6), '|',
                     markersize=7, color='0.4')
            for x in np.linspace(0,60,7):
                plt.text(x_M16 + s*dx_dl + s*x, fy_M12(d)+0.6*wdl,
                         str(int(2*x)), size=6, va='bottom',
                         ha='center', color='k')
            # -- DL restrictions
            for k in DLconstraints.keys():
                if str(d) in k:
                    #print(d, k, DLconstraints[k])
                    if 'min' in k:
                        plt.fill_between(x_M16 + s*dx_dl +
                                         s*np.array([0,DLconstraints[k]/2]),
                                         fy_M12(d)-np.array([0.9, 0.9])*wdl,
                                         fy_M12(d)-np.array([0.1, 0.1])*wdl,
                                         color='k', hatch='////', alpha=0.3)
                        plt.text(x_M16 + s*dx_dl + s*DLconstraints[k]/2,
                                 fy_M12(d)-2.5*wdl, str(DLconstraints[k]),
                                 color='r', size=8, va='bottom', ha='center')
                    elif 'max' in k:
                        plt.fill_between(x_M16 + s*dx_dl +
                                         s*np.array([60,DLconstraints[k]/2]),
                                    fy_M12(d)-np.array([0.9, 0.9])*wdl,
                                    fy_M12(d)-np.array([0.1, 0.1])*wdl,
                                    color='k', hatch='////', alpha=0.3)
                        plt.text(x_M16 + s*dx_dl + s*DLconstraints[k]/2,
                                 fy_M12(d)-2.5*wdl, str(DLconstraints[k]),
                                 color='r', size=8, va='bottom', ha='center')

        # -- used range:
        for i,d in enumerate(dl):
            #print(d, np.min(where2('dl', d)), np.max(where2('dl', d)))
            s = (-1)**(d==3 or d==4)
            # -- DL range
            plt.plot([x_M16 + s*dx_dl + s*np.min(where2('dl', d)),
                     x_M16 + s*dx_dl + s*np.max(where2('dl', d))],
                     [fy_M12(d)-wdl/2, fy_M12(d)-wdl/2], '-',
                     color=colors[i], alpha=0.5, linewidth=8,
                     label='range %d'%(d))

        # -- light path: T -> M12 -> cart -> M16
        for i, t  in enumerate(T):
            y_M12 = fy_M12(dl[i])
            x_IP = fx_IP(ip[i])
            s = (-1)**(dl[i]==3 or dl[i]==4)
            if lstCart is None:
                x_cart = x_M16 + s*dx_dl + s*np.mean(where2('dl', dl[i])) # middle position
            else:
                x_cart = x_M16 + s*dx_dl + s*(np.interp(lstCart, res['lst'], res['dl'][dl[i]])-0.5)
            # -- light path
            plt.plot([layout[t][0], layout[t][0], x_M16, x_cart+s*1, x_cart+s*1, x_IP, x_IP],
                     [layout[t][1], y_M12, y_M12, y_M12, y_M12-wdl, y_M12-wdl, y_lab],
                     '-', alpha=0.5, linewidth=1.5, color=colors[i])
            # -- cart
            plt.plot(x_cart, y_M12-wdl/2, color=colors[i], markersize=22,
                    marker=DLm_W if (dl[i]==3 or dl[i]==4) else DLm_E)

            # -- write name of station where light is coming from
            if layout[t][1]>y_lab: # north of the tracks
                plt.text(layout[t][0], y_lab-wdl, t, color=colors[i],
                         va='top', ha='center')
            else: # south of the tracks
                plt.text(layout[t][0], fy_M12(7)+wdl, t, color=colors[i],
                         va='bottom', ha='center')
        # -- DL tunnel walls:
        U0 = layout['A0'][0] - 5
        U1 = layout['M0'][0] + 10
        color = (0.1,0.2,0.7)
        plt.fill_between([x_M16-110, x_M16+90], fy_M12(7)*np.array([1,1]),
                         (fy_M12(7)-10)*np.array([1,1]),
                         color=color, alpha=0.2, hatch='//')
        plt.fill_between([x_M16-110, x_M16+90], y_lab*np.array([1,1]),
                         (y_lab+10)*np.array([1,1]),
                         color=color, alpha=0.2, hatch='//')
        # end walls
        plt.fill_between((U0-10, U0), (fy_M12(7), fy_M12(7)), (y_lab, y_lab),
                    color=color, alpha=0.2, hatch='//')
        plt.fill_between((U1, U1+10), (fy_M12(7), fy_M12(7)), (y_lab, y_lab),
                    color=color, alpha=0.2, hatch='//')

        plt.ylim(fy_M12(7)-4*wdl, y_lab+4*wdl)
        plt.xlim(U0-4, U1+4)
        #plt.title('Delay line tunnel')
        ax2.get_yaxis().set_visible(False)
        ax2.get_xaxis().set_visible(False)

        #-- VCM pressure for each DL:
        # ax4 = plt.subplot(224, sharex=ax1)
        # for i,d in enumerate(dl):
        #     #plt.plot(res['lst'], res['vcm'][d], '-',label='VCM'+str(d))
        #     plt.plot(where('lst'), where2('vcm', d),
        #                 '-',markersize=8,label='VCM'+str(d),
        #                 alpha=0.7, color=colors[i], linewidth=3)
        #
        # plt.legend(loc='upper left', prop={'size':10}, numpoints=1,
        #            ncol=6)
        # plt.ylabel('DL VCM pressure %s(bar)'%('for AT/STS or UT' if STS else ''))
        # plt.xlabel('lst (h)')
        # ax1.set_xlim(res['lst'].min(), res['lst'].max())
        # plt.grid()
        # plt.ylim(0,5.0)
        # plt.hlines(2.5, res['lst'].min(), res['lst'].max(), color='k',
        #            alpha=0.4, linestyle='dashed', linewidth=3)

        print('tracking time:', res['lst'][np.where(res['observable'])[0][-1]] -
            res['lst'][np.where(res['observable'])[0][0]])
    else:
        return res

def skyCoverage(T, max_OPD=100, fig=1, flexible=False,
                disp_vcmPressure=False, min_altitude=None,
                max_vcmPressure=None, verbose=False, STS=False,
                createXephemHorizon=False, plotIss=False,
                DLconstraints=None):
    """
    plot the sky coverage of a VLTI configuration (2+ telescopes). By default,
    will display only the sky coverage (from 0 to 70 degrees in zenithal angle)
    and the drawing of the baselines on the platform.

    - T is a list of telescopes, such as ['A0', 'K0', 'G1'] or including Delay
      Lines ['A0DL3', 'B2DL2']

    - max_OPD = maximum OPD correction for each baseline, in meters
      (default=100,)

    LIMITATIONS:
    - there is no limitations in the number of telescopes, 2 is the minimum.
    - AT+UT is not accurate, even though it will plot something!
    - there is an option (max_OPD=) to give the maximum OPD compensation.
      Currently, I set it to 100m (from 11m to 111m). You can play with the
      parameter if you want to see the effect on the double pass for instance,
      by setting it to 200.

    CHECK:
    iss@wvgvlti > issguiTestDlRange
    """
    global horizon, horizonEVM
    if min_altitude is None:
        min_altitude = _min_alt
        if plotIss:
            min_altitude = 0

    color=(0.2,0.3,0.5)
    if all(['DL' in x for x in T]):
        dl = [int(t.split('DL')[1][0]) for t in T]
        T = [t.split('DL')[0] for t in T]
        if disp_vcmPressure and max_vcmPressure is None:
            max_vcmPressure=10.
    else:
        dl = None

    # -- baseline length:
    allB = {}
    for t1 in T:
        for t2 in T:
            if t1<t2:
                allB[t1+t2]=round(np.sqrt((layout[t1][0]-layout[t2][0])**2+
                            (layout[t1][1]-layout[t2][1])**2), 1)
    if verbose:
        print('baselines (%1.0f -> %1.0f m):'%(np.min([b[1] for b in allB.items()]),
                                   np.max([b[1] for b in allB.items()])), allB)

    if disp_vcmPressure or max_vcmPressure:
        N=45
    elif createXephemHorizon:
        N=180
        min_altitude=0
    else:
        N=45

    dec = np.linspace(-90, 45, N)
    res_alt = []
    res_az = []
    vcm_alt = []
    vcm_az = []
    vcm_pM = []
    for d in dec:
        lst = np.linspace(-12,12,1+2*int(N*np.cos(d*np.pi/180)))
        if dl is None:
            T_=T
        else:
            T_ = [T[k]+'DL'+str(dl[k]) for k in range(len(T))]
        x = nTelescopes(T_, [0.0, d], lst, min_altitude=min_altitude-5,
                        max_OPD=max_OPD, STS=STS, flexible=flexible,
                        DLconstraints=DLconstraints)
        w = np.where((x['alt']>x['horizon'])*(1-np.float64(x['observable'])))
        res_alt.extend(x['alt'][w])
        res_az.extend(x['az'][w])
        if 'vcm' in x.keys():
            w = np.where((x['alt']>20)*(np.float64(x['observable'])))
            vcm_alt.extend(x['alt'][w])
            vcm_az.extend(x['az'][w])
            tmp = 0
            for d in dl:
                tmp = np.maximum(tmp, x['vcm'][d][w])
            vcm_pM.extend(tmp)

    Bground = [np.sqrt(np.sum(x['ground'][b]**2)) for b in x['ground'].keys()]
    Bground = np.array(Bground)

    res_alt = np.array(res_alt)
    res_az = np.array(res_az)
    if 'vcm' in x.keys():
        vcm_alt = np.array(vcm_alt)
        vcm_az = np.array(vcm_az)
        vcm_pM = np.array(vcm_pM)

    #plt.subplots_adjust(wspace=0.02, top=0.9, left=0.02,
    #                    right=0.98, bottom=0.07)

    # ------------------------------
    #ax = plt.subplot(121,polar=1)
    # if disp_vcmPressure:
    #     plt.close(fig)
    #     plt.figure(fig, figsize=(9.5,8))
    #     ax = plt.axes([0.05, 0.05, 0.55, 0.9], polar=1)
    # else:
    #     plt.close(fig)
    #     plt.figure(fig, figsize=(8.2,5.3))
    #     ax = plt.axes([0.02, 0.05, 0.58, 0.9], polar=1)

    # # -- VCM pressure as color
    # if 'vcm' in x.keys() and disp_vcmPressure:
    #     plt.scatter((vcm_az-90)*np.pi/180, 90-vcm_alt, s=40,
    #                 c=vcm_pM, marker='h', cmap='spectral', edgecolor=None,
    #                 vmin=0,vmax=4.5, label='max VCMs pressure', alpha=0.5)

    #     plt.colorbar(orientation='vertical', shrink=0.7,
    #                  ticks=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5])

    # if 'vcm' in x.keys() and not max_vcmPressure is None and not disp_vcmPressure:
    #     w = np.where(vcm_pM>max_vcmPressure)
    #     ax.plot((vcm_az[w]-90)*np.pi/180, 90-vcm_alt[w], '+',
    #         markersize=8, alpha=0.4, color=(0.8, 0.4, 0.0),
    #         label='VCMs>%3.1fbar'%(max_vcmPressure),
    #         markeredgewidth=2)

    allAz, allAlt = list(res_az), list(res_alt)          ########   res_az-90   az-90  #######

    # -- non observable domain
    label = 'DL limits'
    if not DLconstraints is None:
        label += '; '+', '.join([k+':'+str(DLconstraints[k]) for k in DLconstraints.keys()])
    plt.plot((res_az-90)*np.pi/180, 90-res_alt, 'x', alpha=0.4,
        color=(0.4, 0.1, 0.2), label=label, markersize=8,
        markeredgewidth=1.5)
              
    # -- plot horizon
    az = np.linspace(0,360,361)
    label_ = True
    for t in T:
        h = np.interp(az%360, horizon[t][0], horizon[t][1])
        plt.fill_between((az-90)*np.pi/180, 90-h, np.ones(len(az))*90,
                        color=(0.3, 0.14, 0), alpha=0.2,
                        label='shadowing' if label_ else '')
        # h2 = np.interp(az%360, horizonEVM[t][0], horizonEVM[t][1])
        # ax.fill_between((az-90)*np.pi/180, 90-h2, np.ones(len(az))*90,
        #                 color=(0., 0., 1.), alpha=0.2,
        #                 label='shadowingm EVM' if label_ else '')

        label_ = False
        allAz.extend(az)
        allAlt.extend(h)

    allAz = np.array(allAz)
    allAlt = np.array(allAlt)

    # -- make overall horizon
    alt = []
    az = np.linspace(0,360,N+1)
    for i in range(len(az)-1):
        alt.append(max(0, allAlt[np.where((allAz>=az[i])*(allAz<=az[i+1]))].max()))
    alt = np.array(alt)
    if createXephemHorizon:
        # -- debug plot:
        #ax.plot((az[:-1]-90)*np.pi/180, 90-alt, '-',
        #    color=(0.7, 0.0, 0.4), linewidth=3)
        filename = os.path.join(os.path.expanduser('~'),'.xephem/',''.join(T)+'.hzn')
        f = open(filename, 'w')
        for a in np.linspace(1,360, 360):
            f.write('%3.0f %3.0f\n'%(a, np.interp((180-a)%360, az[:-1], alt)))
        print(filename, 'has been written')
        f.close()
        #plt.close(fig)
        #return

    # -- plot the horizon:
    # if plotIss:
    #     horizon_ = 0
    # else:
    #     print(_min_alt)
    #     horizon_ = _min_alt - 2

    # ax.set_ylim(0,90-horizon_)
    # ax.text(np.pi/2, 80-horizon_, 'N', size='x-large', color='k', weight='black',
    #             horizontalalignment='center',
    #             verticalalignment='center',)
    # ax.text(0.0, 80-horizon_, 'E', size='x-large', color='k', weight='black',
    #             horizontalalignment='center',
    #             verticalalignment='center')
    # ax.text(np.pi, 80-horizon_, 'W', size='x-large', color='k', weight='black',
    #             horizontalalignment='center',
    #             verticalalignment='center')

    # ax.legend(loc='lower left', prop={'size':9}, numpoints=1, ncol=3)
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)

      # if not plotIss:
      #   color2=(0.1, 0.2, 0.4)
      #   color3=(0.0, 0.3, 0.6)

        # -- declination circles:
        # allX = []
        # lst = np.linspace(-8,8,17*2+1)
        # for d in [-75, -60, -45, -30, -15, 0, 15, 30]:
        # #for d in [-80, -70, -60, -50, -40, -30, -20,-10, 0, 10, 20, 30]:
        #     x = nTelescopes(T_, [0.0, d], lst,min_altitude =min_altitude,
        #                     max_OPD=max_OPD, flexible=flexible, DLconstraints=DLconstraints)
        #     allX.append(x)
        #     allX.append(nTelescopes(T_, [0.0, d+5], lst,min_altitude =min_altitude,
        #                     max_OPD=max_OPD, flexible=flexible, DLconstraints=DLconstraints))
        #     allX.append(nTelescopes(T_, [0.0, d+10], lst,min_altitude =min_altitude,
        #                     max_OPD=max_OPD, flexible=flexible, DLconstraints=DLconstraints))
        #     if d == 30:
        #         allX.append(nTelescopes(T_, [0.0, d+15], lst,min_altitude =min_altitude,
        #                     max_OPD=max_OPD, flexible=flexible, DLconstraints=DLconstraints))

        #     ax.plot(np.radians(x['az']-90), 90-x['alt'],
        #              '-', color=color2, alpha=0.25, linewidth=2)
        #     ax.text(np.radians(x['az'][len(lst)//2]-90),
        #             90-x['alt'][len(lst)//2], str(d)+r'$^\mathrm{o}$', ha='center',
        #             va='center', color=color3, alpha=0.8, size=10)

              #if d == -30: # -- hour angle ticks
    # hs = [-4,-3,-2,-1,1,2,3,4]
    # for h in hs:
    #     x = nTelescopes(T_, [0.0, d], [h], flexible=flexible,
    #                                  DLconstraints=DLconstraints)
    #     plt.text(np.radians(x['az'][0]-90),
    #                          90-x['alt'][0], str(h)+'h', ha='center',
    #                          va='bottom', color='#24448c', alpha=0.8,
    #                          rotation=h*9, size=10)

        # -- RA lines
        # for k in range(len(lst))[1::2]:
        #     ax.plot([np.radians(x['az'][k]-90) for x in allX],
        #             [90-x['alt'][k] for x in allX],
        #             '-', color=color2, alpha=0.25, linewidth=2)

        # # -- south pole
        # x = nTelescopes(T_, [0.0, -90], [0], flexible=flexible,
        #                 DLconstraints=DLconstraints)
        # plt.plot(np.radians(x['az']-90), 90-x['alt'],'*k',
        #           markersize=8, markeredgewidth=2, alpha=0.4)

        # -- airmass circles:
        # alt = np.linspace(90,10,200)
        # czt = np.cos(np.pi*(90-alt)/180.)
        # airmass = 1/czt*(1-.0012*(1/czt**2-1))

        # __el = np.array([75, 60, 45, 30])
        # #print(np.round(np.interp(__el, alt[::-1], airmass[::-1]),2))
        # color3=(0.1, 0.4, 0.2)
        # for a in [1.04,1.15,1.40,2]:
        # #for a in [1.06,1.15,1.3,1.55,2,2.9]:
        #     #print(a, round(np.interp(a, airmass, alt), 1))
        #     ax.plot(np.linspace(0,np.pi*2,100),
        #              90-np.interp(a, airmass, alt)*np.ones(100),
        #              color=color3, alpha=0.25, linewidth=2)
        #     ax.text(-np.pi/4., 90-np.interp(a, airmass, alt),
        #              str(a), color=color3, alpha=1, rotation=45,
        #              ha='center', va='center', size=10)
        #     ax.text(-np.pi/4.+np.pi, 90-np.interp(a, airmass, alt),
        #              str(a), color=color3, alpha=1, rotation=45,
        #              ha='center', va='center', size=10)
        #     ax.text(np.pi/4., 90-np.interp(a, airmass, alt),
        #              r'%2.0f$^\mathrm{o}$'%(round(np.interp(a, airmass, alt)/2.5,0)*2.5),
        #              color=color3, alpha=1, rotation=-45,
        #              ha='center', va='center', size=10)
        #     ax.text(np.pi/4.+np.pi, 90-np.interp(a, airmass, alt),
        #              r'%2.0f$^\mathrm{o}$'%(round(np.interp(a, airmass, alt)/2.5,0)*2.5),
        #              color=color3, alpha=1, rotation=-45,
        #              ha='center', va='center', size=10)
        # -- zenith hole
    plt.fill_between(np.linspace(np.pi/2,5*np.pi/2,100), 4*np.ones(100), 0*np.ones(100),
                        color=(0.8,0.3,0.2), alpha=0.25, linewidth=0, label='zenith')
    # else:
        # -- same as Paranal
        # ax.plot([0,np.pi], [90,90], color='0.5', alpha=0.5, linewidth=2)
        # ax.plot([np.pi/2,3*np.pi/2], [90,90], color='0.5', alpha=0.5, linewidth=2)

        # for a in [10,30,50,70]:
        #     ax.plot(np.linspace(0,np.pi*2,100), 90-a*np.ones(100),
        #             color='0.5', alpha=0.5, linewidth=2)
        #     ax.text(np.pi/4., 90-a,
        #              str(a)+r'$^\mathrm{o}$', color='0.2', alpha=1, rotation=-45,
        #              ha='center', va='center')
        #     ax.text(5*np.pi/4., 90-a,
        #              str(a)+r'$^\mathrm{o}$', color='0.2', alpha=1, rotation=-45,
        #              ha='center', va='center')
        #     czt = np.cos(np.pi*(90-a)/180.)
        #     airmass = 1/czt*(1-.0012*(1/czt**2-1))
        #     ax.text(3*np.pi/4., 90-a, '%3.1f'%airmass, color='0.2',
        #             alpha=1, rotation=45, ha='center', va='center')
        #     ax.text(-np.pi/4., 90-a, '%3.1f'%airmass, color='0.2',
        #             alpha=1, rotation=45, ha='center', va='center')

    # ------------------------------
    #ax = plt.subplot(122)
    # ax2 = plt.axes([0.6, 0.05, 0.4, 0.9])

    # ax2.set_axis_off()
    # title = '-'.join(T_)+(' (STS)' if STS else '')
    # title += ': %2.0f -> %2.0fm '%(Bground.min(), Bground.max())
    # if _test_dl34_plus30:
    #     title += '\nWARNING DL3,4 range += 30m'
    # ax2.set_title(title, fontdict={'fontsize':12})
    # ax2.set_aspect('equal')
    # # -- stations:
    # ax2.plot([layout[k][2] for k in layout.keys() if 'U' in k],
    #     [layout[k][3] for k in layout.keys() if 'U' in k], 'o', color='0.7',
    #     markersize=30)
    # ax2.plot([layout[k][2] for k in layout.keys() if not 'U' in k],
    #     [layout[k][3] for k in layout.keys() if not 'U' in k], 'o', color='0.7',
    #     markersize=10)

    # -- name of station
    # for k in layout.keys():
    #     ax2.text(layout[k][2], layout[k][3], k,
    #                 ha='center', va='center',
    #                 color='k', fontsize=15 if 'U' in k else 8, weight='black',
    #                 rotation=-layout_orientation)


    # -- baselines:
    # for i in range(len(T)+1):
    #     for j in range(len(T))[i+1:]:
    #         ax2.plot([layout[T[i]][2], layout[T[j]][2]],
    #             [layout[T[i]][3], layout[T[j]][3]],'-', linewidth=3,
    #             alpha=0.8, color=color)
    # plt.plot([0,0,7], [80,70,70], '-', linewidth=1.5, color='0.5')
    # plt.text(0, 80, 'N', va='bottom', ha='center')
    # plt.text(8, 70, 'E', va='center', ha='left')
    # _x, _y = np.array([48, 48]), np.array([70,-120])
    # plt.plot(np.cos(np.deg2rad(layout_orientation))*_x + np.sin(np.deg2rad(layout_orientation))*_y,
    #         -np.sin(np.deg2rad(layout_orientation))*_x + np.cos(np.deg2rad(layout_orientation))*_y,
    #         color='k', linestyle='dotted', alpha=0.2)

    # -- ruler
    # x0, y0 = -10, -110
    # N=14
    # for i in range(N):
    #     if i%2==0:
    #         plt.text(x0+i*10, y0-5, '%2dm'%(i*10), ha='center', va='bottom', size=8)
    #     plt.plot([x0+i*10, x0+(i+1)*10], [y0,y0],
    #              color='0.8' if i%2==0 else '0.2', linewidth=3)
    # plt.text(x0+N*10, y0-5, '%dm'%(N*10), ha='center', va='bottom', size=8)

    # ax2.set_ylim(-110,110)
    # ax2.set_xlim(-30,150)
    # return

def uvCoverage(quadruplets=None, decUV=-24, maxB=150, fig=1, beam=False,
               tObs=None, flexible=False, max_OPD=100, withSky=True):
    """
    - quadruplets: list of configs, such as
       ['U1-U2-U3-U4','A1-G1-K0-J3','D0-H0-G1-I1','A1-B2-C1-D0']
    """

    global directory
    if quadruplets is None:
        #-- current:
        #quadruplets=['A1-G1-K0-J3','D0-H0-G1-I1','A1-B2-C1-D0']
        quadruplets = ['A0-B5-J2-J6','A0-G1-K0-J2','D0-G2-J3-K0','A0-B2-C1-D0'] # 2024
        max_OPD = [200, 100, 100, 100]
        maxB = 210

    # -- colors based on "spectral map"
    #colors = plt.cm.spectral(np.linspace(0.05,0.95,len(quadruplets)))
    #colors = [[0.2+0.8*c[k]**2 if k<3 else c[k] for k in range(4)] for c in colors]

    # -- color map
    #colors = plt.cm.gnuplot(np.linspace(0,0.9,len(quadruplets)))
    #colors = plt.cm.RdYlGn(np.linspace(0,1,len(quadruplets)))
    #colors = [[x, 1-((x-0.5)/0.5)**2, (1-x)] for x in np.linspace(0,1.0,len(quadruplets))]

    # -- gist_stern map
    #colors = plt.cm.gist_stern(np.linspace(0,0.9,len(quadruplets)))
    #colors = [[0.75*c[k]**2 if k<3 else c[k] for k in range(4)] for c in colors]

    colors = ['0.5', 'orange', 'c', 'r', 'm', 'g', 'b', 'y']

    hatchs = ['||', '//', '--', 'o', '*']

    plt.close(fig)
    if beam:
        plt.figure(fig, figsize=(14,4.5), facecolor='#eeefef')
        plt.clf()
        plt.subplots_adjust(wspace=0.1, top=0.98, left=0.04,
                            right=0.96, bottom=0.04)
    elif withSky:
        plt.figure(fig, figsize=(10.6,4.3), facecolor='#eeefef')
        plt.clf()
        plt.subplots_adjust(wspace=0.05, hspace=0.0, top=0.95, left=0.01,
                            right=0.99, bottom=0.01)
    else:
        plt.figure(fig, figsize=(7.5,4.3), facecolor='#eeefef')
        plt.clf()
        plt.subplots_adjust(wspace=0.05, hspace=0.0, top=0.85, left=0.01,
                            right=0.95, bottom=0.01)

    if beam:
        ax0 = plt.subplot(143, polar=True)
        ax1 = None #plt.subplot(232)
    elif withSky:
        ax0 = plt.subplot(133, polar=True)
        ax1 = None
    else:
        ax0 = plt.subplot(122, polar=True)
        ax1 = None

    u, v = [], []
    bins = np.arange(18)*10
    #bins = np.log10(5*1.1**np.arange(40))
    ball = []

    if not max_OPD is None and (type(max_OPD)==int or type(max_OPD)==float):
        max_OPD = [max_OPD]*len(quadruplets)
    print('max_OPD', max_OPD)
    for i,q in enumerate(quadruplets):
        # -- observability
        n = 6 # -- compute every 60/n minutes
        b = nTelescopes(q.split('-'), (0.0, decUV), np.linspace(-12,12,24*n+1), plot=False,
                        flexible=flexible, max_OPD=max_OPD[i], min_altitude=_min_alt)
        if not tObs is None:
            if isinstance(tObs, list) and len(tObs)==len(quadruplets):
                _tObs = tObs[i]
            else:
                _tObs = tObs
            # -- select the tObs hours closest to transit
            if np.sum(b['observable'])/float(n) > _tObs:
                # -- keep tObs hours closest to transit:
                w = np.abs(b['lst'][b['observable']]).argsort()
                lstMin = np.min(b['lst'][b['observable']][w][:int(n*_tObs)])
                lstMax = np.max(b['lst'][b['observable']][w][:int(n*_tObs)])
                b['observable'] *= (b['lst']<=lstMax)*(b['lst']>=lstMin)
            else:
                w = []

        bmin, bmax = None, None
        bq = []
        for j,t in enumerate(b['baseline']):
            LST = b['lst'][b['observable']]
            PA = (90-b['PA'][t][b['observable']])*np.pi/180
            B = b['B'][t][b['observable']]
            u.extend(b['u'][t][b['observable']])
            v.extend(b['v'][t][b['observable']])
            if j==0:
                ax0.plot(PA, B, color=colors[i], label=q, linewidth=4,
                        alpha=1./np.sqrt(len(q.split('-'))))
            else:
                ax0.plot(PA, B, color=colors[i], linewidth=4,
                        alpha=1./np.sqrt(len(q.split('-'))),)
            ax0.plot(PA+np.pi, B, color=colors[i], linewidth=4,
                        alpha=1./np.sqrt(len(q.split('-'))),)
            bq.extend(list(B))
            ball.extend(list(B))
            if bmin is None:
                bmin = B
            if bmax is None:
                bmax = B
            bmin = np.minimum(bmin, B)
            bmax = np.maximum(bmax, B)

    ax0.set_ylim(0,maxB)
    title = '[u,v], dec=%5.1f'%(decUV)
    if not tObs is None:
        if isinstance(tObs, list) and len(tObs)==len(quadruplets):
            title += '\nfor '+', '.join(['%3.1f'%float(x) for x in tObs])+'h'
        else:
            title += ' for %3.1fh'%float(tObs)
    ax0.set_title(title)
    ax0.get_xaxis().set_visible(False)

    if beam:
        ax1 = plt.subplot(141)
    elif withSky:
        ax1 = plt.subplot(131)
    else:
        ax1 = plt.subplot(121)

    ax1.set_axis_off()
    ax1.set_aspect('equal')
    #-- draw UTs
    #ax1.plot([layout[k][2] for k in layout.keys() if 'U' in k],
    #    [layout[k][3] for k in layout.keys() if 'U' in k], 'o', color='0.8',
    #    markersize=25)
    _t = np.linspace(0, np.pi, 20)
    for k in layout:
        if k.startswith('U'):
            ax1.plot(layout[k][2] + 16*np.cos(_t),
                     layout[k][3] + 16*np.sin(_t), '-', color='b', alpha=0.3)
            ax1.plot(layout[k][2] + 16*np.cos(-_t),
                     layout[k][3] + 16*np.sin(-_t), '-', color='b', alpha=0.3)
            ax1.fill_between(layout[k][2] + 16*np.cos(_t),
                             layout[k][3] - 16*np.sin(_t),
                             layout[k][3] + 16*np.sin(_t), color='b', alpha=0.1)

    #-- draw ATs
    ax1.plot([layout[k][2] for k in layout.keys() if not 'U' in k],
        [layout[k][3] for k in layout.keys() if not 'U' in k], 'o', color='w',
        markersize=10)
    #-- draw ruler
    x0, y0 = 17,43
    for i in range(14):
        x1, x2 = i*10., (i+1)*10.
        y1, y2 = 0,0
        X1 =  x1*np.cos(layout_orientation*np.pi/180) + y1*np.sin(layout_orientation*np.pi/180)
        X2 =  x2*np.cos(layout_orientation*np.pi/180) + y2*np.sin(layout_orientation*np.pi/180)
        Y1 = -x1*np.sin(layout_orientation*np.pi/180) + y1*np.cos(layout_orientation*np.pi/180)
        Y2 = -x2*np.sin(layout_orientation*np.pi/180) + y2*np.cos(layout_orientation*np.pi/180)

        ax1.plot([X1-x0,X2-x0], [Y1-y0,Y2-y0], color='0.8' if i%2==0 else '0.2',
                 linewidth=2)
        if i%2==0:
            ax1.text(X1-x0, Y1-y0, '%2dm'%(i*10), ha='center', va='bottom', size=8,
                     rotation=-layout_orientation)
        # ax1.plot([30+i*10, 30+(i+1)*10], [-110,-110],
        #          color='0.8' if i%2==0 else '0.2', linewidth=2)
    ax1.text(X2-x0, Y2-y0, '%2dm'%(i*10+10), ha='center', va='bottom', size=8,
             rotation=-layout_orientation)

    if False: # draw stations with STS M12 as of 2014
        sts = ['A0', 'A1', 'G2', 'I1', 'J2']
        # ax1.plot([layout[k][2] for k in sts],
        #          [layout[k][3] for k in sts], '*',
        #          color=(0.2,0.3, 0.8), markersize=22, alpha=0.8)
        for q in quadruplets:
            for k in q.split('-'):
                if not k in sts:
                    ax1.plot(layout[k][2], layout[k][3], 'p',
                            color=(0.8,0.3,0.1), markersize=18, alpha=0.8)
                else:
                    ax1.plot(layout[k][2], layout[k][3], '*',
                            color=(0.2,0.3,0.8), markersize=22, alpha=0.8)

    #-- name of station
    for k in layout.keys():
        ax1.text(layout[k][2], layout[k][3], k, ha='center', va='center',
                    color='k', fontsize=8+4*('U' in k), weight='black',
                    rotation=-layout_orientation)
    #-- baselines:
    for k,q in enumerate(quadruplets):
        T = q.split('-')
        label = q
        for i in range(len(T)+1):
            for j in range(len(T))[i+1:]:
                plt.plot([layout[T[i][:2]][2], layout[T[j][:2]][2]],
                    [layout[T[i][:2]][3], layout[T[j][:2]][3]],'-', linewidth=3,
                    alpha=1./np.sqrt(len(T)), color=colors[k], label=label)
                label=''
    ax1.legend(loc='upper left', prop={'size':11})
    plt.plot([-20,-20,-13], [40,30,30], '-', linewidth=1.5, color='0.5')
    plt.text(-20, 41, 'N', va='bottom', ha='center')
    plt.text(-12, 30, 'E', va='center', ha='left')
    _x, _y = np.array([48, 48]), np.array([70,-120])
    plt.plot(np.cos(np.deg2rad(layout_orientation))*_x + np.sin(np.deg2rad(layout_orientation))*_y,
            -np.sin(np.deg2rad(layout_orientation))*_x + np.cos(np.deg2rad(layout_orientation))*_y,
            color='k', linestyle='dotted', alpha=0.2)

    ax1.set_xlim(-32,140)
    ax1.set_ylim(-103,103)


    if 'U5' in layout.keys():
        ax1.set_xlim(-60,200)
        ax1.set_ylim(-180,120)
        # -- show control building
        Ur, Vr = np.array(cb[::2]), np.array(cb[1::2])
        Xr = np.cos(layout_orientation*np.pi/180)*Ur+\
            np.sin(layout_orientation*np.pi/180)*Vr
        Yr = -np.sin(layout_orientation*np.pi/180)*Ur+\
             np.cos(layout_orientation*np.pi/180)*Vr
        plt.plot(Xr, Yr, '-', color='m', linewidth=3, alpha=0.2)
        # -- show road
        Ur, Vr = np.array(road1[::2]), np.array(road1[1::2])
        Xr = np.cos(layout_orientation*np.pi/180)*Ur+\
            np.sin(layout_orientation*np.pi/180)*Vr
        Yr = -np.sin(layout_orientation*np.pi/180)*Ur+\
             np.cos(layout_orientation*np.pi/180)*Vr
        plt.plot(Xr, Yr, '-', color='0.5', linewidth=5, alpha=0.1)

        Ur, Vr = np.array(road2[::2]), np.array(road2[1::2])
        Xr = np.cos(layout_orientation*np.pi/180)*Ur+\
            np.sin(layout_orientation*np.pi/180)*Vr
        Yr = -np.sin(layout_orientation*np.pi/180)*Ur+\
             np.cos(layout_orientation*np.pi/180)*Vr
        plt.plot(Xr, Yr, '-', color='0.5', linewidth=5, alpha=0.1)

        # -- edge of the mountain
        Ur, Vr = np.array(edge[::2]), np.array(edge[1::2])
        _c = np.polyfit(Ur, Vr, 4)
        _Ur = np.linspace(min(Ur), max(Ur), 100)
        _Vr = np.polyval(_c, _Ur)
        #Xr = np.cos(layout_orientation*np.pi/180)*Ur+\
        #    np.sin(layout_orientation*np.pi/180)*Vr
        #Yr = -np.sin(layout_orientation*np.pi/180)*Ur+\
        #     np.cos(layout_orientation*np.pi/180)*Vr
        #plt.plot(Xr, Yr, 'o', color='r', linewidth=2, alpha=0.1)
        Xr = np.cos(layout_orientation*np.pi/180)*_Ur+\
            np.sin(layout_orientation*np.pi/180)*_Vr
        Yr = -np.sin(layout_orientation*np.pi/180)*_Ur+\
             np.cos(layout_orientation*np.pi/180)*_Vr
        plt.plot(Xr, Yr, '-', color='r', linewidth=2, alpha=0.1)



    if 'J7' in layout.keys():
        ax1.set_xlim(-24, 140+100)
        ax1.set_ylim(-103-60, 103+40)

    # -- HA coverage
    polarPlot = True
    if beam:
        ax2 = plt.subplot(142, polar=polarPlot)
    elif withSky:
        ax2 = plt.subplot(132, polar=polarPlot)
    else:
        return

    if not polarPlot:
        # ---------------------------
        # -- HA / Dec plot
        dec = np.linspace(-85,40,100)
        lst = np.linspace(-8,8,100)
        obsAll = dec[:,None]*lst[None,:]*0.0
        airmass = dec[:,None]*lst[None,:]*0.0
        for k,q in enumerate(quadruplets):
            obs = dec[:,None]*lst[None,:]*0.0
            for i,d in enumerate(dec):
                #__c = np.cos(d*np.pi/180.)
                __c = 1
                x = nTelescopes(q.split('-'),[0.0, d],
                                np.maximum(np.minimum(lst/__c, 8),-8),
                                flexible=flexible, max_OPD=max_OPD[k],
                                min_altitude=_min_alt)
                w = np.where(x['observable'])
                if len(w[0])>0:
                    obs[i,w]+=1.0
                airmass[i,:] = np.minimum(x['airmass'], 5)

            ax2.contour(lst, dec, obs, 1, colors=[colors[k]],
                            linewidths=3, label=q, alpha=0.8)
            obsAll += obs

            cdict = {'red'  : ((0.,1,1),
                               (1.,colors[k][0],colors[k][0])),
                     'green': ((0.,1,1),
                               (1.,colors[k][1],colors[k][1])),
                     'blue' : ((0.,1,1),
                               (1.,colors[k][2],colors[k][2]))}
            myCmap = LinearSegmentedColormap('myCmap'+str(k), cdict)
            ax2.pcolormesh(lst,dec, obs, cmap=myCmap, alpha=0.3,
                           edgecolors='None')

        #ax2.pcolormesh(lst,dec, obsAll, cmap='bone', vmin=vmin, vmax=vmax)
        levels = [1.05, 1.15, 1.30]
        CS = ax2.contour(lst, dec , airmass, levels, colors='k',
                         linewidths=2, alpha=0.5)
        ax2.clabel(CS, levels, fmt='%3.2f', inline=1)
        levels = [1.5]
        CS = ax2.contour(lst, dec , airmass, levels, colors='k',
                         linewidths=2, alpha=0.5)
        ax2.clabel(CS, levels, fmt='%3.1f', inline=1)
        levels = [2.0, 2.5]
        CS = ax2.contour(lst, dec , airmass, levels, colors='k',
                         linewidths=2, alpha=0.5)
        ax2.clabel(CS, levels, fmt='%3.1f', inline=1)

        ax2.set_ylim(-88,40)
        ax2.set_xlim(-6.5,6.5)
        ax2.set_xlabel('Hour Angle')
        ax2.set_ylabel('declination')
        ax2.grid()
    else:
        # --------------------
        # -- sky plot
        dec = np.linspace(-85,60,180)
        lst = np.linspace(-8,8,180)
        obsAll = dec[:,None]*lst[None,:]*0.0
        airmass = dec[:,None]*lst[None,:]*0.0
        alt, az = dec[:,None]*lst[None,:]*0.0, dec[:,None]*lst[None,:]*0.0
        for k,q in enumerate(quadruplets):
            obs = dec[:,None]*lst[None,:]*0.0
            for i,d in enumerate(dec):
                #__c = np.cos(d*np.pi/180.)
                __c = 1
                x = nTelescopes(q.split('-'),[0.0, d],
                                np.maximum(np.minimum(lst/__c, 8),-8),
                                flexible=flexible, max_OPD=max_OPD[k],
                                min_altitude=_min_alt)
                w = np.where(x['observable'])
                if len(w[0])>0:
                    obs[i,w]+=1.0
                airmass[i,:] = np.minimum(x['airmass'], 5)
                alt[i,:] = x['alt']
                az[i,:] = x['az']
            az_ = np.linspace(0,360,45)
            alt_ = 0.0*az_
            for i in range(len(az_)):
                alt_[i] = np.max((np.abs(az-az_[i])<np.diff(az).mean())*alt*(1-obs))
            ax2.fill_between((az_-90)*np.pi/180, 90-alt_, 90,
                       color=colors[k], linewidths=2, label=q,
                       alpha=0.35, hatch=hatchs[k%len(hatchs)])
        ax2.set_ylim(0,90-_min_alt)
        ax2.get_yaxis().set_visible(False)
        ax2.get_xaxis().set_visible(False)
        # -- lines:
        altaz = []
        lst = np.linspace(-8,8,101)
        for d in [-75, -60, -45, -30, -15, 0, 15, 30]:#, 45]:
            x = nTelescopes(['U1','U2'] ,[0.0, d], lst,
                            max_OPD=max_OPD[k], min_altitude=_min_alt)
            ax2.plot((x['az']-90)*np.pi/180, 90-x['alt'], '-', color='k',
                     linewidth = 1+float(d==0), alpha=0.5)
            if d<31: # print DEC
               ax2.text((x['az'][len(lst)//2]-90)*np.pi/180,
                        90-x['alt'][len(lst)//2],
                        '%2d$^o$'%d, color='k',
                        ha='center', va='bottom', alpha=0.5, fontsize=10)

            x = nTelescopes(['U1','U2'] ,[0.0, d], np.arange(-8,9),
                            max_OPD=max_OPD[k], min_altitude=_min_alt)
            altaz.append((x['az'], x['alt']))
            if d==-30: # print HA
                for l in range(len(x['az'])):
                    if np.abs(x['lst'][l])<5 and x['lst'][l]!=0:
                        plt.text((x['az'][l]-90)*np.pi/180,
                                 90-x['alt'][l], '%1dh'%int(x['lst'][l]),
                                 color='k', rotation=x['lst'][l]*5,
                                 va='bottom', ha='center', alpha=0.5,
                                 fontsize=10)

        for k in range(len(altaz[0][0])):
            plt.plot([(a[0][k]-90)*np.pi/180 for a in altaz],
                     [90-a[1][k] for a in altaz], '-', color='k',
                     alpha=0.5)

        alt = np.linspace(90,20,100)
        czt = np.cos(np.pi*(90-alt)/180.)
        airmass = 1/czt*(1-.0012*(1/czt**2-1))

        color3=(0.1, 0.2, 0.4)
        for a in [1.04,1.15,1.5]:
            ax2.plot(np.linspace(0,np.pi*2,100),
                     90-np.interp(a, airmass, alt)*np.ones(100),
                     color=color3, alpha=0.25, linewidth=2)
            ax2.text(-np.pi/4., 90-np.interp(a, airmass, alt),
                     str(a), color=color3, alpha=1, rotation=45,
                     ha='center', va='center', size=10)
            ax2.text(-np.pi/4.+np.pi, 90-np.interp(a, airmass, alt),
                     str(a), color=color3, alpha=1, rotation=45,
                     ha='center', va='center', size=10)
            ax2.text(np.pi/4., 90-np.interp(a, airmass, alt),
                     r'%2.0f$^\mathrm{o}$'%(round(np.interp(a, airmass, alt)/5,0)*5),
                     color=color3, alpha=1, rotation=-45,
                     ha='center', va='center', size=10)
            ax2.text(np.pi/4.+np.pi, 90-np.interp(a, airmass, alt),
                     r'%2.0f$^\mathrm{o}$'%(round(np.interp(a, airmass, alt)/5,0)*5),
                     color=color3, alpha=1, rotation=-45,
                     ha='center', va='center', size=10)

    if beam:
        ax4 = plt.subplot(144)
        beamFromUV([np.array(u),np.array(v)], ax=ax4)
    return

def beamFromUV(uv, wl=2.0, ax=None):
    """
    compute the dirty beam for given u,v tracks.

    uv: [u, v] tracks in meters
    wl: wavelength in microns
    """
    import dpfit
    N, Nuv = 60, 120

    uvmax = np.sqrt(max(uv[0]**2+uv[1]**2))
    _u, _v = np.linspace(-uvmax,uvmax,Nuv), np.linspace(-uvmax,uvmax,Nuv)
    _uv = np.meshgrid(_u,_v)
    grid = np.zeros((Nuv,Nuv))*1j
    grid[Nuv//2, Nuv//2] = 1.
    for k in range(len(uv[0])):
        i = np.argmin(np.abs(_u-uv[0][k]))
        j = np.argmin(np.abs(_v-uv[1][k]))
        grid[i, j] = 1
        i = np.argmin(np.abs(_u+uv[0][k]))
        j = np.argmin(np.abs(_v+uv[1][k]))
        grid[i, j] = 1

    # -- map in mas
    x, y = np.linspace(-10,10,N), np.linspace(-10,10,N)
    xy = np.meshgrid(x,y)

    ima = np.zeros((N,N))*1j

    for i in range(N):
        for j in range(N):
            dw = np.array([0.9, 0.95, 1, 1.05, 1.1])
            ima[i,j] += np.sum(grid[:,:,None]*np.exp(1j*2.*np.pi*np.pi/(180*3600.*1000)*
                 (_uv[0][:,:,None]*xy[0][i,j] +
                  _uv[1][:,:,None]*xy[1][i,j])/(wl*dw[None,None,:]*1e-6)))

    #ima = vis.mean(axis=2) # sum on the u,v coordinates
    ima = np.abs(ima*np.conj(ima))
    ima /= np.max(ima)

    #Bmax = np.sqrt(np.max(uv[0]**2+uv[1]**2))
    #print('BMAX=', Bmax)
    #print('lambda/Bmax = ', wl*1e-6/Bmax*180*3600*1000/np.pi, 'mas')

    #-- plot
    if ax is None:
        plt.figure(10)
        plt.clf()
        ax = plt.subplot(111)

    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    fig1 = ax.get_figure()
    fig1.add_axes(ax_cb)

    ax.set_aspect('equal')

    if False:
        P = N/8 # half size of the fit, in pixels
        param = {'O':-1.0, 'AMP':1.0,
                 'sX':1.5, 'sY':2.0,
                 'X0':0.0, 'Y0':0.0}

        # -- dirty beam core fit
        fit = dpfit.leastsqFit(gauss2d,
                               [xy[0][N//2-P:N//2+P,N//2-P:N//2+P].flatten(),
                                xy[1][N//2-P:N//2+P,N//2-P:N//2+P].flatten()],
                               param, ima[N//2-P:N//2+P,N//2-P:N//2+P].flatten(),
                               verbose=True, #doNotFit=['X0', 'Y0']
                               )
        # -- dirty beam core model
        mod = gauss2d(np.array(xy), fit['best']).reshape((N,N))
        sX, sY = fit['best']['sX'], fit['best']['sY']
    else:
        P = N/8
        param = {'O':-1.0, 'AMP':1.0,
                 'sX':1.5, 'sY':2.0,
                 'X0':0.0, 'Y0':0.0,
                 'beta':3.}
        # -- dirty beam core fit
        fit = dpfit.leastsqFit(moffat2d,
                               [xy[0][N//2-P:N//2+P,N//2-P:N//2+P].flatten(),
                                xy[1][N//2-P:N//2+P,N//2-P:N//2+P].flatten()],
                               param, ima[N//2-P:N//2+P,N//2-P:N//2+P].flatten(),
                               verbose=True, doNotFit=['beta', 'X0', 'Y0']
                               )
        # -- dirty beam core model
        mod = moffat2d(np.array(xy), fit['best']).reshape((N,N))
        co = np.sqrt(2**(1/fit['best']['beta'])-1)
        sX, sY = co*fit['best']['sX'], co*fit['best']['sY']

    #x /= np.pi/(180*3600.*1000)
    #y /= np.pi/(180*3600.*1000)
    p = ax.pcolormesh(x,y, np.sqrt(ima), cmap='gnuplot', vmin=0, vmax=1)
    #p = ax.pcolormesh(x,y, ima-mod, cmap='gnuplot', vmin=0, vmax=1)

    c = 2*np.sqrt(2*np.log(2))
    ax.text(x.min()+0.02*x.ptp(), y.max()-0.02*y.ptp(),
                "%3.1fx%3.1fmas @%3.1fum"%(c*sX,c*sY, wl), color='y',
                ha='left', va='top', weight='bold', fontsize=18)
    strehl = mod[N//2-4*P:N//2+4*P,N//2-4*P:N//2+4*P].sum()/\
             ima[N//2-4*P:N//2+4*P,N//2-4*P:N//2+4*P].sum()
    strehl = mod.sum()/ima.sum()

    ax.text(x.max()-0.02*x.ptp(), y.min()+0.02*y.ptp(),
                "strehl = %2.0f%%"%(100*strehl), color='y',
                ha='right', va='bottom', weight='bold', fontsize=18)

    #-- ellipse of the beam
    # t = np.linspace(0, 2*np.pi, 100)
    # _x = sX*np.cos(t)*c/2  + fit['best']['X0']
    # _y = sY*np.sin(t)*c/2  + fit['best']['Y0']
    # ax.plot(_x*np.cos(fit['best']['O'])+_y*np.sin(fit['best']['O']),
    #         _y*np.cos(fit['best']['O'])-_x*np.sin(fit['best']['O']),
    #         '-y', linewidth=2, linestyle='dashed')

    # -- fitting box:
    #ax.plot([x[N//2-P], x[N//2-P], x[N//2+P], x[N//2+P], x[N//2-P]],
    #        [y[N//2-P], y[N//2+P], y[N//2+P], y[N//2-P], x[N//2-P]], '-y',
    #        linestyle='dashed')

    _y = np.linspace(0,1,11)
    cbar = plt.colorbar(p, cax=ax_cb, ticks=_y)
    cbar.ax.set_yticklabels(['%4.1f'%(a) for a in _y])

    ax.set_title('dirty beam')
    ax.set_xlabel('(mas)')
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())

    return

def V_UD(bl, a):
    """
    uniform disk diameter interferometric visibility btl is
    Baseline(m)/wavelength(um), a is UD diameter in mas
    """
    if isinstance(a, float):
        a = [a]
    c = np.pi*a[0]*np.pi/(180*3600*1000.0)*1e6
    return 2*(scipy.special.jv(1,c*bl)+\
              np.float64(bl==0)*1e-6)/\
           (c*bl +np.float64(bl==0)*2e-6)

def gauss2d(xy, param):
    """
    'O': orientation (radians)
    'AMP': amplitude
    'X0', 'Y0': center
    'sX', 'sY': FWHM
    """
    X = xy[0]
    Y = xy[1]
    XP = X*np.cos(param['O']) - Y*np.sin(param['O'])
    YP = X*np.sin(param['O']) + Y*np.cos(param['O'])
    res= param['AMP']*np.exp(-(XP-param['X0'])**2/(np.sqrt(2)*param['sX'])**2 -
                             (YP-param['Y0'])**2/(np.sqrt(2)*param['sY'])**2)
    return res

def moffat2d(xy, param):
    """
    'O': orientation (radians)
    'AMP': amplitude
    'X0', 'Y0': center
    'sX', 'sY', 'beta': alpha X, alpha Y, beta

    http://en.wikipedia.org/wiki/Moffat_distribution
    http://www.aspylib.com/doc/aspylib_fitting.html#elliptical-moffat-psf

    fwhm = sX * 2 * np.sqrt(2**(1/beta)-1)
    """
    X = xy[0]
    Y = xy[1]
    XP = X*np.cos(param['O']) - Y*np.sin(param['O'])
    YP = X*np.sin(param['O']) + Y*np.cos(param['O'])
    res = param['AMP']*(1+XP**2/param['sX']**2+YP**2/param['sY']**2)**(-1.*param['beta'])
    return res

def obsSimulator(obs, dec=-20, tConf=None, obLength=20, max_airmass=2.5, fig=10):
    """
    obs is a list of observations coded as: (quadruplet, N, accuracy)
        - quadruplet: one of the name in "quadruplets" dictionnary
        - N: number of observation
        - accuracy is in ['low', 'normal', 'high'], corresponding to S-C,
            C-S-C or C-S-C-S-C sequences

    dec: declination in degrees (-20 default)

    tConf: dict containing the telescope configurations by name. Default is set
    if not given for "small", "medium", "large" and "UT". e.g.
    {'small':'A1-B2-C1-D0', 'medium':'D0-H0-G1-I1', 'large':'A1-K0-G1-I1',
    'UT':'U1-U2-U3-U4'}

    obLength: time to execute an OB (in minutes)

    airmassMax = 2.5 default

    """
    # -- parameters
    if tConf is None:
        tConf = {'small':'A1-B2-C1-D0',
                 'medium':'D0-H0-G1-I1',
                 'large':'A1-K0-G1-I1',
                 'UT':'U1-U2-U3-U4'}
    nOBsaccuracy = {'low':2, 'normal':3, 'high':5}

    # -- compute all observabilities for each offered, HA steps is OB length:
    N = int(7*60./obLength)
    ha = np.linspace(-N*obLength/60., N*obLength/60., 2*N+1)
    observabilities = {k:nTelescopes(tConf[k].split('-'), (0, dec), ha,
                                     max_airmass=max_airmass)
                        for k in tConf.keys()}
    # -- compare time available and requested
    for k in sorted(tConf.keys()):
        # -- available:
        tmp1 = (observabilities[k]['observable'].sum()-1)*obLength/60.
        print('observable %3.1fh on %6s,'%(tmp1,k), end='')
        # -- requested:
        tmp2 = np.sum([o[1]*nOBsaccuracy[o[2]]*obLength for o in obs if o[0]==k])
        print('requested %3.1fh'%(tmp2/60.))

    # -- plot observability
    plt.figure(fig)
    plt.clf()
    for i,k in enumerate(sorted(tConf.keys())):
        w = np.where(observabilities[k]['observable'])
        plt.plot(observabilities[k]['lst'][w],
                 observabilities[k]['observable'][w]+i,
                 'o-')
        plt.text(-6, 1+i, k, color='k', va='center')
        # --
        blocks = []
        for o in obs:
            if k in o[0]:
                blocks.extend([nOBsaccuracy[o[2]]*obLength]*o[1])
        if len(blocks)>0:
            blocks =np.array(blocks)/60.
            pos = _setBlockInRange(blocks,
                                   observabilities[k]['lst'][w].min(),
                                   observabilities[k]['lst'][w].max())
            blocks = blocks[np.argsort(pos)]
            pos = np.array(pos)[np.argsort(pos)]
            for j in range(len(pos)):
                plt.plot([pos[j]-blocks[j]/2., pos[j]+blocks[j]/2.],
                         (1+i+(j+1)/float(len(blocks)+1))*np.ones(2),
                          linewidth=5, alpha=0.5, color='k')


    plt.xlabel('HA (h)')
    plt.xlim(-6.5, 6.5)
    plt.ylim(0.5, len(tConf)+1.5)
    plt.grid()
    return

def _setBlockInRange(blocksLength, range_min, range_max, plot=False):
    """
    put blocks of length contains in the list "blocksLength" within a range of
    range_min, range_max, trying to get as close as 0 as possible.
    """
    global _blocksL, _range_min, _range_max
    _blocksL = np.array(blocksLength)
    _range_min = range_min
    _range_max = range_max

    pos = np.cumsum(blocksLength)
    pos -= blocksLength[0]/2.
    pos -= pos.mean()

    # -- actual fit
    plsq, cov, info, mesg, ier = scipy.optimize.leastsq(_chi2FuncBIR, pos,
                    full_output=True, ftol=1e-4, epsfcn=1e-6)
    #print(plsq)
    #_meritFuncBIR(np.array(plsq), np.array(blocksLength), range_min, range_max, verbose=1)

    if plot:
        plt.figure(0)
        plt.clf()
        for k in range(len(blocksLength)):
            plt.plot([plsq[k]-blocksLength[k]/2.,
                      plsq[k]+blocksLength[k]/2.], [k,k],
                     '-', linewidth=3 )
            plt.vlines([plsq[k]-blocksLength[k]/2.,
                      plsq[k]+blocksLength[k]/2.],
                      -0.5, len(blocksLength)-0.5, linestyle='dotted')
        plt.vlines([range_min, range_max], -0.5, len(blocksLength)-0.5)
        plt.vlines([0], -0.5, len(blocksLength)-0.5,
                   linestyle='dashed')
        plt.xlim(range_min-0.05*(range_max-range_min),
                 range_max+0.05*(range_max-range_min))
    return plsq

def _chi2FuncBIR(pos):
    global _blocksL, _range_min, _range_max
    res = _meritFuncBIR(np.array(pos), np.array(_blocksL), _range_min, _range_max)
    #print('pos:', pos, np.sum(res))
    return res

def _meritFuncBIR(positions, blocksLength, range_min, range_max, verbose=False):
    """
    assumes positions and blocksLength are ndarrays of same length
    """
    # -- distance to 0
    res = list(0.000001*positions**2)
    if verbose:
        print('distance to 0:', np.sum(res))
    # -- overlap
    tmp = 2*(positions[None,:]-positions[:,None])/(blocksLength[None,:]+blocksLength[:,None])
    tmp = tmp**2
    if verbose:
        print(np.round(tmp, 2))
    tmp *= tmp<=1 # keep only overlap: 1 just overlap, 0 complete overlap
    tmp = (1-tmp)*(tmp>0)
    tmp += positions[None,:]==positions[:,None] # complete overlap
    for k in range(len(positions)):
        tmp[k,k]=0
    if verbose:
        print(np.round(tmp,2))
        print('overlap      :', np.sum(tmp))
    res.extend(list(tmp.flatten()))
    # -- outside range:
    tmp = (positions-blocksLength/2.-range_min)/(0.5*blocksLength)
    tmp *= tmp<0
    res.extend(list(1000*tmp**2))
    if verbose:
        print('below range  :', np.sum(tmp**2))
    tmp = (positions+blocksLength/2.-range_max)/(0.5*blocksLength)
    tmp *= tmp>0
    res.extend(list(1000*tmp**2))
    if verbose:
        print('above range  :', np.sum(tmp**2))
        print('-------------------------')
        print('total        :', np.sum(res))
        print('')
    else:
        return np.array(res)

def bCoverage(configs, dec=-24):
    """
    configs = ['U1-U2-U3-U4','A1-G1-K0-J3','D0-H0-G1-I1','A1-B2-C1-D0'] for example

    configs = ['A1-G1-K0', 'A1-G1-J3', 'A1-K0-J3', 'G1-J3-K0',]
    'D0-H0-G1', 'D0-H0-I1','D0-G1-I1'

    """
    lst = np.linspace(-6,6,200)
    colors = [(1,0,0), (0,1,0), (0,0,1),
              (1,0.5,0), (0,1,0.5), (0.5,0,1),
              (1,0,0.5), (0.5,1,0), (0,0.5,1)]
    colors = [(0.3+c[0]*0.5,
               0.3+c[1]*0.5,
               0.3+c[2]*0.5) for c in colors]
    plt.close(0)
    plt.figure(0)
    for i,c in enumerate(configs):
        tmp = nTelescopes(c.split('-'), (0.0, dec), lst)
        w = np.where(tmp['observable'])

        Bmin = lst[w]*0.0 + 1000.
        Bmax = lst[w]*0.0
        Xname = 'lst'
        #Xname = 'airmass'
        X = tmp[Xname]

        for j,k in enumerate(tmp['B']):
            Bmin = np.minimum(Bmin, tmp['B'][k][w])
            Bmax = np.maximum(Bmax, tmp['B'][k][w])
            plt.plot(X[w], tmp['B'][k][w], color=colors[i],
                     linewidth=3, alpha=0.5, label=c if j==0 else '')
        plt.fill_between(X[w], Bmin, Bmax, color=colors[i],
                         alpha=0.5)
    plt.legend(prop={'size':8})
    plt.title('dec = %6.1f'%(dec))
    plt.xlabel(Xname)
    plt.ylabel('baseline (m)')
    plt.grid()
    return

def decCoverage(T, Nhours=2, min_altitude=_min_alt, max_OPD=100, flexible=False):
    """
    return min and max declination for wich an object is observable more than N
    hours.
    """
    N=100
    dec = np.linspace(-90, 40, N)
    coverage=[]
    for d in dec:
        lst = np.linspace(-12,12,10+int(100*np.cos(d*np.pi/180)))
        x = nTelescopes(T,[0.0, d], lst, min_altitude=min_altitude,
                        flexible=flexible, max_OPD=max_OPD)
        w = np.where(x['observable'])
        coverage.append(np.mean(np.diff(x['lst']))*len(w[0]))
    coverage=np.array(coverage)
    if len(T)==2:
        B =  np.sqrt((layout[T[0]][2]-layout[T[1]][2])**2+
                     (layout[T[0]][3]-layout[T[1]][3])**2)
    else:
        B = None
    try:
        return dec[coverage>=Nhours].min(), dec[coverage>=Nhours].max(), B
    except:
        return 0.,0.

def configForFINITO():
    quads = [['A1DL5','G1DL6','K0DL4','J3DL3'],
             ['A1DL5','G1DL6','K0DL4','I1DL3'],
             ['D0DL5','H0DL4','G1DL6','I1DL3'],
             ['A1DL5','B2DL1','C1DL2','D0DL6'],
             ['U1DL1','U2DL2','U3DL3','U4DL4'],
            ]
    for q in quads:
        print('###', '-'.join(q))
        for Ts in itertools.combinations(q, 3):
            #print('-'.join(Ts) , '->', end='')
            for k in [0,1,2]:
                Tp = np.roll(Ts, k)
                if np.sum(baseline(Tp[1][:2], Tp[0][:2])**2)<\
                        np.sum(baseline(Tp[2][:2], Tp[0][:2])**2) and \
                   np.sum(baseline(Tp[2][:2], Tp[1][:2])**2)< \
                        np.sum(baseline(Tp[2][:2], Tp[0][:2])**2):
                    print('-'.join([Tp[t]+'IP'+str(2*t+1) for t in [0,1,2]]),end='')
            print('')
        print('')
    return

def plotAllConfigs(disp_vcmPressure=False, max_vcmPressure=None):
    if disp_vcmPressure or max_vcmPressure:
        quads = [['A1DL5','G1DL6','K0DL4','J3DL3'],
                 ['D0DL5','H0DL4','G1DL6','I1DL3'],
                 ['A1DL5','B2DL1','C1DL2','D0DL6'],
                 ['U1DL1','U2DL2','U3DL3','U4DL4']
                 ]
    else:
        quads = [['A1','G1','K0','J3'],
                 ['D0','H0','G1','I1'],
                 ['A1','B2','C1','D0'],
                 ['U1','U2','U3','U4']]

    filename = 'OBSERVABILITY/'
    if disp_vcmPressure:
        filename += 'vcm_'
    if max_vcmPressure:
        filename += 'vcm%3.1f_'%max_vcmPressure
    filename = filename.replace('.', '')

    for q in quads:
        for t1 in q:
            for t2 in q:
                if t1<t2:
                    print(t1, t2)
                    skyCoverage([t1,t2],
                        disp_vcmPressure=disp_vcmPressure,
                        max_vcmPressure=max_vcmPressure,
                        createXephemHorizon=True)
                    plt.savefig(filename+t1+'-'+t2+'.pdf')
                    for t3 in q:
                        if t2<t3:
                            print(t1, t2, t3)
                            skyCoverage([t1,t2,t3],
                                disp_vcmPressure=disp_vcmPressure,
                                max_vcmPressure=max_vcmPressure,
                                createXephemHorizon=True)
                            plt.savefig(filename+t1+'-'+t2+'-'+t3+'.pdf')
                            for t4 in q:
                                if t3<t4:
                                    print(t1, t2, t3, t4)
                                    skyCoverage([t1,t2,t3,t4],
                                        disp_vcmPressure=disp_vcmPressure,
                                        max_vcmPressure=max_vcmPressure,
                                        createXephemHorizon=True)
                                    plt.savefig(filename+t1+'-'+t2+'-'+t3+'-'+t4+'.pdf')
    return

def solveDLposition(dl, opd, dlPosMin=11/2., dlPosMax=111/2.,
                    stations=None, STS=False, constraints=None):
    """
    dl: list of delay lines numbers, like [1,3,2]
    opd: dict of OPDs for pairs of delay lines, in meters, for example,
    {(1,2):12.0, (3,1):-34.22, (3,2):0.0}. Convention is 2*(DL2-DL1)=12.0,
    2*(DL1-DL3)=-34.22, because the OPL is twice the DL position

    if a list of stations is given (same order as dl), VCM pressure will be
    computed.

    constraints: specific constraints for each DL, override the dlPosMin and
    dlPosMax which are global. e.g.: constraints={'min3':20, 'max5':40} means
    DL3 must be atleast at 20 m (position, i.e. 40m delay) and DL5 has to be
    at most at position 40 (i.e. delay 80)

    a bit smarter than the previous version: check len(dl) configuration,
    every time putting one of the DL closest to the lab.

    Author: amerand
    """
    # -- build default range constraints
    cons = {}
    for d in [1,2,3,4,5,6]:
        cons['min'+str(d)] = dlPosMin
        cons['max'+str(d)] = dlPosMax
        if _test_dl34_plus30 :
            # -- test the effect of extending DL3,4 for 30 OPL:
            if d in [3,4]:
                cons['max'+str(d)] += 30

    if not constraints is None and isinstance(constraints, dict):
        for k in constraints.keys():
            if k in cons.keys():
                cons[k] = constraints[k]/2.
            else:
                print('WARNING: unknown constraint', k)

    # -- initialise dict used to track optimum solution:
    best = {}
    for d in dl:
        best[d] = np.nan
        best['vcm'+str(d)] = 5

    for i in range(len(dl)): # -- for each DL
        # -- try to put the DL dl[i] at minimum position:
        dlPos = {dl[i]:cons['min'+str(dl[i])]}
        # -- explore all other DL positions
        while len(dlPos.keys())!=len(dl):
            for o in opd.keys(): # -- for each requested OPD
                # -- set DL pos according to OPD
                if o[0] in dlPos.keys() and not o[1] in dlPos.keys():
                    dlPos[o[1]] = dlPos[o[0]] - opd[o]/2.
                elif o[1] in dlPos.keys() and not o[0] in dlPos.keys():
                    dlPos[o[0]] = dlPos[o[1]] + opd[o]/2.
                #print(dlPos, dlPos.keys(), o[0], o[1])
        # -- check that DL carts are within optional constraints:
        test = all([dlPos[k]<=cons['max'+str(k)] for k in dlPos.keys()])
        test = all([dlPos[k]>=cons['min'+str(k)] for k in dlPos.keys()]) and test

        # -- find best position
        if test:
            if not stations is None:
                #-- compute VCM pressure
                for k,d in enumerate(dl):
                    dlPos['vcm'+str(d)] = computeVcmPressure(stations[k],
                                                             d,1,dlPos[d], STS=STS)
                # -- try to keep ALL VCM pressure to a minimum
                if np.max([dlPos['vcm'+str(d)] for d in dl]) <=\
                   np.max([best['vcm'+str(d)] for d in dl]):
                   best = dlPos.copy()
                # -- try to keep DL closer to the lab
                #if np.max([dlPos[d] for d in dl])<=np.max([best[d] for d in dl]):
                #    best = dlPos.copy()

            else:
                # -- set vcm pressure to nan
                for d in dl:
                    dlPos['vcm'+str(d)] = np.nan
                if np.mean([dlPos[d] for d in dl])<=np.mean([best[d] for d in dl]):
                    best = dlPos.copy()
                #if np.max([dlPos[d] for d in dl]) <= np.max([best[d] for d in dl]):
                #    best = dlPos.copy()
        else:
            #print dlPos
            #print opd
            pass
    return best

#------------------------------------------------------------------------------
# VCM functions
# adapted from ComputeVCMParameters3.m, provided by Philippe Gitton
# email 12/03/2013
#------------------------------------------------------------------------------

def checkAllVCM():
    """
    Annex A of N Schuhler Memo's
    http://vlti.pl.eso.org/tec_doc/VLTISE-NSC-2012-013.pdf
    """
    OPL = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    conf = ['DL1-U1', 'DL1-B2', 'DL2-U2', 'DL2-C1', 'DL3-U3', 'DL3-I1',
            'DL3-J3', 'DL4-U4', 'DL4-K0', 'DL4-H0', 'DL5-A1', 'DL5-D0',
            'DL6-D0', 'DL6-G1']

    print('     ', end='')
    for o in OPL:
        print('%4.0f'%o, end='')
    print('')
    for c in conf:
        print(c, end='')
        for o in OPL:
            tmp = computeVcmPressure(c.split('-')[1], int(c[2]), 1, o/2.)
            tmp = max(min(tmp, 2.5), 0)
            print('%4.2f'%tmp,end='')
        print('')
    return

def computeVcmPressure(station, dl, ip, dlPos, verbose=False,
                       returnCurvature=False, STS=False):
    """
    dlPos is actually the DL position, so half the OPL!!! dl is a integer

    Author: amerand, adapted from pgitton
    """
    #assert dlPos>=-10 and dlPos <=70.0, 'dlPos must be in [0..60] m'
    #dlPos = min(dlPos, 60)
    #dlPos = max(dlPos, 0)

    global layout
    uvSta = {'u':layout[station][0],
             'v':layout[station][1]}
    uvDl = dlPosUV(dl)
    if dl in [1,2,5,6]:
        uvDl['u'] = uvDl['u0']+dlPos
    else:
        uvDl['u'] = uvDl['u0']-dlPos
    # Compute distance of relayed telescope exit pupil to DL input aperture
    if 'U' in station or STS:
        # ---------- UT and STS cases
        UTExitPupiluCoordinate = pupilUTuPos(dl)
        # Compute distance between u-position of DL and u-position of telescope
        # relayed exit pupil:
        DistanceInputPupil0 = abs(uvDl['u'] - UTExitPupiluCoordinate) - dlPos
        DistanceInputPupil = abs(uvDl['u'] - UTExitPupiluCoordinate)
    else:
        # ---------- AT case
        du, dv = pupilATdudv(station)
        # Set dTelPup for AT (distance between M11 and relayed pupil)
        dTelPup = 1.23;
        # Compute distance between entrance aperture of DL and telescope relayed
        # exit pupil:
        DistanceInputPupil0 = abs(uvDl['u'] - (uvSta['u'] + du)) +\
                         abs(uvDl['vInAstro'] - (uvSta['v'] + dv)) -\
                         dTelPup - dlPos
        DistanceInputPupil = abs(uvDl['u'] - (uvSta['u'] + du)) +\
                         abs(uvDl['vInAstro'] - (uvSta['v'] + dv)) -\
                         dTelPup
    # Compute distance between DL input aperture and re-imaged pupil ("behind
    # M16")

    # Get re-imaged pupil position (depending on the lab input channel selected
    # and use of BC)
    ReimagedPupilPositionWithBC = pupilReimagedPos(ip, BC=True)
    ReimagedPupilPositionNoBC = pupilReimagedPos(ip, BC=False)
    inputChannelUcoord = ReimagedPupilPositionWithBC['u']
    if 'U' in station or STS:
        # ---------- UT case
        DistanceExitPupil0 = abs(uvDl['u'] - ReimagedPupilPositionWithBC['u'])+\
                      abs(uvDl['vOutRef'] - ReimagedPupilPositionWithBC['v'])-\
                      dlPos
        DistanceExitPupil = abs(uvDl['u'] - ReimagedPupilPositionWithBC['u']) +\
                      abs(uvDl['vOutRef'] - ReimagedPupilPositionWithBC['v'])
    else:
        # ---------- AT case
        DistanceExitPupil0 = abs(uvDl['u'] - inputChannelUcoord) + \
                        abs(inputChannelUcoord - ReimagedPupilPositionNoBC['u'])+\
                        abs(uvDl['vOutAstro'] - ReimagedPupilPositionNoBC['v'])\
                        - dlPos
        DistanceExitPupil = abs(uvDl['u'] - inputChannelUcoord) + \
                        abs(inputChannelUcoord - ReimagedPupilPositionNoBC['u'])+\
                        abs(uvDl['vOutAstro'] - ReimagedPupilPositionNoBC['v'])

    radiusVCM=computeRadiusOfCurvatureVCM(DistanceInputPupil, DistanceExitPupil)
    # I checked these 3 values with respect to the original Matlab function,
    # and it is correct
    if verbose:
        print('DistanceInputPupil (m):', DistanceInputPupil)
        print('DistanceExitPupil (m):', DistanceExitPupil)
        print('radiusVCM (m):',radiusVCM)
        print('curvature (1/m):',1/radiusVCM)
    if returnCurvature:
        return 1./radiusVCM # m-1
    else:
        return vcmCurv2Press(1/radiusVCM, dl)

def computeRadiusOfCurvatureVCM(DistanceInputPupil, DistanceExitPupil):
    """
    Author: amerand, adapted from pgitton
    """
    # Optical parameters of the DL
    # ----------------------------
    f1  = 0.9
    f2  = -266.66e-3
    d12 = 0.7
    d23 = 0.8

    # f1 = optical focal length of M13 (DL primary)
    # f2 = optical focal length of M14 (DL secondary)
    # d12 = distance between vertices of M13 and M14 (measured along u-axis)
    # d23 = distance between vertices of M14 and M15 (the VCM) (measured along u-axis)

    # Do the computation
    # ------------------

    # Pin = distance between relayed telescope pupil and M13 vertex (measured along
    # u-axis) Pout = distance between exit pupil and M13 vertex (measured along
    # u-and v-axis (folded around M16)) Din, Dout = image distances (measured w. r.
    # t. M14 vertex) of the pupil image DinDash, DoutDash = same as Din and Dout,
    # but measured w. r. t. the vertex of M15 (VCM)

    Pin  = DistanceInputPupil + d12
    Pout = DistanceExitPupil + d12
    Din = f2 * (d12 - f1*Pin / (Pin - f1)) / \
            ((d12 - f1*Pin / (Pin - f1)) - f2)
    Dout = f2 * (d12 - f1*Pout / (Pout - f1)) / \
            ((d12 - f1*Pout / (Pout - f1) - f2))
    DinDash  = d23 - Din
    DoutDash = d23 - Dout
    return  -2*(DinDash * DoutDash) / (DinDash + DoutDash)

def pupilUTuPos(dl):
    """
        Author: amerand, adapted from pgitton
    """
    return {1:50.625,
            2:49.875,
            3:53.375,
            4:54.125,
            5:52.0, # dummy, for STS
            6:52.0, # dummy, for STS
            }[dl]

def pupilATdudv(station):
    """
    Author: amerand, adapted from pgitton
    """
    global layout
    if layout[station][1] >-40:
        # northern station
        return -0.889, 0.3594
    else:
        # southern station
        return 0.0889, -0.3594

def pupilReimagedPos(ip, BC=False):
    """
    Author: amerand, adapted from pgitton
    """
    # -- with beam compressor:
    if BC:
        return {1:{'u':52.320, 'v':-36.903},
                2:{'u':52.560, 'v':-36.903},
                3:{'u':52.800, 'v':-32.170},
                4:{'u':53.040, 'v':-32.170},
                5:{'u':53.280, 'v':-27.438},
                6:{'u':53.520, 'v':-27.438},
                7:{'u':53.760, 'v':-22.705},
                8:{'u':54.000, 'v':-22.705}}[ip]
    else:
        return {1:{'u':46.560, 'v':-34.215},
                2:{'u':46.560, 'v':-35.175},
                3:{'u':46.560, 'v':-33.975},
                4:{'u':46.560, 'v':-34.935},
                5:{'u':46.560, 'v':-33.735},
                6:{'u':46.560, 'v':-34.695},
                7:{'u':46.560, 'v':-33.495},
                8:{'u':46.560, 'v':-34.455}}[ip]

def dlPosUV(dl):
    """
    Author: amerand, adapted from pgitton
    """
    res={1:{'v':-37.125, 'u0':58.92},
         2:{'v':-37.875, 'u0':58.92},
         3:{'v':-38.625, 'u0':45.08},
         4:{'v':-39.375, 'u0':45.08},
         5:{'v':-40.625, 'u0':58.92},
         6:{'v':-41.375, 'u0':58.92},
         7:{'v':-42.125, 'u0':45.08},
         8:{'v':-42.875, 'u0':45.08}}[dl]
    if dl in [1,2,5,6]:
        res['vInRef']  = res['v'] - 0.12
        res['vOutRef'] = res['v'] + 0.12
    else: # -- for DL3 and DL4:
        res['vInRef']  = res['v'] + 0.12
        res['vOutRef'] = res['v'] - 0.12
    res['vInAstro']=res['vOutRef']
    res['vOutAstro']=res['vInRef']
    return res

def vcmCurv2Press(cu,dl=1):
    """
    Author: amerand, adapted from pgitton
    """
    C = {1:(0.41148610, 5.81454e-4, 4.42791e-6,  0.13007918),
         2:(0.43840233, 1.12713e-3, 5.34034e-6,  0.22333076),
         3:(0.39729838, 8.04944e-4, 4.55080e-6, -0.04608024),
         4:(0.43717096, 9.12421e-4, 4.00211e-6,  0.10589429),
         5:(0.40438133, 8.11113e-4, 4.37047e-6,  0.25118364),
         6:(0.39226216, 1.16507e-3, 5.13692e-6,  0.27546452)}
    return C[dl][3] + C[dl][0]*cu + C[dl][1]*cu**3 + C[dl][2]*cu**5
