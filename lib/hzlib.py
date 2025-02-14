# -- compute VLTI horizon

import numpy as np
from matplotlib import pyplot as plt
import os

# --     station   U        V        E        N       A0
layout_orientation = -18.984 # degrees
VST_d = 154
VST_p = 0.692
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
           # -- XY are correct, A0 is guessed!
          'I2':(80, -24, 83.456, 3.330, 98),
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
          'VS':(0,0,VST_d*np.sin(VST_p), VST_d*np.cos(VST_p), 0)}

# -- all in meters
AT_M1_HEIGHT = 4
AT_AX_HEIGHT = AT_M1_HEIGHT + 0.7
AT_M1_DIAM = 1.8
# http://www.eso.org/sci/facilities/paranal/telescopes/ut/vltfv6.gif
UT_M1_HEIGHT = 10.0
UT_AX_HEIGHT = 13.0
UT_M1_DIAM = 8.2
# http://www.eso.org/sci/facilities/paranal/telescopes/ut/vltsv6.gif
UT_DOME_HEIGHT = 28.5
# -- wrong vlaue from the original document!
#UT_DOME_HEIGHT = 25
UT_DOME_DIAM = 29.0

def UTshadow(S, plot=False, compare=False, partial=True):
    at = layout[S][2:4]
    uts = ['U1','U2','U3','U4','VS']
    if 'U' in S:
        uts.remove(S)
        TEL = (UT_AX_HEIGHT, UT_M1_HEIGHT, UT_M1_DIAM)
    else:
        TEL = None
    if plot:
        plt.close(0)
        if plot>1:
            plt.figure(0, figsize=(12,4))
            plt.subplots_adjust(bottom=0.12, right=0.96, left=0.08)
            p0 = plt.subplot(121)
        else:
            plt.figure(0, figsize=(8,5))
            plt.subplots_adjust(bottom=0.12, right=0.96, left=0.08)
            p0 = plt.subplot(111)

    az, el = None, None
    for i, u in enumerate(uts):
        if u == 'VS':
            UT = (17.0,20)
        else:
            UT = None

        ut = layout[u][2:4]
        if plot>1 and u!='VS':
            # p = 3+i if i<2 else 5+i
            # p1 = plt.subplot(2,4,p)
            p1 = plt.subplot(4,2,)
            p1.set_aspect('equal')

        if az is None:
            az, el = shadow(at, ut, TEL=TEL,
                            plot=plot>1 and u!='VS', partial=partial)
            if plot:
                p0.plot(az, el, '-b', linestyle='dotted', linewidth=2)
                if plot>1 and u!='VS':
                    p1.set_title(u)

        else:
            _az, _el = shadow(at, ut, TEL=TEL,
                        plot=plot>1 and u!='VS', partial=partial,UT=UT)
            if plot:
                p0.plot(_az, _el, '-b', linestyle='dotted', linewidth=2)
                if plot>1 and u!='VS':
                    p1.set_title(u)
            el = np.maximum(el, _el)

    if compare:
        hfiles = os.listdir('VLTI_horizons')
        hfiles = filter(lambda x: '.horizon' in x, hfiles)
        horizon = {}
        for h in hfiles:
            f = open(os.path.join('VLTI_horizons', h), 'r')
            lines = f.read().split('\n')[:-3] # last 2 lines are weird
            X = np.array([float(l.split()[0]) for l in lines])
            Y = np.array([float(l.split()[1]) for l in lines])
            horizon[h[:2]] = (X,Y)
            f.close()

    if plot:
        p0.plot(az, el, '-b', linewidth=3, label='computed horizon', alpha=0.8)
        p0.set_title('horizon for '+S)
        p0.set_ylabel(r'altitude (deg)')
        p0.set_xlabel(r'azimuth (deg)')
        if compare:
            p0.plot(horizon[S][0], horizon[S][1],
                    '-r', linewidth=2, alpha=0.5,
                    label='horizon file (from JMMC)')
        p0.set_xlim(360, 0)
        p0.set_ylim(0,90)
        p0.grid()
        p0.legend(fontsize=12, loc='upper left')
    else:
        return az, el

def shadow(at, ut, N=721, UT=None, TEL=None, partial=True, plot=False):
    """
    at, ut = (E,N) coordinates in m
    """
    # -- UT dome as a cylinder: diameter, height (m)
    # https://www.eso.org/sci/facilities/paranal/telescopes/ut/vltsv6.gif
    if UT is None:
        UT = (UT_DOME_DIAM, UT_DOME_HEIGHT)

    # -- AT M1 height above ground and diameter (m)
    # http://www.eso.org/sci/facilities/paranal/telescopes/vlti/documents/VLT-MAN-ESO-15000-4552_v97.pdf fig 4
    if TEL is None:
        TEL = (AT_AX_HEIGHT, AT_M1_HEIGHT, AT_M1_DIAM)

    # -- compute coordinates of top ring of the dome:
    # -- iterative to take into account inclination of the M1
    t = np.linspace(0,2*np.pi,2*N)
    el = np.ones(2*N)*45 # -- first approximation
    az = np.linspace(0,1,2*N)
    for i in range(3):
        if np.median(az)>270 or np.median(az)<90:
            az0 = 180
        else:
            az0 = 0.0
        az = (az+az0)%360
        alpha = abs(2*(az - az.mean())/np.ptp(az)) # in [0..1]
        az = (az-az0+360)%360

        # -- center of the M1:
        # == AX height (TEL[0]) when pointing at the horizon
        # == M1 height (TEL[1]) when pointing at zenith
        se = np.sin(el*np.pi/180)
        ce = np.cos(el*np.pi/180)
        ath = TEL[0] - se*(TEL[0]-TEL[1])

        # -- AT mirror center:
        atx, aty, atz = at[0], at[1], ath

        if partial: # -- include partial vignetting, approx but pessimistic
            # -- add TEL M1 diam/2 to dimensions,
            # -- with some dependence with az and el
            rx = ut[0] + (UT[0]+TEL[2]*alpha**2)/2*np.cos(t)
            ry = ut[1] + (UT[0]+TEL[2]*alpha**2)/2*np.sin(t)
            rz = UT[1] + TEL[2]*ce/2
        else: # -- vignetting of the center of M1
            rx = ut[0] + (UT[0])/2*np.cos(t)
            ry = ut[1] + (UT[0])/2*np.sin(t)
            rz = UT[1]

        # -- vector M1-ring:
        v = np.array([rx-atx, ry-aty, rz-atz])
        # -- correct for mirror going black
        # -- M1 recoil position
        ad = -ce*(TEL[0]-TEL[1])
        absv = np.sqrt(np.sum(v**2))
        v *= (absv - ad) / absv

        # -- azimuth and elevation in degrees
        az = 180 - np.arctan2(v[0], v[1])*180/np.pi
        el = np.arctan2(v[2], np.sqrt(v[0]**2 + v[1]**2))*180/np.pi

    # -- continuous azimuth coverage (interpolation)
    _az = np.linspace(0,360,N)
    _el = np.zeros(len(_az))
    for i,a in enumerate(_az):
        try:
            _el[i] = el[np.abs(az-a)<(np.diff(_az).mean()/2.)].max()
        except:
            _el[i] = 0.0
    if plot:
        # -- assume plot is already created!
        # -- TEL elevation axis
        plt.plot(0, TEL[0], '+g', linewidth=1)

        # -- UT dome
        d = np.sqrt((at[0]-ut[0])**2 + (at[1]-ut[1])**2 )
        X = np.array([-UT_DOME_DIAM/2, -UT_DOME_DIAM/2,
                        UT_DOME_DIAM/2, UT_DOME_DIAM/2])
        plt.plot(X+d,[0, UT_DOME_HEIGHT, UT_DOME_HEIGHT, 0], '-k', linewidth=1)
        plt.fill_between(X+d,[0,0,0,0], [0, UT_DOME_HEIGHT, UT_DOME_HEIGHT, 0],
                            color='k', alpha=0.5)
        # -- clearing top of the dome:
        se = np.sin(el.max()*np.pi/180)
        ce = np.cos(el.max()*np.pi/180)
        # -- telescope M1 height
        ath = TEL[0] - se*(TEL[0]-TEL[1])
        # -- telescope -x position
        ad = -ce*(TEL[0]-TEL[1])

        # -- rays:
        d += UT_DOME_DIAM/2
        # plt.plot([ad, ad + d*ce], [ath, ath + d*se],
        #     '-y', linewidth=1)
        plt.plot([ad+TEL[2]/2*se, ad+TEL[2]/2*se + d*ce] ,
            [ath-TEL[2]/2*ce, ath-TEL[2]/2*ce + d*se],
                '-y', linewidth=1)
        plt.plot([ad-TEL[2]/2*se, ad+d*ce-TEL[2]/2*se],
            [ath+TEL[2]/2*ce, ath+d*se+TEL[2]/2*ce],
                '-y', linewidth=1)
        # -- mirror:
        plt.plot([ad + TEL[2]/2*se, ad - TEL[2]/2*se],
                 [ath - TEL[2]/2*ce, ath + TEL[2]/2*ce], '-b', linewidth=2)
    return _az, _el

def tabulateAll():
    import time
    keys =  layout.keys()
    keys = sorted(keys)
    hz = {}
    for k in keys:
        hz[k] = UTshadow(k)
    f = open('newVltiHzn.txt', 'w')
    f.write('# VLTI horizon file: ')
    f.write('computed by amerand@eso.org, '+time.asctime()+'\n')
    f.write('# Azimuth convention: 0deg is South, 90deg is East\n')
    f.write('# UT DOME HEIGHT   = %4.1f m\n'%UT_DOME_HEIGHT)
    f.write('# UT DOME DIAMETER = %4.1f m\n'%UT_DOME_DIAM)
    f.write('# UT EL AXIS HEIGHT= %4.1f m\n'%UT_AX_HEIGHT)
    f.write('# UT M1 HEIGHT     = %4.1f m\n'%UT_M1_HEIGHT)
    f.write('# UT M1 DIAMETER   = %4.1f m\n'%UT_M1_DIAM)
    f.write('# AT EL AXIS HEIGHT= %4.1f m\n'%AT_AX_HEIGHT)
    f.write('# AT M1 HEIGHT     = %4.1f m\n'%AT_M1_HEIGHT)
    f.write('# AT M1 DIAMETER   = %4.1f m\n'%AT_M1_DIAM)
    f.write('# az    '+'    '.join(keys)+'\n')
    for i in range(len(hz['A0'][0])):
        f.write('%5.1f'%hz['A0'][0][i])
        for k in keys:
            f.write(' %5.2f'%hz[k][1][i])
        f.write('\n')
    f.close()
    return

def mira():
    import vlti
    UTshadow('D0', plot=1, compare=1)
    lst = np.linspace(-12,12,200)
    m = vlti.nTelescopes(['A0','B2','C1','D0'],'Mira', lst, plot=0); plt.ylim(0,70)
    plt.plot(m['az'], m['alt'], '-g', linestyle='dashed', linewidth=4,
    alpha=0.5, label='Mira')
    plt.legend(loc='upper right', fontsize=12);
    plt.xlim(225+50, 225-50); plt.ylim(0,60)
    return
