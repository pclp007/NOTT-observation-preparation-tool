import vlti
import ephem
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm
#import simbad
import itertools
from astropy import log
log.setLevel('ERROR')
from astroquery.simbad import Simbad

def vltiAscii(T, targets, date=None, moonDist=20.0, useColor=True,
              windDir=None, lstStep=10, sortByRA=True, useUnicode=False,
              minAltitude=30, max_vcmPressure=None, max_OPD=110, info=True,
              noPrint=False, comments=[], invalidLST=[], UTtime=False, clearScreen=True):
    """
    T: list of telescopes, e.g. ['A0', 'K0', 'G1'], or config as 'U1-U2-U3'

    targets: list of target Simbad can resolve.

    date=None -> Now (default)
    date='2012/03/20'
    date='2012/03/20 03:00:00' (UT time)

    moonDist: minimum distance to the Moon (degrees)

    windDir: direction of the wind in case of wind restriction. Can be
    in degrees (0 is North and 90 is East, same convention as Paranal
    ASM) or given litteraly: 'N', 'SW', 'NE' etc. None (default) means
    no restrictions.

    info: plots a table with info about the targets

    sortByRA: sort targets by RA, otherwise keep the original order

    comments: list of strings, one comment per target

    invalidLST: list of list on unvalid LST range, one for each target
        eg: [[(8.5, 9.0), (10, 10.5)], [], etc.]
    """
    vlt = ephem.Observer()
    vlt.lat = '-24:37:38'
    vlt.long = '-70:24:15'
    vlt.elevation = 2635.0

    dVCM = 0.5 # upper limit for VCMChr = max_vcmPressure+dVCM

    # http://stackoverflow.com/questions/287871/print-in-terminal-with-colors-using-python
    colors = {'GREEN':'\033[92m',   'GREENBG':'\033[42m',
              'BLUE':'\033[94m',    'BLUEBG':'\033[44m',
              'RED':'\033[91m',     'REDBG':'\033[41m',
              'YELLOW':'\033[93m',  'YELLOWBG':'\033[43m',
              'GRAY':'\033[90m',    'CYAN':'\033[36m',
              'CYANBG':'\033[46m', 'MAGENTABG':'\033[45m',
              'MAGENTA':'\033[95m', 'NO_COL':'\033[0m' }

    if not useColor or noPrint:
        for k in colors.keys():
            colors[k] = ''

    if isinstance(windDir, str):
        direction = {'':None, 'N':0, 'S':180, 'E':90, 'W':270,
                     'NE':45, 'NW':315, 'SW':225, 'SE':135}
        windDir = direction[windDir]

    windArrow = {0:8659, 45:8665, 90:8656, 135:8662,
                 180:8657, 225:8663, 270:8658, 315:8664,
                 360:8659}

    if useUnicode and not windDir is None:
        windK = int(45*round(windDir/45))
        windChr = chr(windArrow[windK])
    else:
        windChr='W'
    # ephemeris of the Sun
    sun = ephem.Sun()
    sun_eph={}

    # now
    if date is None:
        vlt.date = ephem.date(ephem.now())
        _now = True
    else:
        vlt.date = int(ephem.date(date))+(24-7.25)/24. # middle of the night
        _now = False

    sun_eph['NOW'] = vlt.date

    sun_eph['SUNSET'] = vlt.next_setting(sun) # sun sets
    vlt.horizon = '-18:0'
    sun_eph['TWI END']=vlt.next_setting(sun) # TWI ends
    vlt.horizon = '0'
    sun_eph['SUNRISE'] = vlt.next_rising(sun) # sun sets
    vlt.horizon = '-18:0'
    sun_eph['TWI START']=vlt.next_rising(sun) # TWI starts
    vlt.horizon = '0:0'

    # -- check that the events are in the right order
    for k in sun_eph.keys():
        if float(sun_eph[k]) > float(vlt.date) + 0.5:
            sun_eph[k] = ephem.date(sun_eph[k]-1)
        if float(sun_eph[k]) < float(vlt.date) - 0.5:
            sun_eph[k] = ephem.date(sun_eph[k]+1)

    # offset between LST and UT
    lstOffset = float(vlt.sidereal_time())/(2*np.pi) - \
                float(vlt.date)%1 - 0.5
    #print('LST - UT:', lstOffset*24)

    # position of the Moon
    moon = ephem.Moon()
    moon.compute(vlt)

    # unicode sympol of the moon: 0 is full, 50 is new
    moonChr = {'00':9673, '25':5613, '40':9789, '50':9712,
               '60':9790, '75':5610, '100':9673}

    #moonK = np.argmin([np.abs(float(k)-moon.phase) for k in moonChr.keys()])

    if useUnicode:
        # does not work very well:(
        #moonC = chr(moonChr[moonChr.keys()[moonK]])
        moonC = chr(9790); moonC=chr(9686)
        upChr = chr(8900); #upChr = chr(9624)
        downChr = ' '# chr(8901)
        nowChr = chr(9731)
        lineChr = chr(8901)
        tickChr = [chr(9312), chr(9313), chr(9314)]
        transitChr = chr(10023)
        sunChr = [chr(9728), chr(9728)]
        twiChr = [chr(10024), chr(10024)]
        notobsChr = chr(8901)
        vignetChr = '"' # vignetted by telescope
        vcmChr = chr(11048) #'v' # vcm pressure sligtly too high
        VCMChr = chr(11047) #'V' # vcm pressure too high
        zenithChr = 'Z'
        invalidChr = 'o'
    else:
        moonC = 'm'
        upChr = '='
        downChr = ' ' # below minimum altitude
        nowChr = '@'
        lineChr = '-' # line for LST
        tickChr = ['Q', 'H', 'Q']
        transitChr = 'T'
        zenithChr = 'Z'
        sunChr = ['(', ')']
        twiChr = ['[', ']']
        notobsChr = '_' # no delay
        vignetChr = '^' # vignetted by telescope
        vcmChr = 'v' # vcm pressure sligtly too high
        VCMChr = 'V' # vcm pressure too high
        invalidChr = 'o' # avoid this LST (already observed)

    sun_eph_lst = {}
    for k in sun_eph.keys():
        sun_eph_lst[k] = (sun_eph[k]+lstOffset-0.5)%1*24

    kz= ['SUNSET', 'TWI END', 'TWI START', 'SUNRISE']
    for k in range(3):
        #print kz[k], sun_eph_lst[kz[k]]
        if sun_eph_lst[kz[k+1]]<sun_eph_lst[kz[k]]:
            sun_eph_lst[kz[k+1]] += 24

    # -- from sunset to sunrise
    # lst = np.arange(np.floor(sun_eph_lst['SUNSET'])-0,
    #                 np.ceil(sun_eph_lst['SUNRISE'])+1./lstStep,
    #                 1./lstStep)
    lst = np.arange(np.floor(sun_eph_lst['SUNSET']*lstStep)/lstStep,
                    np.ceil(sun_eph_lst['SUNRISE']*lstStep)/lstStep,
                    1./lstStep)
    ut = lst-lstOffset*24

    sun_eph_lst['1/2'] = 0.5*sun_eph_lst['TWI START'] +\
                         0.5*sun_eph_lst['TWI END']
    sun_eph_lst['1/4'] = 0.25*sun_eph_lst['TWI START'] +\
                         0.75*sun_eph_lst['TWI END']
    sun_eph_lst['3/4'] = 0.75*sun_eph_lst['TWI START'] +\
                         0.25*sun_eph_lst['TWI END']
    ## if useUnicode:
    ##     lstString1 = []
    ##     for l in lst:
    ##         if np.abs(int(l)-l)<1e-6:
    ##             print l,l%24.0, (l%24)>20,
    ##             if (l%24)<21:
    ##                 lstString1.append(chr(9311+int(l%24)))
    ##             else:
    ##                 lstString1.append(chr(12860+int(l%24)))
    ##         else:
    ##             lstString1.append(' ')
    ##     lstString2 = ''
    ## else:

    if UTtime:
        lstString1 = [str(int((l%24+0.01)/10.))
                      if np.abs(round(l,0)-l)<0.5/lstStep else ' ' for l in ut]
        lstString2 = [str(int((l%24+0.01)%10.))
                      if np.abs(round(l,0)-l)<0.5/lstStep else ' ' for l in ut]
    else:
        lstString1 = [str(int((l%24+0.01)/10.))
                      if np.abs(round(l,0)-l)<0.01 else ' ' for l in lst]
        lstString2 = [str(int((l%24+0.01)%10.))
                      if np.abs(round(l,0)-l)<0.01 else ' ' for l in lst]
    lstString3 = [lineChr for l in lst]

    lstString3[np.abs(lst-sun_eph_lst['SUNSET']).argmin()] = sunChr[0]
    lstString3[np.abs(lst-sun_eph_lst['SUNRISE']).argmin()] = sunChr[1]
    lstString3[np.abs(lst-sun_eph_lst['TWI START']).argmin()] = twiChr[1]
    lstString3[np.abs(lst-sun_eph_lst['TWI END']).argmin()] = twiChr[0]
    lstString3[np.abs(lst-sun_eph_lst['1/4']).argmin()] = tickChr[0]
    lstString3[np.abs(lst-sun_eph_lst['1/2']).argmin()] = tickChr[1]
    lstString3[np.abs(lst-sun_eph_lst['3/4']).argmin()] = tickChr[2]

    if date is None:
        lstString3[np.abs(lst-sun_eph_lst['NOW']).argmin()] = nowChr

    __c = lineChr
    for i in range(len(lstString3)):
        if lstString3[i] == sunChr[0]:
            __c = '-'
        elif lstString3[i] == sunChr[1]:
            __c = lineChr
        elif lstString3[i] == twiChr[0]:
            __c = '='
        elif lstString3[i] == twiChr[1]:
            __c = '-'
        if lstString3[i] == lineChr:
            lstString3[i] = __c

    obsStrings = []
    if isinstance(targets, str): # single target -> list of one element
        targets = [targets]

    RAs = []

    simTargets = []
    # -- loop on targets, check observability
    allObs = []
    if type(T)==str and '-' in T:
        T = T.split('-')

    for __i, target in enumerate(targets):
        if info:
            if type(target)==str:
                simTargets.append(Simbad.query_object(target))
            else:
                pass
        tmp = vlti.nTelescopes(T, target, lst, min_altitude=minAltitude,
                               max_vcmPressure=max_vcmPressure,
                               max_OPD=max_OPD, flexible=True)

        tmp['ut'] = tmp['lst']+lstOffset

        allObs.append(tmp)
        dra = (moon.ra*12/np.pi-tmp['ra'])*np.cos(tmp['dec']*np.pi/180)*15
        ddec = (moon.dec*180/np.pi-tmp['dec'])
        dist = np.sqrt(dra**2+ddec**2)
        RAs.append(tmp['ra'])

        obsString = []
        for k in range(len(tmp['alt'])):
            if tmp['observable'][k]:
                if dist < moonDist:
                    obsString.append(moonC)
                else:
                    obsString.append(upChr)
            else:
                # -- why is it not observable?
                if tmp['horizon'][k]>0 and \
                      tmp['horizon'][k]>tmp['alt'][k] and\
                      minAltitude<tmp['alt'][k]: # BEHIND UT
                    obsString.append(vignetChr)
                elif 'vcm' in tmp.keys() and not max_vcmPressure is None and\
                        tmp['horizon'][k]<tmp['alt'][k] and \
                        minAltitude<tmp['alt'][k] and \
                        max([np.abs(tmp['opd'][d][k]) for d in tmp['opd'].keys()])<=max_OPD and \
                        max([tmp['vcm'][d][k] for d in tmp['vcm'].keys()])>=max_vcmPressure:
                        # VCM pressure too high
                    if max([tmp['vcm'][d][k] for d in tmp['vcm'].keys()])>=max_vcmPressure+dVCM:
                        obsString.append(VCMChr)
                    else:
                        obsString.append(vcmChr)
                elif tmp['alt'][k]>=minAltitude:
                    obsString.append(notobsChr) # DL LIMIT
                else:
                    obsString.append(downChr) # TOO LOWW

        # -- altitude ticks
        alt = np.int_(np.floor(tmp['alt']/10.)*(tmp['alt']>=minAltitude))
        for k in range(len(obsString))[1:]:
            if alt[k]>alt[k-1]:
                obsString[k]=str(alt[k])
            elif alt[k]<alt[k-1]:
                obsString[k]=str(alt[k-1])

        # -- transit / zenith avoidance
        iT = tmp['alt'].argmax()
        if not iT==0 and not iT==len(tmp['alt']):
            obsString[iT] = transitChr
        for k in range(len(obsString)):
            if tmp['alt'][k]>=87.0:
                obsString[k] = zenithChr

        if date is None:
            __in = np.abs(lst-sun_eph_lst['NOW']).argmin()
            #print __in, obsString[__in],  obsString[__in] == downChr
            if obsString[__in] == downChr or obsString[__in] == notobsChr:
                obsString[__in] = nowChr
            #print obsString[__in]

        # -- HOUR ticks for unobservable slots
        _lst = np.floor(tmp['lst'])

        for k in range(len(obsString))[1:]:
            if not tmp['observable'][k] and \
                not obsString[k-1] in ['1','2','3','4','5','6','7','8','9','T','Z',nowChr] and\
                            _lst[k-1]!=_lst[k]:
                    obsString[k-1] = colors['GRAY']+'|'+colors['NO_COL']

        # -- 1/2H ticks for unobservable slots
        if lstStep > 10 and lstStep%2==0:
            _lst = np.floor(tmp['lst']+0.5)

            for k in range(len(obsString))[1:]:
                if not tmp['observable'][k] and \
                    not obsString[k-1] in ['1','2','3','4','5','6','7','8','9','T','Z',nowChr] and\
                            _lst[k-1]!=_lst[k]:
                        obsString[k-1] = colors['GRAY']+':'+colors['NO_COL']

        # -- wind restriction:
        if not windDir is None:
            _obs = np.logical_and(tmp['observable'],
                     np.abs((-tmp['az'][k]-windDir)%360-180)>=90)
            # AZ convention are different between wind and Target!
            for k in range(len(obsString)):
                if obsString[k] == upChr and \
                       np.abs((-tmp['az'][k]-windDir)%360-180)<=90:
                    obsString[k] = windChr
        else:
            _obs = tmp['observable']

        # -- remaining time (incl wind)
        if _now and any(_obs) and \
                    np.interp(sun_eph_lst['NOW'], tmp['lst'], _obs)>0:
            _remain = max(np.max(tmp['lst'][_obs])-sun_eph_lst['NOW'], 0)

            _remain = ' [%dh%02.0f]'%(int(_remain), 60*(_remain-int(_remain)))
            if useColor:
                _remain = colors['CYAN']+_remain
            #_remain = ' (%.2fh)'%(max(np.max(tmp['lst'][tmp['observable']])-sun_eph_lst['NOW'], 0))
        else:
            _remain = ''

        # -- invalid LST
        if len(invalidLST)==len(targets):
            invalid = invalidLST[__i]
            for lims in invalid:
                for k in range(len(obsString)):
                    if lst[k]>=lims[0] and lst[k]<=lims[1]:
                        obsString[k] = invalidChr

        if useColor:
            for k in range(len(obsString)):
                if obsString[k]==downChr:
                    obsString[k]=colors['GRAY']+obsString[k]+colors['NO_COL']
                elif obsString[k]==notobsChr:
                    obsString[k]=colors['RED']+obsString[k]+colors['NO_COL']
                elif obsString[k]==vignetChr:
                    obsString[k]=colors['RED']+obsString[k]+colors['NO_COL']
                elif obsString[k]==VCMChr:
                    obsString[k]=colors['RED']+obsString[k]+colors['NO_COL']
                elif obsString[k]==vcmChr:
                    obsString[k]=colors['BLUE']+obsString[k]+colors['NO_COL']
                elif obsString[k]==moonC:
                    obsString[k]=colors['YELLOWBG']+obsString[k]+colors['NO_COL']
                elif obsString[k]==upChr:
                    obsString[k]=colors['CYAN']+obsString[k]+colors['NO_COL']
                elif obsString[k]==transitChr:
                    obsString[k]=colors['CYANBG']+obsString[k]+colors['NO_COL']
                elif obsString[k]==windChr:
                    obsString[k]=colors['REDBG']+obsString[k]+colors['NO_COL']
                elif obsString[k] in ['1','2','3','4','5','6','7','8','9']:
                    obsString[k]=colors['YELLOW']+obsString[k]+colors['NO_COL']
                elif obsString[k]==zenithChr:
                    obsString[k]=colors['REDBG']+obsString[k]+colors['NO_COL']
                elif obsString[k]==invalidChr:
                    obsString[k]=colors['BLUEBG']+obsString[k]+colors['NO_COL']

        obsString = ''.join(obsString)
        if isinstance(target, str):
            targ_name = target
        else:
            # assume [ra dec]
            targ_name = '%2d:%2d %3d:%2d' % (int(target[0]),
                                             int(60*(target[0]-int(target[0]))),
                                             int(target[1]),
                                             int(60*(abs(target[1])-
                                                     int(abs(target[1]))))
                                            )
        if len(comments)==len(targets):
            Nt = max([len(t) for t in targets])
            fmt = '%%%ds | %%s'%Nt
            targ_name = fmt%(targ_name, comments[__i])
        obsStrings.append(obsString+' '+colors['YELLOW']+
                          targ_name+_remain+colors['NO_COL'])
    RAs = np.array(RAs)

    if sortByRA:
        obsStrings = [obsStrings[k] for k in np.argsort((RAs-float(sun_eph_lst['1/2']-12))%24)]
        if info and len(simTargets)>0:
            simTargets = [simTargets[k] for k in np.argsort((RAs-float(sun_eph_lst['1/2']-12))%24)]

    lstString1 = ''.join(lstString1)
    if len(lstString2)>1:
        lstString2 = ''.join(lstString2)

    _lst = np.floor(lst)
    for k in range(len(lstString3))[1:]:
        if lineChr in lstString3[k-1] and _lst[k-1]!=_lst[k]:
            lstString3[k-1] = colors['GRAY']+'|'+colors['NO_COL']

    lstString3 = ''.join(lstString3)

    if not noPrint and clearScreen:
        # -- clear screen
        print("\033c")


    legendString = colors['BLUE']+'-'.join(T)+' '+str(sun_eph['NOW'])
    legendString += ' '+upChr+'/'+notobsChr
    legendString += ':obs/not[OPD<%2.0fm ALT<%.0fº]'%(max_OPD, minAltitude)+' '
    legendString += vignetChr+':behind UT '
    legendString += moonC+':Moon(%1.0f%%)<%2.0fº'%(moon.phase, moonDist)+' '
    if not max_vcmPressure is None:
        print(vcmChr+':VCM>%3.1fbar'%(max_vcmPressure),end=' ')
        print(VCMChr+':VCM>%3.1fbar'%(max_vcmPressure+dVCM),end=' ')
    if not windDir is None:
        legendString += windChr+':wind restr. (%3d)'%(windDir)+' '
    legendString += sunChr[0]+sunChr[1]+':sunset/rise '+twiChr[0]+twiChr[1]+':twilight'
    legendString += colors['NO_COL']

    if noPrint:
        res = [legendString]
        if UTtime:
            res.append(lstString1+' UT')
        else:
            res.append(lstString1+' LST')

        if lstString2!='':
            res.append(lstString2)
        res.append(' '+lstString3)
    else:
        print(legendString)
        if UTtime:
            print(colors['GRAY']+' '+lstString1[1:]+' UT '+colors['NO_COL'])
        else:
            print(colors['GRAY']+' '+lstString1[1:]+' LST'+colors['NO_COL'])
        if lstString2!='':
            print(colors['GRAY']+' '+lstString2[1:]+colors['NO_COL'])
        print(' '+lstString3)

    # if useUnicode:
    #     # -- zodiac strip -> does not work...
    #     lstString4 = [chr(9800+int(((l-3)%24)/2.)) if
    #                   np.abs(int(l)-l)<1e-3 and np.abs((l+1)%2. < 1e-3)
    #                   else ' ' for l in lst]
    #     lstString4 = ''.join(lstString4)
    #     print('Z'+lstString4)

    for k in range(len(targets)):
        #if len(targets)>4 and k%5==0 and k>0:
        #    print ' '+lstString3
        if noPrint:
            res.append(' '+obsStrings[k])
        else:
            print(' '+obsStrings[k])

    if noPrint:
        return res

def allConfVLTI(target, nTel=2, lst=None, UTs=False, nPA=4, dB=20.0):
    """
    nPA: number of PA zones (from 0 to pi)
    dB: slice in baselines (in meters)
    """
    if lst is None:
        lst = np.linspace(0,24,int(24*4)+1) #
    # list of quadruplets
    if UTs:
        quadruplets = [['U1','U2','U3','U4']]
    else:
        quadruplets = [['A1','K0','G1','J3'],
                       ['A1','B2','C1','D0'],
                       ['D0','H0','G1','I1']]
    # dict of u,v zones
    allZones = {}

    plt.figure(0)
    plt.clf()
    ax1=plt.subplot(121, polar=True)
    ax2=plt.subplot(122, polar=True)
    for q in quadruplets:
        for t in itertools.combinations(q,nTel):
            n = vlti.nTelescopes(t,target,lst, flexible=True)
            w = np.where(n['observable'])
            zones={}
            for k in n['baseline']:
                PA = (90-n['PA'][k][w])*np.pi/180
                B =  n['B'][k][w]
                ax1.plot(PA, B, 'or', alpha=0.5)
                ax1.plot(PA, -B, 'or', alpha=0.5)
                PA_ = np.int_(np.round_(PA/np.pi*nPA))
                PA_ = PA_%(2*nPA)
                #PA_ = PA_%(2*np.pi)
                B_ = np.int_(B/dB)*dB +dB/2
                # count points in each zone
                for k in range(len(PA_)):
                    z = (PA_[k], B_[k])
                    if not z in zones.keys():
                        zones[z] = 1.
                    else:
                        zones[z] += 1.
                    z = ((PA_[k]+nPA)%(2*nPA), B_[k])
                    if not z in zones.keys():
                        zones[z] = 1.
                    else:
                        zones[z] += 1.
            #print t, zones
            for z in zones.keys():
                if zones[z]>=4: # at least observable 4 points == 1h
                    if z in allZones.keys():
                        allZones[z]+=1.0
                    else:
                        allZones[z]=1.0

    ax2.scatter([z[0]*np.pi/nPA for z in allZones.keys()],
                [z[1] for z in allZones.keys()],
                c=[allZones[z] for z in allZones.keys()],
                cmap='hot', s=400, vmin=1,vmax=5)
    ax2.scatter([z[0]*np.pi/nPA for z in allZones.keys()],
                [-z[1] for z in allZones.keys()],
                c=[allZones[z] for z in allZones.keys()],
                cmap='hot', s=400, vmin=1,vmax=5)
    for z in allZones.keys():
        ax2.text(z[0]*np.pi/nPA, z[1], str(int(allZones[z])),
                 ha='center',va='center', color='0.5', weight='black')
        #ax2.text(z[0], -z[1], str(int(allZones[z])),
        #         ha='center',va='center',color='r')

    ax1.set_ylim(0,150)
    ax2.set_ylim(0,150)
    ax2.set_title('number of %dT config with 1h observing time'%nTel)
    #plt.subplot(122)
    #plt.colorbar()
    return allZones

def checkMoonVLT(target, startDate=None, NDays=1, minDist=20.0, period=None,
                 verbose=True, maxPhase=70, printTitle=False, plot=False):
    """
    Check if a target (string name for Simbad) gets closer than minDist to the
    Moon, between startDate (y,m,d) and for NDays.

    startDate with format "YYYY/MM/DD"

    only valid at VLT.

    if Verbose sets to False, returns a list of dates with the moon phase, the
    distance to the Moon and a boolean if target is observable.

    dates are the one of the night starting (Paranal convention), not the UT date of observations!
    """

    vlt = ephem.Observer()
    vlt.lat = '-24:37:38'
    vlt.long = '-70:24:15'
    vlt.elevation = 2635.0

    if not period is None:
        year = 2024+(period-113)//2
        if period%2==1:
            month = 4
        else:
            month = 10
        startDate = '%d/%d/1'%(year, month)
        NDays = 183

    if startDate is None:
        startDate = str(ephem.now()).split()[0]
    vlt.date = int(ephem.date(startDate)) + (24 - 7.25)/24. # middle of the night

    f = ephem.FixedBody()
    f._epoch = 2000.0

    if not type(target)==tuple:
        #s = simbad.query(target)
        s = Simbad.query_object(target)
        try:
            ra = np.sum(np.float64(s['RA'][0].split())*np.array([1, 1/60., 1/3600.]))
            dec = np.float64(s['DEC'][0].split())*np.array([1, 1/60., 1/3600.])
            dec = np.sum(np.abs(dec))*np.sign(dec[0])
        except:
            print('WARNING: cannot get coordinates for', target)
            ra, dec = 0, 0
        #print('RA', s['RA'][0], '->', ra, 'hours')
        #print('Dec', s['DEC'][0], '->', dec, 'degrees')

        f = ephem.FixedBody()
        f._ra = ra*np.pi/12
        f._dec = dec*np.pi/180
    else:
        # assumes (ra, dec)
        f._ra = 15*target[0]*np.pi/180
        f._dec = target[1]*np.pi/180

    moon = ephem.Moon()
    if verbose:
        print(target, f._ra, f._dec, ':')

    res = []
    for k in range(NDays):
        moon.compute(vlt)
        f.compute(vlt)
        # -- separation in degress
        dist = ephem.separation(moon, f)*180/np.pi
        # -- debug:
        #print('%3.0f%% at %5.1fº'%(moon.phase, dist))
        obse = not (dist<minDist and moon.phase>maxPhase)
        if verbose and not obse:
            if printTitle:
                print('# min distance to the Moon: %4.1fdeg'%float(minDist))
                print('# max illumination of the Moon: %3.0f%%'%float(maxPhase))
                print('# actual date/time of conjunction, ESO observation date is the day before!')
                print('# date | phase | dist |')
                printTitle = False
            print(' UT %-10sT04:00:00 | %3.0f | %4.1f | '%(str(vlt.date).split(' ')[0],
                                                           moon.phase, dist))
        vlt.date -= 1
        res.append((str(vlt.date).split()[0], moon.phase,
                    dist, obse))
        vlt.date += 2
    if plot:
        plt.plot([r[2] for r in res], [r[1] for r in res], '-k', alpha=0.2)
        plt.xlabel('distance to Moon (degrees)')
        plt.ylabel('Moon phase (percent)')
        plt.title(target+' starting on '+str(startDate)+' for %d days'%NDays+
            '\nESO date of observations, i.e. 1 day before actual UT date/time of cunjuction')
        plt.fill_between([0, minDist], [maxPhase, maxPhase], [100, 100], color='r', alpha=0.1)
        C = ['r', 'g', 'b', 'y', 'c', 'm']
        C = matplotlib.cm.Set3(np.arange(12)/12+1./24)
        for r in res:
            color = C[(int(r[0].split('/')[1])-1)%len(C)]
            if int(r[0].split('/')[2]) in [1, 10, 20]:
                plt.plot(r[2], r[1], 'o', color=color, alpha=0.2, markersize=20)
                plt.text(r[2], r[1], '/'.join(r[0][5:].split('/')[::-1]),
                    ha='center', va='center', color='k')
            else:
                plt.plot(r[2], r[1], 'o', color=color, alpha=0.3, markersize=10)
        plt.tight_layout()
    if not verbose:
        return res

def imagingLST(min_alt=35, OBs=1.0, configs=None):
    """
    for each dec, show how long the objetc is observable on each configuration
    """
    if configs is None:
        #configs=['A0-B2-C1-D0', 'D0-G2-J2-K0', 'A0-G1-J2-J3', 'A0-G1-J3-K0']
        #configs=['A0-B2-C1-D0', 'A0-B5-D0-G1', 'A0-G1-J3-K0', 'A0-B5-J2-J6']
        configs=['A0-B2-C1-D0', 'D0-G2-J2-K0', 'A0-G1-J3-K0', 'A0-B5-J2-J6']

    hmax, step = 6, 0.1
    lst = np.linspace(-hmax, hmax, int(2*hmax/step+1))

    # -- Plots
    dec = np.linspace(-90,40, 50)
    data = {c:[] for c in configs}
    plt.clf()
    colors = ['g', 'b', 'r', '0.5']
    for i,c in enumerate(configs):
        for d in dec:
            t = vlti.nTelescopes(c.split('-'), (0, d), lst,
                                 min_altitude=min_alt,
                                 max_OPD=200 if 'J6' in c else 110)
            data[c].append(sum(t['observable'])*step)
        plt.plot(dec, data[c], label=c, alpha=0.5, linewidth=3,
                color=colors[i])
    plt.legend()
    plt.grid()
    plt.xlabel(r'declination ($^{\circ}$)')
    plt.ylabel('number of hours observable')
    plt.title(r'above %.0f$^{\circ}$ elevation'%min_alt)
    plt.xlim(-90)

    # -- table
    dec = [-76, -74, -72, -70, -55, -40, -25, -10, 0, 10, 15, 20, 25]
    dec = [-82, -80, -75, -70, -55, -40, -25, -10, 0, 10, 15, 20, 25]

    print('%% maximum numbers of %.0fmin concatenations'%(60.*OBs, ))
    print('%% min elevation=%.1f deg'%(min_alt, ))
    print('dec & '+' & '.join(configs)+'\\\\')
    for d in dec:
        tmp = []
        print('%3d & '%d, end='')
        for c in configs:
            t = vlti.nTelescopes(c.split('-'), (0, d), lst,
                                 min_altitude=min_alt)
            tmp.append('   %3.0f   '%(sum(t['observable'])*step/OBs))
        print(' & '.join(tmp)+'\\\\')
