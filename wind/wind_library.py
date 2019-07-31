''' 
Utilities for reading wind data from observation stations, converting between 
vector and polar representations, basic data processing, estimating
10 m wind speeds, etc.

Allie King, SFEI, March 14, 2019
'''

#--------------------------------------------------------------------------------------#
# Import packages
#--------------------------------------------------------------------------------------#

import numpy as np
import pandas as pd
import datetime as dt
from netCDF4 import Dataset
import re    
import os             

#--------------------------------------------------------------------------------------#
# Define functions
#--------------------------------------------------------------------------------------#

#--------------------------------------------------------------------------------------#
def wind_vector(ws, wdir):
    '''
    Usage:
    
        u, v = wind_vector(ws, wdir)
        
    Computes wind speed to east (u) and to north (v) given wind speed (ws)
    and wind direction (wdir) where wind direction is in degrees and follows
    meteorological convention
    '''
    
    u = ws * np.cos((270. - wdir)*np.pi/180.)

    v = ws * np.sin((270. - wdir)*np.pi/180.)
    
    return (u, v)

#--------------------------------------------------------------------------------------#  
def wind_speed_and_direction(u, v):

    '''
    Usage:
       
        ws, wdir = wind_speed_and_direction(u, v)
        
    Computes the wind speed (ws) and direction (wdir) given the wind vector
    components (u, v). Wind direction follows the meteorological convention.
    '''
    
    ws = np.sqrt(u**2 + v**2)
    wdir = 180. + 180./np.pi * np.arctan2(u,v)
    
    return (ws, wdir)
    

#--------------------------------------------------------------------------------------#
def psit_26(zet):

    '''translated from coare36vn_zrf.m -- Allie King, SFEI, March 12, 2019
    computes temperature structure function'''
    
    psi=np.ones(np.shape(zet),dtype=float) 
    k=zet>=0 # stable
    dzet=np.minimum(50,0.35*zet[k]) 
    psi[k]=-((1+0.6667*zet[k])**1.5+0.6667*(zet[k]-14.28)*np.exp(-dzet)+8.525)
    k=zet<0 # unstable
    x=(1-15*zet[k])**0.5
    psik=2*np.log((1+x)/2)
    x=(1-34.15*zet[k])**0.3333
    psic=1.5*np.log((1+x+x**2)/3)-np.sqrt(3)*np.arctan((1+2*x)/np.sqrt(3))+4*np.arctan(1)/np.sqrt(3)
    f=zet[k]**2/(1+zet[k]**2)
    psi[k]=(1-f)*psik+f*psic
    
    return psi

#--------------------------------------------------------------------------------------#
def psiu_26(zet):

    '''translated from coare36vn_zrf.m -- Allie King, SFEI, March 12, 2019
    computes velocity structure function'''
    
    psi=np.ones(np.shape(zet),dtype=float) 
    k=zet>=0 # stable
    dzet=np.minimum(50,0.35*zet[k]) # stable
    a=0.7
    b=3/4
    c=5
    d=0.35
    psi[k]=-(a*zet[k]+b*(zet[k]-c/d)*np.exp(-dzet)+b*c/d)
    k=zet<0 # unstable
    x=(1-15*zet[k])**0.25
    psik=2*np.log((1+x)/2)+np.log((1+x*x)/2)-2*np.arctan(x)+2*np.arctan(1)
    x=(1-10.15*zet[k])**0.3333
    psic=1.5*np.log((1+x+x**2)/3)-np.sqrt(3)*np.arctan((1+2*x)/np.sqrt(3))+4*np.arctan(1)/np.sqrt(3)
    f=zet[k]**2/(1+zet[k]**2)
    psi[k]=(1-f)*psik+f*psic
    
    return psi

#--------------------------------------------------------------------------------------#
def psiu_40(zet):

    '''translated from coare36vn_zrf.m -- Allie King, SFEI, March 12, 2019
    computes velocity structure function'''
    
    psi=np.ones(np.shape(zet),dtype=float) 
    k=zet>=0 # stable
    dzet=np.minimum(50,0.35*zet[k]) 
    a=1
    b=3/4
    c=5
    d=0.35
    psi[k]=-(a*zet[k]+b*(zet[k]-c/d)*np.exp(-dzet)+b*c/d)
    k=zet<0 # unstable
    x=(1-18*zet[k])**0.25
    psik=2*np.log((1+x)/2)+np.log((1+x*x)/2)-2*np.arctan(x)+2*np.arctan(1)
    x=(1-10*zet[k])**0.3333
    psic=1.5*np.log((1+x+x**2)/3)-np.sqrt(3)*np.arctan((1+2*x)/np.sqrt(3))+4*np.arctan(1)/np.sqrt(3)
    f=zet[k]**2/(1+zet[k]**2)
    psi[k]=(1-f)*psik+f*psic
    
    return psi

#--------------------------------------------------------------------------------------#
def bucksat(T,P,Tf):

    '''translated from coare36vn_zrf.m -- Allie King, SFEI, March 12, 2019
    computes saturation vapor pressure [mb] 
    given T [degC] and P [mb] Tf is freezing pt'''
    
    exx=6.1121*np.exp(17.502*T/(T+240.97))*(1.0007+3.46e-6*P)
    ii=T<Tf
    exx[ii]=(1.0003+4.18e-6*P[ii])*6.1115*np.exp(22.452*T[ii]/(T[ii]+272.55))#vapor pressure ice

    return exx

#--------------------------------------------------------------------------------------#
def qsat26sea(T,P,Ss,Tf):

    '''translated from coare36vn_zrf.m -- Allie King, SFEI, March 12, 2019
    computes surface saturation specific humidity [g/kg]
    given T [degC] and P [mb]'''
    
    ex=bucksat(T,P,Tf)
    fs=1-0.02*Ss/35# reduction sea surface vapor pressure by salinity
    es=fs*ex 
    qs=622*es/(P-0.378*es)
    
    return qs
    
#--------------------------------------------------------------------------------------#
def qsat26air(T,P,rh):

    '''translated from coare36vn_zrf.m -- Allie King, SFEI, March 12, 2019
    computes saturation specific humidity [g/kg] given 
    T [degC] and P [mb]'''
    
    Tf=0#assumes relative humidity for pure water
    es=bucksat(T,P,Tf)
    em=0.01*rh*es
    q=622*em/(P-0.378*em)
    
    return (q,em)

#--------------------------------------------------------------------------------------#
def grv(lat):

    '''translated from coare36vn_zrf.m -- Allie King, SFEI, March 12, 2019
    computes g [m/sec^2] given lat in deg'''
    
    gamma=9.7803267715
    c1=0.0052790414
    c2=0.0000232718
    c3=0.0000001262
    c4=0.0000000007
    phi=lat*np.pi/180
    x=np.sin(phi)
    g=gamma*(1+c1*x**2+c2*x**4+c3*x**6+c4*x**8)
    
    return g
    
#--------------------------------------------------------------------------------------#
def RHcalc(T,P,Q,Tf):

    '''translated from coare36vn_zrf.m  -- Allie King, SFEI, March 12, 2019
    computes relative humidity given T,P, & Q'''
    
    es=6.1121*np.exp(17.502*T/(T+240.97))*(1.0007+3.46e-6*P)
    ii=T<Tf#ice case
    es[ii]=6.1115*np.exp(22.452*T[ii]/(T[ii]+272.55))*(1.0003+4.18e-6*P[ii])
    em=Q*P/(0.378*Q+0.622)
    RHrf=100*em/es
    
    return RHrf

#--------------------------------------------------------------------------------------#
def make_array(var, N):

    '''if var is a scalar, returns var as an array of length N;
    if var is a list, returns var as an array created from that
    list. raises an exception if list/array input does not have
    length N. -- added by Allie King, SFEI, March 12, 2019'''

    # if var is a scalar, make it an array of length N
    if not hasattr(var, "__len__"):
        var = var*np.ones(N,dtype=float)
    
    # if var is a list, make it an array
    elif isinstance(var,list):
        var = np.array(var)
        
    # check that var has length N
    if not len(var)==N:
        raise('all input variables must either be scalars or be the same length as u')

    return var

#--------------------------------------------------------------------------------------#

def coare36(u,zu,t,zt,rh,zq,ts,Ss,P=1013.25,lat=37.7749,zi=600.,**kwargs):
    
    '''Vectorized version of COARE 3.6 code (Fairall et al, 2003) with 
    modification based on the CLIMODE, MBL and CBLAST experiments 
    (Edson et al., 2012). 
    
    The wave parameterizations are based on fits to the Banner-Norison 
    wave model and the Fairall-Edson flux database.  It also allows 
    salinity as a input.  Open ocean Ss=35; Great Lakes Ss=0; Option to 
    handle flow over land has been added. To trigger over-land flow 
    scenario simply specify roughness lengths for momentum, temperature,
    and humidity as keyword arguments. For flow over water, these 
    quantities are estimated from wind speed so they do not need to be
    specified by the user.
    
    ********************************************************************
    Note the jcool=1 skin cooling option has been disabled since there 
    is no heat flux info. The jcool=1 option attempted to convert bulk
    water temperature measurements to skin temperature. This version of
    the code assumes the water temperature input represents the skin 
    temperature. This assumption may introduce errors ~0.5 degC
    ********************************************************************
    
    The code assumes u is an array. Other input variables may be scalars
    or arrays same size as u.
    
    Usage:
        
        For wind measured over water:
    
            U10 = coare36(u,zu,t,zt,rh,zq,ts,Ss,P,lat,zi)
        
        For wind measured over land:
        
            U10 = coare36(u,zu,t,zt,rh,zq,ts,0.0,P,lat,zi,z0=z0,z0t=z0t,z0q=z0q)
    
    Input:  
    
        u = mean wind vector magnitude (m/s) at height zu(m)
        t = bulk air temperature (degC) at height zt(m)
       rh = relative humidity (%) at height zq(m)
       ts = water temperature (degC) 
       Ss = water surface salinity (PSU) (set to zero for flow over land)
        P = surface air pressure (mb) (default = 1013.25)
      lat = latitude (default = +37.7749 N for San Francisco)
       zi = planetary boundary layer depth (default 600 m)
       zu, zt, zq heights of the observations (m)
       z0 = momentum roughness length (m)
      z0t = temperature roughness length (m)
      z0q = humidity roughness length (m)
    
    Output:  
    
      U10 = wind speed at 10 m
    
    References:
    
     Fairall, C.W., E.F. Bradley, J.E. Hare, A.A. Grachev, and J.B. 
     Edson (2003), Bulk parameterization of air sea fluxes: updates 
     and verification for the COARE algorithm, J. Climate, 16, 571-590.
    
     Edson, J.B., J. V. S. Raju, R.A. Weller, S. Bigorre, A. Plueddemann, 
     C.W. Fairall, S. Miller, L. Mahrt, Dean Vickers, and Hans Hersbach, 
     2013: On the Exchange of momentum over the open ocean. J. Phys. 
     Oceanogr., 43, 1589-1610. doi: 
     http://dx.doi.org/10.1175/JPO-D-12-0173.1 
    
    Code history:
     
    1. 12/14/05 - created based on scalar version coare26sn.m with input
       on vectorization from C. Moffat.  
    2. 12/21/05 - sign error in psiu_26 corrected, and code added to use 
       variable values from the first pass through the iteration loop for 
       the stable case with very thin M-O length relative to zu (zetu>50) 
       (as is done in the scalar coare26sn and COARE3 codes).
    3. 7/26/11 - S = dt was corrected to read S = ut.
    4. 7/28/11 - modification to roughness length parameterizations based 
       on the CLIMODE, MBL, Gasex and CBLAST experiments are incorporated
    5. New wave parameterization added 9/20/2017  based on fits to wave 
       model
    6. 3/12/19 (Allie King, SFEI) Translated to Python and simplified to 
       return only the 10m wind speed. Cool skin temperature jcool=1 option 
       was disabled because it requires estimates of heat exchange which 
       vary in time and are not measured at most wind observation stations, 
       so it is assumed that the water temperature represents the skin 
       temperature. This could introduce error ~0.5oC. Also added an option 
       for flow over land, in which roughness lengths are specified by the 
       user instead of calculated.
    -----------------------------------------------------------------------
    '''        
    
    # check if roughness length has been specified in optional keyword 
    # arguments (**kwargs), and if so, assign it.
    # also create a flag that indicates whether zo has been specified, 
    # meaning that flow is over land
    if ('z0' in kwargs) or ('z0t' in kwargs) or ('z0q' in kwargs):
        z0 = kwargs.get('z0')
        z0t = kwargs.get('z0t')
        z0q = kwargs.get('z0q')
        on_land_flag = True
    else:
        on_land_flag = False     

    # make all input into arrays the same length as wind speed 
    if not hasattr(u, "__len__"):
        u=np.array([u])
    elif isinstance(u, list):
        u = np.array(u)
    N = len(u)
    t=make_array(t, N)
    rh=make_array(rh, N)
    P=make_array(P, N)
    ts=make_array(ts, N)
    lat=make_array(lat, N)
    zu=make_array(zu, N)
    zt=make_array(zt, N)
    zq=make_array(zq, N)
    Ss=make_array(Ss, N)
    if on_land_flag:
        z0=make_array(z0,N)
        z0t=make_array(z0t,N)
        z0q=make_array(z0q,N)
    
    # convert rh to specific humidity
    Tf=-0.0575*Ss+1.71052E-3*Ss**1.5-2.154996E-4*Ss*Ss # freezing point of seawater
    Qs = qsat26sea(ts,P,Ss,Tf)/1000.  # surface water specific humidity (g/kg)                  
    Q,Pv  = qsat26air(t,P,rh) # specific humidity of air (g/kg).  Assumes rh relative to ice T<0
    Q=Q/1000.
    
    # specify fized roughness for ice
    ice=np.zeros(N,dtype=bool)
    iice=ts<Tf
    ice[iice]=True
    zos=5.e-4    
    
    #***********  set constants **********************************************
    Beta = 1.2
    von  = 0.4
    fdg  = 1.00 # Turbulent Prandtl number
    tdk  = 273.16
    grav = grv(lat)
    
    #***********  air constants **********************************************
    visa = 1.326e-5*(1.+6.542e-3*t+8.301e-6*t**2-4.84e-9*t**3)
    
    #****************  begin bulk loop ***************************************
    
    #***********  first guess ************************************************
    du = u-0.0  # subtract zero water velocity
    dt = ts-t-.0098*zt
    dq = Qs-Q
    ta = t+tdk
    ug = 0.5
    ut    = np.sqrt(du**2+ug**2)
    if on_land_flag:
        u10   = ut*np.log(10/z0)/np.log(zu/z0)
    else:
        u10   = ut*np.log(10/1e-4)/np.log(zu/1e-4)
    usr   = 0.035*u10
    if on_land_flag:
        zo10 = z0
        zot10 = z0t
    else:
        zo10  = 0.011*usr**2/grav + 0.11*visa/usr
        Cd10  = (von/np.log(10/zo10))**2
        Ch10  = 0.00115
        Ct10  = Ch10/np.sqrt(Cd10)
        zot10 = 10/np.exp(von/Ct10)
    Cd    = (von/np.log(zu/zo10))**2
    Ct    = von/np.log(zt/zot10)
    CC    = von*Ct/Cd
    Ribcu = -zu/zi/.004/Beta**3
    Ribu  = -grav*zu/ta*(dt+.61*ta*dq)/ut**2
    zetu = CC*Ribu*(1+27/9*Ribu/CC)
    k50=zetu>50      # stable with very thin M-O length relative to zu 
    k=Ribu<0 
    zetu[k]=CC[k]*Ribu[k]/(1+Ribu[k]/Ribcu[k]) 
    L10 = zu/zetu
    gf=ut/du
    usr = ut*von/(np.log(zu/zo10)-psiu_40(zu/L10))
    tsr = -dt*von*fdg/(np.log(zt/zot10)-psit_26(zt/L10))
    qsr = -dq*von*fdg/(np.log(zq/zot10)-psit_26(zq/L10))
    
    #************************************************************
    #  The following gives the new formulation for the
    #  Charnock variable: COARE 3.5 wind speed dependent charnock
    #************************************************************
    umax=19
    a1=0.0017
    a2=-0.0050
    charn=a1*u10+a2
    k=u10>umax
    charn[k]=a1*umax+a2
    
    
    #**************  bulk loop ***********************************
    
    nits=10 # number of iterations
    
    for i in range(nits):
    
        zet=von*grav*zu/ta*(tsr +.61*ta*qsr)/(usr**2)
        L=zu/zet
        if on_land_flag:
            zo=z0
        else:
            zo=charn*usr**2/grav+0.11*visa/usr # surface roughness
            zo[iice]=zos
        rr=zo*usr/visa
        if on_land_flag:
            zot=z0t
            zoq=z0q
        else:
            # These thermal roughness lengths give Stanton and
            # Dalton numbers that closely approximate COARE 3.0:
            zoq=np.minimum(1.6e-4,5.8e-5/rr**.72)       
            zot=np.copy(zoq)                            
        cdhf=von/(np.log(zu/zo)-psiu_26(zu/L))
        cqhf=von*fdg/(np.log(zq/zoq)-psit_26(zq/L))
        cthf=von*fdg/(np.log(zt/zot)-psit_26(zt/L))
        usr=ut*cdhf
        qsr=-dq*cqhf
        tsr=-dt*cthf
        tvsr=tsr+0.61*ta*qsr
        tssr=tsr+0.51*ta*qsr
        Bf=-grav/ta*usr*tvsr
        ug=0.2*np.ones(N)
        k=Bf>0
        ug[k]=Beta*(Bf[k]*zi)**.333
        ut=np.sqrt(du**2+ug**2)
        gf=ut/du
        u10N = usr/von/gf*np.log(10./zo)
        charn=a1*u10N+a2
        k=u10N>umax
        charn[k]=a1*umax+a2
        
        # save first iteration solution for case of zetu>50
        if i==0:  
            usr50=usr[k50]
            tsr50=tsr[k50]
            qsr50=qsr[k50]
            L50=L[k50]
            zet50=zet[k50]
           
    # insert first iteration solution for case with zetu>50
    usr[k50]=usr50
    tsr[k50]=tsr50
    qsr[k50]=qsr50
    L[k50]=L50
    zet[k50]=zet50
    
    #*********************************
    #  Find the stability functions
    #*********************************
    psi    = psiu_26(zu/L)
    psi10  = psiu_26(10./L)
    gf = ut/du
    
    #*********************************************************
    #  Determine the wind speeds. 
    #  Note that usr is the friction velocity that includes 
    #  gustiness usr = np.sqrt(Cd) S, which is equation (18) in
    #  Fairall et al. (1996)
    #*********************************************************
    S = np.copy(ut)
    S10 = S + usr/von*(np.log(10./zu)-psi10+psi)
    U10 = S10/gf
    
    return U10
    
#--------------------------------------------------------------------------------------#
def make_outliers_nan(y):

    ''' 
    Usage:
    
        y = make_outliers_nan(y)
        
    Argument y must be a floating point array. Returns y with all data
    outside +/- 3 sigma from the mean replace by np
    .nan.
    '''
    
    mean = np.mean(y)
    stdv = np.sqrt(np.var(y)) 
    y[np.logical_or(y<(mean-3*stdv), y>(mean+3*stdv))] = np.nan
    
    return y

#--------------------------------------------------------------------------------------#
def read_cimis_station(station_name, datadir, start_time, end_time):

    '''
    Usage: 
        
        time_days, ws, wdir, Ta, Ts, rh = read_cimis_station(station_name, datadir, start_time, end_time)
        
    Input:
        
        station_name = name of CIMIS station
        datadir = directory containing CIMIS data files 
        start_time = start time in PST as a datetime object
        end_time = end time in PST as a datetime object
        
    Output:
    
        time_days = time in days since start time
        ws = wind speed at 2 m (measured, m/s)
        wdir = wind direction
        Ta = air temperature (oC)
        Ts = soil temperature (oC)
        rh = relative humidity (%)
        
    Notes: 
    
    Outliers, defined as outside 6 sigma from the mean, are removed.
    
    CIMIS time stamps are in PST and “Hourly data reflects the previous 
    hour's 60 minutes of readings.” We keep the PST time zone and correct the 
    time stamp by subtracting 30 minutes so it represents the middle of the 
    60-minute data averaging interval instead of the end.
    
    Returns an exception if no good wind data is found inside the time window.
    '''

    # set data file name
    datafilename = os.path.join(datadir, str(station_name) + '.csv')

    # load wind data from file, and if there is a problem loading it, raise
    # an exception and exit this function
    try:
        df = pd.read_csv(datafilename)
    except:
        raise
        
    # if there is no data in the data frame, raise an exception
    if len(df)==0:
        raise Exception('no wind data available from Station %s' % station_name)
        
    # extract data relevant to 10m wind vector along with qc codes
    # ... wind speed
    ind = df.keys().get_loc('Wind Speed (m/s)')  
    ws = df[df.keys()[ind]].values
    ws_qc = df[df.keys()[ind+1]].values
    # ... wind direction
    ind = df.keys().get_loc('Wind Dir (0-360)')  
    wdir = df[df.keys()[ind]].values
    wdir_qc = df[df.keys()[ind+1]].values
    # ... air temperature
    ind = df.keys().get_loc('Air Temp (C)')  
    Ta = df[df.keys()[ind]].values
    Ta_qc = df[df.keys()[ind+1]].values
    # ... soil temperature
    ind = df.keys().get_loc('Soil Temp (C)')  
    Ts = df[df.keys()[ind]].values
    Ts_qc = df[df.keys()[ind+1]].values
    # ... relative humidity
    ind = df.keys().get_loc('Rel Hum (%)')  
    rh = df[df.keys()[ind]].values
    rh_qc = df[df.keys()[ind+1]].values
    # ... time stamp
    date = df['Date'].values
    hour = df['Hour (PST)'].values/100 - 1
    datetimes = [(dt.datetime.strptime((date[i] + (' %02d:00' % hour[i])),'%m/%d/%Y %H:%M')) for i in range(len(hour))]
    
    # set bad data (indicated by the qc code) to nan
    ind = np.logical_or(ws_qc == 'I', ws_qc == 'S')
    ws[ind] = np.nan
    ind = np.logical_or(wdir_qc == 'I', wdir_qc == 'S')
    wdir[ind] = np.nan
    ind = np.logical_or(Ta_qc == 'I', Ta_qc == 'S')
    Ta[ind] = np.nan
    ind = np.logical_or(Ts_qc == 'I', Ts_qc == 'S')
    Ts[ind] = np.nan
    ind = np.logical_or(rh_qc == 'I', rh_qc == 'S')
    rh[ind] = np.nan
    
    # compute time in days since start date (note time zone is already PST)
    time_days = np.array([(datetime - start_time).total_seconds()/3600./24. for datetime in datetimes])
    
    # correct time so it reflects the midpoint of the measurement window instead
    # of the end point of the measurement window. CIMIS website says "Hourly 
    # data reflects the previous hour's 60 minutes of readings"
    time_days = time_days - 0.5/24.
    
    # compute end time in days since start date
    end_days = (end_time - start_time).total_seconds()/3600./24.
    
    # select data in date range specified with one day buffer on either end, 
    # and if there is no data in the date range, raise an exception
    ind = np.logical_and(time_days>=-1., time_days<=(end_days+1.))
    if not any(ind):
        raise Exception('no wind data available from Station %s during time period specified' % station_name)
    time_days = time_days[ind]
    ws = ws[ind]
    wdir = wdir[ind]
    Ta = Ta[ind]
    Ts = Ts[ind]
    rh = rh[ind]
    
    # remove repeats 
    time_days, ind = np.unique(time_days, return_index=True)
    ws = ws[ind]
    wdir = wdir[ind]
    Ta = Ta[ind]
    Ts = Ts[ind]
    rh = rh[ind]
    
    # compute wind vector components from wind speed and direction
    u, v = wind_vector(ws, wdir)
    
    # make outliers (outside +/- 3 sigma from the mean) nan, provided there are at least three non-nan data points
    ind = ~np.isnan(u)
    if any(ind):
        u[ind] = make_outliers_nan(u[ind])
    ind = ~np.isnan(v)
    if any(ind):
        v[ind] = make_outliers_nan(v[ind])
    ind = ~np.isnan(Ta)
    if any(ind):
        Ta[ind] = make_outliers_nan(Ta[ind])
    ind = ~np.isnan(Ts)
    if any(ind):
        Ts[ind] = make_outliers_nan(Ts[ind])
    ind = ~np.isnan(rh)
    if any(ind):
        rh[ind] = make_outliers_nan(rh[ind])
    
    # compute wind speed and direction again from wind vector components
    ws, wdir = wind_speed_and_direction(u, v)
    
    # eliminate nan winds, and if there is no non-nan wind, raise an exception
    ind = np.logical_and(~np.isnan(ws), ~np.isnan(wdir))
    if not any(ind):
        raise Exception('no good wind data available from Station %s during time period specified' % station_name)
    time_days = time_days[ind]
    ws = ws[ind]
    wdir = wdir[ind]
    Ta = Ta[ind]
    Ts = Ts[ind]
    rh = rh[ind]
   
    # return wind and other met data
    return (time_days, ws, wdir, Ta, Ts, rh)
    
#--------------------------------------------------------------------------------------#
def read_ndbc_station(station_name, datadir, start_time, end_time, timeshift_minutes):

    '''
    Usage: 
        
        time_days, ws, wdir, Ta, Ts, rh, Pa = read_ndbc_station(station_name, datadir, start_time, end_time, timeshift_minutes)
        
    Input:
        
        station_name = name of NDBC station
        datadir = directory containing NDBC station data files
        start_time = start time in PST as a datetime object
        end_time = end time in PST as a datetime object
        timeshift_minutes = difference between the time stamp (which comes at the 
                            end of the wind averaging period), and the center 
                            time of the wind averaging period. For NOAA CO-OPS
                            and C-MAN stations, timeshift_minutes = -1.0, but 
                            stations run by other organizations can have different 
                            time shifts
        
    Output:
    
        time_days = time in days since start time
        ws = wind speed (m/s)
        wdir = wind direction
        Ta = air temperature (oC)
        Ts = water surface temperature (oC)
        rh = relative humidity (%)
        Pa = air pressure (mbar) 
        
    Note: outliers, defined as outside 6 sigma from the mean, are removed.
    
    Time is converted from UTC to PST, and time stamp is shifted from the end of the
    averaging interval to middle of averaging interval based on the timeshift_minutes
    input variable.
    
    Returns an exception if no good wind data is found inside the time window.
    '''
    
    # find years in date range (add one day on either end as buffer)
    years = list(range((start_time-dt.timedelta(1)).year, (end_time+dt.timedelta(1)).year+1))
    
    # create list of column headers
    colnames_pre2005 = ['YYYY','MM','DD','hh','WDIR','WSPD','GST','WVHT','DPD',
                       'APD','MWD','PRES','ATMP','WTMP','DEWP','VIS','TIDE']
    colnames = ['YYYY','MM','DD','hh','mm','WDIR','WSPD','GST','WVHT','DPD',
                       'APD','MWD','PRES','ATMP','WTMP','DEWP','VIS','TIDE']
    
    # Loop through years, load data file from each year into a dataframe, and 
    # make a list of all the dataframes. First try to load the "continuous wind
    # data", which is what NDBC calls data that has been standardized to 10m 
    # height and 10min time averages. If that data does not exist, try to open 
    # the regular historical data instead. If no file exists or there is 
    # another problem loading into the data frame, print the exception to the 
    # screen but do not exit this function
    df_list = []
    for year in years:
        datafilename = os.path.join(datadir, station_name + 'h' + str(year) + '.txt')
        try:
            if year<=2004: # before 2005, there is no minutes field, so manually add it
                df = pd.read_csv(datafilename,sep='\s+',skiprows=1,names=colnames_pre2005)
                df['mm'] = 0
            elif year<=2006: # before 2007, there are no units, so only one header line
                df = pd.read_csv(datafilename,sep='\s+',skiprows=1,names=colnames)
            else:          # 2007 and beyond, there are units, so two header lines      
                df = pd.read_csv(datafilename,sep='\s+',skiprows=2,names=colnames)
            df = df.sort_index(axis=1)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(e)
        else:
            df_list.append(df)
            
    # if the list of dataframes is empty, return an exception and exit this 
    # function, otherwise compile all the dataframes in the list together into 
    # one dataframe        
    if len(df_list)==0:
        raise Exception('no wind data available from NDBC Station %s during time period specified' % station_name)
    else:
        df = df_list[0]
        for i in range(1,len(df_list)):
            df = df.append(df_list[i])
            
    # if the compiled dataframe itself is empty, return an exception
    if len(df)==0:
        raise Exception('no wind data available from NDBC Station %s during time period specified' % station_name)
                
    # extract relevant arrays from dataframe
    year = df['YYYY'].values
    month = df['MM'].values
    day = df['DD'].values
    hour = df['hh'].values
    minute = df['mm'].values
    wdir = df['WDIR'].values.astype(float) # wind direction in degrees
    ws = df['WSPD'].values # wind speed in m/s
    Ta = df['ATMP'].values # air temperature in oC
    Ts = df['WTMP'].values # water surface temperature in oC
    Td = df['DEWP'].values # dewpoint temperature in oC
    Pa = df['PRES'].values # atmospheric pressure in mbar (same as hPa)
    
    # make missing data nan
    ws[ws>90.] = np.nan
    wdir[wdir>900.] = np.nan
    Ta[Ta>90.] = np.nan
    Ts[Ts>90.] = np.nan
    Td[Td>90.] = np.nan
    Pa[Pa>9000.] = np.nan
    
    # compute the relative humidity from air temperature and dewpoint temperature
    rh = 100.*(np.exp((17.625*Td)/(243.04+Td))/np.exp((17.625*Ta)/(243.04+Ta)))
    
    # compute time as a datetime object
    datetimes = [dt.datetime(year[i],month[i],day[i],hour[i],minute[i]) for i in range(len(year))]
    
    # compute time in days since start date
    time_days = np.array([(datetime - start_time).total_seconds()/3600./24. for datetime in datetimes])
    
    # convert from UTC time to PST
    time_days = time_days - 8.0/24.
    
    # adjust the time stamp to the center of the wind averaging interval
    time_days = time_days + timeshift_minutes/60./24.
    
    # compute end time in days since start date
    end_days = (end_time - start_time).total_seconds()/3600./24.
    
    # select data in date range specified with one day buffer on either end, 
    # and if there is no data in the date range, raise an exception
    ind = np.logical_and(time_days>=-1., time_days<=(end_days+1.))
    if not any(ind):
        raise Exception('no wind data available from NDBC Station %s during time period specified' % station_name)
    time_days = time_days[ind]
    ws = ws[ind]
    wdir = wdir[ind]
    Ta = Ta[ind]
    Ts = Ts[ind]
    rh = rh[ind]
    Pa = Pa[ind]
    
    # remove repeats 
    time_days, ind = np.unique(time_days, return_index=True)
    ws = ws[ind]
    wdir = wdir[ind]
    Ta = Ta[ind]
    Ts = Ts[ind]
    rh = rh[ind]
    Pa = Pa[ind]
    
    # compute wind vector components from wind speed and direction
    u, v = wind_vector(ws, wdir)
    
    # make outliers (outside +/- 3 sigma from the mean) nan, provided there are at least three non-nan data points
    ind = ~np.isnan(u)
    if any(ind):
        u[ind] = make_outliers_nan(u[ind])
    ind = ~np.isnan(v)
    if any(ind):
        v[ind] = make_outliers_nan(v[ind])
    ind = ~np.isnan(Ta)
    if any(ind):
        Ta[ind] = make_outliers_nan(Ta[ind])
    ind = ~np.isnan(Ts)
    if any(ind):
        Ts[ind] = make_outliers_nan(Ts[ind])
    ind = ~np.isnan(rh)
    if any(ind):
        rh[ind] = make_outliers_nan(rh[ind])
    ind = ~np.isnan(Pa)
    if any(ind):
        Pa[ind] = make_outliers_nan(Pa[ind])
    
    # compute wind speed and direction again from wind vector components
    ws, wdir = wind_speed_and_direction(u, v)
    
    # eliminate nan winds, and if there is no non-nan wind, raise an exception
    ind = np.logical_and(~np.isnan(ws), ~np.isnan(wdir))
    if not any(ind):
        raise Exception('no good wind data available from Station %s during time period specified' % station_name)
    time_days = time_days[ind]
    ws = ws[ind]
    wdir = wdir[ind]
    Ta = Ta[ind]
    Ts = Ts[ind]
    rh = rh[ind]
    Pa = Pa[ind]
 
    # return time and met variables
    return (time_days, ws, wdir, Ta, Ts, rh, Pa)  

#--------------------------------------------------------------------------------------#    
def read_asos_station(station_name, datadir, start_time, end_time):

    '''
    Usage: 
        
        time_days, ws, wdir, Ta, rh, Pa = read_asos_station(station_name, datadir, start_time, end_time)
        
    Input:
        
        station_name = name of ASOS station
        datadir = directory containing ASOS station data files
        start_time = start time in PST as a datetime object
        end_time = end time in PST as a datetime object
        
    Output:
    
        time_days = time in days since start time
        ws = wind speed (m/s)
        wdir = wind direction
        Ta = air temperature (oC)
        rh = relative humidity (%)
        Pa = air pressure (mbar) 
        
    Note: outliers, defined as outside 6 sigma from the mean, are removed.
    
    Time is converted from UTC to PST, and we subtract one minute to convert 
    the end of the 2-minute averaging interval to the center of the interval 
    based on the following note from Daryl Herzmann [akrherz@iastate.edu]:
    “The wind speed value would represent 5 second observations averaged over 
    the previous two minutes for those two minutes only. The gust value is, 
    I believe, the peak 5 second value in that two minute window. There is no 
    accumulated hourly wind average or such.” 
    
    Returns an exception if no good wind data is found inside the time window   
     
    ASOS winds are already measured at 10m with some variance allowed 
    (anemometers are at 27ft or 33ft depending on local regulations). 
    '''
    
    # find years in date range (add one day on either end as buffer)
    years = list(range((start_time-dt.timedelta(1)).year, (end_time+dt.timedelta(1)).year+1))
    
    # Loop through years, load data file from each year into a dataframe, and 
    # make a list of all the dataframes. If a file doesn't exist or there is 
    # another problem loading into the data frame, print the exception to the 
    # screen but do not exit this function
    df_list = []
    for year in years:
        datafilename = os.path.join(datadir, str(year) + '/' + station_name + '.txt')
        try:
            df = pd.read_csv(datafilename,sep=',',comment='#',na_values=['M','T'],low_memory=False)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(e)
        else:
            df_list.append(df)
    
    # if the list of dataframes is empty, return an exception and exit this 
    # function, otherwise compile all the dataframes in the list together into 
    # one dataframe        
    if len(df_list)==0:
        raise Exception('no wind data available from ASOS Station %s during time period specified' % station_name)
    else:
        df = df_list[0]
        for i in range(1,len(df_list)):
            df = df.append(df_list[i])
                
    # extract timestamp, wind speed, and wind direction, converting wind speed to m/s from knots
    timestamp = df['valid'].values
    ws = df['sknt'].values * 0.514444  # convert knots to m/s
    wdir = df['drct'].values.astype(float) # wind direction (met convention)
    Ta = (df['tmpf'].values - 32.)*5./9.  # air temperature - convert F to C
    rh = df['relh'].values # relative humidity (%)
    Pa = df['mslp'].values # sea level pressure (mbar)
    
    # compute datatime from timestamp
    datetimes = [dt.datetime.strptime(ts,'%Y-%m-%d %H:%M') for ts in timestamp]
    
    # compute time in hours since start date
    time_days = np.array([(datetime - start_time).total_seconds()/3600./24. for datetime in datetimes])
    
    # convert from UTC time to PST. ASOS time stamps already correspond to the 
    # time of the measurement, so don't need time shift for this
    time_days = time_days - 8.0/24.
    
    # subtract one minute from time stamp to represent time in the middle of the
    # 2-minute measurement interval
    time_days = time_days - 1.0/60./24.
    
    # compute end time in days since start date
    end_days = (end_time - start_time).total_seconds()/3600./24.
    
    # select data in date range specified with one day buffer on either end, 
    # and if there is no data in the date range, raise an exception
    ind = np.logical_and(time_days>=-1., time_days<=(end_days+1.))
    if not any(ind):
        raise Exception('no wind data available from ASOS Station %s during time period specified' % station_name)
    time_days = time_days[ind]
    ws = ws[ind]
    wdir = wdir[ind]
    Ta = Ta[ind]
    rh = rh[ind]
    Pa = Pa[ind]
    
    # remove repeats 
    time_days, ind = np.unique(time_days, return_index=True)
    ws = ws[ind]
    wdir = wdir[ind]
    Ta = Ta[ind]
    rh = rh[ind]
    Pa = Pa[ind]
    
    # compute wind vector components from wind speed and direction
    u, v = wind_vector(ws, wdir)
    
    # make outliers (outside +/- 3 sigma from the mean) nan, provided there are at least three non-nan data points
    ind = ~np.isnan(u)
    if any(ind):
        u[ind] = make_outliers_nan(u[ind])
    ind = ~np.isnan(v)
    if any(ind):
        v[ind] = make_outliers_nan(v[ind])
    ind = ~np.isnan(Ta)
    if any(ind):
        Ta[ind] = make_outliers_nan(Ta[ind])
    ind = ~np.isnan(rh)
    if any(ind):
        rh[ind] = make_outliers_nan(rh[ind])
    ind = ~np.isnan(Pa)
    if any(ind):
        Pa[ind] = make_outliers_nan(Pa[ind])
    
    # compute wind speed and direction again from wind vector components
    ws, wdir = wind_speed_and_direction(u, v)
    
    # eliminate nan winds, and if there is no non-nan wind, raise an exception
    ind = np.logical_and(~np.isnan(ws), ~np.isnan(wdir))
    if not any(ind):
        raise Exception('no good wind data available from Station %s during time period specified' % station_name)
    time_days = time_days[ind]
    ws = ws[ind]
    wdir = wdir[ind]
    Ta = Ta[ind]
    rh = rh[ind]
    Pa = Pa[ind]
    
    # return time and met variables
    return (time_days, ws, wdir, Ta, rh, Pa) 

#--------------------------------------------------------------------------------------#   
def read_amu_amv(windfilename, max_time_steps=25000):

    ''' 
    Reads a Delft3D-FM wind input file (i.e., *.amu or *.amv) and returns
    the grid as well as the times and wind vector components. Usage:
    
        ref_time, time_zone, time_days, xg, yg, wind = read_amu_amv(windfilename, max_time_steps)
        
    Input:
        
        windfilename    =   name of an amu or amv file including the path
                            (e.g. '/home/alliek/windinput/winds.amu'), 
        max_time_steps  =   maximum number of time steps (used to initialize
                            output arrays to speed aggregation of the wind data,
                            must be greater than the actual number of time steps
                            in the wind input file
        
    Output:
    
        ref_time        =   reference time (datetime object)
        time_zone       =   time zone of reference time (string)
        time_days       =   time in days since reference time
        xg              =   x-coordinate grid (units specified in *.amu/*.amv file
                            header)
        yg              =   y-coordinate grid (units specified in *.amu/*.amv file
                            header)
        wind            =   x-component (if *.amu) or y-component (if *.amv) of 
                            wind vector on grid over time (units specified in 
                            *.amu/*.amv file header)
  
    '''
    
    with open(windfilename, 'r') as f:
        
        # read lines until the end of the header, extracting important grid info
        flag = True
        while flag:
            line = f.readline()
            if '### END OF HEADER' in line:
                flag = False
            elif 'n_rows' in line:
                n_rows = int(line.split()[2])
            elif 'n_cols' in line:
                n_cols = int(line.split()[2])
            elif 'x_llcorner' in line:
                x_llcorner = float(line.split()[2])
            elif 'y_llcorner' in line:
                y_llcorner = float(line.split()[2])                
            elif 'dx' in line:
                dx = float(line.split()[2])
            elif 'dy' in line:
                dy = float(line.split()[2])
        
        
        # initialize the time and wind output arrays 
        time_days = np.nan*np.ones(max_time_steps)
        wind = np.nan*np.ones((max_time_steps, n_rows, n_cols),dtype=float)
        
        # continue to read times and chunks of wind data until the end of the file
        time_count = -1
        for line in f.readlines():
        
            # if it is the line with the time stamp, read the new time and store the previous wind matrix
            if 'TIME' in line:
            
                # advance time counter
                time_count = time_count + 1
            
                # parse the time stamp
                timestring, equalstring, timestamp_float, timeunit, sincestr, refdatestr, reftimestr, refzonestr = line.replace('=',' = ').split()
                # extract the time as a float
                timestamp_float = float(timestamp_float)
                # extract the reference time (for now ignore the time zone)
                ref_time = dt.datetime.strptime(refdatestr + ' ' + reftimestr, '%Y-%m-%d %H:%M:%S')
                time_zone = refzonestr
                # compute the number of days since the input reference time, depending on the units
                # of the time stamp
                if timeunit=='seconds':
                    time_days[time_count] = dt.timedelta(seconds=timestamp_float).total_seconds()/3600./24.
                elif timeunit=='minutes':
                    time_days[time_count] = dt.timedelta(minutes=timestamp_float).total_seconds()/3600./24.
                elif timeunit=='hours':
                    time_days[time_count] = dt.timedelta(hours=timestamp_float).total_seconds()/3600./24.
                elif timeunit=='days':
                    time_days[time_count] = dt.timedelta(days=timestamp_float).total_seconds()/3600./24.
                else:
                    raise('time units in wind file time stamps are not recognized')
                
                # initialize row counter for reading the rows of the wind matrix that follows
                # and also initialize the wind matrix
                row_count = 0
                wind_matrix = np.nan*np.ones((n_rows,n_cols),dtype=float)
            
            # if it is not a line with a time stamp, read another row into the wind matrix for this time step
            else:
            
                # read the next line into the wind matrix and advance row number
                wind_matrix[row_count,:] = np.array(line.split()).astype(float)
                row_count = row_count + 1
                
                # if this was the last row, store the wind matrix for this time step in the
                # wind time history matrix 
                if row_count==n_rows:
                    wind[time_count,:,:] = wind_matrix
                
        # at the end of the file, trim the unused times
        ind = ~np.isnan(time_days)
        time_days = time_days[ind]
        wind = wind[ind,:,:]        
        
        # create the x, y grid corresponding to the wind points, noting that 
        # wind input follows ARC/INFO ASCII grid convention that x increases 
        # from left to right and y increases from bottom to top, so y decreases 
        # as we move from the start to farther along in the file
        x = np.linspace(x_llcorner, x_llcorner + (n_cols-1)*dx, n_cols)
        y = np.flipud(np.linspace(y_llcorner, y_llcorner + (n_rows-1)*dy, n_rows))
        xg, yg = np.meshgrid(x, y)

        # return the time, x, y, and wind
        return (ref_time, time_zone, time_days, xg, yg, wind)
 
#--------------------------------------------------------------------------------------#       
def write_winds_to_amu_amv(ref_time, tzone, time_days, xg, yg, U10, V10, outfileprefix='wind', grid_unit='m', wind_unit='m s-1'):
                           
    ''' 
    Writes the Delft3D-FM wind input files (*.amu and *.amv)
    
    Usage: 
    
        write_winds_to_amu_amv(ref_time, tzone, time_days, xg, yg, U10, V10, outfileprefix='wind', grid_unit='m', wind_unit='m s-1')
        
    Input:
        
        ref_time      =   datetime object specifying the reference time, i.e., the time at which time_days[0]=0
        tzone         =   time zone in hours w.r.t. UTC (for PST, tzone = -8)
        time_days     =   1D array with time in days since ref_time [n_time]
        xg, yg        =   regular on which wind input is specified [n_rows x n_cols]
        U10           =   10m wind vector component to the east [n_time x n_rows x n_cols]
        V10           =   10m wind vector component to the north [n_time x n_rows x n_cols]
        outfileprefix =   prefix for *.amu/*.amv files including file path (default is just 'wind')
        grid_unit     =   units of grid (default is 'm')
        wind_unit     =   units of wind data (default is 'm s-1')
        
    Output:
    
        Writes files outfileprefix.amu and outfileprefix.amv to disk.
  
    '''
                           
    # determine size of input data arrays and make sure they match
    n_time_t = len(time_days)
    n_rows_x, n_cols_x = np.shape(xg)
    n_rows_y, n_cols_y = np.shape(yg)
    n_time_u, n_rows_u, n_cols_u = np.shape(U10)
    n_time_v, n_rows_v, n_cols_v = np.shape(V10)
    if not n_rows_x == n_rows_y & n_rows_x == n_rows_u & n_rows_x == n_rows_v: 
        raise('error: row dimension must match xg[row,:], y[row,:], U10[:,row,:], V10[:,row,:]')
    if not n_cols_x == n_cols_y & n_cols_x == n_cols_u & n_cols_x == n_cols_v:
        raise('error: row dimension must match xg[:,col], y[:,col], U10[:,:,col], V10[:,:,col]')
    if not n_time_t == n_time_u & n_time_t == n_time_v:
        raise('error: time dimension must match time_days[time], U10[time,:,;], V10[time,:,:]')
    n_time = np.copy(n_time_t)
    n_rows = np.copy(n_rows_x)
    n_cols = np.copy(n_cols_x)
    
    # note grid properties and make sure it is oriented correctly
    x_llcorner = np.min(xg)
    y_llcorner = np.min(yg)
    dx = xg[0,1]-xg[0,0]
    dy = yg[0,0]-yg[1,0]
    if dx<0 or dy<0:
        raise('error: minimum of xg and yg must be in lower left corner')
    
    # make a format string for writing rows of wind matrices
    rowformat = '%0.1f' + ' %0.1f'*(n_cols-1) + '\n'
    
    # write amu file                  
    with open(outfileprefix + '.amu','wt+') as f:
        f.write('### START OF HEADER\n')
        f.write('FileVersion = 1.03\n')
        f.write('Filetype = meteo_on_equidistant_grid\n')
        f.write('NODATA_value = -999\n')
        f.write('n_cols = %d\n' % n_cols)
        f.write('n_rows = %d\n' % n_rows)
        f.write('x_llcorner = %0.2f\n' % x_llcorner)
        f.write('y_llcorner = %0.2f\n' % y_llcorner)
        f.write('dx = %0.2f\n' % dx)
        f.write('dy = %0.2f\n' % dy)
        f.write('n_quantity = 1\n')
        f.write('quantity1 = x_wind\n')
        f.write('unit1 = %s\n' % wind_unit)
        f.write('### END OF HEADER\n')
        for it in range(n_time):
            if tzone>=0:
                f.write('TIME=%013.6f hours since %s\n' % (time_days[it]*24.,ref_time.strftime('%Y-%m-%d %H:%M:%S') + ' +%02d:00' % tzone))
            else:
                f.write('TIME=%013.6f hours since %s\n' % (time_days[it]*24.,ref_time.strftime('%Y-%m-%d %H:%M:%S') + ' -%02d:00' % abs(tzone)))
            for ir in range(n_rows):
                f.write(rowformat % tuple(U10[it,ir,:]))
    
    # write amv file            
    with open(outfileprefix + '.amv','wt+') as f:
        f.write('### START OF HEADER\n')
        f.write('FileVersion = 1.03\n')
        f.write('Filetype = meteo_on_equidistant_grid\n')
        f.write('NODATA_value = -999\n')
        f.write('n_cols = %d\n' % n_cols)
        f.write('n_rows = %d\n' % n_rows)
        f.write('x_llcorner = %0.2f\n' % x_llcorner)
        f.write('y_llcorner = %0.2f\n' % y_llcorner)
        f.write('dx = %0.2f\n' % dx)
        f.write('dy = %0.2f\n' % dy)
        f.write('n_quantity = 1\n')
        f.write('quantity1 = y_wind\n')
        f.write('unit1 = %s\n' % wind_unit)
        f.write('### END OF HEADER\n')
        for it in range(n_time):
            if tzone>=0:
                f.write('TIME=%013.6f hours since %s\n' % (time_days[it]*24.,ref_time.strftime('%Y-%m-%d %H:%M:%S') + ' +%02d:00' % tzone))
            else:
                f.write('TIME=%013.6f hours since %s\n' % (time_days[it]*24.,ref_time.strftime('%Y-%m-%d %H:%M:%S') + ' -%02d:00' % abs(tzone)))
            for ir in range(n_rows):
                f.write(rowformat % tuple(V10[it,ir,:]))

#--------------------------------------------------------------------------------------#
def extract_observation_station_winds_from_amu_amv(xg, yg, U10, V10, xobs, yobs):

    '''
    U10_obs, V10_obs = extract_observation_station_winds_from_amu_amv(
                                                xg, yg, U10, V10, xobs, yobs)
    
    Input:
    
        xg, yg      = grid that has been extracted from *.amu, *.amv files
                      via the read_amu_amv function defined above -- units are 
                      specified in the *.amu/*.amv file headers)
        U10, V10    = wind field that has been extracted from *.amu, *.amv files
                      via the read_amu_amv function defined above -- units are
                      specified in the *.amu/*.amv file headers
        xobs, yobs  = location of observation station where the user wants
                      a wind time history extracted from the *.amu/*.amv data
                      (must have same units as xg, yg); may be 1D arrays 
                      representing multiple stations    
        
    Output:
    
        U10_obs, V10_obs = wind vector from *.amu, *.amv at observation station                                    
    
    '''

    # convert grid coordinates to 1D arrays
    x = xg[0,:]
    y = yg[:,0]
    
    # find number of points in time and number of observation stations
    Ntime = np.shape(U10)[0]
    if not hasattr(xobs, "__len__"):
        Nstations = 1
    else:
        Nstations = len(xobs)
        if not len(yobs)==Nstations:
            raise('xobs and yobs must be the same length')
    
    # if only one station, find its location and output winds at that location
    if Nstations==1:
    
        # locate station
        ix = np.argmin((x-xobs)**2)
        iy = np.argmin((y-yobs)**2)    
    
        # extract wind for all times from observation station location in 
        # *.amu *.amv file contents
        U10_obs = U10[:,iy,ix]
        V10_obs = V10[:,iy,ix]
    
    # if more than one station, find location of each and compile data at station 
    # locations
    else:
    
        # initialize observation station output
        U10_obs = np.nan*np.ones((Ntime,Nstations))
        V10_obs = np.nan*np.ones((Ntime,Nstations))
    
        # loop through all the stations 
        for snum in range(Nstations):
        
            # locate station
            ix = np.argmin((x-xobs[snum])**2)
            iy = np.argmin((y-yobs[snum])**2)
   
            # extract wind for all times from observation station location in 
            # *.amu *.amv file contents
            U10_obs[:,snum] = U10[:,iy,ix]
            V10_obs[:,snum] = V10[:,iy,ix]
                    
    # return wind values at observation stations
    return (U10_obs, V10_obs)

#--------------------------------------------------------------------------------------#
def read_10m_wind_data_from_csv(filepathandprefix, start_time, end_time):

    '''
    Reads the csv files containing 10m wind data from stations around SF Bay/Delta
    and returns time and wind speed
    
    Usage: 
    
        time_days, station_names, windspeed_10m = read_10m_wind_data_from_csv(filepathandprefix, start_time, end_time)
        
    Input:
        
        filepathandprefix = path and prefix of 10m wind csv file. Full file name have a year and ".csv"
                            appended to this path and prefix. Example if
                            filepathandprefix = '/Compiled_Data_10m/winddata/SFB_hourly_U10_'
                            then file name for year 2011 will be 
                            '/Compiled_Data_10m/winddata/SFB_hourly_U10_2011.csv'
        start_time = start of time window desired, also used as reference time (datetime object)
        end_time = end of time window desired (datetime object)
        
    Output:
    
        time_days = time in days since start time
        station_names = Organization-Name of observation stations across SFB/Delta where wind speeds are measured
        windspeed_10m = wind speed data at times and stations
    
    '''

    # loop through all the years
    isfirstyear = True
    for year in range(start_time.year, end_time.year+1):
    
        # read data frame from this year
        df = pd.read_csv(filepathandprefix + str(year) + '.csv',comment='#',sep=',',header=[0,1]) 
        
        # if this is the first year, extract the station names from the header
        if isfirstyear:
            headers_and_units = list(df)
            headers = [header_and_unit[0] for header_and_unit in headers_and_units]
            station_names = headers[1:]
        
        # extract all the data and divide it into time and winds data    
        data_plus_time = df.values
        jday = data_plus_time[:,0]
        winddata = data_plus_time[:,1:]
        
        # compute time in days since start date
        time_days = np.array([])
        for it in range(len(jday)):
            time_days_1 = (dt.datetime(year, 1, 1) + dt.timedelta(days=(jday[it] - 1)) - start_time).total_seconds()/3600./24
            time_days = np.append(time_days,time_days_1)
        
        # check if this is the first year, and if so, initilaize compiled
        # data set, otherwise, append
        if isfirstyear:
            time_days_all = time_days
            winddata_all = winddata
            isfirstyear=False
        else:
            time_days_all = np.append(time_days_all, time_days)
            winddata_all = np.append(winddata_all, winddata, axis=0)
            
    # now that all the data is compiled, select only the data in the time window specified
    end_day = (end_time - start_time).total_seconds()/24./3600.
    ind = np.logical_and(time_days_all>=0.0, time_days_all<=end_day)
    time_days = time_days_all[ind]
    windspeed_10m = winddata_all[ind,:]
    
    # return the data
    return (time_days, station_names, windspeed_10m)
    
#--------------------------------------------------------------------------------------# 
def read_met_data_from_netcdf(filepathandprefix, start_time, end_time):

    '''
    Reads netcdf files containing observed met variables and estimated 10m wind speed from 
    stations around SF Bay/Delta and returns time, met variables, and estimated 10m wind in
    the time window specified 
    
    Usage: 
        
        (
        station_name, station_backup, station_description, organization, latitude, longitude, 
        za, zt, zq, z0, z0t, z0q, time_days, u, v, Ta, Ts, rh, Pa, u10, v10 
                         = read_met_data_from_netcdf(filepathandprefix, start_time, end_time)
        )
        
    Input:
        
        filepathandprefix = path and prefix of netcdf files containing met data. Full file name 
                            has a year and ".nc" appended to this path and prefix. Example if
                            filepathandprefix = '/winddata/SFB_hourly_wind_and_met_data_'
                            then file name for year 2011 will be 
                            '/winddata/SFB_hourly_wind_and_met_data_2011.nc'
        start_time = start of time window desired, also used as reference time (datetime object)
        end_time = end of time window desired (datetime object)
        
    Output:
    
        station_name            = station names (Nstations)
        station_backup          = station used for temperature data where main data is missing (Nstations)
        station_description     = long station description (Nstations)
        organization            = organization reporting wind station data (CIMIS, NDBC, or ASOS)
        latitude                = latitude of station (Nstations)
        longitude               = longitude of station (Nstations)
        za                      = anemometer height (m) (Nstations)
        zt                      = temperature measurement height (m) (Nstations)
        zq                      = humidity measurement height (m) (Nstations)
        z0                      = momentum roughness height for land stations (m) (Nstations)
        z0t                     = temperature roughness height for land stations (m) (Nstations)
        z0q                     = humidity roughness height for land stations (m) (Nstations)
        time_days               = time in days since start time (Ntime)
        u                       = wind vector component to east at anemometer height (m/s) (Ntime, Nstations)
        v                       = wind vector component to north at anemometer height (m/s) (Ntime, Nstations)
        Ta                      = air temperature (oC) (Ntime, Nstations)
        Ts                      = water temperature at water stations, soil temperature at land stations (oC) (Ntime, Nstations)
        rh                      = relative humidity (%) (Ntime, Nstations)
        Pa                      = air pressure (mbar) (Ntime, Nstations)
        u10                     = wind vector component to east at 10m (m/s) (Ntime, Nstations)
        v10                     = wind vector component to east at 10m (m/s) (Ntime, Nstations)
    
    '''

    # loop through all the years
    isfirstyear = True
    for year in range(start_time.year, end_time.year+1):
    
        # read netcdf file from this year
        dataset = Dataset(filepathandprefix + str(year) + '.nc','r')
        
        # if it's the first time step, extract time-independent data from file
        if isfirstyear:
            station_name = dataset.variables['station_name'][:]
            station_backup = dataset.variables['station_backup'][:]
            station_description = dataset.variables['station_description'][:]
            organization = dataset.variables['organization'][:]
            latitude = dataset.variables['latitude'][:].data
            longitude = dataset.variables['longitude'][:].data
            za = dataset.variables['anemometer_height'][:].data
            zt = dataset.variables['temperature_height'][:].data
            zq = dataset.variables['humidity_height'][:].data
            z0 = dataset.variables['z0'][:].data
            z0t = dataset.variables['z0t'][:].data
            z0q = dataset.variables['z0q'][:].data
            
        # extract time-dependent data from the file
        jday_1 = dataset.variables['time'][:].data
        u_1 = dataset.variables['u'][:,:].data
        v_1 = dataset.variables['v'][:,:].data
        Ta_1 = dataset.variables['Ta'][:,:].data
        Ts_1 = dataset.variables['Ts'][:,:].data
        rh_1 = dataset.variables['rh'][:,:].data
        Pa_1 = dataset.variables['Pa'][:,:].data
        u10_1 = dataset.variables['u10'][:,:].data
        v10_1 = dataset.variables['v10'][:,:].data
        
        # compute days since start date
        time_days_1 = np.array([(dt.datetime(year,1,1) + dt.timedelta(days=(j-1)) - start_time).total_seconds()/3600./24. for j in jday_1])
        
        # if this is the first year, initialize compiled variables, otherwise append
        if isfirstyear:
            isfirstyear = False
            time_days = np.copy(time_days_1)
            u = np.copy(u_1)
            v = np.copy(v_1)
            Ta = np.copy(Ta_1)
            Ts = np.copy(Ts_1)
            rh = np.copy(rh_1)
            Pa = np.copy(Pa_1)
            u10 = np.copy(u10_1)
            v10 = np.copy(v10_1)
        else:
            time_days = np.append(time_days, time_days_1)
            u = np.append(u, u_1, axis=0)
            v = np.append(v, v_1, axis=0)
            Ta = np.append(Ta, Ta_1, axis=0)
            Ts = np.append(Ts, Ts_1, axis=0)
            rh = np.append(rh, rh_1, axis=0)
            Pa = np.append(Pa, Pa_1, axis=0)
            u10 = np.append(u10, u10_1, axis=0)
            v10 = np.append(v10, v10_1, axis=0)       
      
    # return data      
    return (station_name, station_backup, station_description, organization, latitude, longitude, za, zt, zq, z0, z0t, z0q, time_days, u, v, Ta, Ts, rh, Pa, u10, v10)
            
#--------------------------------------------------------------------------------------#            
     
