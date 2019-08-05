# trying to back out some wave parameters from https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2010JC006172

f=1.0 # average frequency
hrms=0.25 # typical

a=hrms/2.0
h=2.2 # Benthic station msl

omega=2*np.pi*f
grav=9.8
k0=1.0
while 1:
    k1=omega**2/(grav*math.tanh(k0*h))
    if np.abs(k0-k1)<1e-5: break
    k0=k1
k=k1    
c=omega/k
ustok=omega*k*a**2

print(f"u_stok={ustok}  c={c}")

##

# Maybe better to take a look at fetch-limited formulas
# https://planetcalc.com/4442/
# or in unstruc.f90 of DFM
#  - Hurdle, Stive formulation?
#    calculates fetch length and fetch depth (~line 25599)
#   there is a function hurdlestive,
#      appears to take U10 wind, fetch length, fetch depth, and maybe return Hsig and Tsig
#      ! taken from Hurdle, Stive 1989 , RESULTS SEEM VERY SIMILAR TO THOSE OF DELWAQ CODE ABOVE

##

# https://www.valleywater.org/sites/default/files/Appx%20C-E/Appx%20C-E/E7_Appx%20D%20Hydrology%20and%20Hydraulics%20%281%29.pdf

hsig=0.3028*0.6
Tsig=1.5
f=1./Tsig

a=hsig/2.0
h=10*0.3028 # 'point 7'

omega=2*np.pi*f
grav=9.8
k0=1.0
while 1:
    k1=omega**2/(grav*math.tanh(k0*h))
    if np.abs(k0-k1)<1e-5: break
    k0=k1
k=k1    
c=omega/k
ustok=omega*k*a**2

print(f"u_stok={ustok}  c={c}")

# That is much more reasonable.  Stokes drift of 0.06 m/s.
# typical wind drift effect between 1 and 5% (1% for drifters like in Sagami.)
Uwind=8.9 # mph=> m/s
# so in this case stokes drift is 0.6% wind speed.

