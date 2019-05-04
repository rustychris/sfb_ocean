from stompy.io.local import coamps
from stompy.spatial import field

import numpy as np

import six
six.moves.reload_module(coamps)

##
args=dict(start=np.datetime64("2017-06-02"),
          stop=np.datetime64("2017-06-02"),
          cache_dir="cache",fields=["air_temp","rltv_hum","sol_rad"])
gribs=coamps.fetch_coamps_wind(**args)
files=coamps.coamps_files(**args)
## 

# air temp:
gribs[0]

# this seems legit - has a good range, seems likely it's from the
# right elevation
air_temp=field.GdalGrid(gribs[0])

# also seems legit -- near 100 over the ocean, near 10 in Nevada...
rltv_hum=field.GdalGrid(gribs[1])

# first one is 0 everywhere.
sol_rads=[f for f in gribs if f.endswith('sol_rad')]

## 
for i,rec in enumerate(coamps.coamps_files(**args)):
    plt.figure(i).clf()
    sol_rad=field.GdalGrid(rec['sol_rad']['local'])
    img=sol_rad.plot(cmap='jet',clim=[0,1300])
    plt.gca().set_title(str(rec['sol_rad']['timestamp']))
    plt.colorbar(img)
    plt.savefig('sol_rad_%02d.png'%i)

##

# maybe there is something wrong with 2017-06-01
# hour 4 looks like start of night
# hour 13 is dawn
# 


# maybe in W/m2?  ranges up to 500.
sol_rad=field.GdalGrid(sol_rads[13])
plt.clf()
img=sol_rad.plot(cmap='jet')
plt.colorbar(img)



##

# Air temperature:
#  US058GMET-GR1dyn.COAMPS-CENCOOS_CENCOOS-n3-c1_00000F0NL2017060100_0100_010000-000000air_temp   
#  US058GMET-GR1dyn.COAMPS-CENCOOS_CENCOOS-n3-c1_00000F0NL2017060100_0105_000020-000000air_temp
#  US058GMET-GR1dyn.COAMPS-CENCOOS_CENCOOS-n3-c1_00200F0NL2017060100_0105_000020-000000air_temp
#
# Relative humidity -- basically the same as air temp
#  US058GMET-GR1dyn.COAMPS-CENCOOS_CENCOOS-n3-c1_00000F0NL2017060100_0100_002000-000000rltv_hum
#  US058GMET-GR1dyn.COAMPS-CENCOOS_CENCOOS-n3-c1_00000F0NL2017060100_0100_005000-000000rltv_hum
#  US058GMET-GR1dyn.COAMPS-CENCOOS_CENCOOS-n3-c1_00000F0NL2017060100_0100_007000-000000rltv_hum

#                                               hour of run
#                                               |  ......YYYYMMDDHH
#                                               |        |         .elev_code
#                                               |        |          |    elev--.......field_name
# US058GMET-GR1dyn.COAMPS-CENCOOS_CENCOOS-n3-c1_00000F0NL2017060100_0100_007000-000000rltv_hum

# so air temperature and rltv humidity: if it exists at elev_code=102 that's best, 

# I think this is a surface measurement?
# US058GMET-GR1dyn.COAMPS-CENCOOS_CENCOOS-n3-c1_00400F0NL2017060100_0001_000000-000000sol_rad
# 

##

# import merged_sun

