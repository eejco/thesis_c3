import pandas as pd
import xarray as xr
import numpy as np
import glob

from load_meop import MEOP
from stericheight.data_handler import GEBCO

argofolder = '../data/ARGO/'
fname = '../data/qc_profile_ds_2dbar.nc'


def load_profiles(extent=[139, 150,-68,-65.5],maxz=1000):

    print('Loading elevation...')
    elevation_ = GEBCO(coarsen_factor=1).ds.elevation
    elevation = elevation_.sel(latitude = slice(extent[2], extent[3]), longitude = slice(extent[0],extent[1]))

    def crop2shelf(ds,lat='lat',lon='lon'):
        ds['elevation'] = elevation.interp(latitude=ds[lat],longitude=ds[lon]).drop(['longitude','latitude'])
        return ds.where(ds.elevation > -1000,drop=True)
    
    pax = np.linspace(0,2+maxz,2)

    print('Loading MEOP...')
    meop = MEOP(extent)
    meop.load_profiles_for_spira(pax)
    meopds = crop2shelf(meop.ds)


    print('Loading Spira...')
    ds = xr.open_dataset(fname).swap_dims({'n_prof':'time'})
    spira = ds.where(
        (ds.lon > extent[0]) &
        (ds.lon < extent[1]) &
        (ds.lat > extent[2]) &
        (ds.lat < extent[3]), drop=True
    )
    spira = crop2shelf(spira)
    spira = spira.sel(pres=slice(0,maxz))


    print('Loading Argo...')
    # also add argo
    profiles=[]
    # also add argo
    profiles=[]
    for file in glob.glob(argofolder + '*nc'):
        dsa = xr.open_dataset(file)
        for t in dsa.TIME:
            if t < pd.Timestamp(2022,1,1):
                dst = dsa.sel(TIME=t)
                if (dst.LATITUDE < extent[3]) & (dst.LONGITUDE < extent[1]):
                    dstn = dst.swap_dims({'DEPTH':'PRES_ADJUSTED'}).dropna(dim='PRES_ADJUSTED')
                    if elevation.interp({'latitude':dstn.LATITUDE,'longitude':dstn.LONGITUDE}) > -1000:
                        if len(dstn.PRES)>0:
                            pnew = dstn.expand_dims(['TIME']).drop_vars([
        'TRAJECTORY', 'TIME_QC', 'POSITION_QC', 'DC_REFERENCE', 'DIRECTION', 'PRES_QC', 'PRES_ADJUSTED_QC', 'TEMP_ADJUSTED_QC','PSAL_ADJUSTED_QC','PRES_CORE','PRES'
        ])
                            pnew = pnew.assign_coords(
                                LATITUDE=("TIME", [pnew.LATITUDE.values]),
                                LONGITUDE=("TIME", [pnew.LONGITUDE.values])
                            )
                            profiles.append(pnew.interp({'PRES_ADJUSTED':pax}))

    argods = xr.merge(profiles).rename({
        'TIME':'time',
        'LONGITUDE':'lon',
        'LATITUDE':'lat',
        'PRES_ADJUSTED':'pres',
        'TEMP_ADJUSTED':'temp',
        'PSAL_ADJUSTED':'psal'})
    
    argods = crop2shelf(argods)

    print('Merging...')
    df=pd.concat([spira.to_dataframe(),argods.to_dataframe(),meopds.to_dataframe()])
    pres=[i[1] for i in df.index]
