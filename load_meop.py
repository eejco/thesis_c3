import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import compress
from gsw import density


ROOT = '/nfs/b0133/eejco/data/MEOP_2024/media/disk2/roquet/MEOP_public/MEOP-CTD_2024-03-08/'
FNAME_PROFILES = ROOT + 'list_profiles.csv'
FNAME_TAGS = ROOT + 'list_tags.csv'
FNAME_DEPLOYMENTS = ROOT + 'list_deployments.csv'
MAXZ = 300

class Profile():
    def __init__(self,ds,filename,full=False):
        def calc_gpha(t,s,p,maxz):
            try:
                crop = lambda da: da.where(p<maxz,drop=True)
                rho = density.rho(crop(s),crop(t),crop(p))
                rho_ref = density.rho(35,0,crop(p))

                vs = (1/rho - 1/rho_ref)/9.82
                pres_pa = crop(p) * 10000

                vs = vs.assign_coords({'PRES_PA':pres_pa})
                return vs.integrate('PRES_PA').item()
            except:
                return np.nan

        self.latitude = ds.LATITUDE.item()
        self.longitude = ds.LONGITUDE.item()
        self.time = np.datetime64(ds.JULD.item().strftime('%Y-%m-%dT%H:%M:%S'))
        self.filename = filename
        self.gph = calc_gpha(ds.TEMP_ADJUSTED,ds.PSAL_ADJUSTED,ds.PRES_ADJUSTED,MAXZ)

        if full:
            coords = {'pressure': ds.PRES_ADJUSTED}
            self.ds = xr.Dataset(
                {'temperature': xr.DataArray(ds.TEMP_ADJUSTED, coords=coords),
                'salinity': xr.DataArray(ds.PSAL_ADJUSTED, coords=coords)}
            )

class MEOP():

    def __init__(self,extent,minz=0):

        self.extent = extent

        #expand extent to capture more profiles
        expand_lon = 5
        expand_lat = 5
        self.expanded_extent = [extent[0]-expand_lon,
                                extent[1]+expand_lon,
                                extent[2]-expand_lat,
                                extent[3]+expand_lat]

        # read deployments and trim to area we care about
        self.deployments = pd.read_csv(FNAME_DEPLOYMENTS)
        #deployments = deployments[deployments.LATITUDE < -30]

        # read profiles and trim to latitudes we care about
        profiles_ = pd.read_csv(FNAME_PROFILES)
        profiles_ = profiles_[self.is_in(profiles_,self.expanded_extent)]
        profiles_ = profiles_[(~np.isnan(profiles_.N_TEMP))&(~np.isnan(profiles_.N_PSAL))].reset_index(drop=True)

        self.profile_codes=profiles_.SMRU_PLATFORM_CODE.unique()

    def is_in(self,profile,extent):
        return ((profile.LATITUDE <= extent[3]) & 
                (profile.LATITUDE >= extent[2]) &
                (profile.LONGITUDE <= extent[1]) &
                (profile.LONGITUDE >= extent[0])
        )

    def load_profiles(self,full=False):

        # define methods to retrieve info one each profile
        deployment = lambda platform: str.split(platform,'-')[0]
        country = lambda platform: str(self.deployments[self.deployments.DEPLOYMENT_CODE == deployment(platform)].COUNTRY.item())
        folder = lambda platform: ROOT + country(platform) + '/DATA/'
        filename = lambda platform: platform + '_all_prof.nc'

        # gather profiles in desired area, discard others
        profiles = []
        for pc in tqdm(self.profile_codes):
            fname = folder(pc)+filename(pc)
            ds = xr.open_dataset(fname)

            in_area = self.is_in(ds,self.extent)

            if any(in_area):
                for n in ds.N_PROF.where(in_area,drop=True):
                    profiles.append(Profile(ds.sel(N_PROF=int(n)),fname,full=full))

        self.profiles = profiles

        lats = [p.latitude for p in profiles]
        lons = [p.longitude for p in profiles]
        filename = [p.filename for p in profiles]
        gph = [p.gph for p in profiles]
        times = [p.time for p in profiles]

        self.df = pd.DataFrame({'latitude':lats,
                                'longitude':lons,
                                'filename':filename,
                                'gph':gph},
                                index=times)

    def filter_profiles(self,start,end):
        filtered_profiles=[]
        for p in self.profiles:
            if (p.time >= start) and (p.time < end):
                filtered_profiles.append(p)

        return filtered_profiles
    
    def plot_monthly_profiles(self,year):
        fig, ax = plt.subplots(2,5,figsize=(16,6))

        ax = ax.flat
        for m in [2,3,4,5,6,7,8,9,10,11]:
            fp=self.filter_profiles(pd.Timestamp(year,m,1),pd.Timestamp(year,m+1,1))
            for p in fp:
                ax[m-2].plot(p.ds.temperature,p.ds.pressure*-1)

            ax[m-2].set_ylim([-800,0])
            ax[m-2].set_xlim([-2,1.5])
            ax[m-2].set_title('month={0}, n_profiles={1}'.format(m,len(fp)))

    def plot_ts_for_years(self,years):
        fig, ax = plt.subplots(1,4,figsize=(18,4))

        for i,y in enumerate(years):
            idx = (np.equal(years,y))
            for profile in list(compress(self.profiles,idx)):
                im=ax[i].scatter(profile.ds.salinity,profile.ds.temperature,5,profile.ds.pressure,vmin=0,vmax=500)
                ax[i].set_title(str(y))
                ax[i].set_ylim([-2.4,1.4])
                ax[i].set_xlim([33,34.75])

    def grid_profiles(self,lons,lats,lonwindow=None,latwindow=None):

        if (lonwindow is None) and (latwindow is None):
            if (len(lons) > 1) and (len(lats) > 1):
                lonwindow = (lons[1] - lons[0])/2
                latwindow = (lats[1] - lats[0])/2
            else:
                ValueError('lon and lat insufficient length to compute lat and lon windows: please provide')

        def get_in(df,lat,lon):
            return df[(df.longitude > lon - lonwindow) &
                    (df.longitude < lon + lonwindow) &
                    (df.latitude > lat - latwindow) &
                    (df.latitude < lat + latwindow)]

        def resample(df_in):
            df = df_in.resample('MS').mean(numeric_only=True)
            count = df_in.resample('MS').count()
            return df.dropna(), count[count > 0].dropna()

        def create_ds(df,df_ct,lat,lon):
            time = df.index.to_numpy()
            data_vars={}
            def add_var(name):
                data_vars['profile_'+name]=(["longitude","latitude","time"], np.expand_dims(df[name].to_numpy(),axis=(0,1)))
            add_var('gph')
            add_var('latitude')
            add_var('longitude')
            data_vars['profile_cnt']=(["longitude","latitude","time"], np.expand_dims(df_ct['gph'].to_numpy(),axis=(0,1)))
            return xr.Dataset(
                data_vars = data_vars,
                coords=dict(
                    longitude=(["longitude"], [lon]),
                    latitude=(["latitude"], [lat]),
                    time=(["time"],time),
                    )
                )
                
        def build_ds(lat,lon):
            data_ = get_in(self.df,lat,lon)

            if len(data_) > 0:
                data, ct = resample(data_)
                ds = create_ds(data,ct,lat,lon)
            else:
                ds = None

            return ds

        dses = []
        for lon in tqdm(lons):
            for lat in lats:
                ds = build_ds(lat,lon)
                if ds is not None:
                    dses.append(ds)
                # tmp_dses = list(map(build_ds,lats,[lon for _ in lats]))
                # ds_for_lon = xr.merge(list(filter(lambda item: item is not None, tmp_dses)))
                # dses = dses + [ds_for_lon]

        print('merging...')
        self.gridded_profiles = xr.merge(dses)

