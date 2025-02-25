import pandas as pd
import xarray as xr
import numpy as np

from tqdm import tqdm


ROOT = '/nfs/b0133/eejco/data/MEOP_2024/media/disk2/roquet/MEOP_public/MEOP-CTD_2024-03-08/'
FNAME_PROFILES = ROOT + 'list_profiles.csv'
FNAME_TAGS = ROOT + 'list_tags.csv'
FNAME_DEPLOYMENTS = ROOT + 'list_deployments.csv'


class Profile():
    def __init__(self,ds):
        self.latitude = ds.LATITUDE.item()
        self.longitude = ds.LONGITUDE.item()
        self.time = np.datetime64(ds.JULD.item().strftime('%Y-%m-%dT%H:%M:%S'))

        coords = {'pressure': ds.PRES_ADJUSTED}
        self.ds = xr.Dataset(
            {'temperature': xr.DataArray(ds.TEMP_ADJUSTED, coords=coords),
             'salinity': xr.DataArray(ds.PSAL_ADJUSTED, coords=coords)}
        )

class MEOP():

    def __init__(self,extent):

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

    def load_profiles(self):

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
                    profiles.append(Profile(ds.sel(N_PROF=int(n))))

        self.profiles = profiles

        lats = [p.latitude for p in profiles]
        lons = [p.longitude for p in profiles]
        sst = [p.ds.temperature.isel(N_LEVELS=0).item() for p in profiles]
        times = [p.time for p in profiles]

        self.df = pd.DataFrame({'latitude':lats,
                                'longitude':lons,
                                'sst': sst},
                                index=times)

    def filter_profiles(self,start,end):
        filtered_profiles=[]
        for p in self.profiles:
            if (p.time >= start) and (p.time < end):
                filtered_profiles.append(p)

        return filtered_profiles