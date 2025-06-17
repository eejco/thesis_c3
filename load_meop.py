import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs

from tqdm import tqdm
from itertools import compress
from gsw import density

from stericheight.plotting_fns import PlottingFns
pfns = PlottingFns()

ROOT = '/nfs/b0133/eejco/data/MEOP_2024/media/disk2/roquet/MEOP_public/MEOP-CTD_2024-03-08/'
FNAME_PROFILES = ROOT + 'list_profiles.csv'
FNAME_TAGS = ROOT + 'list_tags.csv'
FNAME_DEPLOYMENTS = ROOT + 'list_deployments.csv'
MAXZ = 300

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

    def load_profiles(self, profile_code=None):
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

        # define methods to retrieve info one each profile
        deployment = lambda platform: str.split(platform,'-')[0]
        country = lambda platform: str(self.deployments[self.deployments.DEPLOYMENT_CODE == deployment(platform)].COUNTRY.item())
        folder = lambda platform: ROOT + country(platform) + '/DATA/'
        filename = lambda platform: platform + '_all_prof.nc'

        # gather profiles in desired area, discard others
        profiles = []
        for pc in tqdm(self.profile_codes if profile_code==None else [profile_code]):
            fname = folder(pc)+filename(pc)
            ds = xr.open_dataset(fname)

            in_area = self.is_in(ds,self.extent)

            if any(in_area):
                for n in ds.N_PROF.where(in_area,drop=True):
                    ds_ = ds.sel(N_PROF=int(n))
                    t = ds_.TEMP_ADJUSTED
                    s = ds_.PSAL_ADJUSTED
                    p = ds_.PRES_ADJUSTED

                    if not (np.all(np.isnan(t)) or np.all(np.isnan(s))):
                        coords = {'pressure': p}

                        vars = {'platform':pc,
                                'latitude':ds_.LATITUDE.item(),
                                'longitude':ds_.LONGITUDE.item(),
                                'time':np.datetime64(ds_.JULD.item().strftime('%Y-%m-%dT%H:%M:%S')),
                                'filename':fname,
                                'temperature':[xr.DataArray(t, coords=coords)],
                                'salinity':[xr.DataArray(s, coords=coords)],
                                'gph':calc_gpha(t,s,p,MAXZ)
                        }
                        profiles.append(vars)

        df = pd.DataFrame(profiles).set_index('time')

        # prepare common pressure dimension
        allpres = np.concatenate([p[0].pressure for p in df.temperature])
        values, counts = np.unique(allpres[~np.isnan(allpres)], return_counts=True)
        pressure_df = pd.DataFrame(counts,values)
        self.pressure_axis= pressure_df[(pressure_df>1000).to_numpy().transpose().flatten()].index.to_numpy()

        self.df = df
        return df

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

    def dissect_month(self,ts,background_da,ns_section_lat=140,tmin=-2.2,tmax=-0.9,smin=33.8,smax=34.8):

        # in order to combine many profiles to create a transect, we need to interpolate them onto the same vertical grid
        # create a pressure index using the most commonly occurring values for profiles
        allpres = np.concatenate([p.ds.pressure.to_numpy() for p in self.profiles])
        values, counts = np.unique(allpres[~np.isnan(allpres)], return_counts=True)
        pressure_df = pd.DataFrame(counts,values)
        pressure_axis= pressure_df[(pressure_df>1000).to_numpy().transpose().flatten()].index.to_numpy()

        # function to take a profile and turn it into a ds with uniform z-axis and tagged with the longitude
        def reindex_profile(profile,dim='longitude'):
            return profile.ds \
                    .rename({'N_LEVELS':'pressure'}) \
                    .dropna(dim='pressure',how='all') \
                    .interp(coords={'pressure':pressure_axis}) \
                    .assign_coords({dim:profile.get(dim)}) \
                    .expand_dims(dim)
        
        # Set up figre
        fig = plt.figure(figsize=(24,14))
        axs=[]

        gs = fig.add_gridspec(3,6)

        # Restrict data to within our chosen month
        print('Preparing data...')
        profiles = self.filter_profiles(ts,ts+pd.DateOffset(months=1))
        profiles_df = self.df[(self.df.index > ts) & (self.df.index < ts+pd.DateOffset(months=1))]

        # Extract profile data
        dses = []
        for p in tqdm(profiles):
            if not (np.all(np.isnan(p.ds.temperature)) and np.all(np.isnan(p.ds.salinity))):
                dses.append(reindex_profile(p))

        ds = xr.merge(dses)

        # Get data for selected transect
        print('Preparing data for n-s transect...')
        dses_transect = []
        for p in tqdm(profiles):
            if not (np.all(np.isnan(p.ds.temperature)) and np.all(np.isnan(p.ds.salinity))):
                if (p.longitude > (ns_section_lat - 0.2)) and (p.longitude < (ns_section_lat + 0.2)):
                    dses_transect.append(reindex_profile(p,dim='latitude'))

        ds_transect = xr.merge(dses_transect)

        print('Creating figures...')
        # FIG A) SLA + PROFILE LOCATION
        axs.append(fig.add_subplot(gs[0,0:4],projection=ccrs.Mercator()))

        im=pfns.sp(ax=axs[0],da=background_da.sel(time=ts),extent=self.extent,vmax=10)
        axs[0].scatter(profiles_df.longitude, profiles_df.latitude, transform=ccrs.PlateCarree())
        axs[0].set_title(ts)
        cbar = plt.colorbar(im, label='SLA')

        axs[0].add_patch(mpatches.Rectangle(xy=[ns_section_lat-0.2, self.extent[2]],
                                    width=0.4,
                                    height=self.extent[3]-self.extent[2],
                                    facecolor='none', edgecolor='r',
                                    transform=ccrs.PlateCarree()))

        # FIG B) COASTAL TRANSECT
        axs.append(fig.add_subplot(gs[1,0:4]))
        ds.temperature.plot.contourf(ax=axs[1],x='longitude',y='pressure',vmin=tmin,vmax=tmax,cmap='magma')
        axs[1].invert_yaxis()
        axs[1].set_xlabel('Longitude')
        axs[1].set_ylabel('Pressure/dbar')

        # FIG C) MEAN TEMPERATURE PROFILE
        axs.append(fig.add_subplot(gs[0:2,4]))

        for lon in ds.longitude:
            ds_ = ds.sel(longitude=lon)
            axs[2].plot(ds_.temperature,ds_.pressure,color='thistle')
        ds_mean=ds.mean('longitude')
        axs[2].plot(ds_mean.temperature,ds_mean.pressure,color='mediumvioletred')
        axs[2].invert_yaxis()
        axs[2].set_ylabel('Pressure')
        axs[2].set_xlabel('Temperature')
        axs[2].set_xlim([tmin,tmax])

        # FIG d) MEAN salinity PROFILE
        axs.append(fig.add_subplot(gs[0:2,5]))

        for lon in ds.longitude:
            ds_ = ds.sel(longitude=lon)
            axs[3].plot(ds_.salinity,ds_.pressure,color='lightblue')
        ds_mean=ds.mean('longitude')
        axs[3].plot(ds_mean.salinity,ds_mean.pressure,color='steelblue')
        axs[3].invert_yaxis()
        axs[3].set_ylabel('Pressure')
        axs[3].set_xlabel('Salinity')
        axs[3].set_xlim([smin,smax])

        # FIG E) T-S PLOT
        axs.append(fig.add_subplot(gs[2,0:2]))
        for profile in profiles:
            im=axs[4].scatter(profile.ds.salinity,profile.ds.temperature,5,profile.ds.pressure,vmin=0,vmax=500,cmap='winter')
        axs[4].set_ylim([tmin,tmax])
        axs[4].set_xlim([smin,smax])
        cbar = plt.colorbar(im, label='Pressure')

        # FIG F) N-S Transect Temperature
        axs.append(fig.add_subplot(gs[2,2:4]))
        ds_transect.temperature.plot.contourf(ax=axs[5],x='latitude',y='pressure',vmin=tmin,vmax=tmax,cmap='magma')
        axs[5].invert_yaxis()
        axs[5].set_xlabel('Latitude')
        axs[5].set_ylabel('Pressure/dbar')
        axs[5].set_title('N-S transect at {}E'.format(ns_section_lat))

        # FIG F) N-S Transect Salinity
        axs.append(fig.add_subplot(gs[2,4:6]))
        ds_transect.salinity.plot.contourf(ax=axs[6],x='latitude',y='pressure',vmin=smin,vmax=smax)
        axs[6].invert_yaxis()
        axs[6].set_xlabel('Latitude')
        axs[6].set_ylabel('Pressure/dbar')
        axs[6].set_title('N-S transect at {}E'.format(ns_section_lat))

        print('Rendering..')

        plt.tight_layout()

    def reindex_profiles(self,df=None):
        def reindex_single_profile(profile,time):
            return profile \
                    .swap_dims({'N_LEVELS':'pressure'}) \
                    .dropna(dim='pressure',how='all') \
                    .interp(coords={'pressure':self.pressure_axis}) \
                    .assign_coords({'time':time}) \
                    .expand_dims('time')
        if df is None:
            df = self.df
        #remove duplicates
        #not very nice.. think about this more later
        filtered_df = df[~df.index.duplicated(keep='first')]

        print('reindexing temps..')
        temps_list = [reindex_single_profile(p[0],l) for p,l in zip(filtered_df.temperature, filtered_df.index)]
        print('reindexing psals...')
        psal_list = [reindex_single_profile(p[0],l) for p,l in zip(filtered_df.salinity, filtered_df.index)]

        print('merging...')
        temp_da = xr.merge(temps_list).TEMP_ADJUSTED
        psal_da = xr.merge(psal_list).PSAL_ADJUSTED

        return temp_da, psal_da, filtered_df