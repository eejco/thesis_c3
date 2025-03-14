import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import numpy as np
import cmocean
import pandas as pd

from stericheight.plotting_fns import PlottingFns
from stericheight.data_handler import GEBCO
from stericheight.utils import Utils, Color as uc
from tqdm.notebook import tqdm_notebook

class Region():
    def __init__(self,name,extent=None,elevation=None,min_elevation=None,max_elevation=None):
        
        """
        extent = [minlon,maxlon,minlat,maxlat]
        """
        pfns = PlottingFns()
        
        self.name = name

        if (name in pfns.SEAS.keys()) and (extent is None):
            extent = pfns.SEAS[name]
        
        self.minlat = extent[2]
        self.maxlat = extent[3]
        self.minlon = extent[0]
        self.maxlon = extent[1]
        
        self.extent = extent
        
        self.min_elevation = min_elevation
        self.max_elevation = max_elevation
        self.elevation = elevation if elevation is not None else GEBCO().ds.elevation
        
        self.wrapped = self.minlon > self.maxlon
        
    def crop_da_xy(self,da,ignore_elevation=False):
        if not self.max_elevation and not self.min_elevation:
            ignore_elevation = True

        if self.wrapped:
            rtn = da.where((da.longitude < self.maxlon) | (da.longitude > self.minlon))   
        else:
            rtn = da.where((da.longitude > self.minlon) & (da.longitude < self.maxlon))
            
        rtn = rtn.where((da.latitude > self.minlat) & (da.latitude < self.maxlat))
            
        # interpolate elevation to da being cropped
        if not ignore_elevation:
            elev_interp = self.elevation.interp({'latitude': da.latitude,'longitude': da.longitude})
            if self.min_elevation:
                rtn = rtn.where(elev_interp > self.min_elevation,drop=True)
            if self.max_elevation:
                rtn = rtn.where(elev_interp < self.max_elevation,drop=True)
                
        return  rtn

    def crop_da(self,da=None,ignore_elevation=False, return_posneg=False):
        
        if return_posneg and not self.wrapped:
            return_posneg = False
            
        if da is None:
            interp_elev = False
            da = xr.ones_like(self.elevation)
            maxlon=self.maxlon
            minlon=self.minlon
            maxlat=self.maxlat
            minlat=self.minlat
        else:
            # check we have increasing co-ordinates
            if np.any(np.diff(da.latitude)<0) or np.any(np.diff(da.longitude)<0):
                raise ValueError("Latitude and longitide co-ordinates must be in ascending order")
            
            interp_elev = True
            maxlon = da.sel(longitude=self.maxlon,method='nearest').longitude
            minlon = da.sel(longitude=self.minlon,method='nearest').longitude
            if maxlon==minlon:
                maxlon = minlon + 0.5
                print('MINLON {0} AND MAXLON {1} TOO CLOSE TOGETHER - CHANGING MAXLON TO {2}'.format(self.minlon,self.maxlon,maxlon))
            maxlat = da.sel(latitude=self.maxlat,method='nearest').latitude
            minlat = da.sel(latitude=self.minlat,method='nearest').latitude
            if maxlat==minlat:
                maxlat = minlat + 0.25
                print('MINLAT {0} AND MAXLAT {1} TOO CLOSE TOGETHER - CHANGING MAXLAT TO {2}'.format(self.maxlat,self.maxlat,maxlat))

        """ elevation must have co-ords lat and lon which match da """
        if self.wrapped:
            pos = da.sel(longitude=slice(-180,maxlon)).sel(latitude=slice(minlat,maxlat))
            neg = da.sel(longitude=slice(minlon,180)).sel(latitude=slice(minlat,maxlat))
            rtn = xr.concat([pos,neg],dim='longitude')
        else:
            rtn = da.sel(longitude=slice(minlon,maxlon)).sel(latitude=slice(minlat,maxlat))
        
        if not ignore_elevation:
            elevation = self.elevation.interp(latitude=rtn.latitude,longitude=rtn.longitude) if interp_elev else self.elevation
            if self.min_elevation is not None:
                if return_posneg:
                    pos = pos.where(elevation > self.min_elevation,drop=True)
                    neg = neg.where(elevation > self.min_elevation,drop=True)
                else:  
                    rtn = rtn.where(elevation > self.min_elevation,drop=True)
            if self.max_elevation is not None:
                if return_posneg:
                    pos = pos.where(elevation < self.max_elevation,drop=True)
                    neg = neg.where(elevation < self.max_elevation,drop=True)
                else:
                    rtn = rtn.where(elevation < self.max_elevation,drop=True)
                
        if return_posneg:
            return pos, neg
        else:     
            return  rtn
    
    def crop_df(self,df):
        df = df[(df.lat > self.minlat) & (df.lat < self.maxlat)]
        if not self.wrapped:
            df = df[(df.lon > self.minlon) & (df.lon < self.maxlon)]
        else:
            df = df[(df.lon > self.minlon) | (df.lon < self.maxlon)]
            
        if self.min_elevation is not None:
            try:
                df = df[df.elevation > self.min_elevation]
            except:
                print("df has no column 'elevation'. trying to interp...")
                interp_elev = lambda lat,lon: self.elevation.sel(latitude=lat,longitude=lon,method='nearest').item()
                elevations = list(map(interp_elev,tqdm_notebook(df.lat),df.lon))
                #df = utls.add_elevation(df,self.elevation)
                df = df[np.array(elevations) > self.min_elevation]
            
        return df
        
    def plot_region(self,da=None,da_lbl=None,vmin=None,vmax=None,sea='Southern Ocean', extent=None,ax=None,cmap='Spectral_r',drop_cbar=False):
        """ da is dataset to plot region onto """
        if da is None:
            da = self.elevation
            da_lbl = None if drop_cbar else 'Elevation (m)'
            vmin = -5000
            vmax = 0
            cmap=cmocean.cm.deep_r
        
        title = "{0} for {1}".format(da_lbl,self.name)
        
        createfig = ax is None
        
        if ax == None:
            fig,ax = plt.subplots(figsize=(8,6),subplot_kw={'projection': ccrs.SouthPolarStereo()})

        if vmax is None:
            if not vmin is None:
                vmax = -vmin

        pfns = PlottingFns()
        im1 = pfns.sp(ax,da,vmin=vmin,vmax=vmax,title=title,cbar=da_lbl,cbar_orientation="vertical",cmap=cmap,sea=sea,bathy=self.elevation,extent=extent)
        if self.wrapped:
            cr_pos,cr_neg = self.crop_da(return_posneg=True)#xr.ones_like(self.elevation).where(
            im3 = cr_neg.plot(ax=ax,add_colorbar=False,cmap=cmocean.cm.matter,x='longitude',y='latitude',transform=ccrs.PlateCarree(),alpha=0.70)#,hatches='xx')
        else:
            cr_pos = self.crop_da()
        im2 = cr_pos.plot(ax=ax,add_colorbar=False,cmap=cmocean.cm.matter,x='longitude',y='latitude',transform=ccrs.PlateCarree(),alpha=0.70)#,hatches='xx')
        #im2 = ax.pcolormesh(cr.longitude,cr.latitude,cr.to_numpy(),alpha=0.5,cmap='Spectral',transform=ccrs.PlateCarree())
        #im2 = pfns.sp(ax,self.crop_da())
#         ax.contourf(
#             cr.longitude.to_numpy(), cr.latitude.to_numpy(), cr,
#             transform=ccrs.PlateCarree(),
#             colors='none',
#             levels=[.5, 1.5],
#             hatches='///////',
#         )
#         minlon = self.maxlon if self.wrapped else self.minlon
#         width = self.maxlon - self.minlon
#         width = 360 - width if self.wrapped else width
#         ax.add_patch(mp.Rectangle(xy=[minlon, self.minlat], width=width, height=self.maxlat-self.minlat,
#                                         facecolor='yellow',
#                                         edgecolor='yellow',
#                                         linewidth=6,
#                                         alpha=0.6,
#                                         transform=ccrs.PlateCarree())
#                      )
        return ax

    def plot_cmtlgy(self,da,ylabel='Mean SHA',meandims=('latitude','longitude')):
        fig,ax = plt.subplots(figsize=(14,4))
        xlbl = ['J','F','M','A','M','J','J','A','S','O','N','D']
        
        da = self.crop_da(da)
        cmtlgy = da.groupby('time.month').mean('time',skipna=True)

        def plot_cmtlgys(da,ax):
            cmt = cmtlgy.mean(meandims)
            meansh = da.mean(meandims)

            for grp in meansh.groupby('time.year'):
                data = grp[1].groupby('time.month').mean()
                ax.plot(data.month,data,'x--',label=grp[0])
            ax.plot(cmt.month,cmt,c='k',lw=2,label='mean')
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.15),ncol=9)
            ax.set_ylim([-6,7])
            ax.set_xticks(np.arange(1,13))
            ax.set_xticklabels(xlbl)

        plot_cmtlgys(da,ax)
        ax.grid()
        ax.set_ylabel(ylabel)

        return fig, ax

    def timeseries_compare(self,ds_in,annual=True,crop_times=False,crop_profile_times=True,mask_to_profiles=False,sha_lbl='sha',fig=None, ax=None):

        ds = self.crop_da(ds_in)

        if crop_times:
            if not crop_profile_times:
                ds = ds.sel(time=slice('2008-04-01','2022-03-31'))
            else:
                ds = ds.sel(time=slice('2008-01-01','2017-12-31'))
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12,4))

        # restrict altimetry to only where there are profiles
        if mask_to_profiles:
            data_masked = ds.where(~np.isnan(ds.profile_gpha))
        else:
            data_masked = ds.where(~np.isnan(ds.profile_gpha) & ~np.isnan(ds.sha))
                
        # take annual mean
        data_spatial = data_masked.groupby('time.year').mean('time',skipna=True) if annual else data_masked

        # spatial average
        data = data_spatial.mean(['latitude','longitude'])
        
        plot_time = data.year if annual else data.time
        
        # plot sh and gph on different axes
        def plot(ax,var,c,title,marker='x-'):
            ax.plot(plot_time,data[var].to_numpy(),marker,c=c,label=title)
            
        # plot bar to show number of profiles
        if annual:
            plot(ax,sha_lbl,uc.SHA.value[0],'SHA')
            plot(ax,'profile_gpha',uc.GPHA.value[0],'GPH')
            ax.grid()
            ax.legend()
            ax.set_ylabel('Annual height anomaly (cm)')
        else:
            plot(ax,sha_lbl,uc.SHA.value[1],'SHA Monthly',marker='x')
            plot(ax,'profile_gpha',uc.GPHA.value[1],'GPHA Monthly',marker='x')
            data.sha.rolling(time=12,min_periods=2,center=True).mean().plot(ax=ax,label="SHA 12-Month Average",c=uc.SHA.value[0])
            data.profile_gpha.rolling(time=12,min_periods=2,center=True).mean().plot(ax=ax,label="GPHA 12-Month Average",c=uc.GPHA.value[0])
            ax.legend(loc = 'upper left',fontsize='small')
            ax.grid()
            ax.set_ylabel('Monthly height anomaly (cm)')
            ax.set_xlabel(None)
        
        return fig, ax

    
    def timeseries_bar(self,ds_in,annual=True,crop_profile_times=True,mask_to_profiles=False):
    
        ds = self.crop_da(ds_in)

        if not crop_profile_times:
            ds = ds.sel(time=slice('2008-04-01','2022-03-31'))
        else:
            ds = ds.sel(time=slice('2008-01-01','2017-12-31'))
        
        fig, axs = plt.subplots(2,1,figsize=(12,6))

        # restrict altimetry to only where there are profiles
        if mask_to_profiles:
            data_masked = ds.where(~np.isnan(ds.profile_gpha))
        else:
            data_masked = ds.where(~np.isnan(ds.profile_gpha) & ~np.isnan(ds.sha))
                
        # take annual mean
        data_spatial = data_masked.groupby('time.year').mean('time',skipna=True) if annual else data_masked

        # spatial average
        data = data_spatial.mean(['latitude','longitude'])
        
        # sum counts for bar plot
        counts_annual = data_spatial.profile_cnt.sum(['latitude','longitude']).to_numpy()
        
        plot_time = data.year if annual else data.time
        
        if not annual:
            counts_monthly = data_masked.profile_cnt.sum(['latitude','longitude']).to_numpy()
            counts_quarterly = sum(counts_monthly.reshape([3,int(len(counts_monthly)/3)]))
            time_quarterly = data.time.to_numpy().reshape([int(len(counts_monthly)/3),3])[:,1]
        
        # plot sh and gph on different axes
        def plot(ax,var,c,title,marker='x-'):
            ax.plot(plot_time,data[var].to_numpy(),marker,c=c,label=title)
            
        
        # plot bar to show number of profiles
        if annual:
            plot(axs[0],'sha',uc.SHA.value[0],'SHA')
            plot(axs[0],'profile_gpha',uc.GPHA.value[0],'GPH')
            axs[0].grid()
            axs[0].legend()
            axs[0].set_ylabel('Annual height anomaly (cm)')
            axs[1].bar(plot_time,counts_annual,color=uc.GPHA.value[0])
            axs[1].set_ylabel('Number of profiles')
            axs[1].grid()
            axs[1].set_xlim(axs[0].get_xlim())
        else:
            plot(axs[0],'sha',uc.SHA.value[1],'SHA',marker='x')
            plot(axs[0],'profile_gpha',uc.GPHA.value[1],'GPHA',marker='x')
            axs[0].legend()
            data.sha.rolling(time=12,min_periods=2,center=True).mean().plot(ax=axs[0],label="SHA 12-Mo. Moving Av.",c=uc.SHA.value[0])
            data.profile_gpha.rolling(time=12,min_periods=2,center=True).mean().plot(ax=axs[0],label="GPHA 12-Mo. Moving Av.",c=uc.GPHA.value[0])
            axs[0].grid()
            axs[0].set_ylabel('Monthly height anomaly (cm)')
            
            axs[1].bar(np.arange(len(counts_quarterly)),counts_quarterly,color=uc.GPHA.value[0])
            axs[1].set_ylabel('Number of profiles')
            axs[1].grid()
            axs[1].set_xticks(np.arange(len(counts_quarterly))[::4])
            axs[1].set_xticklabels(map(lambda t: pd.to_datetime(t).strftime('%m-%Y'),time_quarterly[::4]))
        
        return fig, axs

    def plot_profiles_for_region(self, gebco_coarse, profiles_meop, profiles_argo, sta, fin, markersize=3, markerstyle='.', figsize=(14,6), useall=False, fig=None, ax=None, sps=False):
        
        if sps:
            projection = ccrs.SouthPolarStereo()
        else:
            projection = ccrs.Mercator()
        #print('Cropping gebco..')
        gebco_region = self.crop_da(gebco_coarse,ignore_elevation=True)

        #print('Cropping profile dataframes..')
        if not useall:
            # crop to region within satellite data time limits
            meop_crop = self.crop_df(profiles_meop[(profiles_meop.index > pd.to_datetime(sta.item())) & (profiles_meop.index < pd.to_datetime(fin.item()))])
            argo_crop = self.crop_df(profiles_argo[(profiles_argo.index > pd.to_datetime(sta.item())) & (profiles_argo.index < pd.to_datetime(fin.item()))])
        else:
            meop_crop = self.crop_df(profiles_meop)
            argo_crop = self.crop_df(profiles_argo)
        
        # print('Found MEOP profiles with float IDS: ')
        # print(meop_crop.float_id.unique())
        # print('Found ARGO profiles with float IDS: ')
        # print(argo_crop.float_id.unique())
        
        #print('Plotting bathymetry..')
        if ax is None:
            fig,ax = plt.subplots(figsize=figsize,subplot_kw={'projection': projection})
        PlottingFns().sp(ax,gebco_region.elevation,vmin=-5000,vmax=0,cmap=cmocean.cm.deep_r,cbar="Elevation (m)",cbar_orientation='horizontal',extent=self.extent)
        
        #print('Plotting profile locations.. (markerstyle={})'.format(markerstyle))
        ax.plot(meop_crop.lon,meop_crop.lat,markerstyle,ms=markersize,transform=ccrs.PlateCarree(),color=uc.MEOP.value[0],label='MEOP')
        ax.plot(argo_crop.lon,argo_crop.lat,markerstyle,ms=markersize,transform=ccrs.PlateCarree(),color=uc.ARGO.value[0],label='ARGO')
        ax.legend(loc='upper left')
        
        #print('Manifesting')
        return fig,ax