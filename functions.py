import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmocean
from cartopy.feature import LAND
from gsw import geostrophy, density,conversions
from stericheight.data_handler import GEBCO

class Functions():
    def __init__(self,elevation_cf=40):
        self.elevation_ = GEBCO(coarsen_factor=elevation_cf).ds.elevation

    #define helper functions
    def get_trend(self,da: xr.DataArray,ns=True):
        ns_2_yr = lambda x: x * 1e9 * 60 * 60 * 24 * 365.25
        p = da.polyfit(dim='time',deg=1)
        m = p.polyfit_coefficients.sel(degree=1)
        if 'longitude' in da.dims:
            m = m.assign_coords(longitude=da.longitude)
        return ns_2_yr(m) if ns else m

    def crop_to_extent(self,ds, extent):
        if 'lat' in ds.coords:
            lat = 'lat'
            lon = 'lon'
        else:
            lat = 'latitude'
            lon = 'longitude'
        return ds.where((ds[lat] >= extent[2]) & 
                        (ds[lat] <= extent[3]) &
                        (ds[lon] >= extent[0]) &
                        (ds[lon] <= extent[1]), drop=True)

    def season_split(self,da,tlabel='time'):
        # divide by season
        months = da[tlabel].dt.month
        seasons = dict()
        seasons['winter'] = da.where((months >=7) & (months <=9))
        seasons['spring'] = da.where((months >=10) & (months <=12))
        seasons['summer'] = da.where((months >=1) & (months <=3))
        seasons['autumn'] = da.where((months >=4) & (months <=6))
        return seasons

    def seasonal_anomaly(self,da,tlabel='time'):
        return da.groupby(tlabel+'.month') - da.groupby(tlabel+'.month').mean()

    def get_geostrophic_currents(self,eta,lat_name='latitude',lon_name='longitude'):

        g = 9.81
        R = 6371000.0
        omega = 7.292115e-5

        # Ensure coords are present and in degrees
        lat = eta.coords[lat_name]
        lon = eta.coords[lon_name]

        # radians
        lat_r = np.deg2rad(lat)
        lon_r = np.deg2rad(lon)

        # Build helper to take gradients along given axes using the 1D radian coords
        # Works for data with optional leading dims (e.g., time)
        def d_dcoord(da: xr.DataArray, coord: xr.DataArray, axis: int) -> xr.DataArray:
            arr = da.data
            coord_vals = coord.data
            grad = np.gradient(arr, coord_vals, axis=axis, edge_order=2)
            return xr.zeros_like(da) + grad  # keep xarray wrapper/dims

        # Axis indices for lat/lon (support datasets with extra leading dims)
        lat_axis = eta.get_axis_num(lat_name)
        lon_axis = eta.get_axis_num(lon_name)

        # Partial derivatives wrt latitude/longitude in *radians*
        deta_dphi   = d_dcoord(eta, lat_r, lat_axis)         # ∂η/∂φ
        deta_dlambda= d_dcoord(eta, lon_r, lon_axis)         # ∂η/∂λ

        # Coriolis parameter f(φ)
        f = np.sin(lat_r) * omega * 2

        # Broadcast f and cosφ to data shape
        # (xarray auto-broadcasts by coords when wrapped in DataArray with dims)
        f_da = xr.DataArray(f, dims=(lat_name,), coords={lat_name: lat})
        cosphi = xr.DataArray(np.cos(lat_r), dims=(lat_name,), coords={lat_name: lat})

        # Geostrophic components
        ug = (1.0 / R) * deta_dphi * -(g / f_da) 
        vg = (1.0 / (R * cosphi)) * deta_dlambda * (g / f_da)

        ug.name = "ug"; ug.attrs.update(units="m s-1", long_name="zonal geostrophic velocity")
        vg.name = "vg"; vg.attrs.update(units="m s-1", long_name="meridional geostrophic velocity")

        return ug, vg

    def plotc(self,ax,da,title,extent,vmin=None,vmax=None,cmap=cmocean.cm.balance,vector=None,rs=25,lat_name='latitude',lon_name='longitude',ac=True):
        im=da.plot.contourf(x=lon_name,y=lat_name,ax=ax,levels=40,transform=ccrs.PlateCarree(),vmin=vmin,vmax=vmax,add_colorbar=ac,cmap=cmap)
        if vector is not None:
            vector.plot.quiver(ax=ax,x=lon_name,y=lat_name,u='u10', v='v10',transform=ccrs.PlateCarree(),regrid_shape=rs)
        
        self.elevation_.plot.contour(x='longitude',y='latitude',ax=ax,levels=[-1000,-4000],transform=ccrs.PlateCarree(),cmap="copper_r",vmin=-10000,vmax=0,linewidths=1,linestyles='--')

        ax.set_title(title)
        ax.set_extent(extent,crs=ccrs.PlateCarree())
        ax.add_feature(LAND, edgecolor='k')
        gl = ax.gridlines(draw_labels=True)

        gl.top_labels = False
        gl.right_labels = False
        gl.bottom_labels = True
        gl.left_labels = True
        return im


    # find top & bottom 10% of sla and sic
    def get_topbot(self,idx,frac=0.1):
        idx_bq = idx.quantile(frac,dim='time')
        idx_tq = idx.quantile(1-frac,dim='time')
        botboo = idx <= idx_bq
        topboo = idx >= idx_tq
        topfun = lambda da, s=0: da.where(topboo.shift({'time':s},0),drop=True).mean('time')
        botfun = lambda da, s=0: da.where(botboo.shift({'time':s},0),drop=True).mean('time')
        return topboo, botboo, topfun, botfun
    
    def idx_ts(self,idx,frac,lag=0,ylim_idx=12):  
        perc=int(frac*100)
        topboo, botboo, top, bot = self.get_topbot(idx,frac)

        fig,ax = plt.subplots(figsize=(14,3))

        ax.plot(idx.time,idx,color='olive',label='SLA')
        ax.plot(idx.time,np.zeros_like(idx),color='#373e02')

        # highlight regions of positive and negative index
        # need to extend an extra month to capture full month
        def plus1(arr):
            arrin = arr.copy()
            shifted_arr = arr.shift({'time':1},0)
            return arrin | shifted_arr

        # highlight regions of positive and negative index
        ax.fill_between(idx.time, 0,ylim_idx, where=plus1(topboo), alpha=0.4, facecolor='darkkhaki',label='TOP/BOTTOM {}%'.format(perc))
        ax.fill_between(idx.time,-ylim_idx,0, where=plus1(botboo), alpha=0.4, facecolor='darkkhaki')

        # highlight lag regions
        if lag > 0:
            ax.fill_between(idx.time, 0,ylim_idx, where=plus1(topboo).shift({'time':lag}), alpha=0.4, facecolor='coral',label='+ {}-month lag'.format(lag))
            ax.fill_between(idx.time,-ylim_idx,0, where=plus1(botboo).shift({'time':lag}), alpha=0.4, facecolor='coral')
        
        ax.set_ylim([-ylim_idx,ylim_idx])
        ax.set_ylabel('gyre height')
        ax.legend(loc='best')
        ax.set_title('(a)',loc='left')
        ax.grid()


    def run_for_idx(self,idx,frac,da,da_vector,da_title,extent,lag=0,ylim_idx=12,da2=None,da_vector2=None,da_title2=None,ylim2=1,ylim1=1,lat_name='latitude',lon_name='longitude'):  
        perc=int(frac*100)
        topboo, botboo, top, bot = self.get_topbot(idx,frac)

        if da2 is not None:
            fs = 11
            gs = 7
        else:
            fs = 7
            gs = 4

        fig = plt.figure(figsize=(12,fs))
        gs = fig.add_gridspec(gs,2)

        ax = fig.add_subplot(gs[0, :])

        ax.plot(idx.time,idx,color='olive',label='SLA')
        ax.plot(idx.time,np.zeros_like(idx),color='#373e02')

        # highlight regions of positive and negative index
        # need to extend an extra month to capture full month
        def plus1(arr):
            arrin = arr.copy()
            shifted_arr = arr.shift({'time':1},0)
            return arrin | shifted_arr

        # highlight regions of positive and negative index
        ax.fill_between(idx.time, 0,ylim_idx, where=plus1(topboo), alpha=0.4, facecolor='darkkhaki',label='TOP/BOTTOM {}%'.format(perc))
        ax.fill_between(idx.time,-ylim_idx,0, where=plus1(botboo), alpha=0.4, facecolor='darkkhaki')

        # highlight lag regions
        if lag > 0:
            ax.fill_between(idx.time, 0,ylim_idx, where=plus1(topboo).shift({'time':lag}), alpha=0.4, facecolor='coral',label='+ {}-month lag'.format(lag))
            ax.fill_between(idx.time,-ylim_idx,0, where=plus1(botboo).shift({'time':lag}), alpha=0.4, facecolor='coral')

        ax.set_ylim([-ylim_idx,ylim_idx])
        ax.set_ylabel('Gyre Index')
        ax.legend(loc='best')
        ax.set_title('(a)',loc='left')
        ax.grid()

        ax = fig.add_subplot(gs[1:4,0],projection=ccrs.TransverseMercator(central_longitude=145))
        self.plotc(ax,top(da),'{} top {}%'.format(da_title,perc),extent,-ylim1,ylim1,vector=top(da_vector),lat_name=lat_name,lon_name=lon_name,ac=False)
        ax = fig.add_subplot(gs[1:4,1],projection=ccrs.TransverseMercator(central_longitude=145))
        self.plotc(ax,bot(da),'{} bottom {}%'.format(da_title,perc),extent,-ylim1,ylim1,vector=bot(da_vector),lat_name=lat_name,lon_name=lon_name,ac=False)

        if da2 is not None:
            ax = fig.add_subplot(gs[4:7,0],projection=ccrs.TransverseMercator(central_longitude=145))
            self.plotc(ax,top(da2),'{} top {}%'.format(da_title2,perc),extent,-ylim2,ylim2,vector=top(da_vector2),ac=False)
            ax = fig.add_subplot(gs[4:7,1],projection=ccrs.TransverseMercator(central_longitude=145))
            self.plotc(ax,bot(da2),'{} bottom {}%'.format(da_title2,perc),extent,-ylim2,ylim2,vector=bot(da_vector2),ac=False)

        plt.tight_layout()

    def get_asc(da,cropped_elevation,uname='u10',vname='v10',latmin=-66.4,latmax=-65):
        # get -1000 isobath usxing contour
        cc=cropped_elevation.sel(latitude=slice(latmin,latmax)).plot.contour(levels=[-1000])
        lats = xr.DataArray(cc.allsegs[0][0].T[1],coords={'longitude':cc.allsegs[0][0].T[0]})

        #interpolate to data
        asclats = lats.sortby('longitude').interp(longitude=da.longitude,kwargs={'fill_value':'extrapolate'})

        # isolate asc on slope = -1000
        da['asc_latitude'] = xr.DataArray(asclats,coords={'longitude':da.longitude})
        da['asc'] = da[uname].sel(latitude=da.asc_latitude,method='nearest')

        # convert to m
        R = 6371000
        lat_radians = np.deg2rad(da.asc_latitude)

        dx = R * np.cos(lat_radians[:-1]) * np.deg2rad(np.diff(da.longitude))
        dy = R * np.deg2rad(np.diff(da.asc_latitude))

        delta = np.stack([dx,dy],axis=1)
        t_hat = delta / np.linalg.norm(delta, axis=1)[:, None]

    def sanom(self, da):
        return da.groupby('time.month') - da.groupby('time.month').mean()
    
    def sig0range(self,tlims,slims):
        sig0_n = 100
        temprange = np.linspace(tlims[0],tlims[1],sig0_n)
        psalrange = np.linspace(slims[0],slims[1],sig0_n)

        sig0range = np.zeros((sig0_n,sig0_n))
        for i,t in enumerate(temprange):
            for j,s in enumerate(psalrange):
                sig0range[i,j]=density.sigma0(conversions.SA_from_SP(s,0,145,-67),conversions.CT_from_t(s,t,0))
        return temprange,psalrange,sig0range

    def get_seasons(self,ds):
        seasons={}
        seasons['autumn']=ds.where(((ds.month >= 4) & (ds.month <= 6)),drop=True)
        seasons['winter']=ds.where(((ds.month >= 7) & (ds.month <= 9)),drop=True)
        seasons['spring']=ds.where(((ds.month >= 10) & (ds.month <= 12)),drop=True)
        seasons['summer']=ds.where(((ds.month >= 1) & (ds.month <= 3)),drop=True)
        return seasons
    
    def integrate_na(self,da: xr.DataArray):
        da = da.where(~np.isnan(da),drop=True)
        return da.integrate('PRES_Pa') * 100