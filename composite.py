import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmocean
import xesmf as xe
import matplotlib.patches as mpatches
import seaborn as sns
from stericheight.plotting_fns import PlottingFns


class Composite():
    def __init__(self,idx_data,idx_name,std_lim=0.5):
        self.limit = idx_data.std() * std_lim
        self.limit_label = str(std_lim) + r'$\sigma$'

        self.idx_positive = idx_data >= self.limit
        self.idx_negative = idx_data <= -self.limit
        self.idx_name=idx_name
        self.idx_data=idx_data

        self.has_meop=False

    def add_meop(self,meop,tlims=None,slims=None):
        allpres = np.concatenate([p[0].pressure for p in meop.df.temperature])
        values, counts = np.unique(allpres[~np.isnan(allpres)], return_counts=True)
        pressure_df = pd.DataFrame(counts,values)
        self.pressure_axis= pressure_df[(pressure_df>1000).to_numpy().transpose().flatten()].index.to_numpy()
        self.t_pos, self.s_pos, self.df_pos = self.__get_posneg_ts(meop,self.idx_positive)
        self.t_neg, self.s_neg, self.df_neg = self.__get_posneg_ts(meop,self.idx_negative)

        self.df_both = pd.concat([self.df_pos,self.df_neg])

        lims = lambda pos,neg: [min(pos.min().item(),neg.min().item()),max(pos.max().item(),neg.max().item())]
        self.tlims = tlims or lims(self.t_pos,self.t_neg)
        self.slims = slims or lims(self.s_pos,self.s_neg)

        self.has_meop=True


    def composite_timeseries(self,ax,background_data=None,bg_name='',ylim=[-12,12]):
        '''
        da1 = index timeseries which we are using to determine 
        '''
        if background_data is not None:
            ax.plot(background_data.time,background_data,color='#96ae8d',label=bg_name)

        ax.plot(self.idx_data.time,self.idx_data,color='olive',label=self.idx_name)
        ax.plot(self.idx_data.time,np.zeros_like(self.idx_data),color='#373e02')
        ax.fill_between(self.idx_data.time, 0,ylim[1], where=self.idx_positive, alpha=0.4, facecolor='darkkhaki',label='+/- '+ self.limit_label)
        ax.fill_between(self.idx_data.time,ylim[0],0, where=self.idx_negative, alpha=0.4, facecolor='darkkhaki')
        ax.set_ylim(ylim)
        ax.set_ylabel(self.idx_name)
        ax.legend(loc='best')
        ax.set_title('(a)',loc='left')
        ax.grid()
        if self.has_meop:
            ax.plot(self.t_pos.time,[0 for i in self.t_pos],'x')
            ax.plot(self.t_neg.time,[0 for i in self.t_neg],'x')


    def posneg_map(self,axs,data,name: str,vmax,extent,add_profiles=False,label_loc=None):
        pfns = PlottingFns()
        # extract data where positive and negative
        data_pos = data.where(self.idx_positive,drop=True).mean('time')
        data_neg = data.where(self.idx_negative,drop=True).mean('time')
        # plot pos and neg maps
        im1=pfns.sp(axs[0],data_pos,extent=extent,cmap=cmocean.cm.balance,vmax=vmax,vmin=-vmax,land_zorder=3)#title='POSITIVE ' + idx_name + ', (> ' + fill_text + ')')
        im2=pfns.sp(axs[1],data_neg,extent=extent,cmap=cmocean.cm.balance,vmax=vmax,vmin=-vmax,land_zorder=3)#title='NEGATIVE ' + idx_name + ', (< '+ lower_fill_text + ')'
        if not label_loc==None:
            axs[0].text(label_loc[0],label_loc[1],'+'+name,transform=ccrs.PlateCarree(),size=15,c='#373e02',ha='center',bbox=dict(boxstyle="square",facecolor='darkkhaki',alpha=0.5))
            axs[1].text(label_loc[0],label_loc[1],'-'+name,transform=ccrs.PlateCarree(),size=15,c='#373e02',ha='center',bbox=dict(boxstyle="square",facecolor='darkkhaki',alpha=0.5))
        if add_profiles and self.has_meop:
            vmin=self.df_both.index.min().year
            vmax=self.df_both.index.max().year
            im3=axs[0].scatter(x=self.df_pos.longitude,y=self.df_pos.latitude,c=self.df_pos.index.year,vmin=vmin,vmax=vmax,transform=ccrs.PlateCarree())
            im4=axs[1].scatter(x=self.df_neg.longitude,y=self.df_neg.latitude,c=self.df_neg.index.year,vmin=vmin,vmax=vmax,transform=ccrs.PlateCarree())
            plt.colorbar(im1)
            plt.colorbar(im4)
        else:
            plt.colorbar(im1)
            plt.colorbar(im2)
    # filter profiles
    def __get_posneg_ts(self,meop,idx_in):
        # function to take a profile and turn it into a ds with uniform z-axis and tagged with the longitude
        def reindex_profile(profile,time):
            return profile \
                    .rename({'N_LEVELS':'pressure'}) \
                    .dropna(dim='pressure',how='all') \
                    .interp(coords={'pressure':self.pressure_axis}) \
                    .assign_coords({'time':time}) \
                    .expand_dims('time')
        df_months = meop.df.index.to_period('M')
        months = pd.PeriodIndex(idx_in.time.where(idx_in,drop=True).values.astype('datetime64[M]'),freq='M')
        filtered_df = meop.df[df_months.isin(months)].dropna()

        #remove duplicates
        #not very nice.. think about this more later
        filtered_df = filtered_df[~filtered_df.index.duplicated(keep='first')]

        temps_list = [reindex_profile(p[0],l) for p,l in zip(filtered_df.temperature, filtered_df.index)]
        psal_list = [reindex_profile(p[0],l) for p,l in zip(filtered_df.salinity, filtered_df.index)]

        #try:
        filtered_temp = xr.merge(temps_list).TEMP_ADJUSTED
        filtered_psal = xr.merge(psal_list).PSAL_ADJUSTED
        # except:
        #     filtered_temp = temps_list
        #     filtered_psal = psal_list
        return filtered_temp,filtered_psal, filtered_df
    
    def ts_density(self,axs):
        def ts_axes(ax,im,n):
            ax.set_ylim(self.tlims)
            ax.set_xlim(self.slims)
            ax.set_xlabel('Salinity')
            ax.set_ylabel('Temperature')
            ax.set_title('total no. profiles: ' +str(n))

        ts_axes(axs[0],sns.kdeplot(ax=axs[0],x=self.s_pos.values.flatten(), y=self.t_pos.values.flatten(),fill=True, cbar=True),len(self.s_pos))
        ts_axes(axs[1],sns.kdeplot(ax=axs[1],x=self.s_neg.values.flatten(), y=self.t_neg.values.flatten(),fill=True, cbar=True),len(self.s_neg))

    def ts_profiles(self,axs):
        def plot_prof(ax,prof,xlabel,color,mean_color,lims):
            for lon in prof.time:
                ax.plot(prof.sel(time=lon),prof.pressure,color=color)
            ax.plot(prof.mean('time'),prof.pressure,color=mean_color)
            ax.invert_yaxis()
            ax.set_ylabel('Pressure')
            ax.set_xlabel(xlabel)
            ax.set_xlim(lims)

        plot_prof(axs[0],self.t_pos,'Temperature','thistle','mediumvioletred',self.tlims)
        plot_prof(axs[1],self.s_pos,'Salinity','lightblue','steelblue',self.slims)
        plot_prof(axs[2],self.t_neg,'Temperature','thistle','mediumvioletred',self.tlims)
        plot_prof(axs[3],self.s_neg,'Salinity','lightblue','steelblue',self.slims)

    def ts_freq_plot(self,axs):
        def single(ax0,ax1,da_time):
            monthly = da_time.groupby('time.month').count()
            yearly = da_time.groupby('time.year').count()
            ax0.bar(monthly.month,monthly)
            ax0.set_xticks(np.arange(1,13))
            ax0.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
            ax0.set_ylabel('Total number of profiles')

            ax1.bar(yearly.year,yearly)
            ax1.set_xlim([2002,2022])
            ax1.set_ylabel('Total number of profiles')

        single(axs[0],axs[1],self.t_pos.time)
        single(axs[2],axs[3],self.t_neg.time)
            