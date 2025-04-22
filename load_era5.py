import cdsapi
import zipfile
import os
import xarray as xr

FNAME = 'ERA5/'
DATASET = "reanalysis-era5-single-levels-monthly-means"

class ERA5():
    def __init__(self):
        self.client = client = cdsapi.Client()

    def download(self,years,
                 months=["01", "02", "03","04", "05", "06","07", "08", "09","10", "11", "12"],
                 area=[-50,-180,-90,-180]):
        """
        years: ["2012","2013"]
        """
        subfolder = '-'.join(years) + '_' + '-'.join(months) + '_' + '-'.join(str(a) for a in area)
                                                                          
        # submit request
        request = {
            "product_type": ["monthly_averaged_reanalysis"],
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "mean_sea_level_pressure"
            ],
            "year": years,
            "month": months,
            "time": ["00:00"],
            "data_format": "netdcf",
            "download_format": "unarchived",
            "area": area
        }

        client = cdsapi.Client()
        self.fname = client.retrieve(DATASET, request).download()
        self.ds = xr.open_dataset(fname)
