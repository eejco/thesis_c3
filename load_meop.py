import pandas as pd

ROOT = '/nfs/b0133/eejco/data/MEOP_2024/'
FNAME_PROFILES = ROOT + 'list_profiles.csv'
FNAME_TAGS = ROOT + 'list_tags.csv'
FNAME_DEPLOYMENTS = ROOT + 'list_deployments.csv'
FNAME_DATA = ROOT + 'media/disk2/roquet/MEOP_public/MEOP-CTD_2024-03-08/'

class MEOP():

    def get_profiles(extent,start,end):

        minlon = extent[0]
        maxlon = extent[1]
        minlat = extent[2]
        maxlat = extent[3]

        # read deployments and trim to area we care about
        deployments = pd.read_csv(FNAME_DEPLOYMENTS)
        deployments = deployments[deployments.LATITUDE < -30]

        # read profiles and trim to latitudes we care about
        profiles_ = pd.read_csv(FNAME_PROFILES)
        profiles_ = profiles_[profiles_.LATITUDE<-50]
        profiles_ = profiles_[(~np.isnan(profiles_.N_TEMP))&(~np.isnan(profiles_.N_PSAL))].reset_index(drop=True)

        # define methods to retrieve info one each profile
        deployment = lambda platform: str.split(platform,'-')[0]
        country = lambda platform: str(deployments[deployments.DEPLOYMENT_CODE == deployment(platform)].COUNTRY.item())
        folder = lambda platform: data_dir + country(platform) + '/'
        filename = lambda platform: platform + '_all_prof.nc'

        profile_codes=profiles_.SMRU_PLATFORM_CODE.unique()

        flatten = lambda l: [item for sublist in l for item in sublist]  