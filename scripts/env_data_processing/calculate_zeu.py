#%%

import numpy as np
import xarray as xr

# Zeu = ln(0.01)/Kd(490) (Hopkins & Balch, 2018)

ds = xr.open_dataset("/home/mv23682/Documents/Abil_Wiseman2025/scripts/env_data_processing/regridded_data/Kd_490.nc")

Zeu= -np.log(0.01)/ds['Kd_490']

Zeu.name = 'Zeu'
Zeu.attrs['long_name'] = 'Euphotic zone depth'
Zeu.attrs['units'] = 'm'

# Convert to Dataset for saving
Zeu_ds = Zeu.to_dataset()

# Save to NetCDF
Zeu_ds.to_netcdf("/home/mv23682/Documents/Abil_Wiseman2025/scripts/env_data_processing/regridded_data/Zeu.nc")

# %%
