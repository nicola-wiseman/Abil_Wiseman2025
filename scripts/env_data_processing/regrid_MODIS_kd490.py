#%%

import numpy as np
import xarray as xr
import xesmf as xe

# Define the file list
files = [
    '20030101_20250131',
    '20030201_20250228',
    '20030301_20250331',
    '20030401_20250430',
    '20030501_20250531',
    '20030601_20250630',
    '20020701_20250731',
    '20020801_20250831',
    '20020901_20240930',
    '20021001_20241031',
    '20021101_20241130',
    '20021201_20241231'
]

base_path = '/home/mv23682/Documents/Abil_Wiseman2025/scripts/env_data_processing/raw_data/MODIS_Kd490/'

# Create a target grid for regridding
ds_out = xr.Dataset({
    'lat': (['lat'], np.arange(-90, 90, 1)),
    'lon': (['lon'], np.arange(-180, 180, 1))
})

# Create a filename for storing the weights
weights_file = "regridded_data/regrid_weights.nc"

print('Initializing regridder')
# Open the first dataset to create a persistent regridder (only calculated once)
initial_ds = xr.open_dataset(f"{base_path}AQUA_MODIS.{files[0]}.L3m.MC.KD.Kd_490.9km.nc")
regridder = xe.Regridder(initial_ds, ds_out, method="bilinear", periodic=True, filename=weights_file)

ds_all = []

# Loop over each file
for i, file in enumerate(files):
    # Open the dataset
    ds = xr.open_dataset(f"{base_path}AQUA_MODIS.{file}.L3m.MC.KD.Kd_490.9km.nc")
    print(ds)
    # Regrid Kd_490 to the new lat/lon grid (with persistent regridder, reuse weights)
    regridder = xe.Regridder(ds, ds_out, method="bilinear", periodic=True, reuse_weights=True, filename=weights_file)
    dr_out = regridder(ds['Kd_490'])
    
    # Add time and depth dimensions in one step
    dr_out = dr_out.assign_coords(time=i+1).expand_dims(time=[i+1])
    
    # Append the dataset to the list
    ds_all.append(dr_out)

# Concatenate all datasets along the time dimension
ds = xr.concat(ds_all, dim="time")

# Additionally, fill NaN regions (high latitudes during winter) with min or 10% of maximum, whichever is lower
# If you want to skip this, skip lines 64-89 and swtich "filled_ds" to "ds" for lines 90-96
# Calculate the max and min along the time dimension
max_values = ds.max(dim='time', skipna=True)
print(max_values)
min_values = ds.min(dim='time', skipna=True)
print(min_values)

# Assign a name to the DataArray
ds.name = 'Kd_490'

# Save the filled dataset to NetCDF
ds.to_netcdf("/home/mv23682/Documents/Abil_Wiseman2025/scripts/env_data_processing/regridded_data/Kd_490.nc")

print("Finished processing and saved to Kd_490.nc")
# %%
