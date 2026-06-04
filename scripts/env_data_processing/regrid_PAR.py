# %%
from operator import index
import numpy as np
import xarray as xr
import xesmf as xe

path = "/home/mv23682/Documents/Abil_Wiseman2025/scripts/env_data_processing/raw_data/RS_PAR/RS_PAR_ESM-based_fill_monthly_clim_1998-2022_0-200.nc"
name = "PAR"

#load netcdf as xarray
ds1 = xr.open_dataset(path)
ds1 = ds1.drop_vars(['DOI'])
ds1 = ds1.rename(name_dict = {'latitude':'lat', 'longitude':'lon', 'Time':'time', 'Depth':'depth'})

# setup new grid to regrid to:
ds_out = xr.Dataset({'lat': (['lat'], np.arange(-90, 90, 1)),
                    'lon': (['lon'], np.arange(-180, 180, 1)),
                    'depth': (['depth'], np.arange(0, 200, 5)),
                    'time': (['time'], np.arange(1, 13, 1)),
                    })

regridder1 = xe.Regridder(ds1, ds_out, 'conservative', periodic=True)
dr1_out = regridder1(ds1['PAR'],skipna=True, na_thres=0.75)

df1 = dr1_out.to_dataframe(name=name).reset_index()
df1.set_index(['time', 'depth', 'lat', 'lon'], inplace=True)
print(df1)
ds = df1.to_xarray()
print('saving ' + name)
out_dir = "/home/mv23682/Documents/Abil_Wiseman2025/scripts/env_data_processing/regridded_data/"
ds.to_netcdf(out_dir + name + ".nc")

# ── PAR masks ────────────────────────────────────────────────────────────────

# Surface PAR = shallowest depth level (depth=0)
par = ds[name]                                   # dims: (time, depth, lat, lon)
surface_par = par.sel(depth=0)                   # dims: (time, lat, lon)

# Percentage of surface PAR at every depth level
# Broadcast surface_par back across the depth dimension for element-wise division
pct_surface_par = (par / surface_par) * 100      # dims: (time, depth, lat, lon)

# Create boolean masks (True where PAR is BELOW the threshold)
thresholds = {
    "PAR_1prct_mask":    1.0,    # below 1%
    "PAR_0p1prct_mask":  0.1,    # below 0.1%
    "PAR_0p01prct_mask": 0.01,   # below 0.01%
}

for fname, threshold in thresholds.items():
    mask = (pct_surface_par >= threshold).where(pct_surface_par.notnull())
    mask_ds = mask.to_dataset(name="mask")

    # Carry useful metadata
    mask_ds["mask"].attrs.update({
        "long_name":  f"PAR below {threshold}% of surface PAR",
        "description": (
            f"Boolean mask: 1 (True) where PAR >= {threshold}% of the "
            f"surface (depth=0) value at the same time/lat/lon"
        ),
        "threshold_pct": threshold,
        "units": "1",
    })

    out_path = out_dir + fname + ".nc"
    print(f"saving {out_path}")
    mask_ds.to_netcdf(out_path)

print("fin")
# %%
