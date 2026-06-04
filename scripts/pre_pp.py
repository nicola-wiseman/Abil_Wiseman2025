import pandas as pd
import numpy as np
import xarray as xr
from scipy.stats import truncnorm
from abil.pseudo_generation import generate_pseudo_absences


def process_variable(varname):
    """
    Returns a processed DataFrame for either Calcification or Primary_Production
    with coordinates + samples + pseudo-zeros + env merged.
    """
    assert varname in ["Calcification", "Primary_Production"]

    print(f"\n=== Processing {varname} ===\n")

    # Load dataset
    d_raw = pd.read_csv(
        '/user/work/mv23682/Abil/studies/wiseman2024/data/Marsh_et_al_2025_Database_V_1.0.csv',
        skiprows=1,
        names=[
            "PI","Expedition","OS Region","Reference_Author_Published_year","Reference_doi",
            "Date","Sample_ID","Latitude","Longitude","Depth","Irr_Depth",
            "Optical_Depth","Method","Incubation_Length",
            "Calcification","Calcification_Standard_Deviation",
            "Primary_Production","Primary_Production_Standard_Deviation",
            "Total Coccolithophore cell counts [cells mL-1]","Emiliania huxleyi cell counts [cells mL-1]",
            "Chlorophyll-a [mg m-3)","NOx (µM/L)","Silicate (µM/L)","Phosphate (µM/L)",
            "DIC [μmol/kg-seawater]","Total Alkalinity  [μmol/kg-seawater]",
            "Bicarbonate (HCO3) [μmol/kg-seawater]","Carbonate (CO3) [μmol/kg-seawater]",
            "pH","Temperature (degrees C)","Salinity (ppt)"
        ]
    )
    print(f"Initial rows: {len(d_raw)}")

    # Date fields
    d_raw['DateTime'] = pd.to_datetime(d_raw['Date'], dayfirst=False, errors='coerce')
    d_raw['Month'] = d_raw['DateTime'].dt.month
    d_raw['Year'] = d_raw['DateTime'].dt.year

    d = d_raw.copy()
    d = d[~d['Method'].isin(['Diff','Ca45'])]
    print(f"Rows after removing certain methods: {len(d)}")
    
    # QC rules: only apply calcification-specific ones
    if varname == "Calcification":
        denom = d["Emiliania huxleyi cell counts [cells mL-1]"]
        mask = denom.notna() & (d[varname] / denom > 3.5)
        d = d[~mask]
        print(f"Rows after high calcification/Emiliania filter: {len(d)}")

    if varname == "Primary_Production":
        d['Primary_Production'] = d['Primary_Production'] * 1e-3
        print(f"Primary_Production converted to mmol/m3/d")

    # Drop unused columns — keep varname + its SD
    keep = [
        'Latitude','Longitude','Depth','Month','OS Region',
        varname, f"{varname}_Standard_Deviation"
    ]
    d = d[keep]
    print(f"Rows after keeping relevant columns: {len(d)}")

    # CV
    n = 3
    d['CV'] = d[f"{varname}_Standard_Deviation"] / d[varname]

    # Region CV
    region_cv = d.groupby("OS Region")["CV"].mean()
    global_cv = d["CV"].mean()
    d = d.merge(region_cv.rename("region_cv"), on="OS Region", how="left")
    d["region_cv"] = d["region_cv"].fillna(global_cv)

    # Fill missing SD
    m = d[f"{varname}_Standard_Deviation"].isna()
    d.loc[m, f"{varname}_Standard_Deviation"] = d.loc[m, varname] * d.loc[m, "region_cv"]
    print(f"Rows after filling missing SD: {len(d)}")

    # Drop the helper columns
    d.drop(columns=['CV', 'region_cv'], inplace=True)

    # SEM
    d[f"{varname}_SEM"] = d[f"{varname}_Standard_Deviation"] / np.sqrt(n)
    d = d.dropna(subset=[varname, f"{varname}_SEM"])
    print(f"Rows after dropping NAs in varname and SEM: {len(d)}")

    # Resample
    resamples = 50
    mu = d[varname].values.reshape(-1, 1)
    sem = d[f"{varname}_SEM"].values.reshape(-1, 1)

    a = (0 - mu) / sem
    b = np.full_like(a, np.inf)
    random_samples = truncnorm.rvs(a, b, loc=mu, scale=sem, size=(len(d), resamples))
    sample_cols = [f"{varname}_sample_{k+1}" for k in range(resamples)]
    samples_df = pd.DataFrame(random_samples, columns=sample_cols)

    d = pd.concat([d.reset_index(drop=True), samples_df], axis=1)
    d.drop(["OS Region",f"{varname}_Standard_Deviation",f"{varname}_SEM"],axis=1,inplace=True)
    print(f"Rows after adding random samples: {len(d)}")

    # Grid to 180×360×41×12
    depth_bins = np.linspace(0, 205, 42)
    depth_labels = np.linspace(0, 200, 41)
    d["Depth"] = pd.cut(d["Depth"], depth_bins, labels=depth_labels).astype(float)

    lat_bins = np.linspace(-90, 90, 181)
    lat_labels = np.linspace(-90, 89, 180)
    d["Latitude"] = pd.cut(d["Latitude"], lat_bins, labels=lat_labels).astype(float)

    lon_bins = np.linspace(-180, 180, 361)
    lon_labels = np.linspace(-180, 179, 360)
    d["Longitude"] = pd.cut(d["Longitude"], lon_bins, labels=lon_labels).astype(float)

    # Rename columns for consistency
    d = d.groupby(["Latitude", "Longitude", "Depth", "Month"]).mean().reset_index()
    print(f"Rows after grouping and averaging: {len(d)}")
    d = d.rename(columns={"Latitude": "lat", "Longitude": "lon", "Depth": "depth", "Month": "time"})
    d.set_index(["lat", "lon", "depth", "time"], inplace=True)

    # Load environment data
    ds_env = xr.open_dataset('/user/work/mv23682/Abil/studies/wiseman2024/data/env_data.nc')
    df_env = ds_env.to_dataframe().reset_index()
    env_cols = ["temperature","sio4","po4","no3","o2","mld","DIC","TA","PAR"]
    df_env = df_env[env_cols + ["lat","lon","depth","time"]].set_index(["lat","lon","depth","time"])
    print(f"Environment rows: {len(df_env)}")

    # Filter only rows whose environmental data exists AND is fully non-NaN
    valid_env_idx = df_env.dropna().index
    d = d.loc[d.index.intersection(valid_env_idx)]
    print(f"Rows after dropping samples without complete environmental data: {len(d)}")

    # Load mask and convert to DataFrame of zeros
    # mask_ds = xr.open_dataset('/user/work/mv23682/Abil/studies/wiseman2024/data/PAR_1prct_mask.nc')
    # mask = mask_ds["mask"]
    # zeros_df = mask.where(mask == 0, drop=True).to_dataframe().reset_index()
    # zeros_df = zeros_df[zeros_df["mask"] == 0].set_index(["lat","lon","depth","time"])

    # Keep only positions that exist in env data
    # zeros_df = zeros_df.loc[zeros_df.index.intersection(valid_env_idx)]
    # print(f"Zeros rows after intersection with env: {len(zeros_df)}")

    # Sample zeros
    # n_obs = d[varname].notna().sum()
    # zeros_subset = zeros_df.sample(n=n_obs, random_state=42)
    # zs = zeros_subset.copy()
    # zs[varname] = 0
    # for c in sample_cols:
    #     zs[c] = 0
    # zs.drop(columns=["mask"],inplace=True)

    # Concatenate original data and zeros
    # d = pd.concat([d, zs], ignore_index=False)

    # Use AOA to generate pseudo-zeros
    missing_df_env = df_env[~df_env.index.isin(d.index)]
    d = d.join(df_env, how="inner").reset_index()
    d = d.set_index(["lat", "lon", "depth", "time"])
    print(f"Rows after merging with environment: {len(d)}")

    d = generate_pseudo_absences(d, missing_df_env, env_cols, ["Primary_Production"])
    print(f"Rows after adding pseudo-zeros: {len(d)}")

    return d.reset_index(drop=True)


# --- Run variable ---
pp_df   = process_variable("Primary_Production")

# Save combined CSV
outfile = "/user/work/mv23682/Abil/studies/wiseman2024/data/pp_env_aoa_pseudo.csv"
pp_df.to_csv(outfile, index=False)

print("\n=== Finished ===")
print("Saved Primary Production CSV to:")
print(outfile)
# %%
