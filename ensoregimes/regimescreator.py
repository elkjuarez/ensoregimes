import glob
import os
from pathlib import Path
import argparse
import socket
import sys
from datetime import datetime
import numpy as np
import xarray as xr
import xesmf as xe

def parse_args():
    parser = argparse.ArgumentParser(
        description="Running for decade starting on specific year."
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Data year to process, for example 2020.",
    )
    return parser.parse_args()

def sliceimg(ds):
    latSlice = slice(20, 80) #20N, 80N
    lonSlice = slice(180, 330) #180W, 30W
    return ds.sel(lat=latSlice, lon=lonSlice)

def grablevday(ds):
    da = ds["Z"].sel(level=500.)
    da = da.resample(time="D").mean(skipna=True)
    return fixera5(da.to_dataset(name="Z"))

def fixera5(ds):
    rename_map = {}
    if "latitude" in ds.coords:
        rename_map["latitude"] = "lat"
    if "longitude" in ds.coords:
        rename_map["longitude"] = "lon"
    if "valid_time" in ds.dims or "valid_time" in ds.coords:
        rename_map["valid_time"] = "time"
    if rename_map:
        ds = ds.rename(rename_map)
    ds = xr.decode_cf(ds)
    ds = ds.assign_coords(lon=(ds["lon"] % 360)).sortby("lon")
    if ds["lat"][0] > ds["lat"][-1]:
        ds = ds.sortby("lat")
    return ds


args = parse_args()
YEAR = args.year

print("Hello from PBS!")
print(f"Timestamp: {datetime.now()}")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Running on host: {socket.gethostname()}")
print(f"Running job for data year: {YEAR}")

gdex_era5 = '/glade/campaign/collections/gdex/data/d633000/e5.oper.an.pl/'
out_dir = '/glade/derecho/scratch/molina/dailyz500_v052926/'

weight_file = f"{out_dir}weights_bilinear_025_to_150deg.nc"
os.path.exists(weight_file)

degree_to_interp = 1.5
ds_out = xr.Dataset({
    "lat": ("lat", np.arange(-90, 90.1, degree_to_interp)),
    "lon": ("lon", np.arange(0, 360, degree_to_interp)),
})

DECADE = np.linspace(YEAR, YEAR + 9, 10, dtype=int)
MONS = ['01','02','03','04','05','06','07','08','09','10','11','12']

reuse_weights = os.path.exists(weight_file)

for YR in DECADE:
    
    for MO in MONS:
        
        list_z = sorted(
            glob.glob(
                f'{gdex_era5}{YR}{MO}/e5.oper.an.pl.128_129_z.*.nc'
            )
        )

        for fu in list_z:
        
            ds = xr.open_mfdataset(
                fu, preprocess=grablevday
            ).compute()
            
            regridder = xe.Regridder(
                ds, 
                ds_out, 
                "bilinear",
                periodic=True,
                filename=weight_file,
                reuse_weights=reuse_weights,
                unmapped_to_nan=True,
            )
        
            rg_ds = regridder(ds["Z"])
            rg_ds = sliceimg(rg_ds)
            rg_ds.to_dataset(name="Z").to_netcdf(
                f"{out_dir}{Path(fu).name}"
            )