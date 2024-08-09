# Data is NetCDF format NCEP-NCAR Reanalysis from NOAA Physical Sciences Laboratory
# Import Package
import os
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import scipy.stats as st
import xarray as xr
from functools import partial
from progressbar import ProgressBar
from scipy.stats import gumbel_r

# Section 0.1
# Comment section ini jika data sudah terdownload dan ingin me-run ulang sehingga tidak perlu download ulang
# Mendownload data dari NOAA
# Mendefinisikan fungsi untuk mendownload data dari URL dan menyimpannya ke path tertentu
def initiateDownload(url, downloadPath):
    response = requests.get(url, stream=True)
    with open(downloadPath, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)

# Inisiasi path tujuan download
path = 'D:/ITB/AKADEMIK/128/SMT 5/ME3102 ANALISIS DATA CUACA DAN IKLIM I/TUGAS/Tugas Tutorial 3/Data'

# Looping download data temperatur maksimum dari 2003-2023
for year in range(2003, 2024):
    url = f"https://downloads.psl.noaa.gov//Datasets/ncep.reanalysis2/Dailies/gaussian_grid/tmax.2m.gauss.{year}.nc"
    fileName = f"tmax.2m.gauss.{year}.nc"
    directoryDownload = os.path.join(path, fileName)
    print(f"Downloading {fileName}...")
    initiateDownload(url, directoryDownload)
    print(f"Downloaded {fileName}. Saved to {directoryDownload}")
print("Download Completed.")

# Looping download data curah hujan dari 2003-2023
for year in range(2003, 2024):
    url = f"https://downloads.psl.noaa.gov//Datasets/ncep.reanalysis2/Dailies/gaussian_grid/prate.sfc.gauss.{year}.nc"
    fileName = f"prate.sfc.gauss.{year}.nc"
    directoryDownload = os.path.join(path, fileName)
    print(f"Downloading {fileName}...")
    initiateDownload(url, directoryDownload)
    print(f"Downloaded {fileName}. Saved to {directoryDownload}")
print("Download Completed.")

# End of Section 0.1

# Section 0.2
# Membuka data temperatur maksimum dan curah hujan
# Mendefinisikan fungsi untuk memotong batas koordinat wilayah yang ditentukan dari satu dunia
def _preprocess(x, lonBoundaries, latBoundaries):
    return x.sel(lon=slice(*lonBoundaries), lat=slice(*latBoundaries))

# Memotong peta ke batas wilayah Indonesia
desiredLon, desiredLat = (90, 150), (15, -15)
partialFunc = partial(_preprocess, lonBoundaries=desiredLon, latBoundaries=desiredLat)

# Inisiasi path tempat data disimpan
filePath = 'D:/ITB/AKADEMIK/128/SMT 5/ME3102 ANALISIS DATA CUACA DAN IKLIM I/TUGAS/Tugas Tutorial 3/Data/tmax.2m.gauss.*.nc'
filePath2 = 'D:/ITB/AKADEMIK/128/SMT 5/ME3102 ANALISIS DATA CUACA DAN IKLIM I/TUGAS/Tugas Tutorial 3/Data/prate.sfc.gauss.*.nc'

# Membuka seluruh data yang telah didownload dan ditumpuk berdasarkan waktu
tmaxAllData = xr.open_mfdataset(filePath, concat_dim='time', preprocess=partialFunc, combine='nested')
prAllData = xr.open_mfdataset(filePath2, concat_dim='time', preprocess=partialFunc, combine='nested')

# End of Section 0.2

# Section 0.3
# Template Plot dengan Sea Mask
# Membuat mask untuk laut dengan warna biru
oceanFeature = cfeature.NaturalEarthFeature(
    category='physical',
    name='ocean',
    scale='10m',
    edgecolor='none',
    facecolor='skyblue',
)

# Menambahkan garis batas provinsi
provLine = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='10m',
    facecolor='none'
)

# End of Section 0.3

# Tugas 1
# Section 1.1
# Membuat Distribusi Gamma dan Nilai Persentil ke Visualisasi Spasial
# Panjang Lintang dan Bujur
nx = len(tmaxAllData.lon)
ny = len(tmaxAllData.lat)
lon = tmaxAllData.lon
lat = tmaxAllData.lat 

# End of Section 1.1

# Section 1.2
# Mengelompokkan data secara bulanan
monthlyTmaxData = tmaxAllData['tmax'].mean(dim='level').groupby('time.month')
tmaxClimJune = monthlyTmaxData[6] # Mengambil data di bulan Juni saja
tmaxClimJuly = monthlyTmaxData[7] # Mengambil data di bulan Juli saja
tmaxCsJun23 = tmaxClimJune.sel(time='2023-06-15') # Mengambil salah satu data di bulan Juni
tmaxCsJul23 = tmaxClimJuly.sel(time='2023-07-25') # Mengambil salah satu data di bulan Juli

# Mengelompokkan data secara musiman
seasonalTmaxData = tmaxAllData['tmax'].mean(dim='level').groupby('time.season')
tmaxClimJJA = seasonalTmaxData['JJA']

# End of Section 1.2

# Section 1.3
# Comment section ini jika section ini sudah dirunning namun ingin memplot ulang, langsung ke bagian buka data nc
# Menghitung Persentil dari tiap tmax di semua titik di tanggal yang dipilih
pbar = ProgressBar()
percentileJn = xr.DataArray(np.zeros((ny,nx)), dims=("lat", "lon"), coords={"lat":lat, "lon":lon})
for iy in pbar(range(ny)):
    for ix in range(nx):
        tmaxCl = tmaxClimJune[:, iy, ix]
        tmaxCs = tmaxCsJun23[iy, ix]
        if np.isnan(tmaxCs.data).any():
            percentileJn[iy, ix] = np.nan
        else:
            gampar = st.gamma.fit(tmaxCl)
            percentileJn[iy, ix] = st.gamma.cdf(tmaxCsJun23[iy, ix], *gampar)

# Menyimpan hasil persentil dalam NetCDF
percentileJn.to_netcdf('percentile_15062023.nc')

pbar = ProgressBar()
percentileJl = xr.DataArray(np.zeros((ny,nx)), dims=("lat", "lon"), coords={"lat":lat, "lon":lon})
for iy in pbar(range(ny)):
    for ix in range(nx):
        tmaxCl = tmaxClimJuly[:, iy, ix]
        tmaxCs = tmaxCsJul23[iy, ix]
        if np.isnan(tmaxCs.data).any():
            percentileJl[iy, ix] = np.nan
        else:
            gampar = st.gamma.fit(tmaxCl)
            percentileJl[iy, ix] = st.gamma.cdf(tmaxCsJul23[iy, ix], *gampar)

percentileJl.to_netcdf('percentile_25072023.nc')

# End of Section 1.3

# Section 1.4
# Buka data nc
percentileJn = xr.open_dataarray('percentile_15062023.nc')
percentileJl = xr.open_dataarray('percentile_25072023.nc')

# Plot Hasil Persentil
fig = plt.figure(figsize=(16,8))
ax = plt.axes(projection=cartopy.crs.PlateCarree())
percentileJn.plot(transform=cartopy.crs.PlateCarree(), cmap='magma', levels=np.arange(0.0, 1.0, 0.1))
ax.add_feature(oceanFeature)
ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='black', linewidth=2)
ax.add_feature(provLine, edgecolor='gray', linewidth=1)
gl = ax.gridlines(draw_labels=True)
gl.xlabel_style = {'size': 16, 'color':'k'}
gl.ylabel_style = {'size': 16, 'color':'k'}
plt.title(label="Persentil Tmax " + tmaxCsJun23.time.dt.strftime('%Y-%m-%d').values, fontdict={'fontsize':20, 'fontweight':'bold'})
plt.savefig(f'Persentil Tmax pada 15 Juni 2023.png')
plt.close(fig)

fig = plt.figure(figsize=(16,8))
ax = plt.axes(projection=cartopy.crs.PlateCarree())
percentileJl.plot(transform=cartopy.crs.PlateCarree(), cmap='magma', levels=np.arange(0.0, 1.0, 0.1))
ax.add_feature(oceanFeature)
ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='black', linewidth=2)
ax.add_feature(provLine, edgecolor='gray', linewidth=1)
gl = ax.gridlines(draw_labels=True)
gl.xlabel_style = {'size': 16, 'color':'k'}
gl.ylabel_style = {'size': 16, 'color':'k'}
plt.title(label="Persentil Tmax " + tmaxCsJul23.time.dt.strftime('%Y-%m-%d').values, fontdict={'fontsize':20, 'fontweight':'bold'})
plt.savefig(f'Persentil Tmax pada 25 Juli 2023.png')
plt.close(fig)

# End of Section 1.4

# Section 1.5
# Comment section ini jika section ini sudah dirunning namun ingin memplot ulang, langsung ke bagian buka data nc
# Menghitung Persentil dari Tmax tertentu pada tanggal yang dipilih
pbar = ProgressBar()
percentileJnJJA = xr.DataArray(np.zeros((ny,nx)), dims=("lat", "lon"), coords={"lat":lat, "lon":lon})
for iy in pbar(range(ny)):
    for ix in range(nx):
        tmaxCl = tmaxClimJJA[:, iy, ix]
        tmaxCs = tmaxCsJun23[iy, ix]
        if np.isnan(tmaxCs.data).any():
            percentileJnJJA[iy, ix] = np.nan
        else:
            gampar = st.gamma.fit(tmaxCl)
            percentileJnJJA[iy, ix] = st.gamma.cdf(tmaxCsJun23[iy, ix], *gampar)
# Menyimpan hasil persentil
percentileJnJJA.to_netcdf('percentile_JnJJA.nc')

pbar = ProgressBar()
percentileJlJJA = xr.DataArray(np.zeros((ny,nx)), dims=("lat", "lon"), coords={"lat":lat, "lon":lon})
for iy in pbar(range(ny)):
    for ix in range(nx):
        tmaxCl = tmaxClimJJA[:, iy, ix]
        tmaxCs = tmaxCsJul23[iy, ix]
        if np.isnan(tmaxCs.data).any():
            percentileJlJJA[iy, ix] = np.nan
        else:
            gampar = st.gamma.fit(tmaxCl)
            percentileJlJJA[iy, ix] = st.gamma.cdf(tmaxCsJul23[iy, ix], *gampar)

# Menyimpan hasil persentil
percentileJlJJA.to_netcdf('percentile_JlJJA.nc')

# End of Section 1.5

# Section 1.6
# Buka Data nc
percentileJnJJA = xr.open_dataarray('percentile_JnJJA.nc')
percentileJlJJA = xr.open_dataarray('percentile_JlJJA.nc')

# Plot Hasil Persentil
fig2 = plt.figure(figsize=(16,8))
ax = plt.axes(projection=cartopy.crs.PlateCarree())
percentileJnJJA.plot(transform=cartopy.crs.PlateCarree(), cmap='magma', levels=np.arange(0.0, 1.0, 0.1))
ax.add_feature(oceanFeature)
ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='black', linewidth=2)
ax.add_feature(provLine, edgecolor='gray', linewidth=1)
gl = ax.gridlines(draw_labels=True)
gl.xlabel_style = {'size': 16, 'color':'k'}
gl.ylabel_style = {'size': 16, 'color':'k'}
plt.title(label=f"Persentil Tmax Juni terhadap JJA " + tmaxCsJun23.time.dt.strftime('%Y-%m-%d').values, fontdict={'fontsize':20, 'fontweight':'bold'})
plt.savefig(f'Persentil Tmax Juni terhadap JJA')
plt.close(fig2)

fig2 = plt.figure(figsize=(16,8))
ax = plt.axes(projection=cartopy.crs.PlateCarree())
percentileJlJJA.plot(transform=cartopy.crs.PlateCarree(), cmap='magma', levels=np.arange(0.0, 1.0, 0.1))
ax.add_feature(oceanFeature)
ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='black', linewidth=2)
ax.add_feature(provLine, edgecolor='gray', linewidth=1)
gl = ax.gridlines(draw_labels=True)
gl.xlabel_style = {'size': 16, 'color':'k'}
gl.ylabel_style = {'size': 16, 'color':'k'}
plt.title(label=f"Persentil Tmax Juli terhadap JJA " + tmaxCsJul23.time.dt.strftime('%Y-%m-%d').values, fontdict={'fontsize':20, 'fontweight':'bold'})
plt.savefig(f'Persentil Tmax Juli terhadap JJA')
plt.close(fig2)

# End of Section 1.6

# Tugas 2
# Section 2.1
# Menghitung annual maxima dan periode ulang lalu memplot secara spasial
# Menghitung Annual Maxima dan konversi satuan curah hujan
annMax = prAllData['prate'].groupby('time.year').max()
annMax = annMax*24*3600

# End of Section 2.1

# Section 2.2
# Comment section ini jika section ini sudah dirunning namun ingin memplot ulang, langsung ke bagian buka data nc
# Estimasi Periode Ulang
# Fungsi Gumbell
w = 1 # Sampling Frekuensi
R = 100 # Periode Ulang, ubah sesuai variasi (10, 25, 50, dan 100) 
Fx = 1-1/(R*w) 

# Menghitung curah hujan untuk setiap periode ulang
pbar = ProgressBar()
returnPeriod = xr.DataArray(np.zeros((len(annMax.lat), len(annMax.lon))), dims=("lat", "lon"), 
                        coords={"lat":annMax.lat, "lon":annMax.lon})

for iy in pbar(range(len(annMax.lat))):
    for ix in range(len(annMax.lon)):
        prmt = gumbel_r.fit(annMax.sel(lat=lat[iy], lon=lon[ix]))
        Rx = gumbel_r.ppf(Fx, *prmt)
        returnPeriod[iy, ix] = Rx
returnPeriod.to_netcdf('returnPeriod100yr.nc')

# End of Section 2.2

# Section 2.3
# Buka data-data periode ulang
returnPeriod10yr = xr.open_dataarray('returnPeriod10yr.nc')
returnPeriod25yr = xr.open_dataarray('returnPeriod25yr.nc')
returnPeriod50yr = xr.open_dataarray('returnPeriod50yr.nc')
returnPeriod100yr = xr.open_dataarray('returnPeriod100yr.nc')

# Plot Hasil Periode Ulang 10 Tahun
fig3 = plt.figure(figsize=(16,8))
ax = plt.axes(projection=cartopy.crs.PlateCarree())
returnPeriod10yr.plot(transform=cartopy.crs.PlateCarree(), cmap='magma', levels=np.linspace(0, 200, 10))
ax.add_feature(oceanFeature)
ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='black', linewidth=2)
ax.add_feature(provLine, edgecolor='gray', linewidth=1)
gl = ax.gridlines(draw_labels=True)
gl.xlabel_style = {'size': 16, 'color':'k'}
gl.ylabel_style = {'size': 16, 'color':'k'}
plt.title(label="Periode Ulang Curah Hujan (mm/hari) 10 Tahun", fontdict={'fontsize':20, 'fontweight':'bold'})
plt.savefig('Periode Ulang Curah Hujan 10 Tahun.png')
plt.close(fig3)

# Plot Hasil Periode Ulang 25 Tahun
fig4 = plt.figure(figsize=(16,8))
ax = plt.axes(projection=cartopy.crs.PlateCarree())
returnPeriod25yr.plot(transform=cartopy.crs.PlateCarree(), cmap='magma', levels=np.linspace(0, 200, 10))
ax.add_feature(oceanFeature)
ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='black', linewidth=2)
ax.add_feature(provLine, edgecolor='gray', linewidth=1)
gl = ax.gridlines(draw_labels=True)
gl.xlabel_style = {'size': 16, 'color':'k'}
gl.ylabel_style = {'size': 16, 'color':'k'}
plt.title(label="Periode Ulang Curah Hujan (mm/hari) 25 Tahun", fontdict={'fontsize':20, 'fontweight':'bold'})
plt.savefig('Periode Ulang Curah Hujan 25 Tahun.png')
plt.close(fig4)

# Plot Hasil Periode Ulang 50 Tahun
fig5 = plt.figure(figsize=(16,8))
ax = plt.axes(projection=cartopy.crs.PlateCarree())
returnPeriod50yr.plot(transform=cartopy.crs.PlateCarree(), cmap='magma', levels=np.linspace(0, 200, 10))
ax.add_feature(oceanFeature)
ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='black', linewidth=2)
ax.add_feature(provLine, edgecolor='gray', linewidth=1)
gl = ax.gridlines(draw_labels=True)
gl.xlabel_style = {'size': 16, 'color':'k'}
gl.ylabel_style = {'size': 16, 'color':'k'}
plt.title(label="Periode Ulang Curah Hujan (mm/hari) 50 Tahun", fontdict={'fontsize':20, 'fontweight':'bold'})
plt.savefig('Periode Ulang Curah Hujan 50 Tahun.png')
plt.close(fig5)

# Plot Hasil Periode Ulang 100 Tahun
fig6 = plt.figure(figsize=(16,8))
ax = plt.axes(projection=cartopy.crs.PlateCarree())
returnPeriod100yr.plot(transform=cartopy.crs.PlateCarree(), cmap='magma', levels=np.linspace(0, 200, 10))
ax.add_feature(oceanFeature)
ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='black', linewidth=2)
ax.add_feature(provLine, edgecolor='gray', linewidth=1)
gl = ax.gridlines(draw_labels=True)
gl.xlabel_style = {'size': 16, 'color':'k'}
gl.ylabel_style = {'size': 16, 'color':'k'}
plt.title(label="Periode Ulang Curah Hujan (mm/hari) 100 Tahun", fontdict={'fontsize':20, 'fontweight':'bold'})
plt.savefig('Periode Ulang Curah Hujan 100 Tahun.png')
plt.close(fig6)

# End of Section 2.3