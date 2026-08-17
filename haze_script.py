import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from ecmwf.opendata import Client
import folium
from folium import plugins
import alphashape
from shapely.geometry import Polygon, MultiPolygon


# --- 1. 48-Hour Backtrack Logic ---
def perform_48h_backtrack(nc_file, start_lat, start_lon, interval=3):
    ds = xr.open_dataset(nc_file).sortby('step')
    time_numeric = ds.step.values.astype('timedelta64[h]').astype(float)
    if time_numeric.ndim > 1: 
        time_numeric = time_numeric.flatten()

    lats, lons = ds.latitude.values, ds.longitude.values
    if lats[0] > lats[-1]: 
        lats, flip_lat = lats[::-1], True
    else: 
        flip_lat = False

    v_dim = 'isobaricInhPa' if 'isobaricInhPa' in ds.coords or 'isobaricInhPa' in ds.dims else 'pressureLevel'

    # Prepare level slice matrices
    ds_1000 = ds.sel({v_dim: 1000})
    ds_925 = ds.sel({v_dim: 925})
    ds_850 = ds.sel({v_dim: 850})

    # Calculate average vector wind field for (925hPa + 850hPa) / 2
    u_avg_val = (ds_925.u.values + ds_850.u.values) / 2.0
    v_avg_val = (ds_925.v.values + ds_850.v.values) / 2.0

    if flip_lat:
        u_mats = {
            '1000': ds_1000.u.values[:, ::-1, :],
            '925': ds_925.u.values[:, ::-1, :],
            '850': ds_850.u.values[:, ::-1, :],
            '925_850_avg': u_avg_val[:, ::-1, :]
        }
        v_mats = {
            '1000': ds_1000.v.values[:, ::-1, :],
            '925': ds_925.v.values[:, ::-1, :],
            '850': ds_850.v.values[:, ::-1, :],
            '925_850_avg': v_avg_val[:, ::-1, :]
        }
    else:
        u_mats = {
            '1000': ds_1000.u.values,
            '925': ds_925.u.values,
            '850': ds_850.u.values,
            '925_850_avg': u_avg_val
        }
        v_mats = {
            '1000': ds_1000.v.values,
            '925': ds_925.v.values,
            '850': ds_850.v.values,
            '925_850_avg': v_avg_val
        }

    results = []
    latest_t = time_numeric[-1]
    duration = 48  # 48 hours backward integration

    levels_to_run = ['1000', '925', '850', '925_850_avg']

    for lvl in levels_to_run:
        u_func = RegularGridInterpolator((time_numeric, lats, lons), u_mats[lvl])
        v_func = RegularGridInterpolator((time_numeric, lats, lons), v_mats[lvl])

        start_times = np.arange(latest_t, duration - 0.1, -interval)

        for start_t in start_times:
            curr_lat, curr_lon, curr_t = start_lat, start_lon, start_t
            traj_id = f"{lvl}_T+{int(start_t)}h"
            path = [(curr_t, curr_lat, curr_lon, lvl, traj_id)]

            for _ in range(duration):
                u = u_func(np.array([[curr_t, curr_lat, curr_lon]]))[0]
                v = v_func(np.array([[curr_t, curr_lat, curr_lon]]))[0]
                curr_lat -= (v * 3600) / 111320.0
                curr_lon -= (u * 3600) / (111320.0 * np.cos(np.radians(curr_lat)))
                curr_t -= 1
                path.append((curr_t, curr_lat, curr_lon, lvl, traj_id))

            results.append(pd.DataFrame(path, columns=['hour', 'lat', 'lon', 'level', 'traj_id']))

    return pd.concat(results, ignore_index=True)


# --- 2. Folium Interactive Map Builder ---
def build_folium_map_by_level(df_48h, run_time_str, valid_time_str, duration=48, target_lat=1.29, target_lon=103.85):
    m = folium.Map(location=[target_lat, target_lon], zoom_start=6, tiles='cartodbpositron')

    # Styles and display configurations
    lvl_styles = {
        '1000':         {'color': '#e74c3c', 'dash': '0',    'label': '1000 hPa (Boundary Layer)'},
        '925':          {'color': '#f39c12', 'dash': '5, 5', 'label': '925 hPa (Low Level)'},
        '850':          {'color': '#2ecc71', 'dash': '2, 5', 'label': '850 hPa (Mid-Low Level)'},
        '925_850_avg':  {'color': '#9b59b6', 'dash': '10, 5','label': '925 + 850 hPa Avg Wind Layer'}
    }

    # Floating Info Box positioned at bottom-left to avoid blocking LayerControl
    info_html = f'''
    <div style="
        position: fixed; 
        bottom: 30px; left: 10px; 
        width: 250px; height: auto; 
        background-color: rgba(255, 255, 255, 0.95); 
        border: 2px solid #2c3e50; 
        border-radius: 8px;
        z-index: 9999; 
        font-size: 12px; 
        padding: 10px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
        font-family: Arial, sans-serif;
    ">
        <h4 style="margin: 0 0 6px 0; color: #2c3e50; font-size: 13px; border-bottom: 1px solid #ccc; padding-bottom: 4px;">
            <b>🌬️ Haze Transport Model</b>
        </h4>
        <b>Model Run:</b> {run_time_str}<br>
        <b>Forecast Valid:</b> {valid_time_str}<br>
        <b>Backtrack Period:</b> {duration} Hours<br>
        <b>Target:</b> {target_lat}°N, {target_lon}°E
    </div>
    '''
    m.get_root().html.add_child(folium.Element(info_html))

    # Target Origin Marker (Singapore)
    folium.Marker(
        [target_lat, target_lon],
        popup="<b>Target Location:</b> Singapore",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)

    ordered_levels = ['1000', '925', '850', '925_850_avg']

    for lvl in ordered_levels:
        if lvl not in df_48h['level'].unique():
            continue

        style = lvl_styles.get(lvl, {'color': 'gray', 'dash': '0', 'label': f'{lvl} Layer'})
        fg = folium.FeatureGroup(name=style['label'], show=True)
        subset_lvl = df_48h[df_48h['level'] == lvl]

        all_level_pts = []

        # 1. Plot individual trajectory lines
        for tid in subset_lvl['traj_id'].unique():
            traj = subset_lvl[subset_lvl['traj_id'] == tid]
            coords = list(zip(traj['lat'], traj['lon']))
            all_level_pts.extend([(lon, lat) for lat, lon in coords])

            folium.PolyLine(
                locations=coords,
                color=style['color'],
                weight=2,
                opacity=0.7,
                dash_array=style['dash'],
                tooltip=f"Traj: {tid} | Layer: {lvl}"
            ).add_to(fg)

        # 2. Compute Alpha Shape Concave Hull to hug trajectory paths
        if len(all_level_pts) >= 4:
            pts_array = np.array(all_level_pts)
            alpha_shape = alphashape.alphashape(pts_array, alpha=1.2)

            def add_polygon_to_map(poly_geom):
                if poly_geom.is_empty:
                    return
                ext_coords = [[lat, lon] for lon, lat in poly_geom.exterior.coords]
                folium.Polygon(
                    locations=ext_coords,
                    color=style['color'],
                    fill=True,
                    fill_color=style['color'],
                    fill_opacity=0.18,
                    weight=1.5,
                    popup=f"<b>48h Concave Transport Corridor:</b> {style['label']}"
                ).add_to(fg)

            if isinstance(alpha_shape, Polygon):
                add_polygon_to_map(alpha_shape)
            elif isinstance(alpha_shape, MultiPolygon):
                for p in alpha_shape.geoms:
                    add_polygon_to_map(p)

        fg.add_to(m)

    # UI Controls
    folium.LayerControl(collapsed=False).add_to(m)
    plugins.Fullscreen().add_to(m)

    return m


# --- 3. Main Execution ---
def main():
    target_grib = "latest_wind.grib"
    target_nc = "haze_forecast.nc"
    html_map_file = "haze_transport_map.html"

    print("🚀 Requesting 72-hour ECMWF HRES Forecast...")
    client = Client(source="azure")
    client.retrieve(
        type="fc",
        levtype="pl",
        levelist=[850, 925, 1000],
        param=["u", "v"],
        step=list(range(0, 73, 3)),
        target=target_grib
    )

    ds = xr.open_dataset(target_grib, engine="cfgrib")
    base_time = pd.to_datetime(ds.time.values)
    valid_time_max = pd.to_datetime(ds.valid_time.values.max())

    run_time_str = base_time.strftime('%Y-%m-%d %H:%M UTC')
    valid_time_str = valid_time_max.strftime('%Y-%m-%d %H:%M UTC')

    print(f"📊 Model Base Run: {run_time_str}")
    print(f"📊 Forecast Range: Up to {valid_time_str}")

    # Slice SE Asia Domain
    ds_region = ds.sel(latitude=slice(20, -10), longitude=slice(90, 130))

    # OPTION 1 FIX: Clean serialization attributes across variables and coordinates to prevent NetCDF decoding errors
    for var in list(ds_region.variables) + list(ds_region.coords):
        if var in ds_region:
            ds_region[var].attrs.pop('dtype', None)

    ds_region.to_netcdf(target_nc)

    if os.path.exists(target_grib):
        os.remove(target_grib)

    print("🔄 Calculating 48-hour backtracks...")
    df_48h = perform_48h_backtrack(target_nc, 1.29, 103.85)

    print("🗺️ Building interactive Folium map...")
    haze_map = build_folium_map_by_level(
        df_48h, 
        run_time_str=run_time_str, 
        valid_time_str=valid_time_str, 
        duration=48
    )
    haze_map.save(html_map_file)
    print(f"✅ Saved interactive map to {html_map_file}")

    if os.path.exists(target_nc):
        os.remove(target_nc)


if __name__ == "__main__":
    main()

