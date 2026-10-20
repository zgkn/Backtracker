import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from ecmwf.opendata import Client
import folium
from folium import plugins


# --- 1. Backtrack Trajectory Logic ---
# Changed to accept 'ds' (Dataset object) directly instead of 'nc_file' (file path)
def perform_backtrack(ds, start_lat, start_lon, interval=3, duration=48):
    ds = ds.sortby('step')
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

    levels_to_run = ['1000', '925', '850', '925_850_avg']

    for lvl in levels_to_run:
        u_func = RegularGridInterpolator((time_numeric, lats, lons), u_mats[lvl])
        v_func = RegularGridInterpolator((time_numeric, lats, lons), v_mats[lvl])

        start_times = np.arange(latest_t, -0.1, -interval)

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
def build_folium_timestepped_map(df_backtrack, run_time_str, valid_time_str, base_time, duration=48, target_lat=1.29, target_lon=103.85):
    m = folium.Map(location=[target_lat, target_lon], zoom_start=6, tiles='OpenStreetMap')

    # Styles and display configurations
    ordered_levels = ['1000', '925', '850', '925_850_avg']
    lvl_styles = {
        '1000':         {'color': '#e74c3c', 'dash': '0',    'label': '1000 hPa (Boundary Layer)'},
        '925':          {'color': '#f39c12', 'dash': '5, 5', 'label': '925 hPa (Low Level)'},
        '850':          {'color': '#2ecc71', 'dash': '2, 5', 'label': '850 hPa (Mid-Low Level)'},
        '925_850_avg':  {'color': '#9b59b6', 'dash': '10, 5','label': '925 + 850 hPa Avg Wind Layer'}
    }

    # Legend rows mirror each level's actual line color/dash style on the map
    legend_rows = ''.join(
        f'<div style="display:flex; align-items:center; margin-top:4px;">'
        f'<span style="display:inline-block; width:22px; height:0; '
        f'border-top:3px {"dashed" if lvl_styles[lvl]["dash"] != "0" else "solid"} {lvl_styles[lvl]["color"]}; '
        f'margin-right:6px;"></span>{lvl_styles[lvl]["label"]}</div>'
        for lvl in ordered_levels
    )

    # Floating Info Box positioned at top-right, clear of the time slider
    # control (leaflet.timedimension docks bottom-left and spans most of the
    # map width, especially on narrow/mobile viewports).
    info_html = f'''
    <div style="
        position: fixed;
        top: 10px; right: 10px;
        width: 250px; max-width: calc(100vw - 20px); height: auto;
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
        <div style="margin-top:8px; padding-top:6px; border-top:1px solid #ccc;">
            <b>Legend — Wind Level:</b>
            {legend_rows}
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(info_html))

    # Target Origin Marker (Singapore)
    folium.Marker(
        [target_lat, target_lon],
        popup="<b>Target Location:</b> Singapore",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)

    # Each trajectory ('traj_id') is one level's backtrack released at a single
    # forecast lead time ("T+Xh"). Instead of overlaying every lead time at once,
    # give each one a real valid-time stamp so the time slider steps through them
    # one forecast timestep at a time.
    lead_hours = sorted(df_backtrack.groupby('traj_id')['hour'].max().unique())
    step_hours = int(abs(lead_hours[1] - lead_hours[0])) if len(lead_hours) > 1 else 3
    period = f'PT{step_hours}H'

    features = []
    for tid, traj in df_backtrack.groupby('traj_id'):
        lvl = traj['level'].iloc[0]
        style = lvl_styles.get(lvl, {'color': 'gray', 'dash': '0', 'label': f'{lvl} Layer'})

        start_t = traj['hour'].max()
        valid_dt = base_time + pd.Timedelta(hours=float(start_t))
        timestamp = valid_dt.isoformat()

        traj = traj.sort_values('hour', ascending=False)
        coords = traj[['lon', 'lat']].values.tolist()

        features.append({
            'type': 'Feature',
            'geometry': {'type': 'LineString', 'coordinates': coords},
            'properties': {
                'times': [timestamp] * len(coords),
                'style': {
                    'color': style['color'],
                    'weight': 2.5,
                    'opacity': 0.85,
                    'dashArray': style['dash'],
                },
                'popup': f"<b>{style['label']}</b><br>Forecast Step: T+{int(start_t)}h<br>Valid: {valid_dt.strftime('%Y-%m-%d %H:%M UTC')}",
            },
        })

    plugins.TimestampedGeoJson(
        {'type': 'FeatureCollection', 'features': features},
        period=period,
        duration=period,
        transition_time=500,
        auto_play=False,
        loop=False,
        add_last_point=False,
        loop_button=True,
        time_slider_drag_update=True,
        date_options='YYYY-MM-DD HH:mm',
    ).add_to(m)

    plugins.Fullscreen().add_to(m)

    return m


# --- 3. Main Execution ---
def main():
    target_grib = "latest_wind.grib"
    backfill_grib = "previous_wind.grib"
    html_map_file = "haze_transport_map.html"
    BACKTRACK_HOURS = 48
    STEP_INTERVAL = 3

    # 90h is the longest step range the ECMWF open-data HRES pressure-level
    # product supports on every cycle (00/12z reach 240h, but 06/18z cap out
    # at 90h) - going further would make the 06/18z runs fail outright.
    print("🚀 Requesting 90-hour ECMWF HRES Forecast...")
    client = Client(source="azure")
    client.retrieve(
        type="fc",
        levtype="pl",
        levelist=[850, 925, 1000],
        param=["u", "v"],
        step=list(range(0, 91, STEP_INTERVAL)),
        target=target_grib
    )

    ds = xr.open_dataset(target_grib, engine="cfgrib")
    base_time = pd.to_datetime(ds.time.values)
    valid_time_max = pd.to_datetime(ds.valid_time.values.max())

    run_time_str = base_time.strftime('%Y-%m-%d %H:%M UTC')
    valid_time_str = valid_time_max.strftime('%Y-%m-%d %H:%M UTC')

    print(f"📊 Model Base Run: {run_time_str}")
    print(f"📊 Forecast Range: Up to {valid_time_str}")

    # The earliest forecast timesteps of this run don't have BACKTRACK_HOURS
    # of their own history yet, so backfill that history from the previous
    # model run (HRES cycles are 6h apart, so BACKTRACK_HOURS back lands
    # exactly on a cycle as long as it's a multiple of 6) and splice it onto
    # this run's wind field.
    prev_time = base_time - pd.Timedelta(hours=BACKTRACK_HOURS)
    print(f"🚀 Requesting backfill from previous model run ({prev_time.strftime('%Y-%m-%d %H:%M UTC')})...")
    client.retrieve(
        type="fc",
        levtype="pl",
        levelist=[850, 925, 1000],
        param=["u", "v"],
        date=prev_time.strftime('%Y-%m-%d'),
        time=int(prev_time.hour),
        step=list(range(0, BACKTRACK_HOURS + 1, STEP_INTERVAL)),
        target=backfill_grib
    )
    ds_backfill = xr.open_dataset(backfill_grib, engine="cfgrib")

    # Slice SE Asia Domain
    ds_region = ds.sel(latitude=slice(20, -10), longitude=slice(90, 130))
    ds_backfill_region = ds_backfill.sel(latitude=slice(20, -10), longitude=slice(90, 130))

    # Re-express the backfill run's steps as hours relative to *this* run's
    # base time (negative = before it), drop its last step since it's the
    # same instant as this run's own step=0, and stitch the two onto one
    # continuous time axis.
    ds_backfill_region = ds_backfill_region.assign_coords(
        step=ds_backfill_region.step - np.timedelta64(BACKTRACK_HOURS, 'h')
    ).isel(step=slice(0, -1))
    ds_region = xr.concat(
        [ds_backfill_region.drop_vars(['time', 'valid_time'], errors='ignore'),
         ds_region.drop_vars(['time', 'valid_time'], errors='ignore')],
        dim='step'
    )

    # Pass the dataset directly in-memory (Skipping intermediate NetCDF file entirely!)
    print(f"🔄 Calculating {BACKTRACK_HOURS}-hour backtracks for every forecast timestep...")
    df_backtrack = perform_backtrack(ds_region, 1.29, 103.85, interval=STEP_INTERVAL, duration=BACKTRACK_HOURS)

    print("🗺️ Building interactive Folium map...")
    haze_map = build_folium_timestepped_map(
        df_backtrack,
        run_time_str=run_time_str,
        valid_time_str=valid_time_str,
        base_time=base_time,
        duration=BACKTRACK_HOURS
    )
    haze_map.save(html_map_file)
    print(f"✅ Saved interactive map artifact to {html_map_file}")

    # Cleanup the GRIB downloads
    for f in (target_grib, backfill_grib):
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    main()
