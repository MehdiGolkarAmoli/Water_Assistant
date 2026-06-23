"""
Water Quality Monitoring Application
=====================================
Sentinel-2 based automatic calculation of:
  - Water Turbidity (NDTI)
  - Chlorophyll-a Concentration

Both indices share the same preprocessing pipeline (cloud filtering, snow
masking, waterbody extraction) and are now computed automatically and
sequentially after the user selects an area of interest. The interface is
designed for managers and non-technical decision-makers: no remote-sensing
jargon, no parameter pickers, no processing logs.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import Polygon
import rasterio
import datetime
import math
import ee
import tempfile
import requests
import time
import warnings
import base64
import json
from datetime import date
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st

st.set_page_config(
    layout="wide",
    page_title="Water Quality Monitoring",
    page_icon="🌊"
)

import folium
from folium import plugins
from streamlit_folium import st_folium

# =============================================================================
# CONSTANTS
# =============================================================================
# --- Fixed processing thresholds (no longer user-configurable) -------------
CLOUD_THRESHOLD = 10          # % — fixed cloud coverage threshold (CLOUDY_PIXEL_PERCENTAGE)
CLOUD_PROB_THRESHOLD = 15      # per-pixel cloud probability cutoff

# Water body detection thresholds
NDWI_THRESHOLD_TURBIDITY = 0.1     # NDWI using B3, B12
NDWI_THRESHOLD_CHLOROPHYLL = 0.05  # NDWI using B3, B8

# Snow detection thresholds (preprocessing only — excludes snow from water)
NDSI_THRESHOLD = 0.42
SNOW_B11_THRESHOLD = 0.1

# Parameter identifiers (internal use only — never shown as a user choice)
PARAM_TURBIDITY = "Turbidity (NDTI)"
PARAM_CHLOROPHYLL = "Chlorophyll Index"

# Chlorophyll visualization range
CHL_VMIN = 2
CHL_VMAX = 100

# Download settings
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2
DOWNLOAD_TIMEOUT = 120
CHUNK_SIZE = 8192
MIN_FILE_SIZE = 10000

# Status constants
STATUS_NO_DATA = "no_data"
STATUS_COMPLETE = "complete"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

# =============================================================================
# Session State Initialization
# =============================================================================
if 'drawn_polygons' not in st.session_state:
    st.session_state.drawn_polygons = []
if 'last_drawn_polygon' not in st.session_state:
    st.session_state.last_drawn_polygon = None
if 'ee_initialized' not in st.session_state:
    st.session_state.ee_initialized = False
if 'current_temp_dir' not in st.session_state:
    st.session_state.current_temp_dir = None
if 'downloaded_months' not in st.session_state:
    # nested by parameter: {PARAM_TURBIDITY: {...}, PARAM_CHLOROPHYLL: {...}}
    st.session_state.downloaded_months = {PARAM_TURBIDITY: {}, PARAM_CHLOROPHYLL: {}}
if 'month_statuses' not in st.session_state:
    st.session_state.month_statuses = {PARAM_TURBIDITY: {}, PARAM_CHLOROPHYLL: {}}
if 'results' not in st.session_state:
    # nested by parameter
    st.session_state.results = {PARAM_TURBIDITY: [], PARAM_CHLOROPHYLL: []}
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'selected_region_index' not in st.session_state:
    st.session_state.selected_region_index = 0
if 'processing_in_progress' not in st.session_state:
    st.session_state.processing_in_progress = False
if 'processing_config' not in st.session_state:
    st.session_state.processing_config = None
if 'mean_data' not in st.session_state:
    st.session_state.mean_data = {PARAM_TURBIDITY: {}, PARAM_CHLOROPHYLL: {}}
if 'download_summary' not in st.session_state:
    # simple end-user facing summary: {PARAM_TURBIDITY: (downloaded, available), ...}
    st.session_state.download_summary = {}
if 'last_progress_heartbeat' not in st.session_state:
    # Updated every time processing makes measurable progress. Used to detect
    # a "stuck" processing_in_progress flag left over from a dropped
    # connection (the Streamlit session is killed mid-run, so the normal
    # finally: block that resets the flag never executes).
    st.session_state.last_progress_heartbeat = None

# --- Stale-processing watchdog --------------------------------------------
# If the app thinks a run is "in progress" but no heartbeat has been
# recorded for longer than this, the previous run almost certainly died
# from a dropped browser/server connection rather than still being active
# (the heartbeat is refreshed at fine granularity — every retry attempt and
# every downloaded chunk — so a genuinely alive run never goes this long
# without updating it). Auto-clear the flag so the user is never
# permanently locked out of the buttons.
STALE_PROCESSING_TIMEOUT_SECONDS = 45

if 'show_recovery_notice' not in st.session_state:
    st.session_state.show_recovery_notice = False

if st.session_state.processing_in_progress:
    heartbeat = st.session_state.last_progress_heartbeat
    if heartbeat is None or (time.time() - heartbeat) > STALE_PROCESSING_TIMEOUT_SECONDS:
        st.session_state.processing_in_progress = False
        st.session_state.show_recovery_notice = True


# =============================================================================
# Earth Engine Authentication
# =============================================================================
@st.cache_resource
def initialize_earth_engine():
    """Initialize Earth Engine"""
    try:
        ee.Initialize()
        return True, "Earth Engine initialized"
    except Exception:
        try:
            base64_key = os.environ.get('GOOGLE_EARTH_ENGINE_KEY_BASE64')

            if base64_key:
                key_json = base64.b64decode(base64_key).decode()
                key_data = json.loads(key_json)

                key_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
                with open(key_file.name, 'w') as f:
                    json.dump(key_data, f)

                credentials = ee.ServiceAccountCredentials(key_data['client_email'], key_file.name)
                ee.Initialize(credentials)
                os.unlink(key_file.name)
                return True, "Authenticated with Service Account"
            else:
                ee.Authenticate()
                ee.Initialize()
                return True, "Authenticated"
        except Exception as auth_error:
            return False, f"Auth failed: {str(auth_error)}"


# =============================================================================
# Helper Functions
# =============================================================================
def get_utm_zone(longitude):
    return math.floor((longitude + 180) / 6) + 1


def validate_geotiff_file(file_path, expected_bands=1):
    """Validate that a GeoTIFF file is complete and readable."""
    try:
        if not os.path.exists(file_path):
            return False, "File does not exist"

        file_size = os.path.getsize(file_path)
        if file_size < MIN_FILE_SIZE:
            return False, f"File too small ({file_size} bytes)"

        with rasterio.open(file_path) as src:
            if src.count < expected_bands:
                return False, f"Wrong band count ({src.count}, expected {expected_bands})"

        return True, "File is valid"

    except Exception as e:
        return False, f"Validation error: {str(e)}"


# =============================================================================
# Water Quality Calculation (GEE Server-Side)
# =============================================================================
def create_water_quality_collection(aoi, start_date, end_date, parameter_type, cloudy_pixel_percentage=CLOUD_THRESHOLD):
    """
    Create water quality collection for either Turbidity or Chlorophyll.

    Snow detection is used as a PREPROCESSING step to exclude snow/ice pixels
    from water detection. Snow mask is never downloaded or shown to the user.

    For TURBIDITY (NDTI):
    1. Link S2_SR with S2_CLOUD_PROBABILITY
    2. Apply cloud mask (probability < 15)
    3. Calculate NDSI for snow detection: (B3 - B11) / (B3 + B11)
    4. Create snow mask: NDSI > 0.42 AND B11 > 0.1
    5. Calculate NDWI for water body detection: (B3 - B12) / (B3 + B12) > 0.1, excluding snow
    6. Calculate NDTI (turbidity index): (B4 - B3) / (B4 + B3)

    For CHLOROPHYLL:
    1. Link S2_SR with S2_CLOUD_PROBABILITY
    2. Apply cloud mask (probability < 15)
    3. Calculate NDSI for snow detection: (B3 - B11) / (B3 + B11)
    4. Create snow mask: NDSI > 0.42 AND B11 > 0.1
    5. Calculate NDWI for water body detection: (B3 - B8) / (B3 + B8) >= 0.05, excluding snow
    6. Calculate Chlorophyll Index: 4.26 * (B3/B1)^3.94
    """
    s2_sr = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
             .filterBounds(aoi)
             .filterDate(start_date, end_date)
             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloudy_pixel_percentage)))

    s2_cloud_prob = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                     .filterBounds(aoi)
                     .filterDate(start_date, end_date))

    join_filter = ee.Filter.equals(leftField='system:index', rightField='system:index')
    joined = ee.Join.saveFirst('cloud_probability').apply(
        primary=s2_sr, secondary=s2_cloud_prob, condition=join_filter
    )

    def add_cloud_band(feature):
        img = ee.Image(feature)
        cloud_prob_img = ee.Image(img.get('cloud_probability'))
        return img.addBands(cloud_prob_img.select('probability'))

    s2_joined = ee.ImageCollection(joined.map(add_cloud_band))

    if parameter_type == PARAM_TURBIDITY:
        def calculate_turbidity(img):
            cloud = img.select('probability')
            cloud_free = cloud.lt(CLOUD_PROB_THRESHOLD)

            sr = img.select(['B2', 'B3', 'B4', 'B11', 'B12']).multiply(0.0001)

            ndsi = sr.normalizedDifference(['B3', 'B11']).rename('ndsi')
            is_snow = ndsi.gt(NDSI_THRESHOLD).And(sr.select('B11').gt(SNOW_B11_THRESHOLD))

            ndwi = sr.normalizedDifference(['B3', 'B12']).rename('ndwi')
            water_body = ndwi.gt(NDWI_THRESHOLD_TURBIDITY).And(is_snow.Not())

            ndti = sr.normalizedDifference(['B4', 'B3']).rename('wq_index')

            wq_masked = ndti.updateMask(cloud_free).updateMask(water_body)

            rgb = sr.select(['B4', 'B3', 'B2'])

            combined = (wq_masked
                       .addBands(rgb)
                       .addBands(water_body.rename('water_mask')))

            return combined.clip(aoi).copyProperties(img, ['system:time_start'])

        return s2_joined.map(calculate_turbidity)

    else:  # CHLOROPHYLL
        def calculate_chlorophyll(img):
            cloud = img.select('probability')
            cloud_free = cloud.lt(CLOUD_PROB_THRESHOLD)

            sr = img.select(['B1', 'B2', 'B3', 'B4', 'B8', 'B11']).multiply(0.0001)

            ndsi = sr.normalizedDifference(['B3', 'B11']).rename('ndsi')
            is_snow = ndsi.gt(NDSI_THRESHOLD).And(sr.select('B11').gt(SNOW_B11_THRESHOLD))

            ndwi = sr.normalizedDifference(['B3', 'B8']).rename('ndwi')
            water_body = ndwi.gte(NDWI_THRESHOLD_CHLOROPHYLL).And(is_snow.Not())

            chl_index = sr.expression(
                "4.26 * pow((B03 / B01), 3.94)",
                {
                    "B03": sr.select('B3'),
                    "B01": sr.select('B1')
                }
            ).rename('wq_index')

            wq_masked = chl_index.updateMask(cloud_free).updateMask(water_body)

            rgb = sr.select(['B4', 'B3', 'B2'])

            combined = (wq_masked
                       .addBands(rgb)
                       .addBands(water_body.rename('water_mask')))

            return combined.clip(aoi).copyProperties(img, ['system:time_start'])

        return s2_joined.map(calculate_chlorophyll)


def get_monthly_composite(wq_collection, aoi, year, month):
    """Create monthly composite from water quality collection."""
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, 'month')

    monthly = wq_collection.filterDate(start, end)
    count = monthly.size().getInfo()

    if count == 0:
        return None, 0, "No images"

    composite = monthly.median()

    stats = composite.select('wq_index').reduceRegion(
        reducer=ee.Reducer.mean().combine(
            ee.Reducer.count(), sharedInputs=True
        ).combine(
            ee.Reducer.minMax(), sharedInputs=True
        ),
        geometry=aoi,
        scale=10,
        maxPixels=1e13
    )

    return composite, count, stats


# =============================================================================
# Download Functions
# =============================================================================
def download_band_with_retry(image, band, aoi, output_path, scale=10):
    """Download a single band with retry mechanism."""
    try:
        region = aoi.bounds().getInfo()['coordinates']
    except Exception as e:
        return False, f"AOI bounds error: {e}"

    temp_path = output_path + '.tmp'
    if os.path.exists(temp_path):
        os.remove(temp_path)

    if os.path.exists(output_path):
        is_valid, msg = validate_geotiff_file(output_path, expected_bands=1)
        if is_valid:
            return True, "cached"
        os.remove(output_path)

    last_error = None

    for attempt in range(MAX_RETRIES):
        # Refresh the heartbeat at the start of every attempt — this is the
        # finest-grained unit of real work, much smaller than a full month,
        # so the stale-processing watchdog (see session-state init) never
        # mistakes a slow-but-alive download for an abandoned one.
        st.session_state.last_progress_heartbeat = time.time()
        try:
            url = image.select(band).getDownloadURL({
                'scale': scale, 'region': region, 'format': 'GEO_TIFF', 'bands': [band]
            })

            response = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)

            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')
                if 'text/html' in content_type:
                    last_error = "GEE rate limit"
                    raise Exception(last_error)

                downloaded_size = 0
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            st.session_state.last_progress_heartbeat = time.time()

                if downloaded_size < MIN_FILE_SIZE:
                    last_error = f"File too small ({downloaded_size} bytes)"
                    raise Exception(last_error)

                is_valid, msg = validate_geotiff_file(temp_path, expected_bands=1)
                if is_valid:
                    os.replace(temp_path, output_path)
                    return True, "success"
                else:
                    last_error = f"Validation failed: {msg}"
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    raise Exception(last_error)
            else:
                last_error = f"HTTP {response.status_code}"
                raise Exception(last_error)

        except requests.exceptions.Timeout:
            last_error = "Timeout"
        except requests.exceptions.ConnectionError:
            last_error = "Connection error"
        except Exception as e:
            if last_error is None:
                last_error = str(e)

        for f in [output_path, temp_path]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass

        if attempt < MAX_RETRIES - 1:
            wait_time = RETRY_DELAY_BASE ** (attempt + 1)
            time.sleep(wait_time)

    return False, last_error


def download_monthly_data(composite, aoi, temp_dir, month_name, param_short, scale=10):
    """
    Download monthly composite (Water Quality Index + RGB bands).
    Snow mask is NOT downloaded — used only server-side in GEE.
    No UI status is written here; progress is summarized at a higher level.
    """
    wq_path = os.path.join(temp_dir, f"wq_index_{param_short}_{month_name}.tif")
    rgb_path = os.path.join(temp_dir, f"rgb_{param_short}_{month_name}.tif")

    wq_valid, _ = validate_geotiff_file(wq_path, expected_bands=1)
    rgb_valid, _ = validate_geotiff_file(rgb_path, expected_bands=3)

    if wq_valid and rgb_valid:
        return wq_path, rgb_path, STATUS_COMPLETE, "Cached"

    try:
        success, msg = download_band_with_retry(composite, 'wq_index', aoi, wq_path, scale)
        if not success:
            return None, None, STATUS_FAILED, f"WQ Index download failed: {msg}"

        bands_dir = os.path.join(temp_dir, f"bands_{param_short}_{month_name}")
        os.makedirs(bands_dir, exist_ok=True)

        rgb_bands = ['B4', 'B3', 'B2']
        band_files = []

        for band in rgb_bands:
            band_file = os.path.join(bands_dir, f"{band}.tif")
            success, msg = download_band_with_retry(composite, band, aoi, band_file, scale)

            if not success:
                return None, None, STATUS_FAILED, f"RGB {band} download failed: {msg}"

            band_files.append(band_file)

        with rasterio.open(band_files[0]) as src:
            meta = src.meta.copy()
        meta.update(count=3)

        with rasterio.open(rgb_path, 'w', **meta) as dst:
            for i, band_file in enumerate(band_files):
                with rasterio.open(band_file) as src:
                    dst.write(src.read(1), i+1)

        return wq_path, rgb_path, STATUS_COMPLETE, "Downloaded"

    except Exception as e:
        return None, None, STATUS_FAILED, f"Error: {str(e)}"


# =============================================================================
# Visualization Functions
# =============================================================================
def create_turbidity_colormap():
    colors = ['#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF8000', '#FF0000']
    return LinearSegmentedColormap.from_list('turbidity', colors, N=256)


def create_chlorophyll_colormap():
    colors = ['#9400D3', '#4B0082', '#0000FF', '#00FF00', '#FFFF00', '#FF7F00', '#FF0000']
    return LinearSegmentedColormap.from_list('chlorophyll', colors, N=256)


def generate_thumbnails(wq_path, rgb_path, month_name, parameter_type, max_size=300):
    """Generate RGB and water quality index thumbnails."""
    try:
        with rasterio.open(wq_path) as src:
            wq_data = src.read(1)

        with rasterio.open(rgb_path) as src:
            red = src.read(1)
            green = src.read(2)
            blue = src.read(3)

        rgb = np.stack([red, green, blue], axis=-1)
        rgb = np.nan_to_num(rgb, nan=0.0)

        def percentile_stretch(band, lower=2, upper=98):
            valid = band[band > 0]
            if len(valid) == 0:
                return np.zeros_like(band, dtype=np.uint8)
            p_low = np.percentile(valid, lower)
            p_high = np.percentile(valid, upper)
            if p_high <= p_low:
                p_high = p_low + 0.001
            stretched = np.clip((band - p_low) / (p_high - p_low), 0, 1)
            return (stretched * 255).astype(np.uint8)

        rgb_uint8 = np.zeros_like(rgb, dtype=np.uint8)
        for i in range(3):
            rgb_uint8[:, :, i] = percentile_stretch(rgb[:, :, i])

        wq_valid = np.nan_to_num(wq_data, nan=np.nan)

        valid_wq = wq_valid[~np.isnan(wq_valid) & (wq_valid != 0)]
        mean_value = np.nanmean(valid_wq) if len(valid_wq) > 0 else np.nan
        valid_pixel_count = len(valid_wq)
        total_pixels = wq_data.size
        water_coverage = (valid_pixel_count / total_pixels) * 100 if total_pixels > 0 else 0

        if parameter_type == PARAM_TURBIDITY:
            cmap = create_turbidity_colormap()
            wq_normalized = np.clip((wq_valid + 0.3) / 0.6, 0, 1)
        else:
            cmap = create_chlorophyll_colormap()
            wq_normalized = np.clip((wq_valid - CHL_VMIN) / (CHL_VMAX - CHL_VMIN), 0, 1)

        wq_normalized = np.nan_to_num(wq_normalized, nan=0)

        wq_colored = cmap(wq_normalized)[:, :, :3]
        wq_uint8 = (wq_colored * 255).astype(np.uint8)

        water_mask = (~np.isnan(wq_valid)) & (wq_valid != 0)
        for i in range(3):
            wq_uint8[:, :, i] = np.where(water_mask, wq_uint8[:, :, i], 50)

        pil_rgb = Image.fromarray(rgb_uint8, mode='RGB')
        pil_wq = Image.fromarray(wq_uint8, mode='RGB')

        h, w = pil_rgb.size[1], pil_rgb.size[0]
        if h > max_size or w > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            pil_rgb = pil_rgb.resize((new_w, new_h), Image.LANCZOS)
            pil_wq = pil_wq.resize((new_w, new_h), Image.LANCZOS)

        return {
            'rgb_image': pil_rgb,
            'wq_image': pil_wq,
            'month_name': month_name,
            'mean_value': mean_value,
            'water_coverage': water_coverage,
            'valid_pixels': valid_pixel_count,
            'parameter_type': parameter_type
        }

    except Exception:
        # Quietly skip a month that fails to render rather than surfacing
        # remote-sensing error internals to a non-technical user.
        return None


# =============================================================================
# Main Processing Pipeline (silent — no remote-sensing internals shown)
# =============================================================================
def process_single_parameter(aoi, start_date, end_date, parameter_type, temp_dir,
                              cloudy_pixel_percentage=CLOUD_THRESHOLD, scale=10,
                              progress_callback=None, resume=False):
    """
    Run the full pipeline for one parameter (NDTI or Chlorophyll-a):
    cloud filtering -> snow masking -> waterbody extraction -> index calculation
    -> download -> thumbnail generation.

    RESUME / CONNECTION-DROP SAFETY:
    Every month's outcome (downloaded paths or a no-data/failed status) is
    written into st.session_state immediately after that month finishes, not
    only at the end of the function. This means that if the browser tab loses
    its connection, the process is stopped, or the app reruns mid-way, the
    months already completed are not lost — they are recorded in session
    state and, when resume=True, are skipped entirely on the next run rather
    than being re-downloaded from scratch.

    progress_callback(fraction), if provided, is called with a float in [0, 1]
    representing how far this single parameter's processing has progressed
    (covers both the per-month download loop and the thumbnail-generation loop).

    Returns: (results_list, downloaded_count, available_count)
    No technical logs are written to the UI; this function is silent.
    """
    param_short = "turbidity" if parameter_type == PARAM_TURBIDITY else "chlorophyll"

    start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    total_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)

    wq_collection = create_water_quality_collection(
        aoi, start_date, end_date, parameter_type, cloudy_pixel_percentage
    )

    month_infos = []
    for month_index in range(total_months):
        year = start_dt.year + (start_dt.month - 1 + month_index) // 12
        month = (start_dt.month - 1 + month_index) % 12 + 1
        month_infos.append({'month_name': f"{year}-{month:02d}", 'year': year, 'month': month})

    # Reserve 85% of this parameter's progress for fetching/downloading months,
    # and 15% for generating the visual thumbnails afterward.
    n_months = max(len(month_infos), 1)
    DOWNLOAD_SHARE = 0.85

    # Pull any progress already recorded in session state from a previous,
    # interrupted run of this exact parameter.
    downloaded_months = {}
    month_statuses = {}
    if resume:
        for month_name, paths in st.session_state.downloaded_months.get(parameter_type, {}).items():
            wq_valid, _ = validate_geotiff_file(paths.get('wq_index', ''), 1)
            rgb_valid, _ = validate_geotiff_file(paths.get('rgb', ''), 3)
            if wq_valid and rgb_valid:
                downloaded_months[month_name] = paths
        month_statuses = dict(st.session_state.month_statuses.get(parameter_type, {}))

    available_count = sum(
        1 for s in month_statuses.values() if s.get('status') != STATUS_SKIPPED
    )

    for idx, month_info in enumerate(month_infos):
        month_name = month_info['month_name']

        # Already resolved in a previous run (this session) — skip entirely.
        if month_name in downloaded_months or month_name in month_statuses:
            if progress_callback:
                progress_callback(((idx + 1) / n_months) * DOWNLOAD_SHARE)
            continue

        composite, count, stats = get_monthly_composite(
            wq_collection, aoi, month_info['year'], month_info['month']
        )

        if composite is None or count == 0:
            month_statuses[month_name] = {'status': STATUS_NO_DATA}
            st.session_state.month_statuses.setdefault(parameter_type, {})[month_name] = \
                month_statuses[month_name]
        else:
            available_count += 1

            wq_path, rgb_path, status, message = download_monthly_data(
                composite, aoi, temp_dir, month_name, param_short, scale
            )

            month_statuses[month_name] = {'status': status}
            st.session_state.month_statuses.setdefault(parameter_type, {})[month_name] = \
                month_statuses[month_name]

            if status == STATUS_COMPLETE:
                downloaded_months[month_name] = {'wq_index': wq_path, 'rgb': rgb_path}
                # Checkpoint immediately so this month survives a dropped
                # connection or an interrupted run.
                st.session_state.downloaded_months.setdefault(parameter_type, {})[month_name] = \
                    downloaded_months[month_name]

        if progress_callback:
            progress_callback(((idx + 1) / n_months) * DOWNLOAD_SHARE)

    results = []
    mean_data = {}

    sorted_months = sorted(downloaded_months.keys())
    n_thumbs = max(len(sorted_months), 1)

    for t_idx, month_name in enumerate(sorted_months):
        paths = downloaded_months[month_name]
        thumb = generate_thumbnails(paths['wq_index'], paths['rgb'], month_name, parameter_type)
        if thumb:
            results.append(thumb)
            mean_data[month_name] = {'mean': thumb['mean_value'], 'coverage': thumb['water_coverage']}

        if progress_callback:
            progress_callback(DOWNLOAD_SHARE + ((t_idx + 1) / n_thumbs) * (1 - DOWNLOAD_SHARE))

    if progress_callback:
        progress_callback(1.0)

    return results, mean_data, len(downloaded_months), available_count


def run_full_analysis(aoi, start_date, end_date, cloudy_pixel_percentage=CLOUD_THRESHOLD, scale=10,
                       resume_mode=False):
    """
    Automatically runs preprocessing + both index calculations (NDTI, then
    Chlorophyll-a) in sequence. Displays only a simple, user-friendly summary
    plus a progress bar labeled with the calculation currently running.

    If resume_mode=True, any months already completed and checkpointed in
    session state (from a prior run that was interrupted, e.g. by a dropped
    internet connection) are skipped rather than re-downloaded.
    """
    if st.session_state.current_temp_dir is None or not os.path.exists(st.session_state.current_temp_dir):
        st.session_state.current_temp_dir = tempfile.mkdtemp()
    temp_dir = st.session_state.current_temp_dir

    summary_placeholder = st.empty()
    progress_label = st.empty()
    progress_bar = st.progress(0)
    download_summary = {}

    # Each parameter's calculation occupies half of the overall bar.
    LABEL_TURBIDITY = "🌊 در حال محاسبه شاخص کدورت (NDTI)..."
    LABEL_CHLOROPHYLL = "🌿 در حال محاسبه شاخص کلروفیل..."

    def make_progress_updater(label, offset):
        def update(fraction_done_within_param):
            overall_fraction = offset + (fraction_done_within_param * 0.5)
            progress_bar.progress(min(overall_fraction, 1.0))
            progress_label.markdown(f"**{label}** — {int(min(fraction_done_within_param, 1.0) * 100)}٪")
            # Refresh the heartbeat so the stale-processing watchdog knows
            # this run is genuinely still active, not abandoned mid-flight.
            st.session_state.last_progress_heartbeat = time.time()
        return update

    # --- Turbidity (NDTI) ---
    turb_progress = make_progress_updater(LABEL_TURBIDITY, offset=0.0)
    turb_progress(0)
    turb_results, turb_mean, turb_downloaded, turb_available = process_single_parameter(
        aoi, start_date, end_date, PARAM_TURBIDITY, temp_dir, cloudy_pixel_percentage, scale,
        progress_callback=turb_progress, resume=resume_mode
    )
    st.session_state.results[PARAM_TURBIDITY] = turb_results
    st.session_state.mean_data[PARAM_TURBIDITY] = turb_mean
    download_summary[PARAM_TURBIDITY] = (turb_downloaded, turb_available)

    summary_placeholder.info(
        f"🌊 شاخص کدورت: {turb_downloaded} تصویر از {turb_available} تصویر موجود دریافت شد."
    )

    # --- Chlorophyll-a ---
    chl_progress = make_progress_updater(LABEL_CHLOROPHYLL, offset=0.5)
    chl_progress(0)
    chl_results, chl_mean, chl_downloaded, chl_available = process_single_parameter(
        aoi, start_date, end_date, PARAM_CHLOROPHYLL, temp_dir, cloudy_pixel_percentage, scale,
        progress_callback=chl_progress, resume=resume_mode
    )
    st.session_state.results[PARAM_CHLOROPHYLL] = chl_results
    st.session_state.mean_data[PARAM_CHLOROPHYLL] = chl_mean
    download_summary[PARAM_CHLOROPHYLL] = (chl_downloaded, chl_available)

    progress_label.markdown("**✅ پایش کیفیت آب تکمیل شد**")
    progress_bar.progress(1.0)
    time.sleep(0.4)
    progress_label.empty()
    progress_bar.empty()

    st.session_state.download_summary = download_summary

    has_any_results = bool(turb_results) or bool(chl_results)
    return has_any_results


# =============================================================================
# Legend + Management Guidance (Persian) — always visible, no jargon
# =============================================================================
def render_turbidity_guidance_panel():
    """Permanently visible legend + management guidance for Turbidity (NDTI)."""
    st.markdown("### 🎨 راهنمای رنگ و تفسیر مدیریتی — شاخص کدورت آب")

    col_legend, col_text = st.columns([1, 2])

    with col_legend:
        fig, ax = plt.subplots(figsize=(5, 0.45))
        cmap = create_turbidity_colormap()
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        ax.imshow(gradient, aspect='auto', cmap=cmap)
        ax.set_xticks([0, 128, 255])
        ax.set_xticklabels(['آب شفاف', 'متوسط', 'بسیار کدر'])
        ax.set_yticks([])
        st.pyplot(fig)
        plt.close(fig)

    with col_text:
        st.markdown(
            """
**افزایش کدورت آب:**
- کاهش کیفیت آب قابل استفاده برای کشاورزی و مصارف شهری
- افزایش هزینه‌های تصفیه آب
- کاهش نفوذ نور به آب و آسیب به اکوسیستم و آبزیان
- نشانه احتمالی فرسایش خاک، رسوب‌گذاری یا آلودگی در حوضه آبریز

**کاهش کدورت آب:**
- بهبود کیفیت آب و کاهش هزینه‌های تصفیه
- شرایط مطلوب‌تر برای زیست‌بوم آبی و ماهی‌پروری
- نشانه کنترل مؤثر فرسایش و مدیریت بهتر حوضه آبریز

**چرا پایش این شاخص مهم است؟**
پایش روند کدورت به مدیران امکان می‌دهد قبل از وقوع بحران (مانند رسوب‌گذاری در سدها یا
افزایش هزینه تصفیه) اقدام کنند. تغییرات ناگهانی معمولاً نشانه رویدادهایی مانند بارش‌های
شدید، فعالیت‌های عمرانی در بالادست یا تخلیه پساب است و نیازمند بررسی سریع است.
            """
        )


def render_chlorophyll_guidance_panel():
    """Permanently visible legend + management guidance for Chlorophyll-a."""
    st.markdown("### 🎨 راهنمای رنگ و تفسیر مدیریتی — شاخص کلروفیل")

    col_legend, col_text = st.columns([1, 2])

    with col_legend:
        fig, ax = plt.subplots(figsize=(5, 0.45))
        cmap = create_chlorophyll_colormap()
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        ax.imshow(gradient, aspect='auto', cmap=cmap)
        ax.set_xticks([0, 128, 255])
        ax.set_xticklabels(['کم', 'متوسط', 'بالا (شکوفایی جلبکی)'])
        ax.set_yticks([])
        st.pyplot(fig)
        plt.close(fig)

    with col_text:
        st.markdown(
            """
**افزایش غلظت کلروفیل:**
- احتمال شکوفایی جلبکی و کاهش کیفیت آب آشامیدنی
- افزایش هزینه‌های تصفیه و خطر مسدود شدن فیلترها
- کاهش اکسیژن محلول در آب و خطر برای آبزیان و ماهی‌پروری
- در موارد شدید، احتمال سمیت آب و توقف موقت برداشت آب

**کاهش غلظت کلروفیل:**
- بهبود کیفیت آب و کاهش ریسک‌های بهداشتی
- کاهش هزینه‌های عملیاتی تصفیه‌خانه
- شرایط پایدارتر برای اکوسیستم آبی

**چرا پایش این شاخص مهم است؟**
افزایش ناگهانی کلروفیل معمولاً پیش‌نشانگر شکوفایی جلبکی است که در صورت عدم اقدام به‌موقع
می‌تواند منجر به توقف تأمین آب، هزینه‌های اضطراری تصفیه یا آسیب به صنعت ماهی‌پروری شود.
پایش منظم این شاخص امکان برنامه‌ریزی پیشگیرانه و کاهش ریسک اقتصادی را فراهم می‌کند.
            """
        )


# =============================================================================
# Display: imagery, time-series, and statistics for one parameter
# =============================================================================
def display_side_by_side_imagery(results, parameter_type):
    """Side-by-side processed index image and corresponding RGB image."""
    if not results:
        st.info("داده‌ای برای نمایش در این بازه زمانی وجود ندارد.")
        return

    param_short = "NDTI" if parameter_type == PARAM_TURBIDITY else "Chl-a"

    for r in results:
        if parameter_type == PARAM_TURBIDITY:
            mean_str = f"{r['mean_value']:.4f}" if not np.isnan(r['mean_value']) else "بدون داده"
        else:
            mean_str = f"{r['mean_value']:.2f}" if not np.isnan(r['mean_value']) else "بدون داده"

        cols = st.columns(2)
        cols[0].image(r['wq_image'], caption=f"{r['month_name']} — {param_short}: {mean_str}", use_container_width=True)
        cols[1].image(r['rgb_image'], caption=f"{r['month_name']} — تصویر طبیعی (RGB)", use_container_width=True)


def display_time_series_chart(results, parameter_type):
    """Time series chart of mean index values directly under the imagery."""
    if not results:
        return

    param_short = "NDTI" if parameter_type == PARAM_TURBIDITY else "Chl-a"
    param_unit = "" if parameter_type == PARAM_TURBIDITY else " (µg/L)"
    chart_title = "روند زمانی کدورت آب" if parameter_type == PARAM_TURBIDITY else "روند زمانی کلروفیل"

    months = []
    mean_values = []
    coverage_values = []

    for r in results:
        months.append(r['month_name'])
        mean_values.append(r['mean_value'] if not np.isnan(r['mean_value']) else 0)
        coverage_values.append(r['water_coverage'])

    if not months:
        return

    valid_values = [m for m in mean_values if m != 0]

    fig, ax1 = plt.subplots(figsize=(12, 5))

    color1 = '#1f77b4' if parameter_type == PARAM_TURBIDITY else '#228B22'
    ax1.set_xlabel('Month')
    ax1.set_ylabel(f'Mean {param_short}{param_unit}', color=color1)

    if valid_values:
        ax1.plot(months, mean_values, 'o-', color=color1, linewidth=2, markersize=8, label=f'Mean {param_short}')
        ax1.tick_params(axis='y', labelcolor=color1)

        if parameter_type == PARAM_TURBIDITY:
            ax1.set_ylim(min(mean_values) - 0.02, max(mean_values) + 0.02)
            ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Neutral (NDTI=0)')
        else:
            ax1.set_ylim(0, max(mean_values) * 1.2)
    else:
        ax1.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax1.transAxes, fontsize=12)

    ax1.set_xticklabels(months, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)

    ax1_twin = ax1.twinx()
    color2 = '#2ca02c'
    ax1_twin.set_ylabel('Water Coverage (%)', color=color2)
    ax1_twin.bar(months, coverage_values, alpha=0.3, color=color2, label='Water Coverage')
    ax1_twin.tick_params(axis='y', labelcolor=color2)
    ax1_twin.set_ylim(0, max(coverage_values) * 1.3 if max(coverage_values) > 0 else 100)

    ax1.set_title(chart_title, fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def display_statistics_summary(results, parameter_type):
    """Statistics summary, contained within the same parameter page."""
    if not results:
        return

    param_short = "NDTI" if parameter_type == PARAM_TURBIDITY else "Chl-a"

    months = [r['month_name'] for r in results]
    mean_values = [r['mean_value'] if not np.isnan(r['mean_value']) else 0 for r in results]
    coverage_values = [r['water_coverage'] for r in results]
    valid_values = [m for m in mean_values if m != 0]

    st.markdown("#### 📈 خلاصه آماری")

    col1, col2, col3, col4 = st.columns(4)

    if valid_values:
        if parameter_type == PARAM_TURBIDITY:
            col1.metric(f"میانگین {param_short}", f"{np.mean(valid_values):.4f}")
            col2.metric(f"حداکثر {param_short}", f"{np.max(valid_values):.4f}")
            col3.metric(f"حداقل {param_short}", f"{np.min(valid_values):.4f}")
        else:
            col1.metric(f"میانگین {param_short}", f"{np.mean(valid_values):.2f}")
            col2.metric(f"حداکثر {param_short}", f"{np.max(valid_values):.2f}")
            col3.metric(f"حداقل {param_short}", f"{np.min(valid_values):.2f}")
    else:
        col1.metric(f"میانگین {param_short}", "—")
        col2.metric(f"حداکثر {param_short}", "—")
        col3.metric(f"حداقل {param_short}", "—")

    col4.metric("میانگین پوشش آب", f"{np.mean(coverage_values):.1f}%")

    with st.expander("📋 جدول داده‌های ماهانه"):
        import pandas as pd

        if parameter_type == PARAM_TURBIDITY:
            value_col = [f"{v:.4f}" if v != 0 else "—" for v in mean_values]
        else:
            value_col = [f"{v:.2f}" if v != 0 else "—" for v in mean_values]

        df = pd.DataFrame({
            'ماه': months,
            f'میانگین {param_short}': value_col,
            'پوشش آب (%)': [f"{v:.1f}" for v in coverage_values]
        })
        st.dataframe(df, use_container_width=True)


def render_parameter_page(parameter_type):
    """
    Full page for one parameter, in the required order:
    1. Legend + Management Guidance Panel
    2. Side-by-side imagery
    3. Time-series chart
    4. Statistics Summary
    """
    if parameter_type == PARAM_TURBIDITY:
        render_turbidity_guidance_panel()
    else:
        render_chlorophyll_guidance_panel()

    st.divider()

    results = st.session_state.results.get(parameter_type, [])

    if not results:
        st.info("برای مشاهده نتایج، ابتدا یک منطقه را انتخاب و پایش را اجرا کنید.")
        return

    st.markdown("#### 🖼️ تصاویر پردازش‌شده")
    display_side_by_side_imagery(results, parameter_type)

    st.divider()
    display_time_series_chart(results, parameter_type)

    st.divider()
    display_statistics_summary(results, parameter_type)


# =============================================================================
# Main Application
# =============================================================================
def main():
    st.title("🌊 سامانه پایش کیفیت آب")
    st.markdown(
        "پایش خودکار **کدورت آب** و **غلظت کلروفیل** با استفاده از تصاویر ماهواره‌ای Sentinel-2."
    )

    if st.session_state.show_recovery_notice:
        st.warning(
            "⚠️ به نظر می‌رسد پایش قبلی به دلیل قطعی اینترنت یا اتصال متوقف شده بود. "
            "وضعیت بازنشانی شد — می‌توانید با دکمه «ادامه پایش» بدون از دست رفتن پیشرفت قبلی ادامه دهید، "
            "یا یک پایش جدید شروع کنید."
        )
        st.session_state.show_recovery_notice = False

    # Initialize Earth Engine
    ee_ok, ee_msg = initialize_earth_engine()
    if not ee_ok:
        st.error(ee_msg)
        st.stop()

    # ==========================================================================
    # Cache status (kept minimal — no technical internals)
    # ==========================================================================
    has_cache = bool(st.session_state.results.get(PARAM_TURBIDITY)) or \
                bool(st.session_state.results.get(PARAM_CHLOROPHYLL))

    has_partial = (
        any(
            bool(st.session_state.downloaded_months.get(p)) or bool(st.session_state.month_statuses.get(p))
            for p in (PARAM_TURBIDITY, PARAM_CHLOROPHYLL)
        )
        and not st.session_state.processing_complete
    )

    if has_cache:
        st.sidebar.success("✅ نتایج پایش قبلی موجود است")
    elif has_partial:
        st.sidebar.warning("⏸️ پایش ناتمام — می‌توانید با «ادامه پایش» ادامه دهید")
    else:
        st.sidebar.info("هنوز پایشی انجام نشده است")

    if st.session_state.processing_in_progress:
        st.sidebar.warning("⏳ در حال پردازش...")

    if st.sidebar.button("🗑️ پاک کردن نتایج"):
        st.session_state.downloaded_months = {PARAM_TURBIDITY: {}, PARAM_CHLOROPHYLL: {}}
        st.session_state.month_statuses = {PARAM_TURBIDITY: {}, PARAM_CHLOROPHYLL: {}}
        st.session_state.results = {PARAM_TURBIDITY: [], PARAM_CHLOROPHYLL: []}
        st.session_state.mean_data = {PARAM_TURBIDITY: {}, PARAM_CHLOROPHYLL: {}}
        st.session_state.download_summary = {}
        st.session_state.current_temp_dir = None
        st.session_state.processing_config = None
        st.session_state.processing_complete = False
        st.session_state.processing_in_progress = False
        st.session_state.last_progress_heartbeat = None
        st.rerun()

    # ==========================================================================
    # 1. Region selection
    # ==========================================================================
    st.header("1️⃣ انتخاب منطقه مورد نظر (بدنه آبی)")

    if not st.session_state.processing_in_progress:
        m = folium.Map(location=[35.6892, 51.3890], zoom_start=8)
        plugins.Draw(export=True, position='topleft', draw_options={
            'polyline': False, 'rectangle': True, 'polygon': True,
            'circle': False, 'marker': False, 'circlemarker': False
        }).add_to(m)
        folium.TileLayer(tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                        attr='Google', name='Satellite').add_to(m)
        folium.LayerControl().add_to(m)

        map_data = st_folium(m, width=800, height=500)

        if map_data and map_data.get('last_active_drawing'):
            geom = map_data['last_active_drawing'].get('geometry', {})
            if geom.get('type') == 'Polygon':
                st.session_state.last_drawn_polygon = Polygon(geom['coordinates'][0])
                st.success("✅ منطقه انتخاب شد")

        if st.button("💾 ذخیره منطقه"):
            if st.session_state.last_drawn_polygon:
                is_duplicate = False
                for existing in st.session_state.drawn_polygons:
                    if existing.equals(st.session_state.last_drawn_polygon):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    st.session_state.drawn_polygons.append(st.session_state.last_drawn_polygon)
                    st.success("✅ منطقه ذخیره شد!")
                    st.rerun()
                else:
                    st.warning("⚠️ این منطقه قبلاً ذخیره شده است")
            else:
                st.warning("⚠️ ابتدا یک منطقه را روی نقشه رسم کنید")
    else:
        st.info("🔒 نقشه در حین پردازش قفل است")

    if st.session_state.drawn_polygons:
        st.subheader("📍 مناطق ذخیره‌شده")
        for i, p in enumerate(st.session_state.drawn_polygons):
            c1, c2, c3 = st.columns([3, 1, 1])
            centroid = p.centroid
            c1.write(f"**منطقه {i+1}**: ~{p.area * 111 * 111:.2f} کیلومتر مربع")
            c2.write(f"مرکز: ({centroid.y:.4f}, {centroid.x:.4f})")
            if c3.button("🗑️", key=f"del_{i}", disabled=st.session_state.processing_in_progress):
                st.session_state.drawn_polygons.pop(i)
                if st.session_state.selected_region_index >= len(st.session_state.drawn_polygons):
                    st.session_state.selected_region_index = max(0, len(st.session_state.drawn_polygons) - 1)
                st.rerun()

    # ==========================================================================
    # 2. Time period
    # ==========================================================================
    st.header("2️⃣ بازه زمانی")
    c1, c2 = st.columns(2)
    start = c1.date_input("از تاریخ", value=date(2024, 1, 1), disabled=st.session_state.processing_in_progress)
    end = c2.date_input("تا تاریخ (غیرشامل)", value=date(2025, 1, 1), disabled=st.session_state.processing_in_progress)

    if start >= end:
        st.error("بازه تاریخ نامعتبر است")
        st.stop()

    months = (end.year - start.year) * 12 + (end.month - start.month)
    st.info(f"📅 بازه انتخابی: **{months} ماه**")

    # ==========================================================================
    # 3. Run analysis — fully automatic (preprocessing + both indices)
    # ==========================================================================
    st.header("3️⃣ اجرای پایش")

    selected_polygon = None

    if st.session_state.drawn_polygons:
        region_options = []
        for i, p in enumerate(st.session_state.drawn_polygons):
            area = p.area * 111 * 111
            region_options.append(f"منطقه {i+1} (~{area:.2f} کیلومتر مربع)")

        if st.session_state.selected_region_index >= len(st.session_state.drawn_polygons):
            st.session_state.selected_region_index = 0

        selected_idx = st.selectbox(
            "🎯 انتخاب منطقه",
            range(len(region_options)),
            format_func=lambda i: region_options[i],
            index=st.session_state.selected_region_index,
            disabled=st.session_state.processing_in_progress
        )

        st.session_state.selected_region_index = selected_idx
        selected_polygon = st.session_state.drawn_polygons[selected_idx]

    elif st.session_state.last_drawn_polygon is not None:
        selected_polygon = st.session_state.last_drawn_polygon
        st.info("ℹ️ استفاده از منطقه رسم‌شده (ذخیره‌نشده)")
    else:
        st.warning("⚠️ ابتدا یک منطقه را روی نقشه رسم کنید")

    st.caption("پس از اجرا، پیش‌پردازش (حذف ابر، حذف برف، استخراج بدنه آب)، سپس شاخص کدورت و شاخص کلروفیل به‌طور خودکار محاسبه می‌شوند.")

    # Detect whether there is leftover, partially-completed progress in this
    # session (e.g. the connection dropped or the run was stopped midway).
    has_partial_progress = any(
        bool(st.session_state.downloaded_months.get(p)) or bool(st.session_state.month_statuses.get(p))
        for p in (PARAM_TURBIDITY, PARAM_CHLOROPHYLL)
    ) and not st.session_state.processing_complete

    col_start, col_resume = st.columns(2)

    with col_start:
        start_btn = st.button(
            "🚀 شروع پایش",
            type="primary",
            disabled=st.session_state.processing_in_progress or selected_polygon is None
        )

    with col_resume:
        resume_btn = st.button(
            "🔄 ادامه پایش (در صورت قطعی اینترنت)",
            disabled=st.session_state.processing_in_progress or not has_partial_progress
        )

    if has_partial_progress and not st.session_state.processing_in_progress:
        st.caption("ℹ️ بخشی از پایش قبلی ناتمام مانده است (مثلاً به دلیل قطع اینترنت). با «ادامه پایش» می‌توانید بدون از دست رفتن پیشرفت قبلی، ادامه دهید.")

    should_process = False
    resume_mode = False

    if start_btn:
        # Fresh run: clear all prior progress and start from month 1.
        st.session_state.downloaded_months = {PARAM_TURBIDITY: {}, PARAM_CHLOROPHYLL: {}}
        st.session_state.month_statuses = {PARAM_TURBIDITY: {}, PARAM_CHLOROPHYLL: {}}
        st.session_state.results = {PARAM_TURBIDITY: [], PARAM_CHLOROPHYLL: []}
        st.session_state.mean_data = {PARAM_TURBIDITY: {}, PARAM_CHLOROPHYLL: {}}
        st.session_state.download_summary = {}
        st.session_state.current_temp_dir = None
        st.session_state.processing_complete = False

        st.session_state.processing_config = {
            'polygon_coords': list(selected_polygon.exterior.coords),
            'start_date': start.strftime('%Y-%m-%d'),
            'end_date': end.strftime('%Y-%m-%d'),
        }
        should_process = True
        resume_mode = False

    elif resume_btn:
        # Resume: keep whatever months are already checkpointed in session
        # state and only fetch the ones still missing.
        if st.session_state.processing_config is None:
            st.session_state.processing_config = {
                'polygon_coords': list(selected_polygon.exterior.coords),
                'start_date': start.strftime('%Y-%m-%d'),
                'end_date': end.strftime('%Y-%m-%d'),
            }
        should_process = True
        resume_mode = True

    if should_process:
        config = st.session_state.processing_config
        st.session_state.processing_in_progress = True
        # Seed the heartbeat now, before any Earth Engine calls — those
        # round-trips (e.g. checking image availability) happen before the
        # first download chunk and would otherwise look like silence to the
        # watchdog.
        st.session_state.last_progress_heartbeat = time.time()

        aoi = ee.Geometry.Polygon([[tuple(c) for c in config['polygon_coords']]])

        try:
            success = run_full_analysis(
                aoi,
                config['start_date'],
                config['end_date'],
                CLOUD_THRESHOLD,
                10,
                resume_mode=resume_mode
            )
            st.session_state.processing_complete = success
            if not success:
                st.warning("⚠️ داده‌ای برای این منطقه و بازه زمانی یافت نشد.")
        except Exception:
            st.error(
                "متأسفانه در حین پردازش خطایی رخ داد (مانند قطع اتصال اینترنت). "
                "نگران نباشید — پیشرفت تاکنون ذخیره شده و می‌توانید با دکمه «ادامه پایش» دوباره تلاش کنید."
            )
        finally:
            st.session_state.processing_in_progress = False
            st.rerun()

    # Simple, user-friendly download summary (persists after run)
    if st.session_state.download_summary:
        st.divider()
        turb_d, turb_a = st.session_state.download_summary.get(PARAM_TURBIDITY, (0, 0))
        chl_d, chl_a = st.session_state.download_summary.get(PARAM_CHLOROPHYLL, (0, 0))
        st.info(
            f"🌊 شاخص کدورت: {turb_d} تصویر از {turb_a} تصویر موجود دریافت شد.\n\n"
            f"🌿 شاخص کلروفیل: {chl_d} تصویر از {chl_a} تصویر موجود دریافت شد."
        )

    # ==========================================================================
    # Results — two dedicated tabs
    # ==========================================================================
    if st.session_state.processing_complete:
        st.divider()
        st.header("📊 نتایج پایش")

        tab_turbidity, tab_chlorophyll = st.tabs(["🌊 کدورت آب (NDTI)", "🌿 کلروفیل"])

        with tab_turbidity:
            render_parameter_page(PARAM_TURBIDITY)

        with tab_chlorophyll:
            render_parameter_page(PARAM_CHLOROPHYLL)


if __name__ == "__main__":
    main()
