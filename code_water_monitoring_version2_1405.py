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
import inspect
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

# branca + jinja2 are folium's own dependencies; they are used to inject the
# small piece of JavaScript that shows the live area while a region is being
# drawn. If they are ever unavailable the app simply runs without that label.
try:
    from branca.element import MacroElement as _BrancaMacroElement
    from jinja2 import Template as _JinjaTemplate
    _LIVE_AREA_SUPPORTED = True
except Exception:
    _LIVE_AREA_SUPPORTED = False

# =============================================================================
# CONSTANTS
# =============================================================================
# --- Fixed processing thresholds (no longer user-configurable) -------------
CLOUD_THRESHOLD = 10          # % — fixed cloud coverage threshold (CLOUDY_PIXEL_PERCENTAGE)
CLOUD_PROB_THRESHOLD = 15      # per-pixel cloud probability cutoff

# Water body detection threshold
# AWEIsh (Automated Water Extraction Index, shadow variant) — shared by both
# indices so NDTI and NDCI are computed on the exact same water mask.
# AWEIsh = Blue + 2.5*Green - 1.5*(NIR + SWIR1) - 0.25*SWIR2
AWEI_THRESHOLD = 0.05

# Snow detection thresholds (preprocessing only — excludes snow from water)
NDSI_THRESHOLD = 0.39
SNOW_B11_THRESHOLD = 0.1  # kept for reference; no longer used by is_snow (see below)

# MODIS-heritage water/snow discrimination test (Hall et al., 1995; Riggs et al.),
# translated to Sentinel-2 bands. Water absorbs NIR almost completely regardless
# of turbidity, while snow reflects strongly there, so this is a more robust way
# to keep turbid/sediment-laden water from being flagged as snow than the SWIR
# threshold alone.
NIR_SNOW_THRESHOLD = 0.11    # B8 (NIR) — MODIS band 2 analogue
GREEN_SNOW_THRESHOLD = 0.1   # B3 (Green) — MODIS band 4 analogue

# Parameter identifiers (internal use only — never shown as a user choice)
PARAM_TURBIDITY = "Turbidity (NDTI)"
PARAM_CHLOROPHYLL = "Chlorophyll Index"
PARAM_CDOM = "CDOM"

# Every parameter the app monitors, in the order they are processed and shown.
ALL_PARAMETERS = [PARAM_TURBIDITY, PARAM_CHLOROPHYLL, PARAM_CDOM]

# Chlorophyll visualization range
CHL_VMIN = -1.0
CHL_VMAX = 0.9

# CDOM (Colored Dissolved Organic Matter) visualization range — fallback only.
# Unlike NDTI/NDCI, CDOM is not a normalized-difference index bounded to
# [-1, 1]: it is an absorption coefficient whose plausible range differs a lot
# between water bodies. The app therefore derives the display range from the
# actual data of the run (2nd-98th percentile across all months, so colours stay
# comparable month to month) and only falls back to these constants when that is
# not possible.
CDOM_VMIN = 0.0
CDOM_VMAX = 30.0


def empty_param_dict(factory=dict):
    """A fresh {parameter: empty container} map for every monitored parameter."""
    return {p: factory() for p in ALL_PARAMETERS}


def param_short_name(parameter_type):
    """Short technical label used in captions, tables and the Excel export."""
    if parameter_type == PARAM_TURBIDITY:
        return "NDTI"
    if parameter_type == PARAM_CHLOROPHYLL:
        return "Chl-a"
    return "CDOM"


def param_decimals(parameter_type):
    """How many decimals to show for this parameter's values."""
    return 4 if parameter_type == PARAM_TURBIDITY else 2


def format_param_value(parameter_type, value, empty="—"):
    """Format one mean value for display, honouring the parameter's precision."""
    try:
        if value is None or np.isnan(value):
            return empty
    except TypeError:
        return empty
    return f"{value:.{param_decimals(parameter_type)}f}"

# -----------------------------------------------------------------------------
# Persian UI font (B Nazanin)
# -----------------------------------------------------------------------------
# "B Nazanin" is not a free web font, so it only renders if it happens to be
# installed on the viewer's machine. To make the app look identical for every
# user, drop a copy of the font file next to the app (any ONE of the paths
# below) and it will be embedded directly into the page as a base64 @font-face
# rule — no web server configuration needed.
#
#   Recommended:  assets/BNazanin.woff2      (smallest / fastest)
#   Also fine:    assets/BNazanin.ttf  |  BNazanin.ttf  |  fonts/BNazanin.ttf
#
# If no file is found, the app silently falls back to Vazirmatn (loaded from
# Google Fonts), exactly like the previous version — nothing breaks.
BNAZANIN_FONT_CANDIDATES = [
    "assets/BNazanin.woff2",
    "assets/BNazanin.woff",
    "assets/BNazanin.ttf",
    "assets/B Nazanin.ttf",
    "assets/B-Nazanin.ttf",
    "fonts/BNazanin.woff2",
    "fonts/BNazanin.ttf",
    "BNazanin.woff2",
    "BNazanin.ttf",
    "B Nazanin.ttf",
]

# Optional: if you prefer to host the font on Google Drive (like the app logo),
# put the shareable file ID here and leave the local files out. The file must be
# shared as "Anyone with the link -> Viewer".
BNAZANIN_FONT_DRIVE_ID = ""

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
    st.session_state.downloaded_months = empty_param_dict(dict)
if 'month_statuses' not in st.session_state:
    st.session_state.month_statuses = empty_param_dict(dict)
if 'results' not in st.session_state:
    # nested by parameter
    st.session_state.results = empty_param_dict(list)
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'selected_region_index' not in st.session_state:
    st.session_state.selected_region_index = 0
if 'processing_in_progress' not in st.session_state:
    st.session_state.processing_in_progress = False
if 'processing_config' not in st.session_state:
    st.session_state.processing_config = None
if 'mean_data' not in st.session_state:
    st.session_state.mean_data = empty_param_dict(dict)
if 'download_summary' not in st.session_state:
    # simple end-user facing summary: {PARAM_TURBIDITY: (downloaded, available), ...}
    st.session_state.download_summary = {}
if 'resume_after_interruption' not in st.session_state:
    # True when a previous run was interrupted and can be resumed
    st.session_state.resume_after_interruption = False
if 'cdom_display_range' not in st.session_state:
    # (vmin, vmax) colour range derived from the CDOM data of the current run
    st.session_state.cdom_display_range = None
if 'map_version' not in st.session_state:
    # Bumped whenever a region is saved or deleted. It is part of the map
    # widget's key, so the widget is remounted and drops the shape still held
    # by its own drawing toolbar — otherwise a deleted region would be sent
    # back by the widget and reappear on the map.
    st.session_state.map_version = 0
if 'pending_run' not in st.session_state:
    # 'start' | 'resume' | None — a click arms the run, the next script run
    # executes it (so the page is fully rendered before processing begins)
    st.session_state.pending_run = None
if 'active_page' not in st.session_state:
    # Which of the four top-navigation pages is currently shown:
    # 'setup' | 'turbidity' | 'chlorophyll' | 'chat'
    st.session_state.active_page = 'setup'


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
    Create water quality collection for Turbidity, Chlorophyll or CDOM.

    Snow detection is used as a PREPROCESSING step to exclude snow/ice pixels
    from water detection. Snow mask is never downloaded or shown to the user.

    For TURBIDITY (NDTI):
    1. Link S2_SR with S2_CLOUD_PROBABILITY
    2. Apply cloud mask (probability < 15)
    3. Calculate NDSI for snow detection: (B3 - B11) / (B3 + B11)
    4. Create snow mask (MODIS-heritage water/snow test): NDSI > 0.42 AND B8 (NIR) > 0.11 AND B3 (Green) > 0.1
    5. Calculate AWEIsh for water body detection: B2 + 2.5*B3 - 1.5*(B8+B11) - 0.25*B12 > 0.05, excluding snow
    6. Calculate NDTI (turbidity index): (B4 - B3) / (B4 + B3)

    For CHLOROPHYLL:
    1. Link S2_SR with S2_CLOUD_PROBABILITY
    2. Apply cloud mask (probability < 15)
    3. Calculate NDSI for snow detection: (B3 - B11) / (B3 + B11)
    4. Create snow mask (MODIS-heritage water/snow test): NDSI > 0.42 AND B8 (NIR) > 0.11 AND B3 (Green) > 0.1
    5. Calculate AWEIsh for water body detection: B2 + 2.5*B3 - 1.5*(B8+B11) - 0.25*B12 > 0.05, excluding snow
    6. Calculate Chlorophyll Index (NDCI): (B5 - B4) / (B5 + B4)

    For CDOM (Colored Dissolved Organic Matter):
    1-5. Exactly the same preprocessing chain as above (cloud mask, snow mask,
         AWEIsh water body) so all three parameters are computed on the very
         same water pixels and are directly comparable.
    6. Calculate CDOM from the green/red band ratio:
           CDOM = 537 * exp(-2.93 * (B3 / B4))
       This is the Landsat-heritage band-ratio model of Brezonik, Menken &
       Bauer (2005), applied to the equivalent Sentinel-2 bands. Its output
       approximates the CDOM absorption coefficient at 440 nm (a440), in
       units of 1/m — higher values mean more dissolved organic ("humic")
       matter and darker, tea-coloured water.
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

            sr = img.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']).multiply(0.0001)

            ndsi = sr.normalizedDifference(['B3', 'B11']).rename('ndsi')
            is_snow = (
                ndsi.gt(NDSI_THRESHOLD)                          # NDSI > 0.42 — spectral snow signature
                .And(sr.select('B8').gt(NIR_SNOW_THRESHOLD))     # NIR ~0.11 — excludes water, snow reflects strongly here
                .And(sr.select('B3').gt(GREEN_SNOW_THRESHOLD))   # Green ~0.1 — excludes dark shadow/non-snow surfaces
            )

            awei = sr.expression(
                'BLUE + 2.5 * GREEN - 1.5 * (NIR + SWIR1) - 0.25 * SWIR2',
                {
                    'BLUE': sr.select('B2'),
                    'GREEN': sr.select('B3'),
                    'NIR': sr.select('B8'),
                    'SWIR1': sr.select('B11'),
                    'SWIR2': sr.select('B12'),
                }
            ).rename('awei')
            water_body = awei.gt(AWEI_THRESHOLD).And(is_snow.Not())

            ndti = sr.normalizedDifference(['B4', 'B3']).rename('wq_index')

            wq_masked = ndti.updateMask(cloud_free).updateMask(water_body)

            rgb = sr.select(['B4', 'B3', 'B2'])

            combined = (wq_masked
                       .addBands(rgb)
                       .addBands(water_body.rename('water_mask')))

            return combined.clip(aoi).copyProperties(img, ['system:time_start'])

        return s2_joined.map(calculate_turbidity)

    elif parameter_type == PARAM_CDOM:
        def calculate_cdom(img):
            cloud = img.select('probability')
            cloud_free = cloud.lt(CLOUD_PROB_THRESHOLD)

            sr = img.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']).multiply(0.0001)

            ndsi = sr.normalizedDifference(['B3', 'B11']).rename('ndsi')
            is_snow = (
                ndsi.gt(NDSI_THRESHOLD)                          # NDSI > 0.42 — spectral snow signature
                .And(sr.select('B8').gt(NIR_SNOW_THRESHOLD))     # NIR ~0.11 — excludes water, snow reflects strongly here
                .And(sr.select('B3').gt(GREEN_SNOW_THRESHOLD))   # Green ~0.1 — excludes dark shadow/non-snow surfaces
            )

            awei = sr.expression(
                'BLUE + 2.5 * GREEN - 1.5 * (NIR + SWIR1) - 0.25 * SWIR2',
                {
                    'BLUE': sr.select('B2'),
                    'GREEN': sr.select('B3'),
                    'NIR': sr.select('B8'),
                    'SWIR1': sr.select('B11'),
                    'SWIR2': sr.select('B12'),
                }
            ).rename('awei')
            water_body = awei.gt(AWEI_THRESHOLD).And(is_snow.Not())

            # CDOM from the green/red band ratio (Brezonik et al., 2005):
            #     CDOM = 537 * exp(-2.93 * (B3 / B4))
            # Red reflectance is the denominator, so pixels where it is zero or
            # negative (deep shadow, residual masking artefacts) are excluded
            # rather than producing an infinite ratio.
            red_ok = sr.select('B4').gt(0)

            cdom = sr.expression(
                '537 * exp(-2.93 * (B03 / B04))',
                {
                    'B03': sr.select('B3'),
                    'B04': sr.select('B4'),
                }
            ).rename('wq_index')

            wq_masked = (cdom
                         .updateMask(cloud_free)
                         .updateMask(water_body)
                         .updateMask(red_ok))

            rgb = sr.select(['B4', 'B3', 'B2'])

            combined = (wq_masked
                       .addBands(rgb)
                       .addBands(water_body.rename('water_mask')))

            return combined.clip(aoi).copyProperties(img, ['system:time_start'])

        return s2_joined.map(calculate_cdom)

    else:  # CHLOROPHYLL
        def calculate_chlorophyll(img):
            cloud = img.select('probability')
            cloud_free = cloud.lt(CLOUD_PROB_THRESHOLD)

            sr = img.select(['B1', 'B2', 'B3', 'B4', 'B5', 'B8', 'B11', 'B12']).multiply(0.0001)

            ndsi = sr.normalizedDifference(['B3', 'B11']).rename('ndsi')
            is_snow = (
                ndsi.gt(NDSI_THRESHOLD)                          # NDSI > 0.42 — spectral snow signature
                .And(sr.select('B8').gt(NIR_SNOW_THRESHOLD))     # NIR ~0.11 — excludes water, snow reflects strongly here
                .And(sr.select('B3').gt(GREEN_SNOW_THRESHOLD))   # Green ~0.1 — excludes dark shadow/non-snow surfaces
            )

            awei = sr.expression(
                'BLUE + 2.5 * GREEN - 1.5 * (NIR + SWIR1) - 0.25 * SWIR2',
                {
                    'BLUE': sr.select('B2'),
                    'GREEN': sr.select('B3'),
                    'NIR': sr.select('B8'),
                    'SWIR1': sr.select('B11'),
                    'SWIR2': sr.select('B12'),
                }
            ).rename('awei')
            water_body = awei.gt(AWEI_THRESHOLD).And(is_snow.Not())

            # NDCI (Normalized Difference Chlorophyll Index): (B5 - B4) / (B5 + B4)
            # https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ndci/
            chl_index = sr.normalizedDifference(['B5', 'B4']).rename('wq_index')

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


def create_cdom_colormap():
    """Clear blue water -> yellow -> amber -> dark brown (tea-coloured, humic)."""
    colors = ['#08306B', '#2171B5', '#6BAED6', '#EDF8B1', '#FEE391', '#EC7014', '#662506']
    return LinearSegmentedColormap.from_list('cdom', colors, N=256)


def compute_cdom_display_range(wq_paths, low=2, high=98):
    """
    Derive one common colour range for every CDOM month of a run.

    CDOM is an absorption coefficient, not a bounded index, so a fixed scale
    either saturates or washes out depending on the water body. The 2nd-98th
    percentile of all months pooled together keeps the colours meaningful AND
    comparable between months (which a per-image stretch would not).
    """
    samples = []
    for path in wq_paths:
        try:
            with rasterio.open(path) as src:
                data = src.read(1)[::4, ::4]  # subsample: plenty for a percentile
            finite = data[np.isfinite(data) & (data != 0)]
            if finite.size:
                samples.append(finite)
        except Exception:
            continue

    if not samples:
        return CDOM_VMIN, CDOM_VMAX

    pooled = np.concatenate(samples)
    if pooled.size == 0:
        return CDOM_VMIN, CDOM_VMAX

    vmin = float(np.percentile(pooled, low))
    vmax = float(np.percentile(pooled, high))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return CDOM_VMIN, CDOM_VMAX
    return vmin, vmax


def generate_thumbnails(wq_path, rgb_path, month_name, parameter_type, max_size=300,
                        value_range=None):
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
        elif parameter_type == PARAM_CDOM:
            cmap = create_cdom_colormap()
            vmin, vmax = value_range if value_range else (CDOM_VMIN, CDOM_VMAX)
            if vmax <= vmin:
                vmax = vmin + 1e-6
            wq_normalized = np.clip((wq_valid - vmin) / (vmax - vmin), 0, 1)
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
                              resume=False, progress_callback=None):
    """
    Run the full pipeline for one parameter (NDTI or Chlorophyll-a):
    cloud filtering -> snow masking -> waterbody extraction -> index calculation
    -> download -> thumbnail generation.

    Key resilience behaviours (adapted from old version):
    - Per-month session state writes: progress is preserved after every month so
      that a connection drop never discards completed work.
    - Resume logic: when resume=True, months already present in
      st.session_state.downloaded_months[parameter_type] (and whose files are
      still valid on disk) are skipped entirely.
    - File-level cache: download_monthly_data() validates existing GeoTIFFs on
      disk before attempting a new download — so cached files survive page
      reloads even without session state.
    - EE server calls (.size().getInfo(), get_monthly_composite) are wrapped in
      try/except so a transient network error on one month does not crash the
      whole pipeline; the month is marked STATUS_FAILED and processing continues.

    Returns: (results_list, mean_data_dict, downloaded_count, available_count)
    No technical logs are written to the UI; this function is silent.

    progress_callback(done, total, month_name), if given, is invoked once
    before the download loop starts (to reflect months already recovered
    from cache/resume) and once after every month is handled, so the caller
    can drive a progress bar / stage label.
    """
    param_short = {
        PARAM_TURBIDITY: "turbidity",
        PARAM_CHLOROPHYLL: "chlorophyll",
        PARAM_CDOM: "cdom",
    }[parameter_type]

    start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    total_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)

    # ------------------------------------------------------------------
    # Build the GEE collection (server-side — no network download yet)
    # ------------------------------------------------------------------
    wq_collection = create_water_quality_collection(
        aoi, start_date, end_date, parameter_type, cloudy_pixel_percentage
    )

    month_infos = []
    for month_index in range(total_months):
        year = start_dt.year + (start_dt.month - 1 + month_index) // 12
        month = (start_dt.month - 1 + month_index) % 12 + 1
        month_infos.append({'month_name': f"{year}-{month:02d}", 'year': year, 'month': month})

    # ------------------------------------------------------------------
    # FIX A: Restore already-downloaded months from session state (resume)
    # ------------------------------------------------------------------
    # Ensure the nested dict for this parameter exists in session state
    if not isinstance(st.session_state.downloaded_months.get(parameter_type), dict):
        st.session_state.downloaded_months[parameter_type] = {}
    if not isinstance(st.session_state.month_statuses.get(parameter_type), dict):
        st.session_state.month_statuses[parameter_type] = {}

    downloaded_months = {}

    if resume and st.session_state.downloaded_months.get(parameter_type):
        for month_name, paths in st.session_state.downloaded_months[parameter_type].items():
            if paths.get('wq_index') and paths.get('rgb'):
                wq_valid, _ = validate_geotiff_file(paths['wq_index'], expected_bands=1)
                rgb_valid, _ = validate_geotiff_file(paths['rgb'], expected_bands=3)
                if wq_valid and rgb_valid:
                    downloaded_months[month_name] = paths
                    # Keep the cached status entry as well
                    if month_name not in st.session_state.month_statuses[parameter_type]:
                        st.session_state.month_statuses[parameter_type][month_name] = {
                            'status': STATUS_COMPLETE, 'message': 'Cached'
                        }

    # Also check disk for any months whose files exist but session state was lost
    # (e.g. after a page reload without resume — file-level cache recovery)
    for month_info in month_infos:
        month_name = month_info['month_name']
        if month_name in downloaded_months:
            continue
        wq_path = os.path.join(temp_dir, f"wq_index_{param_short}_{month_name}.tif")
        rgb_path = os.path.join(temp_dir, f"rgb_{param_short}_{month_name}.tif")
        wq_valid, _ = validate_geotiff_file(wq_path, expected_bands=1)
        rgb_valid, _ = validate_geotiff_file(rgb_path, expected_bands=3)
        if wq_valid and rgb_valid:
            downloaded_months[month_name] = {'wq_index': wq_path, 'rgb': rgb_path}
            st.session_state.downloaded_months[parameter_type][month_name] = downloaded_months[month_name]
            st.session_state.month_statuses[parameter_type][month_name] = {
                'status': STATUS_COMPLETE, 'message': 'Cached (disk)'
            }

    # Months not yet downloaded (skip ones already done or already statused as no-data)
    already_statused = {
        m for m, s in st.session_state.month_statuses[parameter_type].items()
        if s.get('status') in (STATUS_NO_DATA, STATUS_COMPLETE)
    }
    months_to_process = [
        m for m in month_infos
        if m['month_name'] not in downloaded_months
        and m['month_name'] not in already_statused
    ]

    available_count = len(downloaded_months)  # start with already-recovered months

    # Progress bookkeeping: months already resolved before the loop (from the
    # resume/disk-cache recovery above) count as "done" immediately so the
    # progress bar reflects real state instead of restarting from zero.
    already_done_count = total_months - len(months_to_process)
    processed_count = already_done_count
    if progress_callback:
        progress_callback(processed_count, total_months, None)

    # ------------------------------------------------------------------
    # FIX B: Per-month EE + download loop with immediate session state writes
    # ------------------------------------------------------------------
    for month_info in months_to_process:
        month_name = month_info['month_name']

        # FIX C: Wrap every EE server call so a transient error skips the month
        try:
            composite, count, stats = get_monthly_composite(
                wq_collection, aoi, month_info['year'], month_info['month']
            )
        except Exception:
            # Network or EE error — mark as failed and continue to next month
            st.session_state.month_statuses[parameter_type][month_name] = {
                'status': STATUS_FAILED, 'message': 'EE request failed'
            }
            processed_count += 1
            if progress_callback:
                progress_callback(processed_count, total_months, month_name)
            continue

        if composite is None or count == 0:
            st.session_state.month_statuses[parameter_type][month_name] = {
                'status': STATUS_NO_DATA, 'message': 'No images'
            }
            processed_count += 1
            if progress_callback:
                progress_callback(processed_count, total_months, month_name)
            continue

        available_count += 1

        wq_path, rgb_path, status, message = download_monthly_data(
            composite, aoi, temp_dir, month_name, param_short, scale
        )

        # FIX D: Write to session state immediately after each month — not at end
        st.session_state.month_statuses[parameter_type][month_name] = {
            'status': status, 'message': message
        }

        if status == STATUS_COMPLETE:
            downloaded_months[month_name] = {'wq_index': wq_path, 'rgb': rgb_path}
            # Persist to session state right away so a crash/reload can recover
            st.session_state.downloaded_months[parameter_type][month_name] = {
                'wq_index': wq_path, 'rgb': rgb_path
            }

        processed_count += 1
        if progress_callback:
            progress_callback(processed_count, total_months, month_name)

    # ------------------------------------------------------------------
    # Thumbnail generation (uses only successfully downloaded months)
    # ------------------------------------------------------------------
    results = []
    mean_data = {}

    # CDOM gets one colour range shared by every month of the run (see
    # compute_cdom_display_range); the two normalized-difference indices keep
    # their fixed scales exactly as before.
    value_range = None
    if parameter_type == PARAM_CDOM and downloaded_months:
        value_range = compute_cdom_display_range(
            [downloaded_months[m]['wq_index'] for m in sorted(downloaded_months.keys())]
        )
        st.session_state.cdom_display_range = value_range

    for month_name in sorted(downloaded_months.keys()):
        paths = downloaded_months[month_name]
        thumb = generate_thumbnails(paths['wq_index'], paths['rgb'], month_name, parameter_type,
                                    value_range=value_range)
        if thumb:
            results.append(thumb)
            mean_data[month_name] = {'mean': thumb['mean_value'], 'coverage': thumb['water_coverage']}

    return results, mean_data, len(downloaded_months), available_count


def run_full_analysis(aoi, start_date, end_date, cloudy_pixel_percentage=CLOUD_THRESHOLD,
                       scale=10, resume=False):
    """
    Automatically runs preprocessing + all three index calculations (NDTI, then
    Chlorophyll-a, then CDOM) in sequence. Displays only a simple,
    user-friendly summary.

    Resilience additions vs. original:
    - resume=True is forwarded to process_single_parameter so cached months are
      skipped rather than re-downloaded.
    - Each parameter block is wrapped in try/except so a hard failure on one
      index does not prevent the others from running.
    - processing_config is written to session state here so the main() button
      handler can pass it to a resume run later.
    """
    if st.session_state.current_temp_dir is None or not os.path.exists(st.session_state.current_temp_dir):
        st.session_state.current_temp_dir = tempfile.mkdtemp()
    temp_dir = st.session_state.current_temp_dir

    summary_placeholder = st.empty()
    download_summary = dict(st.session_state.download_summary)  # preserve any prior summary

    # ------------------------------------------------------------------
    # Progress bar + stage label — shows overall level (%) and which stage
    # (parameter + month) is currently being processed. Total work units are
    # "months × 3 parameters"; months already recovered via resume/disk cache
    # count as already-done so the bar starts from the right place on Resume.
    # ------------------------------------------------------------------
    start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    total_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
    total_units = max(total_months * len(ALL_PARAMETERS), 1)

    progress_bar = st.progress(0)
    stage_text = st.empty()

    def make_progress_callback(stage_label, unit_offset):
        def _callback(done_in_stage, total_in_stage, month_name):
            done_units = unit_offset + done_in_stage
            percent = int(min(1.0, done_units / total_units) * 100)
            if month_name:
                stage_text.markdown(
                    f"**{stage_label}** — در حال پردازش ماه «{month_name}» "
                    f"({done_in_stage} از {total_in_stage}) — {percent}٪"
                )
            else:
                stage_text.markdown(f"**{stage_label}** — در حال آماده‌سازی... — {percent}٪")
            progress_bar.progress(min(1.0, done_units / total_units))
        return _callback

    with st.spinner("در حال پایش کیفیت آب... این فرآیند ممکن است چند دقیقه طول بکشد"):
        # --- Turbidity (NDTI) ---
        turb_results, turb_mean, turb_downloaded, turb_available = [], {}, 0, 0
        try:
            turb_results, turb_mean, turb_downloaded, turb_available = process_single_parameter(
                aoi, start_date, end_date, PARAM_TURBIDITY, temp_dir,
                cloudy_pixel_percentage, scale, resume=resume,
                progress_callback=make_progress_callback("🌊 مرحله ۱ از ۳ — شاخص کدورت (NDTI)", 0)
            )
        except Exception:
            pass  # Partial or zero results; pipeline continues to chlorophyll

        # Merge with any results already in session state (resume case)
        if turb_results:
            st.session_state.results[PARAM_TURBIDITY] = turb_results
            st.session_state.mean_data[PARAM_TURBIDITY] = turb_mean
        download_summary[PARAM_TURBIDITY] = (turb_downloaded, turb_available)

        summary_placeholder.info(
            f"🌊 شاخص کدورت: {turb_downloaded} تصویر از {turb_available} تصویر موجود دریافت شد."
        )

        # --- Chlorophyll-a ---
        chl_results, chl_mean, chl_downloaded, chl_available = [], {}, 0, 0
        try:
            chl_results, chl_mean, chl_downloaded, chl_available = process_single_parameter(
                aoi, start_date, end_date, PARAM_CHLOROPHYLL, temp_dir,
                cloudy_pixel_percentage, scale, resume=resume,
                progress_callback=make_progress_callback("🌿 مرحله ۲ از ۳ — شاخص کلروفیل", total_months)
            )
        except Exception:
            pass  # Partial or zero results; still show whatever was collected

        if chl_results:
            st.session_state.results[PARAM_CHLOROPHYLL] = chl_results
            st.session_state.mean_data[PARAM_CHLOROPHYLL] = chl_mean
        download_summary[PARAM_CHLOROPHYLL] = (chl_downloaded, chl_available)

        summary_placeholder.info(
            f"🌊 شاخص کدورت: {turb_downloaded} تصویر از {turb_available} تصویر موجود دریافت شد.\n\n"
            f"🌿 شاخص کلروفیل: {chl_downloaded} تصویر از {chl_available} تصویر موجود دریافت شد."
        )

        # --- CDOM (Colored Dissolved Organic Matter) ---
        cdom_results, cdom_mean, cdom_downloaded, cdom_available = [], {}, 0, 0
        try:
            cdom_results, cdom_mean, cdom_downloaded, cdom_available = process_single_parameter(
                aoi, start_date, end_date, PARAM_CDOM, temp_dir,
                cloudy_pixel_percentage, scale, resume=resume,
                progress_callback=make_progress_callback(
                    "🍂 مرحله ۳ از ۳ — شاخص مواد آلی محلول (CDOM)", total_months * 2
                )
            )
        except Exception:
            pass  # Partial or zero results; still show whatever was collected

        if cdom_results:
            st.session_state.results[PARAM_CDOM] = cdom_results
            st.session_state.mean_data[PARAM_CDOM] = cdom_mean
        download_summary[PARAM_CDOM] = (cdom_downloaded, cdom_available)

    progress_bar.progress(1.0)
    stage_text.markdown("✅ پردازش هر سه شاخص به پایان رسید — ۱۰۰٪")

    st.session_state.download_summary = download_summary

    has_any_results = any(
        bool(st.session_state.results.get(p)) for p in ALL_PARAMETERS
    )
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


def render_cdom_guidance_panel():
    """Permanently visible legend + management guidance for CDOM."""
    st.markdown("### 🎨 راهنمای رنگ و تفسیر مدیریتی — شاخص مواد آلی محلول رنگی (CDOM)")

    col_legend, col_text = st.columns([1, 2])

    with col_legend:
        fig, ax = plt.subplots(figsize=(5, 0.45))
        cmap = create_cdom_colormap()
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        ax.imshow(gradient, aspect='auto', cmap=cmap)
        ax.set_xticks([0, 128, 255])
        ax.set_xticklabels(['کم (آب زلال)', 'متوسط', 'زیاد (آب چای‌رنگ)'])
        ax.set_yticks([])
        st.pyplot(fig)
        plt.close(fig)

        value_range = st.session_state.get('cdom_display_range')
        if value_range:
            st.caption(
                f"محدوده رنگی این پایش: از {value_range[0]:.1f} تا {value_range[1]:.1f} "
                "(بر متر) — بر پایه‌ی داده‌های همین منطقه و بازه زمانی تعیین شده است."
            )

    with col_text:
        st.markdown(
            """
**افزایش مواد آلی محلول رنگی:**
- تیره و چای‌رنگ شدن آب و کاهش مطلوبیت آب شرب (رنگ، بو و مزه)
- افزایش مصرف مواد ضدعفونی‌کننده و خطر تشکیل فرآورده‌های جانبی گندزدایی در تصفیه‌خانه
- کاهش نفوذ نور به اعماق و تغییر شرایط زیستی بدنه آبی
- نشانه احتمالی ورود رواناب حوضه آبریز، زهکشی اراضی آلی/تالابی، یا تخلیه فاضلاب

**کاهش مواد آلی محلول رنگی:**
- شفاف‌تر شدن آب و کاهش نیاز به مواد شیمیایی در تصفیه
- کاهش ریسک بهداشتی مرتبط با فرآورده‌های جانبی گندزدایی
- نشانه کاهش ورود بار آلی از حوضه آبریز

**چرا پایش این شاخص مهم است؟**
مواد آلی محلول رنگی، برخلاف کدورت، ذرات معلق نیستند و با ته‌نشینی ساده حذف نمی‌شوند؛
بنابراین افزایش آن مستقیماً بر هزینه و پیچیدگی فرایند تصفیه آب شرب اثر می‌گذارد. این شاخص
معمولاً پس از بارش‌های سنگین و ذوب برف (ورود رواناب از خاک و پوشش گیاهی حوضه) افزایش
می‌یابد و یکی از بهترین نشانگرهای ورود بار آلی از سطح حوضه به بدنه آبی است.

*این شاخص از نسبت باند سبز به قرمز سنتینل-۲ و رابطه‌ی Brezonik و همکاران (۲۰۰۵) محاسبه
می‌شود و تقریبی از ضریب جذب مواد آلی محلول در طول موج ۴۴۰ نانومتر (بر متر) است.*
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

    param_short = param_short_name(parameter_type)

    for r in results:
        mean_str = format_param_value(parameter_type, r['mean_value'], empty="بدون داده")

        cols = st.columns(2)
        cols[0].image(r['wq_image'], caption=f"{r['month_name']} — {param_short}: {mean_str}", use_container_width=True)
        cols[1].image(r['rgb_image'], caption=f"{r['month_name']} — تصویر طبیعی (RGB)", use_container_width=True)


def display_time_series_chart(results, parameter_type):
    """Time series chart of mean index values directly under the imagery."""
    if not results:
        return

    # Persian font stack for chart text — mirrors the CSS fallback chain used
    # elsewhere in the app (B Nazanin first, falling back gracefully if it is
    # not installed on the machine rendering the figure).
    PERSIAN_FONT = ['B Nazanin', 'BNazanin', 'Vazirmatn', 'Tahoma', 'DejaVu Sans']

    if parameter_type == PARAM_TURBIDITY:
        param_label_fa = "شاخص کدورت آب (NDTI)"
        param_unit = ""
        chart_title = "روند زمانی کدورت آب"
    elif parameter_type == PARAM_CDOM:
        param_label_fa = "شاخص مواد آلی محلول (CDOM)"
        param_unit = " (بر متر)"
        chart_title = "روند زمانی مواد آلی محلول رنگی"
    else:
        param_label_fa = "شاخص کلروفیل (NDCI)"
        param_unit = " (µg/L)"
        chart_title = "روند زمانی کلروفیل"

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

    color1 = {
        PARAM_TURBIDITY: '#1f77b4',
        PARAM_CHLOROPHYLL: '#228B22',
        PARAM_CDOM: '#8C510A',
    }[parameter_type]
    ax1.set_xlabel('ماه', fontsize=13, fontfamily=PERSIAN_FONT)
    ax1.set_ylabel(f'میانگین {param_label_fa}{param_unit}', color=color1, fontsize=13, fontfamily=PERSIAN_FONT)

    if valid_values:
        ax1.plot(months, mean_values, 'o-', color=color1, linewidth=2, markersize=8,
                  label=f'میانگین {param_label_fa}')
        ax1.tick_params(axis='y', labelcolor=color1)

        if parameter_type == PARAM_TURBIDITY:
            ax1.set_ylim(min(mean_values) - 0.02, max(mean_values) + 0.02)
            ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='خط خنثی (کدورت = ۰)')
        elif parameter_type == PARAM_CDOM:
            # CDOM is an absorption coefficient, not a normalized index: it is
            # always positive and has no meaningful "neutral" line at zero, so
            # the axis is simply scaled to the observed values.
            val_min = min(mean_values)
            val_max = max(mean_values)
            padding = max((val_max - val_min) * 0.15, 0.05)
            ax1.set_ylim(max(0.0, val_min - padding), val_max + padding)
        else:
            # FIX: NDCI (like NDTI) is a normalized-difference index and can be
            # negative — forcing the axis to start at 0 (the old behaviour)
            # clips or completely hides months with a negative mean value.
            # Scale to the actual min/max instead, the same way the turbidity
            # chart already does.
            val_min = min(mean_values)
            val_max = max(mean_values)
            padding = max((val_max - val_min) * 0.15, 0.02)
            ax1.set_ylim(val_min - padding, val_max + padding)
            ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.4, label='خط خنثی (کلروفیل = ۰)')
    else:
        ax1.text(0.5, 0.5, 'داده معتبری موجود نیست', ha='center', va='center',
                  transform=ax1.transAxes, fontsize=12, fontfamily=PERSIAN_FONT)

    ax1.set_xticklabels(months, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)

    ax1_twin = ax1.twinx()
    color2 = '#2ca02c'
    ax1_twin.set_ylabel('پوشش آب (٪)', color=color2, fontsize=13, fontfamily=PERSIAN_FONT)
    ax1_twin.bar(months, coverage_values, alpha=0.3, color=color2, label='پوشش آب')
    ax1_twin.tick_params(axis='y', labelcolor=color2)
    ax1_twin.set_ylim(0, max(coverage_values) * 1.3 if max(coverage_values) > 0 else 100)

    ax1.set_title(chart_title, fontsize=15, fontweight='bold', fontfamily=PERSIAN_FONT)
    legend = ax1.legend(loc='upper left', prop={'family': PERSIAN_FONT, 'size': 10})

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def display_statistics_summary(results, parameter_type):
    """Statistics summary, contained within the same parameter page."""
    if not results:
        return

    param_short = param_short_name(parameter_type)

    months = [r['month_name'] for r in results]
    mean_values = [r['mean_value'] if not np.isnan(r['mean_value']) else 0 for r in results]
    coverage_values = [r['water_coverage'] for r in results]
    valid_values = [m for m in mean_values if m != 0]

    st.markdown("#### 📈 خلاصه آماری")

    col1, col2, col3, col4 = st.columns(4)

    if valid_values:
        col1.metric(f"میانگین {param_short}", format_param_value(parameter_type, np.mean(valid_values)))
        col2.metric(f"حداکثر {param_short}", format_param_value(parameter_type, np.max(valid_values)))
        col3.metric(f"حداقل {param_short}", format_param_value(parameter_type, np.min(valid_values)))
    else:
        col1.metric(f"میانگین {param_short}", "—")
        col2.metric(f"حداکثر {param_short}", "—")
        col3.metric(f"حداقل {param_short}", "—")

    col4.metric("میانگین پوشش آب", f"{np.mean(coverage_values):.1f}%")

    with st.expander("📋 جدول داده‌های ماهانه"):
        import pandas as pd

        value_col = [
            format_param_value(parameter_type, v) if v != 0 else "—" for v in mean_values
        ]

        df = pd.DataFrame({
            'ماه': months,
            f'میانگین {param_short}': value_col,
            'پوشش آب (%)': [f"{v:.1f}" for v in coverage_values]
        })
        _render_persian_dataframe_html(df)


def generate_combined_timeseries_excel():
    """
    Build a single Excel (.xlsx) workbook containing the monthly time-series
    values for ALL monitored parameters — Turbidity (NDTI), Chlorophyll (NDCI)
    and CDOM — as one sheet each. Returns workbook bytes for st.download_button.
    """
    import io
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment, PatternFill
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    wb.remove(wb.active)  # drop the default empty sheet

    header_font = Font(name='Arial', bold=True, color='FFFFFF')
    header_fill = PatternFill(start_color='1F77B4', end_color='1F77B4', fill_type='solid')
    body_font = Font(name='Arial')
    center = Alignment(horizontal='center')

    center_lat, center_lon = _get_roi_center_coordinates()
    lat_out = round(float(center_lat), 6) if center_lat is not None else "—"
    lon_out = round(float(center_lon), 6) if center_lon is not None else "—"

    sections = [
        (PARAM_TURBIDITY, "کدورت (NDTI)", "میانگین NDTI"),
        (PARAM_CHLOROPHYLL, "کلروفیل (NDCI)", "میانگین NDCI"),
        (PARAM_CDOM, "مواد آلی محلول (CDOM)", "میانگین CDOM"),
    ]

    for parameter_type, sheet_name, value_header in sections:
        ws = wb.create_sheet(title=sheet_name)
        ws.sheet_view.rightToLeft = True

        headers = ["ماه", value_header, "پوشش آب (%)", "عرض جغرافیایی مرکز", "طول جغرافیایی مرکز"]
        ws.append(headers)
        for col_idx in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=col_idx)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = center

        results = st.session_state.results.get(parameter_type, [])
        for r in sorted(results, key=lambda x: x['month_name']):
            mean_val = r['mean_value']
            mean_out = round(float(mean_val), 4) if not np.isnan(mean_val) else "بدون داده"
            ws.append([r['month_name'], mean_out, round(float(r['water_coverage']), 1), lat_out, lon_out])

        if not results:
            ws.cell(row=2, column=1, value="داده‌ای موجود نیست.").font = body_font
        else:
            for row in ws.iter_rows(min_row=2):
                for cell in row:
                    cell.font = body_font
                    cell.alignment = center

        column_widths = [14, 18, 16, 20, 20]
        for i, w in enumerate(column_widths, start=1):
            ws.column_dimensions[get_column_letter(i)].width = w

        ws.freeze_panes = "A2"

    buffer = io.BytesIO()
    wb.save(buffer)
    return buffer.getvalue()


def _get_roi_center_coordinates():
    """
    Return (lat, lon) of the center (centroid) of the region of interest used
    for the current/most recent monitoring run, based on the polygon stored
    in st.session_state.processing_config. Returns (None, None) if no run has
    been configured yet.
    """
    config = st.session_state.get('processing_config')
    if not config or not config.get('polygon_coords'):
        return None, None
    try:
        centroid = Polygon(config['polygon_coords']).centroid
        return centroid.y, centroid.x
    except Exception:
        return None, None


def render_parameter_page(parameter_type):
    """
    Full page for one parameter, in the required order:
    1. Statistics Summary (خلاصه آماری)
    2. Legend + Management Guidance Panel
    3. Side-by-side imagery (collapsible)
    4. Time-series chart
    """
    if parameter_type == PARAM_TURBIDITY:
        _render_active_section_badge("🌊", "کدورت آب (NDTI)", "#0B6E76", "#2FC2CE")
    elif parameter_type == PARAM_CDOM:
        _render_active_section_badge("🍂", "مواد آلی محلول رنگی (CDOM)", "#7A4A12", "#D9A05B")
    else:
        _render_active_section_badge("🌿", "شاخص کلروفیل", "#1B7A3D", "#4CC26B")

    results = st.session_state.results.get(parameter_type, [])

    if results:
        display_statistics_summary(results, parameter_type)
        st.divider()

    if parameter_type == PARAM_TURBIDITY:
        render_turbidity_guidance_panel()
    elif parameter_type == PARAM_CDOM:
        render_cdom_guidance_panel()
    else:
        render_chlorophyll_guidance_panel()

    st.divider()

    if not results:
        st.info("برای مشاهده نتایج، ابتدا یک منطقه را انتخاب و پایش را اجرا کنید.")
        return

    with st.expander("🖼️ تصاویر پردازش‌شده (برای نمایش/پنهان‌سازی کلیک کنید)", expanded=False):
        display_side_by_side_imagery(results, parameter_type)

    st.divider()
    display_time_series_chart(results, parameter_type)


# =============================================================================
# نظر متخصص آب — Statistical Analysis Pipeline
# (ported as-is from the separate analysis notebook: Mann-Kendall trend test,
#  MAD-based anomaly detection, seasonal climatology, cross-parameter
#  correlation — only the input source changes: in-memory Excel bytes
#  instead of a file path, so nothing needs to be written to disk on Posit.)
# =============================================================================
def _expert_compute_trend(values, n):
    """Mann-Kendall trend test: standard (always) + seasonal (once >=24 months exist)."""
    import pymannkendall as mk

    result = {}

    if n < 4:
        result["standard"] = {"available": False, "reason": f"Mann-Kendall needs at least ~4 points; only {n} available."}
        result["seasonal"] = {"available": False, "reason": "Not enough data."}
        return result

    mk_res = mk.original_test(values)
    result["standard"] = {
        "method": "Mann-Kendall (non-seasonal)",
        "trend": mk_res.trend,
        "significant": bool(mk_res.h),
        "p_value": float(mk_res.p),
        "tau": float(mk_res.Tau),
        "sen_slope_per_month": float(mk_res.slope),
        "note": ("Does not separate the seasonal cycle from the trend. "
                 "If a strong seasonal pattern exists, prefer the seasonal result below when available.")
    }

    if n >= 24:
        try:
            smk_res = mk.seasonal_test(values, period=12)
            result["seasonal"] = {
                "method": "Seasonal Mann-Kendall (Hirsch-Slack), period=12 months",
                "trend": smk_res.trend,
                "significant": bool(smk_res.h),
                "p_value": float(smk_res.p),
                "sen_slope_per_month": float(smk_res.slope),
                "replicates_per_season": round(n / 12, 1),
                "power_caution": (f"Only ~{n/12:.1f} years of data per calendar month are available. "
                                   "Seasonal Mann-Kendall has low statistical power below ~3-4 years; "
                                   "treat this result as directional, not conclusive.")
            }
        except Exception as e:
            result["seasonal"] = {"available": False, "reason": f"Could not compute: {e}"}
    else:
        result["seasonal"] = {
            "available": False,
            "reason": f"Needs at least 24 months (2 full years) to run; only {n} months available."
        }

    return result


def _expert_detect_anomalies_mad(df, date_col, value_column, threshold=3.5):
    """Median Absolute Deviation based anomaly detection."""
    values = df[value_column]
    median = values.median()
    mad = np.median(np.abs(values - median))

    if mad == 0:
        return []

    modified_z = 0.6745 * (values - median) / mad

    anomalies = []
    for idx, row in df.iterrows():
        if abs(modified_z[idx]) > threshold:
            anomalies.append({
                "date": row[date_col].strftime("%Y-%m"),
                "value": float(row[value_column]),
                "modified_z_score": round(float(modified_z[idx]), 2)
            })
    return anomalies


def _expert_analyze_sheet(df, value_column, date_col=None, water_col=None):
    import pandas as pd

    date_col = date_col or df.columns[0]
    water_col = water_col or df.columns[-1]

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    values = df[value_column]
    n = len(values)

    raw_observations = [
        {
            "date": row[date_col].strftime("%Y-%m"),
            "value": float(row[value_column]),
            "water_coverage_pct": float(row[water_col])
        }
        for _, row in df.iterrows()
    ]

    trend = _expert_compute_trend(values.values, n)
    anomalies = _expert_detect_anomalies_mad(df, date_col, value_column)

    df["MonthNumber"] = df[date_col].dt.month
    seasonal_climatology = df.groupby("MonthNumber")[value_column].mean().round(4).to_dict()

    summary = {
        "period": {
            "start": df[date_col].min().strftime("%Y-%m"),
            "end": df[date_col].max().strftime("%Y-%m"),
            "months": int(n)
        },
        "raw_observations": raw_observations,
        "statistics": {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std()),
            "variance": float(values.var()),
            "min": float(values.min()),
            "max": float(values.max()),
            "range": float(values.max() - values.min())
        },
        "trend": trend,
        "extremes": {
            "minimum": {"date": df.loc[values.idxmin(), date_col].strftime("%Y-%m"), "value": float(values.min())},
            "maximum": {"date": df.loc[values.idxmax(), date_col].strftime("%Y-%m"), "value": float(values.max())}
        },
        "seasonal_climatology": seasonal_climatology,
        "water_coverage": {
            "mean_percent": float(df[water_col].mean()),
            "minimum_percent": float(df[water_col].min()),
            "maximum_percent": float(df[water_col].max())
        },
        "missing_values": int(values.isna().sum()),
        "anomalies": anomalies
    }
    return summary


def _expert_compute_correlation_by_date(df1, value_col1, df2, value_col2, date_col1=None, date_col2=None):
    """Correlate two parameter series matched by actual date, not row position."""
    import pandas as pd

    date_col1 = date_col1 or df1.columns[0]
    date_col2 = date_col2 or df2.columns[0]

    d1 = df1[[date_col1, value_col1]].copy()
    d1[date_col1] = pd.to_datetime(d1[date_col1])
    d1 = d1.rename(columns={date_col1: "date", value_col1: "v1"})

    d2 = df2[[date_col2, value_col2]].copy()
    d2[date_col2] = pd.to_datetime(d2[date_col2])
    d2 = d2.rename(columns={date_col2: "date", value_col2: "v2"})

    merged = pd.merge(d1, d2, on="date", how="inner")
    unmatched_1 = set(d1["date"]) - set(merged["date"])
    unmatched_2 = set(d2["date"]) - set(merged["date"])

    return {
        "pearson_correlation": float(merged["v1"].corr(merged["v2"])),
        "n_matched_dates": int(len(merged)),
        "unmatched_dates_first_only": sorted(d.strftime("%Y-%m") for d in unmatched_1),
        "unmatched_dates_second_only": sorted(d.strftime("%Y-%m") for d in unmatched_2),
    }


def analyze_water_quality_from_bytes(excel_bytes):
    """
    Same logic as the standalone analyze_water_quality(excel_file) from the
    analysis notebook, adapted to read the in-memory Excel bytes produced by
    generate_combined_timeseries_excel() instead of a file on disk.

    Robustness addition (not present in the original notebook): a month with
    no data is written into the Excel export as the text "بدون داده" rather
    than a number, which would otherwise turn the whole column non-numeric
    and silently drop that parameter from the analysis. Those cells are
    coerced to NaN and excluded here instead.

    Also mirrors the notebook's latest addition: the center coordinates of
    the region of interest (columns "عرض جغرافیایی مرکز" / "طول جغرافیایی
    مرکز" in the exported Excel) are lifted into top-level
    "center_latitude" / "center_longitude" keys of the returned summary, so
    the location-aware chat agent can use them directly (e.g. to fetch
    historical weather for that exact point) without re-parsing the sheet.
    """
    import io
    import pandas as pd

    excel_buffer = io.BytesIO(excel_bytes)
    xls = pd.ExcelFile(excel_buffer)
    sheet_names = xls.sheet_names
    results = {}
    cleaned_frames = {}

    for sheet in sheet_names:
        df = pd.read_excel(excel_buffer, sheet_name=sheet)
        numeric_like_cols = [c for c in df.columns if c != df.columns[0]]
        for c in numeric_like_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=numeric_like_cols[:1])  # drop rows with no value for the main indicator

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) < 2 or df.empty:
            continue
        value_col = numeric_cols[0]
        water_col = "پوشش آب (%)" if "پوشش آب (%)" in df.columns else None
        results[sheet] = _expert_analyze_sheet(df, value_col, water_col=water_col)
        cleaned_frames[sheet] = df

    # Correlations between every pair of parameters (turbidity ↔ chlorophyll,
    # turbidity ↔ CDOM, chlorophyll ↔ CDOM), each matched by calendar month.
    if len(cleaned_frames) >= 2:
        sheets = list(cleaned_frames.keys())
        relationships = {}
        for i in range(len(sheets)):
            for j in range(i + 1, len(sheets)):
                sheet_a, sheet_b = sheets[i], sheets[j]
                df1, df2 = cleaned_frames[sheet_a], cleaned_frames[sheet_b]
                try:
                    col1 = df1.select_dtypes(include=np.number).columns[0]
                    col2 = df2.select_dtypes(include=np.number).columns[0]
                    relationships[f"{sheet_a} ↔ {sheet_b}"] = \
                        _expert_compute_correlation_by_date(df1, col1, df2, col2)
                except Exception:
                    continue
        if relationships:
            results["relationships"] = relationships

    # --- Lift region center coordinates to the top level (for the chat agent) ---
    if cleaned_frames:
        first_sheet_df = list(cleaned_frames.values())[0]
        lat_col = "عرض جغرافیایی مرکز"
        lon_col = "طول جغرافیایی مرکز"
        if lat_col in first_sheet_df.columns and lon_col in first_sheet_df.columns:
            lat_vals = pd.to_numeric(first_sheet_df[lat_col], errors='coerce').dropna()
            lon_vals = pd.to_numeric(first_sheet_df[lon_col], errors='coerce').dropna()
            if not lat_vals.empty and not lon_vals.empty:
                results["center_latitude"] = float(lat_vals.iloc[0])
                results["center_longitude"] = float(lon_vals.iloc[0])

    return results


def _expert_results_signature():
    """Cheap signature used to detect when monitoring results changed, so the
    JSON summary + chat history for نظر متخصص آب can be refreshed automatically."""
    sig = []
    for p in ALL_PARAMETERS:
        results = st.session_state.results.get(p, [])
        sig.append(tuple(sorted(
            (r['month_name'], None if np.isnan(r['mean_value']) else round(float(r['mean_value']), 6))
            for r in results
        )))
    return tuple(sig)


# =============================================================================
# نظر متخصص آب — LLM Agent Chat (LangGraph ReAct agent, OpenAI-compatible)
# =============================================================================
# NOTE ON CREDENTIALS: this app is hosted on a Posit server that cannot read a
# local .env file, so the API base URL / keys (OpenAI-compatible LLM, Tavily,
# OpenWeatherMap) are all hardcoded below and read directly from this file
# instead of being loaded through python-dotenv or environment variables, per
# explicit request. If this project's git repo is ever shared or made public,
# consider moving these values to Posit Connect's own "Environment Variables"
# panel (Settings → Vars) instead of leaving live keys in source — that still
# avoids the .env problem without exposing the keys in version control.
OPENAI_BASE_URL = "https://api.avalai.ir/v1"
OPENAI_API_KEY = "aa-xLsSw3ad4txKKuwvHY7cPGy1StemeS3xtuChVf9utHKOd3Cr"
EXPERT_CHAT_MODEL = "gpt-5.2"  # change to whichever model your endpoint provides

# Tavily web search key, used by the agent's general-purpose web search tool
# (qualitative/general climate context). Hardcoded directly here — read from
# the code, not from a .env file — because the Posit server this app is
# deployed on cannot read a local .env file.
TAVILY_API_KEY = "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"

# OpenWeatherMap Geocoding API key, used by the agent's reverse-geocoding
# tool (lat/lon -> city/region/country name). Also hardcoded directly here
# for the same Posit-server reason as above.
OPENWEATHER_API_KEY = "5fce9bd0bcda8e2cd43468bf50755c82"


def _build_agent_system_prompt(analysis_json):
    """
    Persian system prompt for the water-quality expert agent. Combines the
    original evidence-based / no-hallucination rules with instructions for
    when to use each of the three available tools (reverse geocoding, web
    search, historical weather). The statistical analysis JSON (including,
    when available, center_latitude / center_longitude of the monitored
    region) is embedded directly in the prompt, exactly as in the previous
    non-agent version — it is NOT exposed as a separate callable tool.
    """
    return f"""شما یک متخصص باتجربه در زمینه کیفیت آب، سنجش‌ازدور ماهواره‌ای (سنتینل-۲)، و اقلیم‌شناسی هستید.

در ادامه، خلاصه‌ی تحلیل آماری سری زمانی **سه شاخص** کیفیت آب یک بدنه‌ی آبی، به‌صورت JSON در اختیار
شما قرار گرفته است. این خلاصه شامل مختصات مرکز منطقه (کلیدهای center_latitude و center_longitude،
در صورت وجود)، نتیجه‌ی آزمون روند من-کندال، ناهنجاری‌های شناسایی‌شده (بر پایه‌ی انحراف مطلق از
میانه)، الگوی فصلی چندساله، آمار توصیفی، و همبستگی دوبه‌دوی شاخص‌ها (کلید relationships) است.

سه شاخص موجود در داده‌ها (هر سه روی دقیقاً یک ماسک پهنه‌ی آبی و یک زنجیره‌ی پیش‌پردازش یکسان —
حذف ابر، حذف برف، و استخراج پهنه آب با AWEIsh — محاسبه شده‌اند و بنابراین مستقیماً با هم قابل
مقایسه‌اند):

۱. «شاخص کدورت آب» (برگه‌ی «کدورت (NDTI)» در داده‌ها) — اختلاف نرمال‌شده‌ی باند قرمز و سبز؛ بی‌بعد و
   در بازه‌ی ۱- تا ۱+. افزایش آن یعنی آب گل‌آلودتر و ذرات معلق بیشتر (فرسایش خاک، رسوب، رواناب،
   فعالیت عمرانی بالادست).

۲. «شاخص کلروفیل» (برگه‌ی «کلروفیل (NDCI)») — اختلاف نرمال‌شده‌ی باند لبه‌ی قرمز و قرمز؛ بی‌بعد و در
   بازه‌ی ۱- تا ۱+. افزایش آن یعنی زیست‌توده‌ی جلبکی/فیتوپلانکتونی بیشتر و احتمال شکوفایی جلبکی.

۳. «شاخص مواد آلی محلول رنگی» (برگه‌ی «مواد آلی محلول (CDOM)») — از نسبت باند سبز به قرمز سنتینل-۲ و
   با رابطه‌ی CDOM = 537 × exp(−2.93 × (B3/B4)) محاسبه می‌شود (رابطه‌ی باندی Brezonik و همکاران،
   ۲۰۰۵) و تقریبی از ضریب جذب مواد آلی محلول در طول موج ۴۴۰ نانومتر، با واحد «بر متر»، است. این
   شاخص برخلاف دو شاخص قبلی نرمال‌شده نیست، همیشه مثبت است و خط خنثی در صفر ندارد؛ بنابراین مقادیر
   آن را هرگز مانند NDTI/NDCI در بازه‌ی ۱- تا ۱+ تفسیر نکنید.

نکات تفسیری مهم درباره‌ی شاخص مواد آلی محلول رنگی:
- این شاخص «ماده‌ی محلول» را می‌سنجد، نه ذرات معلق. بنابراین افزایش هم‌زمان کدورت و مواد آلی محلول
  معمولاً نشانه‌ی ورود رواناب از سطح حوضه (بارش سنگین یا ذوب برف) است، در حالی که افزایش مواد آلی
  محلول بدون افزایش کدورت بیشتر به زهکشی اراضی آلی/تالابی، تخلیه‌ی فاضلاب یا تجزیه‌ی مواد آلی درون
  خود بدنه‌ی آبی اشاره دارد.
- افزایش آن برای بهره‌بردار آب شرب مهم است: رنگ، بو و مزه‌ی آب را تغییر می‌دهد، مصرف مواد
  گندزدا را بالا می‌برد و خطر تشکیل فرآورده‌های جانبی گندزدایی را افزایش می‌دهد. این مواد با
  ته‌نشینی ساده حذف نمی‌شوند.
- مواد آلی محلول رنگی نور را جذب می‌کند و می‌تواند بر برآورد کلروفیل اثر بگذارد؛ اگر کلروفیل و مواد
  آلی محلول هم‌زمان و به‌شدت همبسته بودند، این احتمال را در تفسیر خود صریحاً ذکر کنید.

داده‌های تحلیل:
{analysis_json}

شما به سه ابزار دسترسی دارید:
۱. reverse_geocode: یافتن دقیق نام شهر/منطقه/کشور بر اساس یک مختصات جغرافیایی (عرض و طول جغرافیایی)،
   با استفاده از سرویس Geocoding شرکت OpenWeatherMap. هر زمان که نیاز به شناسایی نام منطقه‌ی مورد
   مطالعه بر اساس center_latitude / center_longitude موجود در داده‌های تحلیل بالا داشتید، ابتدا از
   همین ابزار استفاده کنید — نه حدس زدن و نه جست‌وجوی وب — چون این ابزار مستقیماً و با دقت بالا نام
   مکان را از روی مختصات برمی‌گرداند.
۲. tavily_search (جست‌وجوی وب عمومی): برای اطلاعات کیفی و کلی اقلیمی (نوع اقلیم، طبقه‌بندی کوپن،
   الگوهای فصلی بارش و دما، رویدادهای خاص مانند سیل یا خشک‌سالی در آن منطقه) که به‌صورت عددی در دسترس
   نیست و باید از منابع وب یافت شود. همچنین برای هر پرسش عمومی دیگری که نیاز به اطلاعات به‌روز از وب
   دارد.
۳. get_monthly_weather_stats: ابزار اصلیِ «تحلیل علّی» شما — داده‌های دقیق هواشناسی تاریخی (بایگانی
   Open-Meteo) را برای یک مختصات جغرافیایی و یک سال/ماه مشخص برمی‌گرداند: دمای بیشینه/کمینه روزانه،
   بارش، برف، سرعت باد و درصد روزهای برفی آن ماه. **نقش اصلی این ابزار پاسخ به سؤال مستقیم کاربر درباره‌ی
   وضعیت هوا نیست، بلکه یافتن دلیل محتمل روند یا ناهنجاری‌های شاخص‌های کیفیت آب است** — یعنی از این ابزار
   باید به‌عنوان بخشی از فرایند عادی و خودکار تحلیل خودتان استفاده کنید، دقیقاً مانند یک متخصص انسانی که
   قبل از تفسیر یک نوسان غیرعادی، بی‌درنگ و بدون نیاز به درخواست کاربر، داده‌ی هواشناسی همان ماه را
   بررسی می‌کند.

دستورالعمل‌های استفاده از ابزارها:
- **استفاده‌ی خودکار و پیش‌کنشانه از get_monthly_weather_stats (نه صرفاً واکنشی):** هر زمان که در
  داده‌های تحلیل بالا با یکی از موارد زیر مواجه شدید، پیش از پاسخ نهایی، به‌طور خودکار get_monthly_weather_stats
  را برای ماه‌(های) مرتبط فراخوانی کنید — حتی اگر کاربر صریحاً کلمه‌ی «هوا» یا «آب‌وهوا» را به کار نبرده
  باشد:
    الف) هر آیتم در فهرست anomalies (ناهنجاری‌های شناسایی‌شده با انحراف مطلق از میانه) — بررسی کنید آیا
         بارش شدید، ذوب برف (برف قابل‌توجه در ماه‌های قبل یا همان ماه)، یا باد شدید می‌تواند توضیح‌دهنده‌ی
         آن جهش/افت ناگهانی باشد.
    ب) ماه‌های extremes.minimum و extremes.maximum — همین بررسی برای مقادیر حداکثر و حداقل کل بازه.
    پ) وقتی trend.standard یا trend.seasonal یک روند معنادار (significant: true) نشان می‌دهد — چند ماه
       نمونه (برای مثال ابتدای، میانه، و انتهای بازه‌ی دارای روند) را از نظر بارش/دما/باد بررسی کنید تا
       ببینید آیا روند مشاهده‌شده با یک روند هواشناسی هم‌راستا (مثلاً کاهش تدریجی بارش، افزایش دما) قابل
       توضیح است یا خیر.
    ت) وقتی کاربر درباره‌ی «چرا» یک تغییر رخ داده، یا درباره‌ی مقایسه‌ی دو ماه/دو فصل سؤال می‌پرسد.
  خلاصه: get_monthly_weather_stats بخشی جدایی‌ناپذیر از تفسیر روند/ناهنجاری شماست، نه یک ابزار جانبی که
  فقط با درخواست مستقیم کاربر برای «وضعیت هوا» فعال می‌شود.
- برای مختصات این ابزار و reverse_geocode، همیشه از center_latitude / center_longitude موجود در داده‌های
  تحلیل بالا استفاده کنید (در صورت نبودن این مقادیر در داده‌ها، صریحاً به کاربر بگویید که مختصات منطقه
  در دسترس نیست). هرگز از کاربر نخواهید مختصات را برایتان ارسال کند.
- سایر ابزارها را فقط زمانی فراخوانی کنید که واقعاً لازم است: reverse_geocode هر زمان که نام منطقه هنوز
  شناسایی نشده؛ tavily_search برای زمینه‌ی کیفی/کلی اقلیمی یا رویدادهای خاص (سیل، خشک‌سالی) که در داده‌ی
  عددی Open-Meteo دیده نمی‌شود.
- هرگز مختصات یا نام منطقه را حدس نزنید. برای شناسایی نام منطقه، همیشه ابتدا از reverse_geocode استفاده
  کنید؛ فقط اگر reverse_geocode نتیجه‌ای نداد یا کاربر اطلاعات کیفی/توصیفی بیشتری خواست، سراغ
  tavily_search بروید.
- ترتیب پیشنهادی هنگام تفسیر یک نوسان یا ناهنجاری: ابتدا در صورت نامشخص بودن نام منطقه، آن را با
  reverse_geocode شناسایی کنید؛ سپس بلافاصله و بدون نیاز به تأیید کاربر، داده‌ی عددی دقیق ماه(های)
  موردنظر را با get_monthly_weather_stats بگیرید؛ و در صورت نیاز، زمینه‌ی کلی اقلیمی یا رویدادهای خاص
  را نیز با tavily_search تکمیل کنید.

دستورالعمل‌های مربوط به تاریخ (تقویم شمسی):
- کاربر با تقویم شمسی (جلالی) صحبت می‌کند و تاریخ‌ها را در پاسخ‌های خود باید به همین صورت (مثلاً «مرداد
  ۱۴۰۳») بیان کنید، مگر آنکه خود کاربر از تقویم میلادی استفاده کند.
- داده‌های تحلیل بالا (raw_observations، anomalies، extremes، seasonal_climatology) و ورودی‌های ابزار
  get_monthly_weather_stats بر پایه‌ی تاریخ میلادی (year, month) هستند. تبدیل بین تقویم شمسی و میلادی را
  همیشه خودتان، به‌صورت داخلی و بی‌صدا انجام دهید.
- هرگز از کاربر معادل میلادی یک تاریخ شمسی، یا مختصات جغرافیایی منطقه را نپرسید؛ این اطلاعات یا در
  داده‌های تحلیل بالا موجود است یا باید خودتان با ابزارها به دست آورید.

دستورالعمل‌های پاسخ‌گویی:
۱. پاسخ خود را در درجه‌ی اول بر پایه‌ی داده‌های JSON بالا و در صورت لزوم نتایج ابزارها بنا کنید و از
   دانش عمومی خود درباره‌ی کیفیت آب، سنجش‌ازدور و علوم محیط‌زیست برای تفسیر و تکمیل پاسخ استفاده کنید.
۲. تفسیر را مبتنی بر شواهد ارائه دهید؛ به‌جای نسبت‌دادن هر نوسان یا ناهنجاری به‌طور پیش‌فرض به «خطای
   حسگر»، ابتدا توضیح‌های محیطی، هیدرولوژیکی و هواشناسی محتمل را با کمک get_monthly_weather_stats در
   نظر بگیرید (رواناب فصلی، بارندگی، ذوب برف، سیل، رسوب‌گذاری، شکوفایی جلبکی و مانند آن).
۳. اگر داده‌ی کافی برای نتیجه‌گیری قطعی وجود ندارد (برای نمونه کمتر از ۲۴ ماه برای آزمون فصلی، یا
   نتایج ابزارها ناکافی/متناقض بود)، این محدودیت را صریح بیان کنید؛ حدس قطعی نزنید و هیچ واقعیتی را از
   خود نسازید.
۴. در پاسخ نهایی، در صورت استفاده از ابزارها، بین «شناسایی منطقه»، «داده‌ی عددی دقیق هواشناسی مرتبط با
   روند/ناهنجاری» و «زمینه‌ی کلی اقلیمی» تمایز قائل شوید.
۵. هر سه شاخص را با هم و در کنار یکدیگر تفسیر کنید، نه جدا از هم. به همبستگی‌های موجود در کلید
   relationships توجه کنید و هر جا الگوی مشترک یا واگرایی معناداری بین شاخص‌ها دیدید (برای نمونه
   افزایش هم‌زمان کدورت و مواد آلی محلول پس از یک ماه پربارش، یا افزایش کلروفیل بدون تغییر دو شاخص
   دیگر)، آن را صریح بیان کرده و توضیح فرایندی آن را ارائه دهید.
۶. اگر پرسش کاربر به زبان فارسی باشد، پاسخ باید کاملاً و فقط به زبان فارسی نوشته شود و از هیچ مخفف یا
   واژه‌ی انگلیسی استفاده نشود (برای نمونه به‌جای NDTI بنویسید «شاخص کدورت آب»، به‌جای NDCI بنویسید
   «شاخص کلروفیل»، و به‌جای CDOM بنویسید «مواد آلی محلول رنگی»؛ به‌جای MAD بنویسید «انحراف مطلق از
   میانه»).
۷. اگر پرسش کاربر به زبان دیگری باشد، به همان زبان پاسخ دهید.
"""


# =============================================================================
# نظر متخصص آب — Cached network helpers (reverse geocoding + historical
# weather)
# =============================================================================
# WHY THIS EXISTS: a monitored waterbody's center coordinates never move, and
# historical (past-month) weather data never changes once the month is over.
# Before this fix, both the reverse-geocode and the monthly-weather calls
# hit their external APIs fresh on every single agent turn — and a single
# user question can trigger several of these calls in sequence (identify
# region -> weather for month A -> weather for month B -> ...). Each call
# also had a generous 30s timeout with no cap on the *total* turn time, so a
# multi-tool question could run 60-90+ seconds while fully blocking
# Streamlit's script thread. During that time Streamlit can't send its
# normal keep-alive traffic, and Posit's reverse proxy (or any proxy in
# front of it) has its own idle/read timeout — commonly ~60s — and simply
# drops the connection from the outside once that's exceeded. Nothing raises
# a Python exception in that case, so the existing try/except in
# render_expert_chat_tab() never fires either: the user just sees no
# response at all.
#
# THE FIX: cache both lookups to disk with st.cache_data(persist="disk").
# The first time a region/month is asked about, the API is still called
# once; every question after that — for that region, for that month, by any
# user in any session on this deployment — is served instantly from cache
# with zero network round-trips. This removes almost all of the latency
# that was tripping the proxy's idle timeout. Per-call timeouts are also
# tightened from 30s to a (connect=5s, read=15s) tuple so an unresponsive
# API fails fast instead of silently eating the whole time budget.
# =============================================================================
@st.cache_data(ttl=None, persist="disk", show_spinner=False)
def _reverse_geocode_cached(lat: float, lon: float) -> list:
    """Cached OpenWeatherMap reverse-geocoding lookup. Coordinates are rounded
    to ~100m precision so nearby points within the same monitored waterbody
    reuse the same cache entry instead of each triggering a fresh request."""
    params = {
        "lat": round(lat, 3),
        "lon": round(lon, 3),
        "limit": 5,
        "appid": OPENWEATHER_API_KEY,
    }
    r = requests.get(
        "https://api.openweathermap.org/geo/1.0/reverse",
        params=params,
        timeout=(5, 15),  # (connect, read) — fail fast instead of hanging up to 30s
    )
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=None, persist="disk", show_spinner=False)
def _fetch_monthly_weather_cached(lat: float, lon: float, year: int, month: int) -> dict:
    """Cached Open-Meteo historical-archive lookup for one lat/lon/year/month.
    Historical weather for a past month never changes, so once fetched it is
    reused for every future question about that exact month — by any user,
    in any session — instead of hitting the API again.

    This helper backs the agent's get_monthly_weather_stats tool, whose main
    purpose is diagnostic: explaining *why* a turbidity/chlorophyll trend or
    anomaly occurred (heavy rainfall, snowmelt, wind-driven mixing, etc.),
    not just answering a direct question about the weather itself."""
    import calendar

    last_day = calendar.monthrange(year, month)[1]
    params = {
        "latitude": round(lat, 3),
        "longitude": round(lon, 3),
        "start_date": f"{year}-{month:02d}-01",
        "end_date": f"{year}-{month:02d}-{last_day}",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,snowfall_sum,wind_speed_10m_max",
        "timezone": "auto",
    }
    r = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params=params,
        timeout=(5, 15),  # (connect, read) — fail fast instead of hanging up to 30s
    )
    r.raise_for_status()
    return r.json()


def _get_expert_agent(analysis_json):
    """
    Build a fresh LangGraph ReAct agent with the three tools: OpenWeatherMap
    reverse geocoding, Tavily web search, and Open-Meteo historical weather.
    Rebuilt on every call since the system prompt embeds the current
    analysis JSON, which changes whenever new monitoring results are
    generated (see _expert_results_signature()). Building a ReAct agent is
    cheap (no network calls happen until a tool is actually invoked), so
    recreating it per question keeps the code simple and avoids
    stale-prompt bugs.

    API keys (TAVILY_API_KEY, OPENWEATHER_API_KEY) are hardcoded constants
    read directly from this file rather than from a .env file, since the
    Posit server this app runs on cannot read a local .env.

    The reverse_geocode and get_monthly_weather_stats tools below call the
    disk-cached helpers defined just above this function
    (_reverse_geocode_cached / _fetch_monthly_weather_cached) instead of
    hitting their APIs directly — see the comment on those helpers for why.

    Requires: langchain, langchain-openai, langgraph, langchain-tavily
    (pip install langchain langchain-openai langgraph langchain-tavily)
    """
    from langchain.chat_models import init_chat_model
    from langchain_core.tools import tool
    from langchain_tavily import TavilySearch
    from langgraph.prebuilt import create_react_agent

    # TavilySearch reads its key from the environment — set it directly from
    # the hardcoded constant above (no .env involved).
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

    model = init_chat_model(
        model=EXPERT_CHAT_MODEL,
        model_provider="openai",
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
        temperature=0.3,
        timeout=45,      # hard per-call cap on the LLM request itself
        max_retries=1,   # avoid a silent retry doubling the wait on a slow endpoint
    )

    tavily_tool = TavilySearch(max_results=5, topic="general")

    @tool
    def reverse_geocode(lat: float, lon: float) -> str:
        """Reverse-geocode a latitude/longitude pair into a place name
        (city/town/village, state/province, and country) using the
        OpenWeatherMap Geocoding API. Use this whenever you need to identify
        the name of the region/city/country for a given set of coordinates
        (e.g. the center_latitude / center_longitude of the monitored water
        body) — it is faster and more precise than a general web search for
        this specific purpose."""
        try:
            results = _reverse_geocode_cached(lat, lon)
        except Exception as e:
            return str({
                "lat": lat, "lon": lon, "results": [],
                "error": f"geocoding service unavailable or timed out: {e}",
            })

        if not results:
            return str({"lat": lat, "lon": lon, "results": [], "note": "No place name found for these coordinates."})

        return str({
            "lat": lat,
            "lon": lon,
            "results": [
                {
                    "name": item.get("name"),
                    "state": item.get("state"),
                    "country": item.get("country"),
                    "local_names": item.get("local_names"),
                }
                for item in results
            ],
        })

    @tool
    def get_monthly_weather_stats(lat: float, lon: float, year: int, month: int) -> str:
        """Get daily historical weather data (max/min temperature, precipitation,
        snowfall, wind speed) for a given latitude/longitude and a specific
        year/month, using the Open-Meteo historical archive API. Also returns
        the number of days with snowfall and the snow-day percentage for that
        month.

        This is your primary DIAGNOSTIC tool, not just a way to answer direct
        weather questions. Its main purpose is to help you explain the likely
        physical cause of a turbidity/chlorophyll trend or anomaly found in
        the statistical analysis JSON (e.g. a spike coinciding with heavy
        precipitation or rapid snowmelt, or a chlorophyll rise coinciding with
        calm, warm, low-wind conditions favorable to algal blooms).

        Call this tool proactively — as a normal, automatic step of your own
        analysis — for the month(s) around any anomaly in `anomalies`, around
        `extremes.minimum` / `extremes.maximum`, and around representative
        months of any statistically significant trend, even if the user never
        explicitly asks about the weather. Do not wait for the user to ask
        'what was the weather like' before checking; treat this the way a
        human water-quality expert would instinctively pull rainfall/snowmelt
        records before offering an interpretation of an unusual reading.

        Always pass the center_latitude / center_longitude already present in
        the analysis JSON, and the Gregorian year/month equivalent of the
        month you are investigating (convert internally from Shamsi/Jalali if
        that is how the user phrased the date) — never ask the user for
        coordinates or for the Gregorian date."""
        try:
            data = _fetch_monthly_weather_cached(lat, lon, year, month)
        except Exception as e:
            return str({"error": f"weather service unavailable or timed out: {e}"})

        daily = data.get("daily", {})
        snowfall = daily.get("snowfall_sum", [])
        total_days = len(daily.get("time", []))
        snow_days = sum(1 for s in snowfall if s and s > 0)
        snow_pct = round(100 * snow_days / total_days, 1) if total_days else None

        data["summary"] = {
            "total_days": total_days,
            "snow_days": snow_days,
            "snow_day_percentage": snow_pct,
            "temperature_max_monthly": daily.get("temperature_2m_max"),
            "temperature_min_monthly": daily.get("temperature_2m_min"),
            "precipitation_monthly": daily.get("precipitation_sum"),
            "snowfall_monthly": snowfall,
        }
        return str(data)

    tools = [reverse_geocode, tavily_tool, get_monthly_weather_stats]
    system_prompt = _build_agent_system_prompt(analysis_json)
    return create_react_agent(model, tools, prompt=system_prompt)


def ask_water_quality_expert(question, analysis_json, chat_history):
    """
    Send `question` to the water-quality expert agent, giving it the running
    chat history for context. chat_history is a list of
    {"role": "user"/"assistant", "content": ...} dicts — the same shape
    already used elsewhere in the app for st.session_state.expert_chat_history,
    so no other call site needs to change.
    """
    from langchain_core.messages import HumanMessage, AIMessage

    agent = _get_expert_agent(analysis_json)

    messages = []
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=question))

    result = agent.invoke({"messages": messages})
    return result["messages"][-1].content


def _inject_persian_chat_css():
    """
    Right-to-left layout + Persian font for the چت با متخصص tab. Streamlit's
    built-in chat elements (st.chat_message / st.chat_input) are LTR by
    default, which misaligns Persian text; this forces RTL direction and a
    Persian-friendly font stack (falls back gracefully if "B Nazanin" is not
    installed on the viewer's system, since it is not a free web font).
    """
    st.markdown(
        """
        <style>
        [data-testid="stChatMessage"],
        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] li,
        [data-testid="stChatMessage"] div,
        [data-testid="stChatMessage"] span {
            direction: rtl;
            text-align: right;
            font-family: "B Nazanin", "BNazanin", "Vazirmatn", Tahoma, sans-serif;
            font-size: 22px;
            line-height: 1.9;
        }
        [data-testid="stChatMessage"] {
            background: #F7FCFD;
            border: 1px solid #CDEBEF;
            border-radius: 16px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.9rem;
            box-shadow: 0 2px 10px rgba(10, 63, 74, 0.07);
        }
        [data-testid="stChatInput"] textarea {
            direction: rtl;
            text-align: right;
            font-family: "B Nazanin", "BNazanin", "Vazirmatn", Tahoma, sans-serif;
            font-size: 20px;
        }
        [data-testid="stChatInput"] {
            border-radius: 16px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_expert_chat_tab():
    """
    صفحه «نظر متخصص آب»: به‌صورت خودکار خروجی اکسل پایش را می‌گیرد، پایپ‌لاین
    تحلیل آماری موجود را روی آن اجرا می‌کند، خلاصه JSON تولید می‌کند، و یک
    رابط گفتگو با یک عامل هوشمند (LangGraph ReAct agent) در اختیار کاربر
    قرار می‌دهد. این عامل علاوه بر خلاصه JSON، به سه ابزار نیز دسترسی دارد:
    شناسایی نام منطقه از روی مختصات (reverse geocoding با OpenWeatherMap،
    نتایج آن روی دیسک کش می‌شوند)، جست‌وجوی وب برای زمینه‌ی کلی اقلیمی
    (Tavily)، و دریافت داده‌های دقیق هواشناسی تاریخی (Open-Meteo، نتایج آن
    نیز روی دیسک کش می‌شوند) برای مختصات مرکز منطقه — این ابزار هواشناسی
    به‌طور خودکار در تحلیل علّی روندها و ناهنجاری‌ها به کار می‌رود، نه فقط در
    پاسخ به سؤال مستقیم درباره‌ی وضعیت هوا.
    """
    _inject_persian_chat_css()

    _render_active_section_badge("💬", "چت با متخصص آب", "#E08E0B", "#F5A524")

    if 'expert_chat_history' not in st.session_state:
        st.session_state.expert_chat_history = []
    if 'expert_analysis_json' not in st.session_state:
        st.session_state.expert_analysis_json = None
    if 'expert_analysis_signature' not in st.session_state:
        st.session_state.expert_analysis_signature = None

    if not any(st.session_state.results.get(p) for p in ALL_PARAMETERS):
        st.info("برای استفاده از این بخش، ابتدا پایش را اجرا کنید تا داده‌ای برای تحلیل وجود داشته باشد.")
        return

    signature = _expert_results_signature()
    if st.session_state.expert_analysis_json is None or st.session_state.expert_analysis_signature != signature:
        with st.spinner("در حال تحلیل آماری داده‌های سری زمانی..."):
            try:
                excel_bytes = generate_combined_timeseries_excel()
                analysis = analyze_water_quality_from_bytes(excel_bytes)
                st.session_state.expert_analysis_json = json.dumps(analysis, ensure_ascii=False, indent=2)
                st.session_state.expert_analysis_signature = signature
                st.session_state.expert_chat_history = []  # data changed -> start a fresh conversation
            except Exception as e:
                st.error(f"خطا در تحلیل داده‌ها: {e}")
                return

    with st.expander("📄 خلاصه تحلیل (JSON) ارسال‌شده به متخصص هوش مصنوعی"):
        st.code(st.session_state.expert_analysis_json, language="json")

    if st.button("🗑️ شروع گفتگوی جدید", key="expert_chat_reset"):
        st.session_state.expert_chat_history = []
        st.rerun()

    st.divider()

    for msg in st.session_state.expert_chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_question = st.chat_input("سؤال خود را درباره کیفیت آب این منطقه بپرسید...")
    if user_question:
        st.session_state.expert_chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("در حال بررسی توسط متخصص هوش مصنوعی..."):
                try:
                    answer = ask_water_quality_expert(
                        user_question,
                        st.session_state.expert_analysis_json,
                        st.session_state.expert_chat_history[:-1]
                    )
                except Exception as e:
                    answer = f"خطا در ارتباط با عامل هوش مصنوعی: {e}"
                st.markdown(answer)

        st.session_state.expert_chat_history.append({"role": "assistant", "content": answer})


# =============================================================================
# Global App Styling — Persian (B Nazanin) font + RTL text + professional
# color palette for general UI chrome only (buttons, headers, inputs,
# sidebar, tabs, metrics, alerts, dataframes, progress bar). This is purely
# presentational: it does not touch the turbidity/chlorophyll colormaps
# (create_turbidity_colormap / create_chlorophyll_colormap) or any of the
# matplotlib figures used to render scientific results, and it does not
# alter any data-processing, calculation, or workflow logic.
# =============================================================================
@st.cache_data(show_spinner=False)
def _load_bnazanin_font():
    """
    Look for a local B Nazanin font file (see BNAZANIN_FONT_CANDIDATES) and, if
    found, return (base64_data, css_format, absolute_path) so it can be embedded
    straight into the page as an @font-face rule. Falls back to the optional
    Google Drive copy, and finally to (None, None, None) — in which case the app
    keeps using the existing "locally installed B Nazanin, else Vazirmatn"
    behaviour and nothing breaks.
    """
    ext_to_format = {
        ".woff2": "woff2",
        ".woff": "woff",
        ".ttf": "truetype",
        ".otf": "opentype",
    }

    base_dir = os.path.dirname(os.path.abspath(__file__))

    for rel_path in BNAZANIN_FONT_CANDIDATES:
        for candidate in (os.path.join(base_dir, rel_path), rel_path):
            if os.path.isfile(candidate):
                ext = os.path.splitext(candidate)[1].lower()
                fmt = ext_to_format.get(ext)
                if not fmt:
                    continue
                try:
                    with open(candidate, "rb") as f:
                        return base64.b64encode(f.read()).decode(), fmt, os.path.abspath(candidate)
                except Exception:
                    continue

    if BNAZANIN_FONT_DRIVE_ID:
        try:
            url = f"https://drive.google.com/uc?export=download&id={BNAZANIN_FONT_DRIVE_ID}"
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            return base64.b64encode(response.content).decode(), "truetype", None
        except Exception:
            pass

    return None, None, None


def _bnazanin_font_face_css():
    """Return the @font-face block for B Nazanin (empty string if unavailable)."""
    font_b64, fmt, font_path = _load_bnazanin_font()
    if not font_b64:
        return ""

    # Make the same font available to matplotlib as well, so the legends and the
    # time-series charts use B Nazanin too instead of the generic fallback.
    if font_path and os.path.splitext(font_path)[1].lower() in (".ttf", ".otf"):
        try:
            import matplotlib.font_manager as fm
            fm.fontManager.addfont(font_path)
        except Exception:
            pass

    return f"""
        @font-face {{
            font-family: 'B Nazanin';
            src: url(data:font/{fmt};charset=utf-8;base64,{font_b64}) format('{fmt}');
            font-weight: normal;
            font-style: normal;
            font-display: swap;
        }}
        @font-face {{
            font-family: 'B Nazanin';
            src: url(data:font/{fmt};charset=utf-8;base64,{font_b64}) format('{fmt}');
            font-weight: bold;
            font-style: normal;
            font-display: swap;
        }}
    """


def _inject_global_app_css():
    """
    Applies a light, water-themed color palette plus the Persian "B Nazanin"
    font (falling back gracefully to Vazirmatn/Tahoma if it is not installed
    on the viewer's system, since it is not a free web font) and right-to-left
    text alignment to Streamlit's general UI chrome. Scoped to text/UI
    elements only — deliberately does not force RTL on the whole document
    body, so widgets such as the folium map, matplotlib figures, and layout
    columns keep their normal structure and behaviour.
    """
    # Embedded B Nazanin (if a font file was supplied) — injected first so the
    # font-family stacks below resolve to the real font for every visitor.
    font_face = _bnazanin_font_face_css()
    if font_face:
        st.markdown(f"<style>{font_face}</style>", unsafe_allow_html=True)

    st.markdown(
        """
        <style>
        /* Reliable Persian web font (loads even when "B Nazanin" is not
           installed locally on the viewer's machine — B Nazanin is still
           tried first via the font-family stack below, this is just a
           good-looking, always-available fallback instead of Tahoma). */
        @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;600;700;800&display=swap');

        /* =====================================================================
           Palette — deep ocean teal + a warm amber accent for emphasis.
           Used only for general UI chrome (never for the turbidity/chlorophyll
           scientific colormaps, which are generated separately in Python).
           ===================================================================== */
        :root {
            --wq-navy:        #0A3F4A;
            --wq-teal-dark:   #0B6E76;
            --wq-teal:        #0E8E99;
            --wq-teal-light:  #2FC2CE;
            --wq-amber:       #F5A524;
            --wq-amber-dark:  #E08E0B;
            --wq-bg-1:        #EAF7F9;
            --wq-bg-2:        #F7FCFD;
            --wq-card:        #FFFFFF;
            --wq-border:      #CDEBEF;
        }

        /* ---- Persian font (applied to text-bearing UI elements) ---- */
        html, body, [class*="css"],
        .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
        .stText, .stCaption, label, .stButton > button, .stDownloadButton > button,
        .stTextInput input, .stNumberInput input, .stDateInput input,
        .stSelectbox div, .stTabs, .stAlert, .stAlert p,
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"],
        [data-testid="stMetricDelta"], [data-testid="stDataFrame"],
        .streamlit-expanderHeader, h1, h2, h3, h4, h5, h6 {
            font-family: "B Nazanin", "BNazanin", "Vazirmatn", Tahoma, sans-serif;
        }

        /* ---- RTL alignment for Persian text blocks (scoped, not global) ---- */
        .stMarkdown, .stMarkdown p, .stMarkdown li,
        .stAlert, .stAlert p, .streamlit-expanderHeader,
        h1, h2, h3, h4, h5, h6, .stCaption, label {
            direction: rtl;
            text-align: right;
        }

        /* ---- Larger, easier-to-read body / subsection text ---- */
        .stMarkdown p, .stMarkdown li {
            font-size: 1.12rem;
            line-height: 2;
        }
        .stAlert p, .stAlert div {
            font-size: 1.08rem;
            line-height: 1.9;
        }
        .stCaption, [data-testid="stCaptionContainer"] {
            font-size: 1rem !important;
        }

        /* ---- App background: soft, professional water-inspired gradient ---- */
        .stApp {
            background: linear-gradient(160deg, var(--wq-bg-1) 0%, var(--wq-bg-2) 55%, #FDF7EC 100%);
        }

        /* ---- Centered content column + general vertical rhythm ---- */
        .block-container {
            max-width: 1250px;
            padding-top: 1.6rem;
            padding-bottom: 3rem;
            margin: 0 auto;
        }
        [data-testid="stElementContainer"] {
            margin-bottom: 0.3rem;
        }
        [data-testid="stHorizontalBlock"] {
            gap: 1.1rem;
        }
        hr {
            margin: 1.7rem 0 !important;
            opacity: 0.55;
        }

        /* ---- Center any embedded iframe widgets (e.g. the folium map) ---- */
        iframe {
            display: block;
            margin-left: auto;
            margin-right: auto;
        }

        /* ---- Main page title ---- */
        h1 {
            color: var(--wq-navy);
            font-weight: 800;
            font-size: 2.1rem;
            font-family: "B Nazanin", "BNazanin", "Vazirmatn", Tahoma, sans-serif;
            background: linear-gradient(90deg, var(--wq-navy) 0%, var(--wq-teal) 60%, var(--wq-teal-light) 100%);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            border-bottom: 3px solid var(--wq-teal-light);
            padding-bottom: 0.5rem;
            display: inline-block;
        }

        /* ---- Big section titles (st.header, e.g. "1️⃣ ...", "2️⃣ ...", "3️⃣ ...") ---- */
        h2 {
            color: var(--wq-navy) !important;
            font-weight: 800;
            font-size: 1.85rem;
            line-height: 1.6;
            background: linear-gradient(90deg, #DFF4F6 0%, #F3FBFC 85%);
            border-right: 6px solid var(--wq-amber);
            border-radius: 10px;
            padding: 0.7rem 1.1rem;
            margin: 1.6rem 0 1rem 0;
            box-shadow: 0 2px 8px rgba(10, 63, 74, 0.08);
        }

        /* ---- Sub-titles (st.subheader / "#### " markdown) ---- */
        h3 {
            color: var(--wq-teal-dark);
            font-weight: 800;
            font-size: 1.45rem;
            line-height: 1.7;
            border-right: 4px solid var(--wq-teal-light);
            padding: 0.35rem 0.9rem;
            margin: 1.3rem 0 0.7rem 0;
            background: linear-gradient(90deg, #EFFBFC 0%, transparent 100%);
            border-radius: 8px;
        }

        /* ---- Sub-section titles (st.markdown "#### ") ---- */
        h4 {
            color: var(--wq-teal-dark);
            font-weight: 700;
            font-size: 1.28rem;
            letter-spacing: 0.2px;
            margin: 1.1rem 0 0.55rem 0;
        }

        /* ---- Sidebar ---- */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, var(--wq-navy) 0%, var(--wq-teal-dark) 100%);
        }
        section[data-testid="stSidebar"] * {
            color: #EAF7F9 !important;
        }
        section[data-testid="stSidebar"] .stButton > button {
            background: rgba(255,255,255,0.10);
            color: #EAF7F9;
            border: 1px solid rgba(255,255,255,0.35);
            box-shadow: none;
        }
        section[data-testid="stSidebar"] .stButton > button:hover {
            background: rgba(255,255,255,0.22);
            transform: none;
        }

        /* ---- Buttons (general) ---- */
        .stButton > button, .stDownloadButton > button {
            background: linear-gradient(135deg, var(--wq-teal) 0%, var(--wq-teal-light) 100%);
            color: #ffffff;
            border: none;
            border-radius: 12px;
            font-weight: 700;
            padding: 0.55rem 1.4rem;
            transition: transform 0.15s ease, box-shadow 0.15s ease, background 0.15s ease;
            box-shadow: 0 3px 10px rgba(14, 142, 153, 0.28);
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            background: linear-gradient(135deg, var(--wq-teal-dark) 0%, var(--wq-teal) 100%);
            box-shadow: 0 6px 16px rgba(14, 142, 153, 0.38);
            transform: translateY(-2px);
        }
        .stButton > button:active, .stDownloadButton > button:active {
            transform: translateY(0);
        }
        .stButton > button:disabled {
            background: #D7E1E3;
            color: #8FA3A8;
            box-shadow: none;
            transform: none;
        }

        /* ---- Primary call-to-action button (e.g. "🚀 شروع پایش") ---- */
        .stButton > button[kind="primary"],
        .stButton > button[kind="primaryFormSubmit"],
        [data-testid="baseButton-primary"] {
            background: linear-gradient(135deg, var(--wq-amber) 0%, var(--wq-amber-dark) 100%) !important;
            color: #ffffff !important;
            box-shadow: 0 4px 12px rgba(224, 142, 11, 0.35) !important;
        }
        .stButton > button[kind="primary"]:hover,
        .stButton > button[kind="primaryFormSubmit"]:hover,
        [data-testid="baseButton-primary"]:hover {
            background: linear-gradient(135deg, var(--wq-amber-dark) 0%, #C97A08 100%) !important;
            box-shadow: 0 7px 18px rgba(224, 142, 11, 0.45) !important;
            transform: translateY(-2px);
        }

        /* ---- Field labels (e.g. "از تاریخ", "تا تاریخ (غیرشامل)", "🎯 انتخاب منطقه") ---- */
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] label,
        .stDateInput label, .stSelectbox label,
        .stTextInput label, .stNumberInput label {
            font-size: 1.15rem !important;
            font-weight: 700 !important;
            color: var(--wq-navy) !important;
        }

        /* ---- Text / date / select inputs ---- */
        .stTextInput input, .stNumberInput input, .stDateInput input {
            border-radius: 10px !important;
            border: 1px solid var(--wq-border) !important;
            font-size: 1.05rem !important;
        }
        .stTextInput input:focus, .stNumberInput input:focus, .stDateInput input:focus {
            border-color: var(--wq-teal) !important;
            box-shadow: 0 0 0 2px rgba(14, 142, 153, 0.18) !important;
        }
        .stSelectbox > div > div {
            border-radius: 10px !important;
            border-color: var(--wq-border) !important;
        }

        /* ---- Tabs ---- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            width: 100%;
            display: flex;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #E1F1F3;
            border-radius: 16px 16px 0 0;
            color: var(--wq-navy);
            font-weight: 800;
            padding: 1.15rem 2.6rem;
            min-height: 3.7rem;
            display: flex;
            align-items: center;
            justify-content: center;
            flex: 1 1 0;
        }
        .stTabs [data-baseweb="tab"] p {
            font-size: 1.65rem !important;
            font-weight: 800 !important;
            font-family: "B Nazanin", "BNazanin", "Vazirmatn", Tahoma, sans-serif !important;
        }
        .stTabs [data-baseweb="tab-panel"] {
            padding-top: 1.3rem;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, var(--wq-teal) 0%, var(--wq-teal-light) 100%) !important;
            color: #ffffff !important;
            box-shadow: 0 4px 14px rgba(14, 142, 153, 0.35) !important;
            border-bottom: 4px solid var(--wq-amber) !important;
            transform: translateY(-2px);
        }

        /* ---- Metrics ---- */
        [data-testid="stMetric"] {
            background: var(--wq-card);
            border: 1px solid var(--wq-border);
            border-top: 3px solid var(--wq-teal-light);
            border-radius: 12px;
            padding: 14px;
            box-shadow: 0 2px 8px rgba(10, 63, 74, 0.07);
        }

        /* ---- خلاصه آماری metric numbers (میانگین/حداکثر/حداقل ...) — larger, Bnazanin ---- */
        [data-testid="stMetricValue"] {
            font-size: 2.1rem !important;
            font-weight: 800 !important;
            font-family: "B Nazanin", "BNazanin", "Vazirmatn", Tahoma, sans-serif !important;
        }
        [data-testid="stMetricLabel"] p {
            font-size: 1.35rem !important;
            font-weight: 700 !important;
            font-family: "B Nazanin", "BNazanin", "Vazirmatn", Tahoma, sans-serif !important;
        }

        /* ---- Alerts / info / success / warning boxes ---- */
        .stAlert {
            border-radius: 12px;
            box-shadow: 0 1px 6px rgba(10, 63, 74, 0.06);
        }

        /* ---- Expanders ---- */
        .streamlit-expanderHeader {
            font-weight: 700;
            color: var(--wq-navy);
        }

        /* ---- Progress bar ---- */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, var(--wq-teal) 0%, var(--wq-amber) 100%);
        }

        /* ---- Dataframes / tables ---- */
        [data-testid="stDataFrame"] {
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid var(--wq-border);
        }

        /* ---- Step headers (workflow sections 1️⃣-4️⃣ replacement) ---- */
        .wq-step-header {
            display: flex;
            align-items: center;
            gap: 0.8rem;
            direction: rtl;
            text-align: right;
            background: linear-gradient(90deg, #DFF4F6 0%, #F3FBFC 85%);
            border-right: 6px solid var(--wq-amber);
            border-radius: 14px;
            padding: 0.95rem 1.4rem;
            margin: 1.9rem 0 1.2rem 0;
            box-shadow: 0 3px 12px rgba(10, 63, 74, 0.10);
        }
        .wq-step-number {
            display: flex;
            align-items: center;
            justify-content: center;
            min-width: 2.2rem;
            height: 2.2rem;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--wq-navy) 0%, var(--wq-teal) 100%);
            color: #ffffff;
            font-weight: 800;
            font-size: 1.05rem;
            flex-shrink: 0;
        }
        .wq-step-icon {
            font-size: 1.75rem;
            line-height: 1;
            flex-shrink: 0;
        }
        .wq-step-title {
            font-size: 1.6rem;
            font-weight: 800;
            color: var(--wq-navy);
            font-family: "B Nazanin", "BNazanin", "Vazirmatn", Tahoma, sans-serif;
        }

        /* ---- Active-page badge (شown at the top of each result tab) ---- */
        .wq-page-badge {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            direction: rtl;
            text-align: right;
            background: linear-gradient(135deg, var(--wq-badge-start) 0%, var(--wq-badge-end) 100%);
            color: #ffffff;
            border-radius: 14px;
            padding: 0.9rem 1.5rem;
            margin: 0.2rem 0 1.5rem 0;
            box-shadow: 0 4px 14px rgba(10, 63, 74, 0.20);
        }
        .wq-page-badge-icon {
            font-size: 1.95rem;
            line-height: 1;
        }
        .wq-page-badge-text {
            font-size: 1.5rem;
            font-weight: 800;
            font-family: "B Nazanin", "BNazanin", "Vazirmatn", Tahoma, sans-serif;
        }

        /* ---- Styled monthly time-series table (replaces st.dataframe) ---- */
        .wq-table-wrapper {
            direction: rtl;
            overflow-x: auto;
            border-radius: 14px;
            border: 1px solid var(--wq-border);
            box-shadow: 0 2px 10px rgba(10, 63, 74, 0.08);
            margin: 0.5rem 0 0.7rem 0;
        }
        .wq-styled-table {
            width: 100%;
            border-collapse: collapse;
            font-family: "B Nazanin", "BNazanin", "Vazirmatn", Tahoma, sans-serif;
            font-size: 1.1rem;
        }
        .wq-styled-table thead th {
            background: linear-gradient(135deg, var(--wq-teal) 0%, var(--wq-teal-dark) 100%);
            color: #ffffff;
            font-weight: 800;
            padding: 0.8rem 1rem;
            text-align: center;
            border: none;
        }
        .wq-styled-table tbody td {
            padding: 0.65rem 1rem;
            text-align: center;
            color: var(--wq-navy);
            border-bottom: 1px solid var(--wq-border);
        }
        .wq-styled-table tbody tr:nth-child(even) {
            background-color: var(--wq-bg-1);
        }
        .wq-styled-table tbody tr:hover {
            background-color: #DDF2F4;
            transition: background-color 0.15s ease;
        }

        /* =====================================================================
           TOP NAVIGATION BAR (the four pages)
           The nav is built from four normal Streamlit buttons laid out in
           columns; everything below is purely the styling that turns them into
           a large, professional, right-to-left tab bar.
           Two selector families are used so the styling works on every recent
           Streamlit version: the modern per-key class (.st-key-wqnav_*) and a
           structural fallback anchored on the hidden .wq-nav-anchor marker.
           ===================================================================== */
        .wq-nav-anchor { display: none; }

        /* Right-to-left order + tighter spacing for the nav row */
        div[class*="st-key-wqnav_"] { margin: 0 !important; }

        [data-testid="stElementContainer"]:has(.wq-nav-anchor) + [data-testid="stHorizontalBlock"],
        .element-container:has(.wq-nav-anchor) + [data-testid="stHorizontalBlock"],
        [data-testid="stElementContainer"]:has(.wq-nav-anchor) + div[data-testid="stHorizontalBlock"] {
            direction: rtl;
            flex-direction: row-reverse;
            gap: 0.55rem !important;
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid var(--wq-border);
            border-radius: 18px;
            padding: 0.5rem;
            box-shadow: 0 4px 18px rgba(10, 63, 74, 0.10);
            margin-bottom: 1.6rem;
            backdrop-filter: blur(4px);
        }

        /* ---- Base (inactive) tab look ---- */
        div[class*="st-key-wqnav_"] .stButton > button,
        [data-testid="stElementContainer"]:has(.wq-nav-anchor) + [data-testid="stHorizontalBlock"] .stButton > button {
            background: #E3F3F5 !important;
            color: var(--wq-navy) !important;
            border: 1px solid var(--wq-border) !important;
            border-radius: 14px !important;
            box-shadow: none !important;
            padding: 0.8rem 0.35rem !important;
            min-height: 4.1rem !important;
            font-family: "B Nazanin", "BNazanin", "Vazirmatn", Tahoma, sans-serif !important;
            font-size: 1.5rem !important;
            font-weight: 800 !important;
            line-height: 1.5 !important;
            direction: rtl !important;
            transition: transform 0.15s ease, box-shadow 0.15s ease, background 0.15s ease;
        }
        div[class*="st-key-wqnav_"] .stButton > button p,
        [data-testid="stElementContainer"]:has(.wq-nav-anchor) + [data-testid="stHorizontalBlock"] .stButton > button p {
            font-family: "B Nazanin", "BNazanin", "Vazirmatn", Tahoma, sans-serif !important;
            font-size: 1.5rem !important;
            font-weight: 800 !important;
            line-height: 1.5 !important;
            margin: 0 !important;
        }

        div[class*="st-key-wqnav_"] .stButton > button:hover:not(:disabled),
        [data-testid="stElementContainer"]:has(.wq-nav-anchor) + [data-testid="stHorizontalBlock"] .stButton > button:hover:not(:disabled) {
            background: #D2ECEF !important;
            border-color: var(--wq-teal-light) !important;
            transform: translateY(-2px);
            box-shadow: 0 5px 14px rgba(14, 142, 153, 0.22) !important;
        }

        /* ---- Active tab (rendered as a "primary" button) ---- */
        div[class*="st-key-wqnav_"] .stButton > button[kind="primary"],
        [data-testid="stElementContainer"]:has(.wq-nav-anchor) + [data-testid="stHorizontalBlock"] .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, var(--wq-navy) 0%, var(--wq-teal) 55%, var(--wq-teal-light) 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-bottom: 5px solid var(--wq-amber) !important;
            box-shadow: 0 6px 18px rgba(10, 63, 74, 0.30) !important;
            transform: translateY(-2px);
        }
        div[class*="st-key-wqnav_"] .stButton > button[kind="primary"]:hover,
        [data-testid="stElementContainer"]:has(.wq-nav-anchor) + [data-testid="stHorizontalBlock"] .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, var(--wq-navy) 0%, var(--wq-teal-dark) 55%, var(--wq-teal) 100%) !important;
        }

        /* ---- Locked tabs (before the monitoring run has produced results) ---- */
        div[class*="st-key-wqnav_"] .stButton > button:disabled,
        [data-testid="stElementContainer"]:has(.wq-nav-anchor) + [data-testid="stHorizontalBlock"] .stButton > button:disabled {
            background: #EDF3F4 !important;
            color: #A3B6BA !important;
            border: 1px dashed #C6D7DA !important;
            box-shadow: none !important;
            transform: none;
            cursor: not-allowed;
        }

        /* ---- Small hint line under the nav bar ---- */
        .wq-nav-hint {
            direction: rtl;
            text-align: center;
            font-family: "B Nazanin", "BNazanin", "Vazirmatn", Tahoma, sans-serif;
            font-size: 1.12rem;
            color: #5C7B80;
            margin: -1.05rem 0 1.5rem 0;
        }

        /* ---- App header (logo + title + subtitle) ---- */
        .wq-app-header {
            display: flex;
            align-items: center;
            gap: 0.85rem;
            direction: rtl;
            margin-bottom: 0.25rem;
        }
        .wq-app-header img {
            height: 3.1rem;
            width: auto;
            border-radius: 10px;
        }
        .wq-app-title {
            font-family: "B Nazanin", "BNazanin", "Vazirmatn", Tahoma, sans-serif;
            font-size: 2.45rem;
            font-weight: 800;
            line-height: 1.5;
            background: linear-gradient(90deg, var(--wq-navy) 0%, var(--wq-teal) 60%, var(--wq-teal-light) 100%);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .wq-app-subtitle {
            direction: rtl;
            text-align: right;
            font-family: "B Nazanin", "BNazanin", "Vazirmatn", Tahoma, sans-serif;
            font-size: 1.22rem;
            color: #3C6067;
            margin: 0 0 1.1rem 0;
            padding-bottom: 0.9rem;
            border-bottom: 3px solid var(--wq-teal-light);
        }

        /* ---- Status strip on the first page (replaces the old sidebar) ---- */
        .wq-status-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            direction: rtl;
            border-radius: 999px;
            padding: 0.55rem 1.25rem;
            font-family: "B Nazanin", "BNazanin", "Vazirmatn", Tahoma, sans-serif;
            font-size: 1.2rem;
            font-weight: 700;
            border: 1px solid transparent;
        }
        .wq-status-ready   { background:#E4F7EC; color:#14713F; border-color:#B9E7CD; }
        .wq-status-idle    { background:#EDF5F6; color:#40646B; border-color:#CFE4E7; }
        .wq-status-running { background:#FEF3DE; color:#96610A; border-color:#F7DDA9; }
        /* Right-align the status chip (RTL) and vertically match the button */
        .wq-status-line {
            direction: rtl;
            text-align: right;
            display: flex;
            align-items: center;
            justify-content: flex-start;
            height: 100%;
            min-height: 2.6rem;
        }
        .wq-status-strip-spacer { height: 0.45rem; }

        /* ---- Results-ready banner on the first page ---- */
        .wq-ready-banner {
            direction: rtl;
            text-align: right;
            background: linear-gradient(135deg, #0B6E76 0%, #2FC2CE 100%);
            color: #ffffff;
            border-radius: 16px;
            padding: 1.1rem 1.5rem;
            margin: 1.2rem 0 0.8rem 0;
            box-shadow: 0 5px 18px rgba(10, 63, 74, 0.22);
            font-family: "B Nazanin", "BNazanin", "Vazirmatn", Tahoma, sans-serif;
            font-size: 1.35rem;
            font-weight: 700;
            line-height: 1.9;
        }

        /* ---- Slightly larger download / action buttons ---- */
        .stDownloadButton > button p {
            font-size: 1.2rem !important;
            font-weight: 700 !important;
        }
        .stButton > button p {
            font-size: 1.1rem;
            font-weight: 700;
        }

        /* ---- Expander headers (تصاویر پردازش‌شده / جدول داده‌های ماهانه ...) ---- */
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary p,
        details summary p {
            direction: rtl;
            font-family: "B Nazanin", "BNazanin", "Vazirmatn", Tahoma, sans-serif !important;
            font-size: 1.25rem !important;
            font-weight: 700 !important;
            color: var(--wq-navy) !important;
        }
        [data-testid="stExpander"] details {
            border-radius: 14px !important;
            border: 1px solid var(--wq-border) !important;
            background: rgba(255, 255, 255, 0.65);
            box-shadow: 0 2px 8px rgba(10, 63, 74, 0.06);
        }

        /* ---- Image captions under the monthly imagery ---- */
        [data-testid="stImageCaption"] {
            direction: rtl;
            text-align: center;
            font-family: "B Nazanin", "BNazanin", "Vazirmatn", Tahoma, sans-serif !important;
            font-size: 1.05rem !important;
            color: var(--wq-teal-dark) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_step_header(step_number, icon, title):
    """Colored, icon-led replacement for the old plain st.header() numbered
    workflow titles (1️⃣/2️⃣/3️⃣ …). Purely presentational."""
    st.markdown(
        f"""
        <div class="wq-step-header">
            <div class="wq-step-number">{step_number}</div>
            <div class="wq-step-icon">{icon}</div>
            <div class="wq-step-title">{title}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_active_section_badge(icon, title, start_hex, end_hex):
    """Colored 'you are here' badge shown at the top of each of the three
    result tabs (Turbidity / Chlorophyll / Expert chat) so the currently
    active page is always visually obvious. Purely presentational."""
    st.markdown(
        f"""
        <div class="wq-page-badge" style="--wq-badge-start:{start_hex}; --wq-badge-end:{end_hex};">
            <span class="wq-page-badge-icon">{icon}</span>
            <span class="wq-page-badge-text">{title}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_persian_dataframe_html(df):
    """Render a DataFrame as a custom-styled, RTL Persian HTML table. This is
    a pure presentation swap for st.dataframe: identical columns and values
    are shown, only the typography/spacing/colors differ (st.dataframe's
    canvas-based grid cannot be restyled with CSS, which is why a plain HTML
    table is used here instead)."""
    html_table = df.to_html(index=False, escape=False, border=0, classes="wq-styled-table")
    st.markdown(f'<div class="wq-table-wrapper">{html_table}</div>', unsafe_allow_html=True)


# =============================================================================
# App logo (replaces the 🌊 wave emoji at the top of the UI)
# =============================================================================
@st.cache_data(show_spinner=False)
def _get_app_logo_base64():
    """Load the app icon from Google Drive and return it as a base64 string
    for inline embedding next to the page title.

    The local assets/app_logo.png file is not present on this deployment
    (Posit Cloud), so the logo is fetched from Google Drive instead using
    the file's shareable ID. The Drive file must be shared as "Anyone with
    the link -> Viewer" for this request to succeed.

    Cached so the file is only downloaded once per session. On any failure
    (network issue, permissions, bad ID) this returns an empty string
    instead of raising, so a missing/unreachable logo never crashes the app
    — the <img> tag will just render nothing instead.
    """
    GOOGLE_DRIVE_LOGO_FILE_ID = "1sX8iMkqYvYDJ-XocOa8Hmr0e6-4R-n6i"
    url = f"https://drive.google.com/thumbnail?id={GOOGLE_DRIVE_LOGO_FILE_ID}&sz=w1000"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return base64.b64encode(response.content).decode()
    except Exception:
        return ""


# =============================================================================
# Top navigation (four pages) — replaces the old single scrolling page
# =============================================================================
# key, icon, Persian title
NAV_PAGES = [
    ("setup",       "🛰️", "تعریف پایش"),
    ("turbidity",   "🌊", "کدورت"),
    ("chlorophyll", "🌿", "کلروفیل"),
    ("cdom",        "🍂", "مواد آلی"),
    ("chat",        "🤖", "چت بات"),
]


def _has_any_results():
    """True once at least one parameter has monitoring results in memory."""
    return any(bool(st.session_state.results.get(p)) for p in ALL_PARAMETERS)


def _render_app_header():
    """Logo + gradient title + one-line subtitle, shown on every page."""
    logo_b64 = _get_app_logo_base64()
    logo_tag = (
        f'<img src="data:image/png;base64,{logo_b64}" alt="">' if logo_b64 else ""
    )
    st.markdown(
        f"""
        <div class="wq-app-header">
            {logo_tag}
            <span class="wq-app-title">سامانه پایش کیفیت آب</span>
        </div>
        <div class="wq-app-subtitle">
            🛰️ پایش خودکار <b>کدورت آب</b> و <b>غلظت کلروفیل</b> با استفاده از تصاویر ماهواره‌ای Sentinel-2
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_top_nav():
    """
    The four-page tab bar at the top of the app. The three result pages stay
    visible at all times (so the structure of the app is obvious from the first
    moment) but remain locked until a monitoring run has produced results.
    """
    results_ready = _has_any_results()
    busy = st.session_state.processing_in_progress

    # Marker used by the CSS to find this row on Streamlit versions that do not
    # expose per-key element classes.
    st.markdown('<div class="wq-nav-anchor"></div>', unsafe_allow_html=True)

    cols = st.columns(len(NAV_PAGES), gap="small")

    for col, (page_key, icon, title) in zip(cols, NAV_PAGES):
        locked = (page_key != "setup") and (not results_ready or busy)
        is_active = (st.session_state.active_page == page_key)
        label = f"{icon}  {title}" + ("  🔒" if locked else "")

        clicked = col.button(
            label,
            key=f"wqnav_{page_key}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
            disabled=locked,
        )
        if clicked and not is_active:
            st.session_state.active_page = page_key
            st.rerun()

    if not results_ready and not busy:
        st.markdown(
            '<div class="wq-nav-hint">🔒 صفحه‌های نتایج پس از اجرای پایش فعال می‌شوند.</div>',
            unsafe_allow_html=True,
        )


def _render_status_strip():
    """
    Run status + 'clear results' action. Shown on the first page directly below
    the «شروع پایش» / «ادامه از محل قطع» buttons (it used to live in the
    sidebar, which has been removed in favour of the top navigation).
    """
    st.markdown('<div class="wq-status-strip-spacer"></div>', unsafe_allow_html=True)

    col_clear, col_status = st.columns([1, 3])

    with col_status:
        if st.session_state.processing_in_progress:
            chip_class, chip_text = "wq-status-running", "⏳ در حال پردازش..."
        elif _has_any_results():
            chip_class, chip_text = "wq-status-ready", "✅ نتایج پایش موجود است"
        else:
            chip_class, chip_text = "wq-status-idle", "ℹ️ هنوز پایشی انجام نشده است"
        st.markdown(
            f'<div class="wq-status-line">'
            f'<span class="wq-status-chip {chip_class}">{chip_text}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_clear:
        if st.button(
            "🗑️ پاک کردن نتایج",
            use_container_width=True,
            disabled=st.session_state.processing_in_progress,
        ):
            st.session_state.downloaded_months = empty_param_dict(dict)
            st.session_state.month_statuses = empty_param_dict(dict)
            st.session_state.results = empty_param_dict(list)
            st.session_state.mean_data = empty_param_dict(dict)
            st.session_state.download_summary = {}
            st.session_state.cdom_display_range = None
            st.session_state.current_temp_dir = None
            st.session_state.processing_config = None
            st.session_state.processing_complete = False
            st.session_state.processing_in_progress = False
            st.session_state.pending_run = None
            st.session_state.expert_analysis_json = None
            st.session_state.expert_analysis_signature = None
            st.session_state.expert_chat_history = []
            st.session_state.active_page = 'setup'
            st.rerun()


# =============================================================================
# Region-of-interest map
# =============================================================================
# The drawings made with the Leaflet Draw toolbar only live inside the browser
# widget, so they vanished on the very next Streamlit rerun (for example right
# after «ذخیره منطقه»). Every saved region — and the freshly drawn, not yet
# saved one — is therefore re-added to the map as a real folium layer on every
# run, which keeps the region permanently visible and lets the map be shown in
# a read-only form while a monitoring run is in progress.
# =============================================================================
ROI_COLOR_SAVED = "#0E8E99"      # teal — a saved region
ROI_COLOR_SELECTED = "#F5A524"   # amber — the region the run will use
ROI_COLOR_DRAFT = "#E04B2F"      # red   — drawn but not saved yet


def _st_folium_compat(fmap, **kwargs):
    """
    Call st_folium while staying compatible with older streamlit-folium
    releases that do not support `key` / `returned_objects`. Limiting the
    returned objects (when available) is what stops harmless panning and
    zooming from triggering a full app rerun.
    """
    try:
        supported = inspect.signature(st_folium).parameters
    except Exception:
        supported = {}
    safe_kwargs = {k: v for k, v in kwargs.items() if not supported or k in supported}
    return st_folium(fmap, **safe_kwargs)


def _polygon_to_latlon(polygon):
    """Shapely stores (lon, lat); folium wants (lat, lon)."""
    return [(y, x) for x, y in polygon.exterior.coords]


def polygon_area_km2(polygon):
    """
    Approximate area of a lat/lon polygon in square kilometres.

    One degree of latitude is ~110.57 km everywhere, but one degree of
    longitude shrinks towards the poles, so the polygon's own mid-latitude is
    used to scale it. (The previous `area * 111 * 111` ignored that and
    overestimated by roughly 20% at Iran's latitudes.)
    """
    try:
        min_lat = polygon.bounds[1]
        max_lat = polygon.bounds[3]
        mean_lat = math.radians((min_lat + max_lat) / 2.0)
        km_per_deg_lat = 110.574
        km_per_deg_lon = 111.320 * math.cos(mean_lat)
        return abs(polygon.area) * km_per_deg_lat * km_per_deg_lon
    except Exception:
        return 0.0


def _format_area_km2(area_km2):
    """Readable area text — more decimals for small regions."""
    if area_km2 >= 100:
        return f"{area_km2:,.0f} کیلومتر مربع"
    if area_km2 >= 10:
        return f"{area_km2:.1f} کیلومتر مربع"
    return f"{area_km2:.2f} کیلومتر مربع"


# =============================================================================
# Live area readout while the user is drawing
# =============================================================================
# Leaflet.draw knows the shape being drawn long before the user finishes it, so
# a small badge is pinned under the bounding box of the in-progress shape and
# updated on every mouse move. The area itself comes from Leaflet.draw's own
# L.GeometryUtil.geodesicArea(), i.e. a proper geodesic area on the ellipsoid —
# not a degree-square approximation.
#
# Everything is wrapped in try/catch: if a future Leaflet.draw release renames
# an internal, the badge silently disappears and drawing keeps working.
_LIVE_AREA_JS = """
{% macro script(this, kwargs) %}
(function () {
  var MAP = {{ this._parent.get_name() }};

  function setup() {
    try {
      if (typeof L === 'undefined' || !MAP || !MAP.getContainer) { return false; }
      var container = MAP.getContainer();
      if (!container) { return false; }

      var label = L.DomUtil.create('div', '', container);
      label.style.cssText = [
        'position:absolute',
        'z-index:1000',
        'display:none',
        'pointer-events:none',
        'white-space:nowrap',
        'direction:rtl',
        'transform:translateX(-50%)',
        'background:rgba(255,255,255,0.97)',
        'color:#0A3F4A',
        'border:2px solid #E04B2F',
        'border-radius:999px',
        'padding:4px 14px',
        "font-family:'B Nazanin','BNazanin','Vazirmatn',Tahoma,sans-serif",
        'font-size:14px',
        'font-weight:700',
        'box-shadow:0 3px 10px rgba(10,63,74,0.40)'
      ].join(';');

      var state = { active: false, type: null, anchor: null, known: null, pts: null };

      // Same spherical-excess formula Leaflet.draw uses, kept local so the
      // badge does not depend on L.GeometryUtil being present.
      function geodesicArea(ring) {
        var n = ring.length, area = 0.0, d2r = Math.PI / 180, i, p1, p2;
        if (n < 3) { return 0; }
        for (i = 0; i < n; i++) {
          p1 = ring[i];
          p2 = ring[(i + 1) % n];
          area += ((p2.lng - p1.lng) * d2r) *
                  (2 + Math.sin(p1.lat * d2r) + Math.sin(p2.lat * d2r));
        }
        return Math.abs(area * 6378137.0 * 6378137.0 / 2.0);
      }

      function formatArea(m2) {
        var km2 = m2 / 1000000.0;
        if (km2 >= 100) { return km2.toFixed(0) + ' کیلومتر مربع'; }
        if (km2 >= 10) { return km2.toFixed(1) + ' کیلومتر مربع'; }
        if (km2 >= 0.01) { return km2.toFixed(2) + ' کیلومتر مربع'; }
        return Math.round(m2) + ' متر مربع';
      }

      function hide() {
        label.style.display = 'none';
      }

      function reset() {
        state.active = false;
        state.type = null;
        state.anchor = null;
        state.known = null;
        state.pts = null;
        hide();
      }

      function draw(ring) {
        if (!ring || ring.length < 3) { hide(); return; }
        var area = geodesicArea(ring);
        if (!isFinite(area) || area <= 0) { hide(); return; }
        var south = ring[0].lat, west = ring[0].lng, east = ring[0].lng, i;
        for (i = 1; i < ring.length; i++) {
          if (ring[i].lat < south) { south = ring[i].lat; }
          if (ring[i].lng < west) { west = ring[i].lng; }
          if (ring[i].lng > east) { east = ring[i].lng; }
        }
        var pt = MAP.latLngToContainerPoint(L.latLng(south, (west + east) / 2));
        label.innerHTML = 'مساحت: ' + formatArea(area);
        label.style.left = pt.x + 'px';
        label.style.top = (pt.y + 10) + 'px';
        label.style.display = 'block';
        state.pts = ring;
      }

      function flatten(latlngs) {
        while (latlngs && latlngs.length && latlngs[0] && latlngs[0].length !== undefined) {
          latlngs = latlngs[0];
        }
        return latlngs || [];
      }

      // Any polygon/polyline layer that appeared after drawing started is the
      // shape currently being drawn. No private Leaflet.draw fields involved.
      function newShapeLayer() {
        var found = null;
        try {
          MAP.eachLayer(function (layer) {
            if (found || !layer || typeof layer.getLatLngs !== 'function') { return; }
            if (state.known && state.known[L.Util.stamp(layer)]) { return; }
            found = layer;
          });
        } catch (err) {
          return null;
        }
        return found;
      }

      function ringFor(cursor) {
        // Rectangle: build it from the anchor corner and the cursor, so the
        // area is live from the very first pixel of the drag.
        if (state.type === 'rectangle' && state.anchor && cursor) {
          var a = state.anchor;
          if (Math.abs(a.lat - cursor.lat) > 1e-12 && Math.abs(a.lng - cursor.lng) > 1e-12) {
            return [
              L.latLng(a.lat, a.lng),
              L.latLng(a.lat, cursor.lng),
              L.latLng(cursor.lat, cursor.lng),
              L.latLng(cursor.lat, a.lng)
            ];
          }
          return null;
        }

        var layer = newShapeLayer();
        if (layer) {
          var ring = flatten(layer.getLatLngs()).slice();
          var isRect = !!(L.Rectangle && layer instanceof L.Rectangle);
          if (!isRect && cursor) { ring.push(cursor); }
          return ring;
        }
        return null;
      }

      function onMove(cursor) {
        if (!state.active || !cursor) { return; }
        draw(ringFor(cursor));
      }

      MAP.on('draw:drawstart', function (e) {
        state.active = true;
        state.type = (e && e.layerType) ? e.layerType : null;
        state.anchor = null;
        state.pts = null;
        state.known = {};
        try {
          MAP.eachLayer(function (layer) { state.known[L.Util.stamp(layer)] = true; });
        } catch (err) {
          state.known = null;
        }
        hide();
      });

      MAP.on('draw:drawstop', reset);
      MAP.on('draw:created', reset);
      MAP.on('draw:canceled', reset);

      MAP.on('mousedown', function (e) {
        if (state.active && !state.anchor && e && e.latlng) { state.anchor = e.latlng; }
      });
      MAP.on('mousemove', function (e) {
        if (e && e.latlng) { onMove(e.latlng); }
      });

      // DOM-level backstop: fires even if Leaflet's own map events are
      // swallowed by the active draw handler.
      container.addEventListener('mousedown', function (ev) {
        if (!state.active || state.anchor) { return; }
        try { state.anchor = MAP.mouseEventToLatLng(ev); } catch (err) { return; }
      }, true);
      container.addEventListener('mousemove', function (ev) {
        if (!state.active) { return; }
        try { onMove(MAP.mouseEventToLatLng(ev)); } catch (err) { return; }
      }, true);

      MAP.on('move', function () { if (state.active && state.pts) { draw(state.pts); } });
      MAP.on('zoom', function () { if (state.active && state.pts) { draw(state.pts); } });

      return true;
    } catch (err) {
      return true;
    }
  }

  // The map variable may not be assigned yet depending on script order, so
  // keep trying briefly instead of giving up on the first pass.
  var tries = 0;
  function attempt() {
    tries += 1;
    var ok = false;
    try { ok = setup(); } catch (err) { ok = true; }
    if (!ok && tries < 40) { setTimeout(attempt, 100); }
  }
  attempt();
})();
{% endmacro %}
"""


if _LIVE_AREA_SUPPORTED:
    class LiveAreaLabel(_BrancaMacroElement):
        """Folium element that injects the live-area script into the map."""
        _template = _JinjaTemplate(_LIVE_AREA_JS)

        def __init__(self):
            super().__init__()
            self._name = "LiveAreaLabel"
else:
    LiveAreaLabel = None


def _add_area_label(layer, polygon, text, color):
    """
    Small floating badge placed just below the polygon showing its area, so the
    user gets an immediate sense of how large the selected region is.

    The badge lives inside the map's own iframe, where the app's stylesheet does
    not reach — hence the inline styling.
    """
    try:
        min_lon, min_lat, max_lon, max_lat = polygon.bounds
        anchor_lat = min_lat
        anchor_lon = (min_lon + max_lon) / 2.0
    except Exception:
        return

    html = (
        f'<div style="'
        f'display:inline-block; white-space:nowrap; direction:rtl;'
        f'background:rgba(255,255,255,0.94); color:#0A3F4A;'
        f'border:2px solid {color}; border-left:none; border-right:none;'
        f'border-top:3px solid {color}; border-bottom:3px solid {color};'
        f'border-radius:999px; padding:3px 10px;'
        f'font-family:\'B Nazanin\',\'BNazanin\',\'Vazirmatn\',Tahoma,sans-serif;'
        f'font-size:13px; font-weight:700; line-height:1.6;'
        f'box-shadow:0 2px 6px rgba(10,63,74,0.35);'
        f'transform:translateX(-50%);'
        f'">{text}</div>'
    )

    folium.Marker(
        location=[anchor_lat, anchor_lon],
        icon=folium.DivIcon(html=html, icon_size=(0, 0), icon_anchor=(0, -6)),
    ).add_to(layer)


def _build_roi_map(interactive=True, highlight_index=None):
    """Build the ROI map with every saved region drawn as a permanent layer."""
    saved = list(st.session_state.drawn_polygons)
    draft = st.session_state.last_drawn_polygon
    draft_is_saved = draft is not None and any(p.equals(draft) for p in saved)

    shapes = saved + ([draft] if (draft is not None and not draft_is_saved) else [])

    if shapes:
        xs_min = min(s.bounds[0] for s in shapes)
        ys_min = min(s.bounds[1] for s in shapes)
        xs_max = max(s.bounds[2] for s in shapes)
        ys_max = max(s.bounds[3] for s in shapes)
        center = [(ys_min + ys_max) / 2, (xs_min + xs_max) / 2]
    else:
        center = [35.6892, 51.3890]

    fmap = folium.Map(location=center, zoom_start=8, control_scale=True)

    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google', name='تصویر ماهواره‌ای'
    ).add_to(fmap)

    if interactive:
        # Only the two drawing tools are shown. The plugin's own edit/delete
        # toolbar and its Export button are switched off: they act on the
        # browser-side layer only, so they never matched what the app actually
        # had stored and were a source of confusion. Deleting a region is done
        # with the 🗑️ button in «مناطق ذخیره‌شده» (saved regions) and with
        # «حذف منطقه رسم‌شده» (a region drawn but not yet saved).
        plugins.Draw(
            export=False,
            position='topleft',
            draw_options={
                'polyline': False, 'rectangle': True, 'polygon': True,
                'circle': False, 'marker': False, 'circlemarker': False,
            },
            edit_options={'edit': False, 'remove': False},
        ).add_to(fmap)

        # Live area readout that follows the shape as it is being drawn
        if LiveAreaLabel is not None:
            try:
                fmap.add_child(LiveAreaLabel())
            except Exception:
                pass

    roi_layer = folium.FeatureGroup(name='مناطق انتخاب‌شده', show=True)

    for i, polygon in enumerate(saved):
        is_selected = (highlight_index is not None and i == highlight_index)
        color = ROI_COLOR_SELECTED if is_selected else ROI_COLOR_SAVED
        area_text = _format_area_km2(polygon_area_km2(polygon))
        label = f"منطقه {i + 1}" + (" — انتخاب‌شده برای پایش" if is_selected else "")
        folium.Polygon(
            locations=_polygon_to_latlon(polygon),
            color=color,
            weight=5 if is_selected else 3,
            opacity=0.95,
            fill=True,
            fill_color=color,
            fill_opacity=0.22 if is_selected else 0.12,
            tooltip=f"{label} — {area_text}",
        ).add_to(roi_layer)
        _add_area_label(roi_layer, polygon, f"منطقه {i + 1} • {area_text}", color)

    if draft is not None and not draft_is_saved:
        draft_area_text = _format_area_km2(polygon_area_km2(draft))
        folium.Polygon(
            locations=_polygon_to_latlon(draft),
            color=ROI_COLOR_DRAFT,
            weight=3,
            dash_array='8,6',
            fill=True,
            fill_color=ROI_COLOR_DRAFT,
            fill_opacity=0.10,
            tooltip=f"منطقه رسم‌شده (ذخیره‌نشده) — {draft_area_text}",
        ).add_to(roi_layer)
        _add_area_label(
            roi_layer, draft, f"منطقه جدید • {draft_area_text}", ROI_COLOR_DRAFT
        )

    roi_layer.add_to(fmap)
    folium.LayerControl(collapsed=True).add_to(fmap)

    if shapes:
        fmap.fit_bounds([[ys_min, xs_min], [ys_max, xs_max]], padding=(25, 25))

    return fmap


def _render_roi_map():
    """
    Render the map. During a monitoring run it stays on screen as a read-only
    view (drawing tools removed, no interaction sent back to the server) instead
    of being replaced by a placeholder message.
    """
    busy = st.session_state.processing_in_progress
    highlight = st.session_state.selected_region_index if st.session_state.drawn_polygons else None

    map_col_l, map_col_c, map_col_r = st.columns([1, 5, 1])

    with map_col_c:
        fmap = _build_roi_map(interactive=not busy, highlight_index=highlight)

        # The version suffix forces a fresh widget after a save or a delete, so
        # the drawing toolbar cannot hand back a region the user just removed.
        version = st.session_state.map_version

        if busy:
            # returned_objects=[] -> the component never sends data back, so the
            # map cannot trigger a rerun in the middle of the processing run.
            _st_folium_compat(
                fmap, key=f"roi_map_locked_{version}", width=700, height=500,
                returned_objects=[],
            )
            st.caption("🔒 نقشه در حین پردازش فقط برای نمایش است؛ منطقه انتخاب‌شده با رنگ نارنجی مشخص شده است.")
            return

        map_data = _st_folium_compat(
            fmap, key=f"roi_map_{version}", width=700, height=500,
            returned_objects=["last_active_drawing"],
        )

        if map_data and map_data.get('last_active_drawing'):
            geom = (map_data['last_active_drawing'] or {}).get('geometry', {}) or {}
            if geom.get('type') == 'Polygon':
                try:
                    new_polygon = Polygon(geom['coordinates'][0])
                except Exception:
                    new_polygon = None
                if new_polygon is not None and (
                    st.session_state.last_drawn_polygon is None
                    or not st.session_state.last_drawn_polygon.equals(new_polygon)
                ):
                    st.session_state.last_drawn_polygon = new_polygon
                    st.rerun()   # redraw immediately as a permanent layer

        has_unsaved_draft = (
            st.session_state.last_drawn_polygon is not None
            and not any(p.equals(st.session_state.last_drawn_polygon)
                        for p in st.session_state.drawn_polygons)
        )

        if st.session_state.last_drawn_polygon is not None:
            draft_area = _format_area_km2(
                polygon_area_km2(st.session_state.last_drawn_polygon)
            )
            if has_unsaved_draft:
                st.info(
                    f"✅ منطقه رسم شد (خط‌چین قرمز) — مساحت تقریبی: **{draft_area}**. "
                    "برای نگه‌داشتن آن، «ذخیره منطقه» را بزنید."
                )
            else:
                st.success("✅ منطقه انتخاب‌شده روی نقشه نمایش داده می‌شود.")

        # The discard button only appears while there is something to discard,
        # so there is never an inactive delete control on screen.
        if has_unsaved_draft:
            save_col, discard_col = st.columns([3, 1])
        else:
            save_col, discard_col = st.container(), None

        if save_col.button("💾 ذخیره منطقه", use_container_width=True):
            if st.session_state.last_drawn_polygon:
                is_duplicate = any(
                    existing.equals(st.session_state.last_drawn_polygon)
                    for existing in st.session_state.drawn_polygons
                )
                if not is_duplicate:
                    st.session_state.drawn_polygons.append(st.session_state.last_drawn_polygon)
                    st.session_state.selected_region_index = len(st.session_state.drawn_polygons) - 1
                    st.session_state.map_version += 1
                    st.success("✅ منطقه ذخیره شد!")
                    st.rerun()
                else:
                    st.warning("⚠️ این منطقه قبلاً ذخیره شده است")
            else:
                st.warning("⚠️ ابتدا یک منطقه را روی نقشه رسم کنید")

        # Removes a region that was drawn by mistake and never saved.
        if discard_col is not None and discard_col.button(
            "✖️ حذف منطقه رسم‌شده",
            use_container_width=True,
            help="منطقه‌ای که رسم شده اما ذخیره نشده است را از نقشه بردارید",
        ):
            st.session_state.last_drawn_polygon = None
            st.session_state.map_version += 1
            st.rerun()


# =============================================================================
# Page 1 — area of interest, time period, and "start monitoring"
# =============================================================================
def render_setup_page():
    # ==========================================================================
    # 1. Region selection
    # ==========================================================================
    _render_step_header(1, "🗺️", "انتخاب منطقه مورد نظر (بدنه آبی)")

    _render_roi_map()

    if st.session_state.drawn_polygons:
        st.subheader("📍 مناطق ذخیره‌شده")
        for i, p in enumerate(st.session_state.drawn_polygons):
            c1, c2, c3 = st.columns([3, 1, 1])
            centroid = p.centroid
            c1.write(f"**منطقه {i+1}**: ~{_format_area_km2(polygon_area_km2(p))}")
            c2.write(f"مرکز: ({centroid.y:.4f}, {centroid.x:.4f})")
            if c3.button("🗑️", key=f"del_{i}", disabled=st.session_state.processing_in_progress):
                removed = st.session_state.drawn_polygons.pop(i)

                # The deleted region must also leave the map. Two things keep a
                # copy of it: the "last drawn" polygon in session state, and the
                # map widget's own drawing toolbar — clear the first, and bump
                # map_version so the widget is remounted and forgets the second.
                if st.session_state.last_drawn_polygon is not None and \
                        st.session_state.last_drawn_polygon.equals(removed):
                    st.session_state.last_drawn_polygon = None
                st.session_state.map_version += 1

                if st.session_state.selected_region_index >= len(st.session_state.drawn_polygons):
                    st.session_state.selected_region_index = max(0, len(st.session_state.drawn_polygons) - 1)
                st.rerun()

    # ==========================================================================
    # 2. Time period
    # ==========================================================================
    _render_step_header(2, "📅", "بازه زمانی")
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
    _render_step_header(3, "🚀", "اجرای پایش")

    selected_polygon = None

    if st.session_state.drawn_polygons:
        region_options = []
        for i, p in enumerate(st.session_state.drawn_polygons):
            region_options.append(f"منطقه {i+1} (~{_format_area_km2(polygon_area_km2(p))})")

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

    # --- Buttons: Start (fresh) and Resume (after interruption) ---
    btn_col1, btn_col2 = st.columns(2)

    start_btn = btn_col1.button(
        "🚀 شروع پایش",
        type="primary",
        disabled=st.session_state.processing_in_progress or selected_polygon is None
    )

    # Show Resume button only when a previous interrupted run exists
    has_partial_cache = any(
        bool(st.session_state.downloaded_months.get(p)) or
        bool(st.session_state.month_statuses.get(p))
        for p in ALL_PARAMETERS
    )
    resume_btn = btn_col2.button(
        "🔄 ادامه از محل قطع",
        disabled=(
            not has_partial_cache or
            st.session_state.processing_config is None or
            st.session_state.processing_in_progress
        ),
        help="اگر اتصال اینترنت قطع شد، پس از اتصال مجدد این دکمه را فشار دهید تا دانلود از همانجا ادامه یابد."
    )

    # --- Auto-continue after an interruption --------------------------------
    # A dropped connection during download/processing does not always surface
    # as a catchable Python exception (e.g. the browser/server connection is
    # cut and the script run is aborted before it reaches its `finally`
    # block). In that case processing_in_progress is left True and survives
    # into the next rerun. We detect that here and resume automatically using
    # the saved processing_config, exactly like pressing "Resume" ourselves.
    # This is the same recovery strategy the old version used (it re-entered
    # processing automatically whenever processing_in_progress was still True
    # on a fresh script run) and is what lets a single click survive an
    # internet interruption instead of leaving the app stuck (Start disabled
    # because processing_in_progress is True, Resume disabled for the same
    # reason).
    auto_continue = (
        not start_btn and not resume_btn
        and st.session_state.processing_in_progress
        and st.session_state.processing_config is not None
    )

    # --- A click only ARMS the run; the work happens on the next script run ---
    # Doing it this way means the whole page (map included) is already rendered
    # in its locked, read-only state before the long download loop begins — the
    # map therefore stays visible for the entire run and can no longer trigger a
    # rerun that would abort the processing halfway through.
    if start_btn:
        st.session_state.downloaded_months = empty_param_dict(dict)
        st.session_state.month_statuses = empty_param_dict(dict)
        st.session_state.results = empty_param_dict(list)
        st.session_state.mean_data = empty_param_dict(dict)
        st.session_state.download_summary = {}
        st.session_state.cdom_display_range = None
        st.session_state.processing_complete = False
        st.session_state.processing_in_progress = True
        st.session_state.resume_after_interruption = False
        st.session_state.current_temp_dir = None
        st.session_state.pending_run = 'start'

        # FIX E: Persist processing config so Resume can reconstruct the AOI and params
        st.session_state.processing_config = {
            'polygon_coords': list(selected_polygon.exterior.coords),
            'start_date': start.strftime('%Y-%m-%d'),
            'end_date': end.strftime('%Y-%m-%d'),
            'cloudy_pixel_percentage': CLOUD_THRESHOLD,
            'scale': 10,
        }
        st.rerun()

    if resume_btn and st.session_state.processing_config is not None:
        st.session_state.processing_in_progress = True
        st.session_state.resume_after_interruption = False
        st.session_state.pending_run = 'resume'
        st.rerun()

    # --- Execute the armed run (fresh start, manual resume, or auto-resume) ---
    pending = st.session_state.get('pending_run')

    if (pending or auto_continue) and st.session_state.processing_config is not None:
        config = st.session_state.processing_config

        # A fresh start runs with resume=False; a manual "ادامه از محل قطع" and
        # the automatic recovery after an interruption both run with
        # resume=True — exactly the original behaviour.
        is_fresh = (pending == 'start')

        st.session_state.pending_run = None
        st.session_state.processing_in_progress = True
        st.session_state.resume_after_interruption = False

        if auto_continue and not pending:
            st.info("🔄 اتصال اینترنت قطع شده بود؛ پایش از همان جا به‌طور خودکار ادامه می‌یابد...")

        aoi = ee.Geometry.Polygon([config['polygon_coords']])

        try:
            success = run_full_analysis(
                aoi,
                config['start_date'],
                config['end_date'],
                config.get('cloudy_pixel_percentage', CLOUD_THRESHOLD),
                config.get('scale', 10),
                resume=not is_fresh   # FIX G: skip already-cached months on resume
            )

            if is_fresh:
                st.session_state.processing_complete = success
                if not success:
                    st.warning("⚠️ داده‌ای برای این منطقه و بازه زمانی یافت نشد.")
            else:
                # Merge with previously completed results still in session state
                has_any = any(
                    bool(st.session_state.results.get(p)) for p in ALL_PARAMETERS
                )
                if has_any:
                    st.session_state.processing_complete = True
                if not success and not has_any:
                    st.warning("⚠️ داده‌ای برای این منطقه و بازه زمانی یافت نشد.")
        except Exception:
            # FIX F: On error, flag that a resume is possible instead of losing progress
            st.session_state.resume_after_interruption = True
            if is_fresh:
                st.error(
                    "متأسفانه اتصال قطع شد یا خطایی رخ داد. "
                    "پس از برقراری اتصال، دکمه «ادامه از محل قطع» را فشار دهید."
                )
            else:
                st.error("اتصال مجدداً قطع شد. لطفاً دوباره تلاش کنید.")
        finally:
            st.session_state.processing_in_progress = False
            st.rerun()

    # Hint when a partial run can be resumed
    if st.session_state.resume_after_interruption and not st.session_state.processing_in_progress:
        st.warning(
            "⚠️ پایش به دلیل قطعی اینترنت متوقف شد. "
            "پس از اتصال مجدد، دکمه «ادامه از محل قطع» را فشار دهید."
        )

    # --- Run status + clear-results action, directly under the run buttons ---
    _render_status_strip()

    # Simple, user-friendly download summary (persists after run)
    if st.session_state.download_summary:
        st.divider()
        turb_d, turb_a = st.session_state.download_summary.get(PARAM_TURBIDITY, (0, 0))
        chl_d, chl_a = st.session_state.download_summary.get(PARAM_CHLOROPHYLL, (0, 0))
        cdom_d, cdom_a = st.session_state.download_summary.get(PARAM_CDOM, (0, 0))
        st.info(
            f"🌊 شاخص کدورت: {turb_d} تصویر از {turb_a} تصویر موجود دریافت شد.\n\n"
            f"🌿 شاخص کلروفیل: {chl_d} تصویر از {chl_a} تصویر موجود دریافت شد.\n\n"
            f"🍂 شاخص مواد آلی محلول: {cdom_d} تصویر از {cdom_a} تصویر موجود دریافت شد."
        )

    # ==========================================================================
    # Results are ready — point the user at the three result pages above
    # ==========================================================================
    if st.session_state.processing_complete and _has_any_results():
        st.divider()
        _render_step_header(4, "📊", "نتایج پایش")

        st.markdown(
            """
            <div class="wq-ready-banner">
                ✅ پایش با موفقیت انجام شد.<br>
                برای مشاهده نتایج، از نوار بالای صفحه یکی از صفحه‌های
                <b>🌊 کدورت</b>، <b>🌿 کلروفیل</b>، <b>🍂 مواد آلی</b> یا
                <b>🤖 چت بات</b> را انتخاب کنید.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --- Download combined time-series (all three indices) as one .xlsx ---
        st.download_button(
            label="⬇️ دانلود سری زمانی کدورت (NDTI)، کلروفیل (NDCI) و مواد آلی محلول (CDOM) — یک فایل Excel",
            data=generate_combined_timeseries_excel(),
            file_name="water_quality_timeseries.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )


# =============================================================================
# Main Application — header, top navigation, then the active page
# =============================================================================
def main():
    _inject_global_app_css()
    _render_app_header()

    # Initialize Earth Engine
    ee_ok, ee_msg = initialize_earth_engine()
    if not ee_ok:
        st.error(ee_msg)
        st.stop()

    # Safety valve: a run can never be "in progress" without a configuration to
    # run — otherwise every button would stay disabled forever.
    if st.session_state.processing_in_progress and st.session_state.processing_config is None \
            and not st.session_state.get('pending_run'):
        st.session_state.processing_in_progress = False

    # While a run is in progress the app always stays on the first page, so the
    # progress bar and the resume logic remain visible to the user.
    if st.session_state.processing_in_progress:
        st.session_state.active_page = 'setup'

    # If results were cleared while a result page was open, fall back to page 1.
    if st.session_state.active_page != 'setup' and not _has_any_results():
        st.session_state.active_page = 'setup'

    _render_top_nav()

    page = st.session_state.active_page

    if page == 'turbidity':
        render_parameter_page(PARAM_TURBIDITY)
    elif page == 'chlorophyll':
        render_parameter_page(PARAM_CHLOROPHYLL)
    elif page == 'cdom':
        render_parameter_page(PARAM_CDOM)
    elif page == 'chat':
        render_expert_chat_tab()
    else:
        render_setup_page()


if __name__ == "__main__":
    main()
