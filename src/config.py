
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "narok_2019.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "hourly.csv"
MODEL_DIR = ROOT / "models"


# Only these columns are read. DHI, both GHI sensors and the other two GTI
# sensors are deliberately left out: they are near-copies of the target (leakage).
COLUMNS = {
    "time": "time",
    "gti_clean": "gti",                     # target: the GTI sensor that is kept clean
    "air_temperature": "temperature",       # deg C
    "relative_humidity": "humidity",        # %
    "barometric_pressure": "pressure",      # hPa
    "precipitation": "precipitation",       # mm per minute
    "wind_speed": "wind_speed",             # m/s
    "wind_from_direction": "wind_direction",  # degrees
    "sensor_cleaning": "cleaning",          # non-zero = sensors being cleaned
}

TARGET = "gti"
WEATHER = ["temperature", "humidity", "pressure", "precipitation", "wind_speed", "wind_direction"]

# Feature set 1 (served by the API): weather + raw time
BASE_FEATURES = WEATHER + ["hour", "day_of_year"]
# Feature set 2 (comparison only): adds cyclical time + GTI history
FULL_FEATURES = BASE_FEATURES + ["hour_sin", "hour_cos", "doy_sin", "doy_cos",
                                 "gti_lag1", "gti_lag24", "gti_roll3"]
# train: Dec 2019-Sep 2020, test: Oct-Dec 2020
TEST_START = "2020-10-01"      
MIN_MINUTES_PER_HOUR = 30      
DAYLIGHT_THRESHOLD = 10        # W/m2; used for daylight-only metrics
RANDOM_STATE = 42
