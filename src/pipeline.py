
import numpy as np
import pandas as pd

from config import (COLUMNS, FULL_FEATURES, MIN_MINUTES_PER_HOUR, PROCESSED_PATH,
                    RAW_PATH, TARGET, WEATHER)


def load(path=RAW_PATH):
    df = pd.read_csv(path, usecols=list(COLUMNS)).rename(columns=COLUMNS)
    df["time"] = pd.to_datetime(df["time"], format="%d/%m/%Y %H:%M")
    return df

def clean(df):
    df = df.drop_duplicates("time").sort_values("time").set_index("time")
    # GTI readings taken while the sensors were being cleaned are not trustworthy
    df.loc[df["cleaning"].fillna(0) != 0, TARGET] = np.nan
    return df.drop(columns="cleaning")
def to_hourly(df):
    
    rad = np.deg2rad(df["wind_direction"])
    df = df.assign(wd_sin=np.sin(rad), wd_cos=np.cos(rad)).drop(columns="wind_direction")

    hourly = df.resample("h").mean()
    hourly["precipitation"] = df["precipitation"].resample("h").sum(min_count=1)
    hourly["wind_direction"] = np.rad2deg(np.arctan2(hourly["wd_sin"], hourly["wd_cos"])) % 360
    hourly = hourly.drop(columns=["wd_sin", "wd_cos"])

    # Discard hourly GTI built from too few minutes; clip small negative night offsets
    enough = df[TARGET].resample("h").count() >= MIN_MINUTES_PER_HOUR
    hourly.loc[~enough, TARGET] = np.nan
    hourly[TARGET] = hourly[TARGET].clip(lower=0)
    return hourly


def add_features(h):
    h = h.copy()
    h[WEATHER] = h[WEATHER].interpolate(limit=3, limit_area="inside")  # fill gaps of <= 3 hours only
    h["hour"] = h.index.hour
    h["day_of_year"] = h.index.dayofyear
    h["hour_sin"] = np.sin(2 * np.pi * h["hour"] / 24)
    h["hour_cos"] = np.cos(2 * np.pi * h["hour"] / 24)
    h["doy_sin"] = np.sin(2 * np.pi * h["day_of_year"] / 365)
    h["doy_cos"] = np.cos(2 * np.pi * h["day_of_year"] / 365)
    # Lags run on the regular hourly index, so a missing hour stays missing (no silent shifting)
    h["gti_lag1"] = h[TARGET].shift(1)
    h["gti_lag24"] = h[TARGET].shift(24)
    h["gti_roll3"] = h[TARGET].shift(1).rolling(3).mean()
    # Same rows for both feature sets, so the comparison is fair
    return h.dropna(subset=FULL_FEATURES + [TARGET])
def main():
    raw = load()
    print(f"raw minute rows:        {len(raw):,}")
    df = clean(raw)
    print(f"after de-duplication:   {len(df):,}")
    hourly = to_hourly(df)
    print(f"hourly rows (calendar): {len(hourly):,}")
    print(f"hours with valid GTI:   {hourly[TARGET].notna().sum():,}")
    final = add_features(hourly)
    print(f"final modelling rows:   {len(final):,}  (needs GTI, weather and GTI history)")
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(PROCESSED_PATH, index_label="time")
    print(f"saved -> {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
