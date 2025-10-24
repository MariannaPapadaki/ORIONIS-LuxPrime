
# =============================================
# ORIONIS — Lux Prime v1.6
# Τελική Έκδοση: drip = Ολισθαίνων Παράγοντας + F9 + 1/12 + Ανοιχτά Όρια
# Δημιουργήθηκε από: Μαρίαννα Παπαδάκη
# =============================================

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Any
import json

@dataclass
class F9Cfg:
    enabled: bool = False
    metallic_rich_rocks: bool = False
    no_water_carrier: bool = False
    geo_zones: list = field(default_factory=lambda: [
        "arctic_circle", "antarctic", "oceania", "usa", "meridians"
    ])
    meridian_proximity_km: float = 100.0

def load_csv(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["time_utc"] = pd.to_datetime(df["time_utc"])
    df = df.sort_values("time_utc").reset_index(drop=True)
    return df

def calculate_drift_norm(df: pd.DataFrame) -> pd.DataFrame:
    df["time_hours"] = (df["time_utc"] - df["time_utc"].iloc[0]).dt.total_seconds() / 3600
    df["lat_diff"] = df["latitude"].diff().fillna(0)
    df["lon_diff"] = df["longitude"].diff().fillna(0)
    df["drift"] = np.sqrt(df["lat_diff"]**2 + df["lon_diff"]**2)
    df["drift_cumsum"] = df["drift"].cumsum()
    df["drift_norm"] = df["drift_cumsum"] / df["drift_cumsum"].max()
    return df

def determine_phase(drift_norm: float, drip_magnitude: float) -> str:
    # Ανοιχτά όρια αν drip > 1.0 km/γεγονός
    if drip_magnitude > 1.0:
        bounds = {"STABLE": 0.0, "DRIFTING": 0.6, "ALERT": 0.9, "RUPTURE": 1.05}
    else:
        bounds = {"STABLE": 0.0, "DRIFTING": 0.8, "ALERT": 1.0, "RUPTURE": 1.125}
    
    if drift_norm < bounds["DRIFTING"]:
        return "STABLE"
    elif drift_norm < bounds["ALERT"]:
        return "DRIFTING"
    elif drift_norm < bounds["RUPTURE"]:
        return "ALERT"
    else:
        return "RUPTURE"

def check_f9_conditions(lat: float, lon: float, geo_info: dict) -> bool:
    metallic = geo_info.get("metallic_rich_rocks", False)
    no_water = geo_info.get("no_water_carrier", False)
    is_arctic = lat >= 66.5
    is_antarctic = lat <= -66.5
    is_oceania = (-50 <= lat <= -10) and (110 <= lon <= 180)
    is_usa = (25 <= lat <= 50) and (-125 <= lon <= -65)
    meridian_dist = abs(lon % 15) * 111.32
    is_meridian = meridian_dist < 100.0
    return (metallic and no_water) or any([is_arctic, is_antarctic, is_oceania, is_usa, is_meridian])

def predict_epicenter_dynamic(df: pd.DataFrame, drift_norm: float, f9_enabled: bool) -> Dict[str, Any]:
    # Μέσος όρος (σταθμισμένος)
    weights = np.exp(np.linspace(0, 1, len(df)))
    weighted_center_lat = np.average(df["latitude"], weights=weights)
    weighted_center_lon = np.average(df["longitude"], weights=weights)
    
    # === drip = Ολισθαίνων Παράγοντας ===
    recent_df = df.tail(max(3, len(df)//4))
    delta_lat = recent_df["latitude"].diff().mean()
    delta_lon = recent_df["longitude"].diff().mean()
    drip_magnitude = np.sqrt(delta_lat**2 + delta_lon**2) * 111.0  # km/γεγονός
    drip_direction_lat = delta_lat * 111.0
    drip_direction_lon = delta_lon * 111.0 * np.cos(np.radians(df["latitude"].mean()))
    
    # drip_factor με 1/12 μείωση
    drip_factor = max(0.0, min(0.8, (drift_norm - 1.0) * 0.8 * (11/12)))
    if f9_enabled:
        drip_factor *= 1.2  # F9 boost
    
    drip_km = drip_magnitude * drip_factor
    
    # Εφαρμογή ολίσθησης
    predicted_lat = weighted_center_lat + (drip_direction_lat / 111.0) * drip_factor
    predicted_lon = weighted_center_lon + (drip_direction_lon / (111.0 * np.cos(np.radians(weighted_center_lat)))) * drip_factor
    
    return {
        "predicted_lat": round(predicted_lat, 6),
        "predicted_lon": round(predicted_lon, 6),
        "drip_magnitude_km": round(drip_magnitude, 3),
        "drip_factor": round(drip_factor, 3),
        "drip_km_applied": round(drip_km, 2),
        "drip_direction_deg": round(np.degrees(np.arctan2(drip_direction_lon, drip_direction_lat)), 1)
    }

def run_orionis_luxprime(filepath: str, geo_info: dict = None) -> Dict[str, Any]:
    if geo_info is None:
        geo_info = {}
    
    df = load_csv(filepath)
    df = calculate_drift_norm(df)
    drift_norm = df["drift_norm"].iloc[-1]
    
    # drip_magnitude για phase bounds
    recent_df = df.tail(max(3, len(df)//4))
    delta_lat = recent_df["latitude"].diff().mean()
    delta_lon = recent_df["longitude"].diff().mean()
    drip_magnitude = np.sqrt(delta_lat**2 + delta_lon**2) * 111.0
    
    # F9 check
    weighted_lat = np.average(df["latitude"], weights=np.exp(np.linspace(0, 1, len(df))))
    weighted_lon = np.average(df["longitude"], weights=np.exp(np.linspace(0, 1, len(df))))
    f9_enabled = check_f9_conditions(weighted_lat, weighted_lon, geo_info)
    
    # Phase
    df["drift_stage"] = df["drift_norm"].apply(lambda x: determine_phase(x, drip_magnitude))
    
    # F9 rule
    recent_mask = df["time_hours"] > (df["time_hours"].max() - 6)
    recent_count = recent_mask.sum()
    
    if f9_enabled and drift_norm >= 1.125 and recent_count >= 3:
        final_stage = "RUPTURE IMMINENT (F9)"
        alert_level = "CRITICAL"
    else:
        final_stage = df["drift_stage"].iloc[-1]
        alert_level = "MONITOR"
    
    # Επίκεντρο με drip
    epicenter = predict_epicenter_dynamic(df, drift_norm, f9_enabled)
    
    # Χρόνος
    predicted_time_hours = df["time_hours"].iloc[-1] + (1.0 / max(0.1, df["drift"].iloc[-1])) * 50
    
    # Summary
    summary = {
        "predicted_epicenter_lat": epicenter["predicted_lat"],
        "predicted_epicenter_lon": epicenter["predicted_lon"],
        "predicted_time_hours": round(predicted_time_hours, 2),
        "final_stage": final_stage,
        "alert_level": alert_level,
        "f9_applied": f9_enabled,
        "drip_magnitude_km": epicenter["drip_magnitude_km"],
        "drip_factor": epicenter["drip_factor"],
        "drip_km_applied": epicenter["drip_km_applied"],
        "drip_direction_deg": epicenter["drip_direction_deg"],
        "phase_drift_max": round(drift_norm, 3),
        "total_events": len(df),
        "time_span_hours": round(df["time_hours"].iloc[-1], 2)
    }
    
    return summary, df
