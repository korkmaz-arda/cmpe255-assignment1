"""Ride estimator tab (F06-F09, S02)."""
from __future__ import annotations

import io
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from taxi import config, landmarks
from taxi.features import build_features
from taxi.predict import predict_batch, predict_trip

from app.interaction import (CUSTOM, DATE_MAX, DATE_MIN, DEFAULT_PICKUP_TIME, DEFAULT_RIDE_DATE, ENDPOINTS,
                             TIME_PRESETS, TRAINING_EXAMPLE_DATE,
                             apply_map_event, combine_pickup, match_landmark)
from app.map_component import route_map
from app.ui_common import GLOSSARY, fmt_seconds

LANDMARK_IDS = landmarks.ids()
CHOICES = LANDMARK_IDS + [CUSTOM]
DEFAULT_CHOICE = {"pickup": "times_square", "dropoff": "jfk"}
MODE_LABELS = {"pickup": "🟢 Pickup", "dropoff": "🔴 Dropoff"}
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _label(choice: str) -> str:
    if choice == CUSTOM:
        return "📍 Custom point (map / coordinates)"
    lm = landmarks.get(choice)
    return f"{lm.icon} {lm.name}"


def init_state() -> None:
    ss = st.session_state
    for e in ENDPOINTS:
        ss.setdefault(f"{e}_choice", DEFAULT_CHOICE[e])
        lm = landmarks.get(DEFAULT_CHOICE[e])
        ss.setdefault(f"{e}_lat", lm.lat)
        ss.setdefault(f"{e}_lon", lm.lon)
    ss.setdefault("map_mode", "dropoff")
    # UI-level defaults (the core never reads the clock): 2026-01-01 at 12:00 AM.
    ss.setdefault("ride_date", DEFAULT_RIDE_DATE)
    ss.setdefault("pickup_time", DEFAULT_PICKUP_TIME)
    ss.setdefault("passengers", 1)


# -- callbacks (run before the script re-executes, so they may set widget state) ----------------------
def _on_choice(endpoint: str) -> None:
    choice = st.session_state[f"{endpoint}_choice"]
    if choice != CUSTOM:
        lm = landmarks.get(choice)
        st.session_state[f"{endpoint}_lat"], st.session_state[f"{endpoint}_lon"] = lm.lat, lm.lon
    st.session_state["map_notice"] = None


def _on_quick_pick(endpoint: str) -> None:
    chip_key = f"chip_{endpoint}"
    choice = st.session_state.get(chip_key)
    if choice:
        st.session_state[f"{endpoint}_choice"] = choice
        _on_choice(endpoint)
    st.session_state[chip_key] = None


def _on_coordinate(endpoint: str) -> None:
    ss = st.session_state
    ss[f"{endpoint}_choice"] = match_landmark(ss[f"{endpoint}_lat"], ss[f"{endpoint}_lon"])
    ss["map_notice"] = None


def _on_map_event() -> None:
    ss = st.session_state
    state = ss.get("route_map")
    event = None
    if state is not None:
        event = state.get("map_event") if hasattr(state, "get") else getattr(state, "map_event", None)
    if not event:
        return
    try:
        update = apply_map_event(event, ss["map_mode"])
    except (ValueError, KeyError) as exc:
        ss["map_notice"] = f"Map point ignored: {exc}"
        return
    ss[f"{update.endpoint}_lat"], ss[f"{update.endpoint}_lon"] = update.lat, update.lon
    ss[f"{update.endpoint}_choice"] = update.choice
    ss["map_notice"] = None


def _on_time_preset() -> None:
    preset = st.session_state.get("time_preset")
    if preset:
        st.session_state["pickup_time"] = TIME_PRESETS[preset]
    st.session_state["time_preset"] = None


def _set_training_date() -> None:
    st.session_state["ride_date"] = TRAINING_EXAMPLE_DATE


def _shift_time(minutes: int) -> None:
    """Move the clock time only; the ride date is deliberately left unchanged (wraps within the day)."""
    t = st.session_state["pickup_time"]
    shifted = (datetime.combine(datetime(2000, 1, 1), t) + timedelta(minutes=minutes)).time()
    st.session_state["pickup_time"] = shifted


def endpoint_payload(endpoint: str) -> dict:
    ss = st.session_state
    choice = ss[f"{endpoint}_choice"]
    label = "custom point" if choice == CUSTOM else landmarks.get(choice).name
    return {"lat": float(ss[f"{endpoint}_lat"]), "lon": float(ss[f"{endpoint}_lon"]), "label": label, "choice": choice}


def banner(bundle) -> None:
    md = bundle.metadata
    vm = md["validation_metrics"]
    with st.container(border=True):
        st.markdown("**The problem:** predict NYC yellow-cab trip duration from pickup/dropoff location, time and "
                    "passengers, as in the Kaggle *NYC Taxi Trip Duration* challenge (2016 trips), scored by RMSLE.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Validation RMSLE", f"{vm['rmsle']:.4f}", help=GLOSSARY["RMSLE"] + " " + GLOSSARY["Validation split"])
        c2.metric("Validation R²", f"{vm['r2']:.3f}", help=GLOSSARY["R²"])
        c3.metric("Validation mean absolute error", f"{vm['mae_s'] / 60:.2f} min", help=GLOSSARY["MAE"])
        if bundle.holdout:
            hm = bundle.holdout["metrics"]
            c4.metric("Final holdout RMSLE (test)", f"{hm['rmsle']:.4f}", help=GLOSSARY["Final holdout (test)"])
        else:
            c4.metric("Final holdout (test)", "Not finalized", help=GLOSSARY["Final holdout (test)"])
        st.caption("Numbers come from the active model version's artifacts. Kaggle leaderboard scores use a "
                   "different test set and protocol, so they are not comparable and are not shown.")


def _endpoint_controls(endpoint: str) -> None:
    title = "Pickup" if endpoint == "pickup" else "Dropoff"
    st.selectbox(title, CHOICES, key=f"{endpoint}_choice", format_func=_label, on_change=_on_choice,
                 args=(endpoint,))
    choice = st.session_state[f"{endpoint}_choice"]
    st.caption("Zone: " + ("custom point chosen on the map or typed" if choice == CUSTOM
                           else landmarks.get(choice).zone))
    st.number_input("Latitude", key=f"{endpoint}_lat", min_value=config.LAT_MIN, max_value=config.LAT_MAX,
                    step=0.0005, format="%.6f", on_change=_on_coordinate, args=(endpoint,))
    st.number_input("Longitude", key=f"{endpoint}_lon", min_value=config.LON_MIN, max_value=config.LON_MAX,
                    step=0.0005, format="%.6f", on_change=_on_coordinate, args=(endpoint,))


def _time_controls() -> None:
    with st.container(border=True, key="when_panel"):
        st.markdown("**When**")
        c_date, c_time = st.columns([1, 1.25], gap="medium")
        c_date.date_input("Ride date", key="ride_date", format="YYYY-MM-DD", min_value=DATE_MIN, max_value=DATE_MAX,
                          help="Sets day of week and month. The model learned from Jan–Jun 2016.")
        c_date.button("📅 2016-03-15 (training period)", key="date_training", on_click=_set_training_date,
                      help=f"Sets {TRAINING_EXAMPLE_DATE:%a %Y-%m-%d} (inside the training period); time is unchanged.")
        c_time.time_input("Pickup time", key="pickup_time", step=60,
                          help="Exact hour and minute. Hour of day, weekday rush (07–09, 16–19) and late night "
                               "(00–05) are model features.")
        with c_time.container(horizontal=True, gap="small", key="time_shift"):
            st.button("−1h", key="t_m60", on_click=_shift_time, args=(-60,), help="One hour earlier (same date)")
            st.button("−15m", key="t_m15", on_click=_shift_time, args=(-15,), help="15 minutes earlier (same date)")
            st.button("+15m", key="t_p15", on_click=_shift_time, args=(15,), help="15 minutes later (same date)")
            st.button("+1h", key="t_p60", on_click=_shift_time, args=(60,), help="One hour later (same date)")
        st.pills("Quick times", list(TIME_PRESETS), key="time_preset", on_change=_on_time_preset)


def _temporal_readout(pickup_dt: datetime, pickup: dict, dropoff: dict, passengers: int) -> None:
    """Show the calendar features the model will actually receive (computed by build_features itself)."""
    req = pd.DataFrame([{"pickup_datetime": pd.Timestamp(pickup_dt), "passenger_count": passengers,
                         "pickup_latitude": pickup["lat"], "pickup_longitude": pickup["lon"],
                         "dropoff_latitude": dropoff["lat"], "dropoff_longitude": dropoff["lon"]}])
    f = build_features(req).iloc[0]
    flags = []
    if f["is_rush_hour"]:
        flags.append("<span class='pill pill-warn'>weekday rush hour</span>")
    if f["is_late_night"]:
        flags.append("<span class='pill'>late night</span>")
    if f["is_weekend"]:
        flags.append("<span class='pill'>weekend</span>")
    if not flags:
        flags.append("<span class='pill'>off-peak</span>")
    st.markdown(f"<div class='hero-sub'>Model receives: <span class='mono'>{pickup_dt:%Y-%m-%d %H:%M}</span> · "
                f"{WEEKDAYS[int(f['day_of_week'])]} · hour {int(f['hour'])} · month {int(f['month'])}</div>"
                + "".join(flags), unsafe_allow_html=True)


def render(bundle) -> None:
    init_state()
    if bundle is None:
        st.info("No trained model is loaded yet, so predictions are unavailable. Run `python scripts/train.py` "
                "(see README), or use **Retrain** in the header once the data is prepared.")
        return
    banner(bundle)

    left, right = st.columns([5, 6], gap="large")
    with left:
        st.subheader("Compose a trip")
        c1, c2 = st.columns(2)
        with c1:
            _endpoint_controls("pickup")
        with c2:
            _endpoint_controls("dropoff")
        st.pills("Quick pickup", landmarks.QUICK_PICKS, key="chip_pickup", format_func=_label,
                 on_change=_on_quick_pick, args=("pickup",))
        st.pills("Quick dropoff", landmarks.QUICK_PICKS, key="chip_dropoff", format_func=_label,
                 on_change=_on_quick_pick, args=("dropoff",))
        _time_controls()
        st.segmented_control("Passengers", list(range(1, 7)), key="passengers", selection_mode="single")

        pickup, dropoff = endpoint_payload("pickup"), endpoint_payload("dropoff")
        passengers = st.session_state.get("passengers")
        pickup_dt = combine_pickup(st.session_state.get("ride_date"), st.session_state.get("pickup_time"))
        result, error = None, None
        if passengers is None or pickup_dt is None:
            error = "Choose a ride date, pickup time and passenger count to get an estimate."
        else:
            _temporal_readout(pickup_dt, pickup, dropoff, passengers)
            try:
                result = predict_trip(bundle, pickup_latitude=pickup["lat"], pickup_longitude=pickup["lon"],
                                      dropoff_latitude=dropoff["lat"], dropoff_longitude=dropoff["lon"],
                                      pickup_datetime=pd.Timestamp(pickup_dt), passenger_count=passengers)
            except Exception as exc:  # keep the view alive (S10)
                error = f"Prediction failed: {exc}"

        if error:
            st.warning(error)
        if result:
            same = (pickup["lat"], pickup["lon"]) == (dropoff["lat"], dropoff["lon"])
            _render_result(result, same)

    with right:
        st.radio("Map click sets", list(ENDPOINTS), key="map_mode", format_func=MODE_LABELS.get, horizontal=True,
                 help="Choose which endpoint a click on the map moves. Dragging a marker always moves that marker.")
        if st.session_state.get("map_notice"):
            st.warning(st.session_state["map_notice"])
        route_map(pickup, dropoff, st.session_state["map_mode"],
                  result["duration"]["seconds"] if result else None, key="route_map", on_event=_on_map_event)
        st.caption("Street map © OpenStreetMap contributors. It needs internet access; predictions "
                   "do not depend on the tiles. Click the map to set the selected endpoint, click a landmark to "
                   "snap to it, or drag the P/D markers. Only coordinates are sent to Python, which validates "
                   "them and computes every feature.")

    _batch_section(bundle)


def _render_result(r: dict, same_place: bool) -> None:
    st.divider()
    if same_place:
        st.info("Pickup and dropoff are the same point; the estimate reflects a zero-distance trip.")
    if r["temporal_extrapolation"]:
        rng = r["training_pickup_range"]
        st.warning(f"**Temporal extrapolation.** This pickup time is outside the model's training period "
                   f"(observed pickups {rng['min'].replace('T', ' ')} to {rng['max'].replace('T', ' ')}). Matching the "
                   "month or weekday does not make it "
                   "in-distribution: traffic in other years may differ.", icon="⚠️")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"<div class='hero-sub'>Predicted duration</div><div class='hero'>{r['duration']['formatted']}</div>",
                    unsafe_allow_html=True)
        b = r["band"]
        st.markdown(f"<div class='hero-sub'>{int(b['level'] * 100)}% prediction band: "
                    f"<span class='mono'>{fmt_seconds(b['low_seconds'])} – {fmt_seconds(b['high_seconds'])}</span></div>",
                    unsafe_allow_html=True, help=GLOSSARY["Prediction band"])
    with c2:
        st.markdown(f"<div class='hero-sub'>Estimated fare</div><div class='hero'>${r['fare']['total']:.2f}</div>",
                    unsafe_allow_html=True)
        st.markdown("<div class='hero-sub'>Illustrative rule-based estimate, not a TLC meter quote</div>",
                    unsafe_allow_html=True)
    rows = "".join(f"<div class='fare-row'><span>{i['label']}</span><span class='mono'>${i['amount']:.2f}</span></div>"
                   for i in r["fare"]["items"] if i["amount"] > 0)
    st.markdown(rows + f"<div class='fare-total'><span>Total</span><span class='mono'>${r['fare']['total']:.2f}</span></div>",
                unsafe_allow_html=True)
    rt = r["route"]
    t1, t2 = st.columns(2)
    t3, t4 = st.columns(2)
    t1.metric("City-block distance", f"{rt['distance_manhattan_km']:.2f} km", f"{rt['distance_manhattan_mi']:.2f} mi",
              delta_color="off", delta_arrow="off", help=GLOSSARY["City-block distance"])
    t2.metric("Great-circle distance", f"{rt['distance_haversine_km']:.2f} km", f"{rt['distance_haversine_mi']:.2f} mi",
              delta_color="off", delta_arrow="off", help=GLOSSARY["Great-circle distance"])
    t3.metric("Compass heading", f"{rt['bearing_deg']:.0f}°")
    t4.metric("Implied avg speed", f"{rt['avg_speed_kmh']:.1f} km/h", f"{rt['avg_speed_mph']:.1f} mph",
              delta_color="off", delta_arrow="off",
              help="City-block distance divided by the predicted duration.")


def _batch_section(bundle) -> None:
    with st.expander("Batch predictions (CSV upload)"):
        st.markdown("Upload a CSV with one trip per row and the columns "
                    f"`{', '.join(config.REQUEST_COLUMNS)}`. Every field is required: rows missing a pickup time or "
                    "passenger count are rejected rather than silently filled in.")
        template = pd.DataFrame([{"pickup_datetime": "2016-03-15 08:30:00", "passenger_count": 1,
                                  "pickup_latitude": 40.7580, "pickup_longitude": -73.9855,
                                  "dropoff_latitude": 40.6413, "dropoff_longitude": -73.7781}])
        st.download_button("Download template CSV", template.to_csv(index=False), "trip_template.csv", "text/csv")
        up = st.file_uploader("Trips CSV", type=["csv"], key="batch_upload")
        if up is None:
            return
        try:
            df = pd.read_csv(io.BytesIO(up.getvalue()))
            df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="raise")
            res = predict_batch(bundle, df)
        except Exception as exc:
            st.error(f"Could not score this file: {exc}")
            return
        st.success(f"Scored {res['count']:,} trips.")
        preds = res["predictions"]
        cols = ["duration_formatted", "duration_seconds", "band_low_seconds", "band_high_seconds",
                "fare_total", "distance_manhattan_km", "distance_haversine_km", "temporal_extrapolation"]
        out = pd.concat([df.reset_index(drop=True), preds[cols]], axis=1)
        st.dataframe(out, width="stretch", hide_index=True)
        st.download_button("Download predictions", out.to_csv(index=False), "trip_predictions.csv", "text/csv")
