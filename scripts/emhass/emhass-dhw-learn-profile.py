#!/usr/bin/env python3
"""Learn weekday and weekend DHW draw profiles for EMHASS.

This Home Assistant runtime helper lives under /config/scripts/emhass/ and is called by
package_emhass_dhw_profile.yaml.
The total-water meter also counts cold water, so it is only an event gate. The thermal draw is
reconstructed from the DHW tank energy balance:

    draw = DHW heat input - tank standby loss - change in stored heat

Only water-use events with a positive inferred thermal draw are added to the profile. Results are
96 quarter-hour kWh values for weekdays and weekends. With --push, the profile is sent to Home
Assistant as the `emhass_dhw_profile_learned` event.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import statistics
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

DEFAULT_HOST = "http://hass.schellekensonline.internal:8123"
DEFAULT_SECRETS = "/config/workspace/home/docker/appdata/emhass/secrets_emhass.yaml"
DEFAULT_TZ = "Europe/Amsterdam"

FLOW_ENTITY = "sensor.watermeter_water_usage"
TANK_ENTITY = "sensor.therma_v_dhw_temperature"
POWER_ENTITY = "sensor.therma_v_thermal_power"
DHW_STATE_ENTITY = "binary_sensor.therma_v_dhw_heating_state"
ENTITIES = [FLOW_ENTITY, TANK_ENTITY, POWER_ENTITY, DHW_STATE_ENTITY]

STEP = timedelta(minutes=5)
SLOT = timedelta(minutes=15)
FLOW_THRESHOLD_LPM = 0.2
MIN_EVENT_LITRES = 0.5
MIN_DRAW_KWH = 0.10
MAX_EVENT_DRAW_KWH = 10.0
TANK_CAPACITY_KWH_PER_K = 0.300 * 997 * 4.184 / 3600
DEFAULT_LOSS_KW = 0.10
WINDOW_PADDING = timedelta(minutes=15)
CLUSTER_EVENT_KWH = 0.20


def read_token(path: str) -> str:
    with open(path) as handle:
        for line in handle:
            if line.startswith("long_lived_token:"):
                return line.split(":", 1)[1].strip().strip("'\"")
    raise SystemExit(f"no long_lived_token in {path}")


def parse_moment(value: str, tz: ZoneInfo) -> datetime:
    moment = datetime.fromisoformat(value)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=tz)
    return moment.astimezone(tz)


def fetch_history(
    host: str, token: str, entities: list[str], start: datetime, end: datetime
) -> list[list[dict]]:
    params = {
        "end_time": end.isoformat(),
        "filter_entity_id": ",".join(entities),
        "minimal_response": "",
        "no_attributes": "",
    }
    url = (
        f"{host}/api/history/period/{urllib.parse.quote(start.isoformat())}?"
        f"{urllib.parse.urlencode(params)}"
    )
    request = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {token}"}
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def numeric_history(rows: list[dict], tz: ZoneInfo) -> tuple[list[datetime], list[float | None]]:
    points: dict[datetime, float | None] = {}
    for row in rows:
        try:
            moment = datetime.fromisoformat(row["last_changed"]).astimezone(tz)
        except (KeyError, ValueError):
            continue
        try:
            value = float(row["state"])
        except (TypeError, ValueError):
            value = None
        points[moment] = value
    ordered = sorted(points.items())
    return [moment for moment, _ in ordered], [value for _, value in ordered]


def state_history(rows: list[dict], tz: ZoneInfo) -> tuple[list[datetime], list[str]]:
    points: dict[datetime, str] = {}
    for row in rows:
        try:
            moment = datetime.fromisoformat(row["last_changed"]).astimezone(tz)
        except (KeyError, ValueError):
            continue
        points[moment] = row["state"]
    ordered = sorted(points.items())
    return [moment for moment, _ in ordered], [state for _, state in ordered]


def value_at(
    moments: list[datetime], values: list[float | None], moment: datetime
) -> float | None:
    index = bisect.bisect_right(moments, moment) - 1
    return values[index] if index >= 0 else None


def state_at(moments: list[datetime], states: list[str], moment: datetime) -> str | None:
    index = bisect.bisect_right(moments, moment) - 1
    return states[index] if index >= 0 else None


def grid(start: datetime, end: datetime) -> list[datetime]:
    count = math.ceil((end - start).total_seconds() / STEP.total_seconds())
    return [start + index * STEP for index in range(count)]


def integrate_dhw_heat(
    moments: list[datetime],
    powers: list[float | None],
    dhw_moments: list[datetime],
    dhw_states: list[str],
    start: datetime,
    end: datetime,
) -> float:
    energy_kwh = 0.0
    for moment in grid(start, end):
        if state_at(dhw_moments, dhw_states, moment) != "on":
            continue
        power = value_at(moments, powers, moment)
        if power is not None and power > 0:
            energy_kwh += power * STEP.total_seconds() / 3600 / 1000
    return energy_kwh


def winsorized_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) < 5:
        return statistics.median(ordered)
    low = ordered[max(0, math.floor((len(ordered) - 1) * 0.10))]
    high = ordered[min(len(ordered) - 1, math.ceil((len(ordered) - 1) * 0.90))]
    clipped = [min(max(value, low), high) for value in ordered]
    return sum(clipped) / len(clipped)


def learn(
    history: list[list[dict]],
    start: datetime,
    end: datetime,
    tz: ZoneInfo,
    loss_kw: float,
    collect_events: bool = False,
) -> dict:
    by_entity = {
        rows[0]["entity_id"]: rows
        for rows in history
        if rows and rows[0].get("entity_id")
    }
    missing = [entity for entity in ENTITIES if entity not in by_entity]
    if missing:
        raise SystemExit(f"history response is missing: {', '.join(missing)}")

    flow_t, flow_v = numeric_history(by_entity[FLOW_ENTITY], tz)
    tank_t, tank_v = numeric_history(by_entity[TANK_ENTITY], tz)
    power_t, power_v = numeric_history(by_entity[POWER_ENTITY], tz)
    dhw_t, dhw_s = state_history(by_entity[DHW_STATE_ENTITY], tz)

    day_profiles: dict[str, dict[object, list[float]]] = {
        "weekday": {},
        "weekend": {},
    }
    day_classes: dict[str, str] = {}
    events: list[dict] = []
    current_day = start.date()
    last_complete_day = (end - timedelta(minutes=15)).date()

    while current_day <= last_complete_day:
        day_start = datetime.combine(current_day, time.min, tzinfo=tz)
        day_end = day_start + timedelta(days=1)
        if day_start < start:
            day_start = start
        if day_end > end:
            day_end = end
        points = grid(day_start, day_end)
        tank_values = [value_at(tank_t, tank_v, moment) for moment in points]
        flow_values = [value_at(flow_t, flow_v, moment) for moment in points]
        complete = (
            sum(value is not None for value in tank_values) >= len(points) * 0.80
            and sum(value is not None for value in flow_values) >= len(points) * 0.80
        )
        if not complete or day_end - day_start < timedelta(hours=20):
            current_day += timedelta(days=1)
            continue

        day_class = "weekend" if current_day.weekday() >= 5 else "weekday"
        profile = [0.0] * 96
        active: list[int] = []
        for index, flow in enumerate(flow_values):
            is_active = flow is not None and flow > FLOW_THRESHOLD_LPM
            if is_active:
                active.append(index)
                continue
            if active:
                _learn_event(
                    active,
                    points,
                    flow_values,
                    tank_t,
                    tank_v,
                    power_t,
                    power_v,
                    dhw_t,
                    dhw_s,
                    profile,
                    events,
                    day_start,
                    loss_kw,
                    day_class,
                )
                active = []
        if active:
            _learn_event(
                active,
                points,
                flow_values,
                tank_t,
                tank_v,
                power_t,
                power_v,
                dhw_t,
                dhw_s,
                profile,
                events,
                day_start,
                loss_kw,
                day_class,
            )

        day_profiles[day_class][current_day] = profile
        day_classes[current_day.isoformat()] = day_class
        current_day += timedelta(days=1)

    weekday_days = len(day_profiles["weekday"])
    weekend_days = len(day_profiles["weekend"])
    windows = {
        day_class: discover_windows(
            events, day_class, len(day_profiles[day_class])
        )
        for day_class in ("weekday", "weekend")
    }
    weekday = aggregate_profile(day_profiles["weekday"], windows["weekday"])
    weekend = aggregate_profile(day_profiles["weekend"], windows["weekend"])
    included_events = [
        event
        for event in events
        if _inside_windows(
            windows[event["day_class"]],
            datetime.fromisoformat(event["start"]),
        )
    ]
    ready = weekday_days >= 10 and weekend_days >= 4
    result = {
        "status": "ready" if ready else "insufficient",
        "weekday": weekday,
        "weekend": weekend,
        "weekday_days": weekday_days,
        "weekend_days": weekend_days,
        "weekday_kwh": round(sum(weekday), 3),
        "weekend_kwh": round(sum(weekend), 3),
        "events": len(included_events),
        "thermal_loss_kw": loss_kw,
        "tank_capacity_kwh_per_k": round(TANK_CAPACITY_KWH_PER_K, 4),
        "history_start": start.isoformat(),
        "history_end": end.isoformat(),
        "learned_at": datetime.now(tz).isoformat(),
        "method": "water-gated tank balance; recurring-event windows; winsorized mean",
        "schedule_windows": {
            day_type: [
                [start.strftime("%H:%M"), end.strftime("%H:%M")]
                for start, end in windows[day_type]
            ]
            for day_type in ("weekday", "weekend")
        },
    }
    if collect_events:
        result["event_details"] = events
    return result


def _learn_event(
    active: list[int],
    points: list[datetime],
    flow_values: list[float | None],
    tank_t: list[datetime],
    tank_v: list[float | None],
    power_t: list[datetime],
    power_v: list[float | None],
    dhw_t: list[datetime],
    dhw_s: list[str],
    profile: list[float],
    events: list[dict],
    day_start: datetime,
    loss_kw: float,
    day_class: str,
) -> None:
    event_start = points[active[0]]
    event_end = points[active[-1]] + STEP
    window_start = event_start - STEP * 3
    window_end = event_end + STEP * 9
    if window_start < day_start:
        window_start = day_start

    tank_start = value_at(tank_t, tank_v, window_start)
    tank_end = value_at(tank_t, tank_v, window_end)
    if tank_start is None or tank_end is None:
        return

    litres = sum(
        (flow_values[index] or 0.0) * STEP.total_seconds() / 60
        for index in active
    )
    if litres < MIN_EVENT_LITRES:
        return

    heat_kwh = integrate_dhw_heat(
        power_t, power_v, dhw_t, dhw_s, window_start, window_end
    )
    duration_h = (window_end - window_start).total_seconds() / 3600
    draw_kwh = (
        heat_kwh
        - loss_kw * duration_h
        - TANK_CAPACITY_KWH_PER_K * (tank_end - tank_start)
    )
    if not MIN_DRAW_KWH <= draw_kwh <= MAX_EVENT_DRAW_KWH:
        return

    weights = [
        max((flow_values[index] or 0.0) * STEP.total_seconds() / 60, 0.0)
        for index in active
    ]
    total_weight = sum(weights)
    if total_weight <= 0:
        return
    for index, weight in zip(active, weights):
        moment = points[index]
        slot = (moment.hour * 60 + moment.minute) // 15
        profile[slot] += draw_kwh * weight / total_weight

    events.append(
        {
            "start": event_start.isoformat(),
            "end": event_end.isoformat(),
            "litres": round(litres, 2),
            "draw_kwh": round(draw_kwh, 3),
            "day_class": day_class,
        }
    )


def discover_windows(
    events: list[dict], day_class: str, complete_days: int
) -> list[tuple[time, time]]:
    """Group recurring substantive events by local start hour."""
    if complete_days < 1:
        return []
    candidates = [
        event
        for event in events
        if event["day_class"] == day_class
        and event["draw_kwh"] >= CLUSTER_EVENT_KWH
    ]
    if not candidates:
        return []

    days_by_hour: dict[int, set[object]] = defaultdict(set)
    for event in candidates:
        start = datetime.fromisoformat(event["start"])
        days_by_hour[start.hour].add(start.date())
    minimum_days = max(4, math.ceil(complete_days * 0.20))
    selected_hours = sorted(
        hour
        for hour, days in days_by_hour.items()
        if len(days) >= minimum_days
    )
    if not selected_hours:
        return []

    groups: list[list[int]] = []
    for hour in selected_hours:
        if not groups or hour != groups[-1][-1] + 1:
            groups.append([hour])
        else:
            groups[-1].append(hour)

    windows: list[tuple[time, time]] = []
    for group in groups:
        start_minute = int(
            group[0] * 60 - WINDOW_PADDING.total_seconds() / 60
        ) % 1440
        end_minute = int(
            (group[-1] + 1) * 60 + WINDOW_PADDING.total_seconds() / 60
        ) % 1440
        windows.append((_time_from_minute(start_minute), _time_from_minute(end_minute)))
    return windows


def _minute_of_day(moment: datetime) -> int:
    return moment.hour * 60 + moment.minute


def _time_from_minute(minute: int) -> time:
    minute %= 1440
    return time(minute // 60, minute % 60)


def _inside_windows(windows: list[tuple[time, time]], moment: datetime) -> bool:
    current = _minute_of_day(moment)
    for start, end in windows:
        start_minute = start.hour * 60 + start.minute
        end_minute = end.hour * 60 + end.minute
        if start_minute <= end_minute:
            if start_minute <= current <= end_minute:
                return True
        elif current >= start_minute or current <= end_minute:
            return True
    return False


def aggregate_profile(
    profiles: dict[object, list[float]],
    windows: list[tuple[time, time]],
) -> list[float]:
    if not windows:
        return [0.0] * 96
    result = [0.0] * 96
    for slot in range(96):
        moment = datetime.combine(
            datetime.now().date(),
            _time_from_minute(slot * 15),
        )
        if not _inside_windows(windows, moment):
            continue
        result[slot] = round(
            winsorized_mean([profile[slot] for profile in profiles.values()]),
            4,
        )
    return result


def push(host: str, token: str, profile: dict) -> None:
    body = json.dumps(profile).encode()
    request = urllib.request.Request(
        f"{host}/api/events/emhass_dhw_profile_learned",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status not in (200, 201):
            raise SystemExit(f"profile push failed with HTTP {response.status}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--secrets", default=DEFAULT_SECRETS)
    parser.add_argument("--timezone", default=DEFAULT_TZ)
    parser.add_argument("--days", type=int, default=42, help="complete days to learn from")
    parser.add_argument("--end", help="end timestamp; defaults to now")
    parser.add_argument("--loss-kw", type=float, default=DEFAULT_LOSS_KW)
    parser.add_argument("--push", action="store_true", help="store the profile in Home Assistant")
    parser.add_argument("--quiet", action="store_true", help="only print the push confirmation")
    parser.add_argument("--out", help="write the learned profile JSON here")
    parser.add_argument("--events-out", help="write individual detected events here")
    args = parser.parse_args()

    if args.days < 1:
        raise SystemExit("--days must be positive")
    tz = ZoneInfo(args.timezone)
    end = parse_moment(args.end, tz) if args.end else datetime.now(tz)
    start = end - timedelta(days=args.days)
    token = read_token(args.secrets)
    history = fetch_history(args.host, token, ENTITIES, start, end)
    profile = learn(
        history,
        start,
        end,
        tz,
        args.loss_kw,
        collect_events=bool(args.events_out),
    )
    events = profile.pop("event_details", [])
    if args.events_out:
        Path(args.events_out).write_text(json.dumps(events, indent=2) + "\n")

    encoded = json.dumps(profile, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(encoded)
    if not args.quiet:
        print(encoded, end="")
    if args.push:
        push(args.host, token, profile)
        print("pushed emhass_dhw_profile_learned", file=__import__("sys").stderr)


if __name__ == "__main__":
    main()
