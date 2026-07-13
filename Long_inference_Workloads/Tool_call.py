​"""
planner_agent.py — Plan → Execute → Synthesize Trip & Daily Agent (v5)
=====================================================================

Bug fixes from v4:
  FIX 1: Model no longer guesses origin city — asks user if missing
  FIX 2: "budget per person" correctly multiplied by num_people before comparison
  FIX 3: Budget verdict is authoritative from tool, model never invents ✅/❌

Free, no-key APIs: Nominatim, OSRM, Open-Meteo, Overpass, Frankfurter, arXiv.

Install:  pip install ollama httpx python-dotenv
Run:      python planner_agent.py --model qwen2.5:14b
          python planner_agent.py --mode bench --input tasks.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import httpx
import ollama
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("planner_agent")

DEFAULT_MODEL  = "qwen2.5:14b"
_CURRENT_MODEL = DEFAULT_MODEL
USER_AGENT     = "PlannerAgent/5.0"

SOURCES = {
    "geocoding": "OpenStreetMap Nominatim — free, no key",
    "routing":   "OSRM public routing — real road distance & drive time",
    "distance":  "Haversine great-circle (straight-line fallback)",
    "weather":   "Open-Meteo Archive API (ERA5, last full year)",
    "forecast":  "Open-Meteo Forecast API (upcoming days)",
    "places":    "OpenStreetMap Overpass API — crowd-sourced",
    "budget":    "Tiered INR rates (metro/tier-2/hill/beach) — not real-time",
    "calendar":  "Python standard-library date arithmetic",
    "itinerary": "Greedy nearest-neighbour proximity grouping",
    "currency":  "Frankfurter / ECB reference rates",
    "units":     "Exact SI conversion factors",
    "time":      "Open-Meteo geocoding timezone + system clock",
    "arxiv":     "arXiv.org Export API",
}


# ══════════════════════════════════════════════════════════════════════════════
# HTTP + reliability helpers
# ══════════════════════════════════════════════════════════════════════════════

def _retry(fn, attempts: int = 3, delay: float = 1.5, label: str = ""):
    last_err = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if i < attempts - 1:
                time.sleep(delay)
    raise RuntimeError(f"{label or 'request'} failed after {attempts} tries: {last_err}")


def _get(url: str, *, params=None, timeout: float = 20.0) -> httpx.Response:
    r = httpx.get(url, params=params, headers={"User-Agent": USER_AGENT},
                  timeout=timeout, follow_redirects=True)
    r.raise_for_status()
    return r


_GEOCODE_CACHE: dict = {}
_TOOL_CACHE:    dict = {}

_PLACE_TYPES = {
    "city", "town", "village", "municipality",
    "suburb", "hamlet", "locality", "state_capital",
}


def _pick_best_geocode(results: list):
    if not results:
        return None
    preferred = [r for r in results
                 if r.get("class") == "place" and r.get("type") in _PLACE_TYPES]
    pool = preferred or results
    return max(pool, key=lambda r: float(r.get("importance", 0) or 0))


def _cache_key(name: str, kwargs: dict) -> str:
    return name + "::" + json.dumps(kwargs, sort_keys=True, default=str)


def _cached(name: str, kwargs: dict, fn):
    key = _cache_key(name, kwargs)
    if key in _TOOL_CACHE:
        return _TOOL_CACHE[key] + "\n[cached]"
    result = fn()
    _TOOL_CACHE[key] = result
    return result


def _geocode(place: str):
    k = place.strip().lower()
    if k in _GEOCODE_CACHE:
        return _GEOCODE_CACHE[k]

    def _do():
        r = _get("https://nominatim.openstreetmap.org/search",
                 params={"q": place, "format": "json", "limit": 5,
                         "addressdetails": 1}, timeout=15)
        data = r.json()
        best = _pick_best_geocode(data)
        if not best:
            raise ValueError(f"Could not geocode {place!r}")
        return float(best["lat"]), float(best["lon"]), best.get("display_name", place)

    result = _retry(_do, label=f"geocode({place})")
    _GEOCODE_CACHE[k] = result
    return result


def _haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(d_lon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _osrm_route(lat1, lon1, lat2, lon2):
    try:
        url = (f"https://router.project-osrm.org/route/v1/driving/"
               f"{lon1},{lat1};{lon2},{lat2}")
        r = _get(url, params={"overview": "false"}, timeout=15).json()
        if r.get("code") == "Ok" and r.get("routes"):
            route = r["routes"][0]
            return round(route["distance"] / 1000, 1), round(route["duration"] / 60)
    except Exception as e:
        log.debug("OSRM failed: %s", e)
    return None


def _route_distance(origin: str, destination: str):
    lat1, lon1, _ = _geocode(origin)
    lat2, lon2, _ = _geocode(destination)
    osrm = _osrm_route(lat1, lon1, lat2, lon2)
    if osrm:
        return osrm[0], osrm[1], "road"
    straight = round(_haversine(lat1, lon1, lat2, lon2), 1)
    return round(straight * 1.25, 1), None, "estimate"


# ══════════════════════════════════════════════════════════════════════════════
# System + planner prompts
# ══════════════════════════════════════════════════════════════════════════════

def build_system_prompt() -> str:
    today = date.today().strftime("%A, %d %B %Y")
    return f"""Today is {today}.

You are Planner, an assistant for trip planning and everyday tasks, backed by
real-data tools. Work in three beats: PLAN what you need, gather it with tools,
then give a clear, decisive answer.

CHOOSING TOOLS
- Pick tools by what the request actually needs. "What's the weather in Goa?"
  needs only weather. "How far is X from Y?" needs only distance.
- A full trip plan needs all four: distance, weather, places, and budget. Then
  build a day-by-day itinerary. Call get_places_to_visit BEFORE plan_daily_itinerary
  and pass its full output into plan_daily_itinerary.
- Call independent tools together so they run in parallel. Build the itinerary
  AFTER you have the place list.
- Trust tool data for all factual numbers — never invent distances, temperatures,
  or costs from memory.

MISSING INFORMATION — CRITICAL RULES:
  ORIGIN: If the user asks to plan a trip but does NOT say where they will travel
  FROM, do NOT guess a city. Reply with exactly this one question:
  "Where will you be travelling from?" and wait for their answer before calling
  any tools. This is the only field that stops execution.
  OTHER FIELDS: If num_days or num_people are missing, assume 3 days and 1 person,
  say so explicitly, and continue normally.

WEATHER: for a trip within ~2 weeks, prefer get_weather_forecast (real upcoming
weather). For trips further out or undated, use get_weather_data (typical seasonal
climate based on last year).

BUDGET — READ CAREFULLY:
  - NEVER write "within budget" or "over budget" yourself.
  - The estimate_trip_budget tool always returns a ✅ or ❌ verdict line when a
    budget is given. Copy that line verbatim into your answer — it is the only
    authoritative source.
  - If the user says "budget per person is Rs X" for N people, pass
    budget_per_person=X and num_people=N. The tool multiplies automatically.
  - If the user gives a total budget for the whole group, pass budget_limit=total.
  - If no budget is stated, do not add a verdict at all — just show the total.
  - NEVER use calculator_calculate for trip costs. Always use estimate_trip_budget
    with place names.

ANSWERING A TRIP
- Open with the verdict: go / adjust / skip, in one line.
- Then: 🗺️ travel (real km + drive time), 🌤️ weather, 📍 places, 🗓️ day-by-day,
  💰 budget — show the FULL breakdown the budget tool returns, copy the ✅/❌ line
  exactly, and state the assumed number of travellers.
- Close with the data sources you used.
Keep non-trip answers to a few sentences."""


def _planner_instruction() -> str:
    return (
        "Before calling any tool, think for one moment and write a short PLAN "
        "(3-6 lines): the goal, the specific information this request needs, and "
        "the tools you will use to get it — noting any tool that must run after "
        "another (e.g. the itinerary runs after the places list). For a full trip "
        "question, your plan should cover travel/distance, weather, attractions "
        "and budget. Do NOT call any tool yet — only plan."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TOOLS
# ══════════════════════════════════════════════════════════════════════════════

# ── Distance ──────────────────────────────────────────────────────────────────
def get_distance_travel(origin: str, destination: str) -> str:
    def _compute():
        km, drive_min, src = _route_distance(origin, destination)
        _, _, n1 = _geocode(origin)
        _, _, n2 = _geocode(destination)
        if km < 2:     mode = "walk"
        elif km < 15:  mode = "auto/cab"
        elif km < 80:  mode = "cab or bus"
        elif km < 300: mode = "train or bus"
        elif km < 800: mode = "overnight train or flight"
        else:          mode = "flight"
        if drive_min is not None:
            h, m = divmod(drive_min, 60)
            drive = f"{h}h {m}m by road" if h else f"{m}m by road"
            dist_line = f"  Road distance: {km} km ({drive}, real routing)"
            src_line  = f"[sources: {SOURCES['geocoding']}; {SOURCES['routing']}]"
        else:
            dist_line = f"  Distance: ~{km} km (straight-line ×1.25 estimate)"
            src_line  = f"[sources: {SOURCES['geocoding']}; {SOURCES['distance']}]"
        return (f"DISTANCE\n"
                f"{dist_line}\n"
                f"  From: {n1.split(',')[0]}\n"
                f"  To:   {n2.split(',')[0]}\n"
                f"  Recommended mode: {mode}\n"
                f"{src_line}")
    return _cached("get_distance_travel",
                   {"origin": origin, "destination": destination}, _compute)


# ── Historical / seasonal weather ─────────────────────────────────────────────
def get_weather_data(location: str, month_start: int, month_end: int) -> str:
    def _compute():
        try:
            lat, lon, display = _geocode(location)
            year  = date.today().year - 1
            ms    = max(1, min(12, int(month_start)))
            me    = max(ms, min(12, int(month_end)))
            start = f"{year}-{ms:02d}-01"
            end   = (f"{year}-12-31" if me == 12 else
                     (datetime.strptime(f"{year}-{me+1:02d}-01", "%Y-%m-%d")
                      - timedelta(days=1)).strftime("%Y-%m-%d"))

            def _do():
                return _get("https://archive-api.open-meteo.com/v1/archive",
                            params={"latitude": lat, "longitude": lon,
                                    "start_date": start, "end_date": end,
                                    "daily": ["temperature_2m_mean", "precipitation_sum"],
                                    "timezone": "auto"}, timeout=20)

            data    = _retry(_do, label="open-meteo").json().get("daily", {})
            temps   = [t for t in data.get("temperature_2m_mean", []) if t is not None]
            rains   = [p for p in data.get("precipitation_sum",   []) if p is not None]
            if not temps:
                return f"No weather data for {location} in that period."
            avg     = round(sum(temps) / len(temps), 1)
            total   = round(sum(rains), 1)
            monthly = round(total / max(me - ms + 1, 1), 1)
            rainy   = sum(1 for p in rains if p > 1)
            names   = ["","Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
            period  = names[ms] if ms == me else f"{names[ms]}-{names[me]}"
            verdict = ("Generally pleasant." if avg < 32 and monthly < 150
                       else "Heat or heavy rain — pack accordingly.")
            return (f"WEATHER (typical, {period} {year}) — {display.split(',')[0]}:\n"
                    f"  Avg temperature:  {avg}°C\n"
                    f"  Total rainfall:   {total} mm\n"
                    f"  Monthly avg rain: {monthly} mm/month\n"
                    f"  Rainy days:       {rainy} (>1mm)\n"
                    f"  Verdict:          {verdict}\n"
                    f"[sources: {SOURCES['geocoding']}; {SOURCES['weather']}]")
        except Exception as e:
            return f"Weather lookup failed: {e}"
    return _cached("get_weather_data",
                   {"location": location, "month_start": month_start,
                    "month_end": month_end}, _compute)


# ── Live upcoming forecast ────────────────────────────────────────────────────
def get_weather_forecast(location: str, days: int = 7) -> str:
    def _compute():
        try:
            lat, lon, display = _geocode(location)
            fdays = max(1, min(16, int(days)))

            def _do():
                return _get("https://api.open-meteo.com/v1/forecast",
                            params={"latitude": lat, "longitude": lon,
                                    "daily": ["temperature_2m_max", "temperature_2m_min",
                                              "precipitation_sum",
                                              "precipitation_probability_max"],
                                    "forecast_days": fdays, "timezone": "auto"}, timeout=20)

            d     = _retry(_do, label="open-meteo-forecast").json().get("daily", {})
            dates = d.get("time", [])
            tmax  = d.get("temperature_2m_max", [])
            tmin  = d.get("temperature_2m_min", [])
            rain  = d.get("precipitation_sum", [])
            pop   = d.get("precipitation_probability_max", [])
            if not dates:
                return f"No forecast available for {location}."
            lines = [f"FORECAST (next {len(dates)} days) — {display.split(',')[0]}:"]
            for i, day in enumerate(dates):
                r = rain[i] if i < len(rain) else 0
                p = pop[i] if i < len(pop) and pop[i] is not None else 0
                lines.append(f"  {day}: {tmin[i]:.0f}-{tmax[i]:.0f}°C, "
                             f"{r:.1f} mm rain ({p:.0f}% chance)")
            wet     = sum(1 for r in rain if r and r > 5)
            verdict = ("Mostly dry — good for travel." if wet <= len(dates) // 3
                       else "Several wet days — plan indoor options.")
            lines.append(f"  Verdict: {verdict}")
            lines.append(f"[sources: {SOURCES['geocoding']}; {SOURCES['forecast']}]")
            return "\n".join(lines)
        except Exception as e:
            return f"Forecast lookup failed: {e}"
    return _cached("get_weather_forecast", {"location": location, "days": days}, _compute)


# ── Places ────────────────────────────────────────────────────────────────────
def get_places_to_visit(location: str, category: str = "tourism") -> str:
    def _compute():
        try:
            lat, lon, display = _geocode(location)
            tag_map = {
                "tourism": '["tourism"~"attraction|museum|viewpoint|zoo|gallery|theme_park"]',
                "food":    '["amenity"~"restaurant|cafe"]',
                "nature":  '["leisure"~"park|garden|nature_reserve"]',
            }
            tag   = tag_map.get(category, tag_map["tourism"])
            query = (f"[out:json][timeout:25];\n"
                     f"(node{tag}(around:15000,{lat},{lon});\n"
                     f" way{tag}(around:15000,{lat},{lon}););\n"
                     f"out center 25;")
            endpoints  = ["https://overpass-api.de/api/interpreter",
                          "https://overpass.kumi.systems/api/interpreter"]
            elements, last_err = None, None
            for ep in endpoints:
                try:
                    def _do(ep=ep):
                        r = httpx.post(ep, data={"data": query},
                                       headers={"User-Agent": USER_AGENT}, timeout=30)
                        r.raise_for_status()
                        return r
                    elements = _retry(_do, attempts=2, delay=2.0,
                                      label="overpass").json().get("elements", [])
                    break
                except Exception as e:
                    last_err = e
            if elements is None:
                return f"Places lookup failed: {last_err}"
            seen, unique = set(), []
            for el in elements:
                tags = el.get("tags", {})
                name = tags.get("name")
                kind = (tags.get("tourism") or tags.get("amenity")
                        or tags.get("leisure") or category)
                elat = el.get("lat") or el.get("center", {}).get("lat")
                elon = el.get("lon") or el.get("center", {}).get("lon")
                if name and elat and elon and name not in seen:
                    seen.add(name)
                    unique.append({"name": name, "kind": kind,
                                   "lat": float(elat), "lon": float(elon)})
            if not unique:
                return f"No {category} places found near {display.split(',')[0]} in OSM."
            lines = [f"PLACES near {display.split(',')[0]} ({category}):"]
            for p in unique[:12]:
                lines.append(f"  • {p['name']} ({p['kind']}) [{p['lat']:.4f},{p['lon']:.4f}]")
            lines.append(f"[source: {SOURCES['places']}]")
            return "\n".join(lines)
        except Exception as e:
            return f"Places lookup failed: {e}"
    return _cached("get_places_to_visit",
                   {"location": location, "category": category}, _compute)


# ── Budget ────────────────────────────────────────────────────────────────────
_METRO = {"mumbai","delhi","bangalore","bengaluru","chennai","kolkata",
          "hyderabad","pune","ahmedabad"}
_HILL  = {"ooty","kodaikanal","manali","shimla","darjeeling","coorg",
          "munnar","mussoorie","nainital","mcleod ganj"}
_BEACH = {"goa","pondicherry","kanyakumari","kovalam","varkala",
          "mahabalipuram","puri","vizag","visakhapatnam"}
_HOTEL = {"metro": 2000, "hill": 1800, "beach": 1600, "tier2": 1200}
_FOOD  = {"metro": 800,  "hill": 700,  "beach": 700,  "tier2": 500}
_MISC  = {"metro": 600,  "hill": 500,  "beach": 500,  "tier2": 350}


def _tier(name: str) -> str:
    n = name.strip().lower()
    if any(c in n for c in _METRO): return "metro"
    if any(c in n for c in _HILL):  return "hill"
    if any(c in n for c in _BEACH): return "beach"
    return "tier2"


def _train_per_km(km: float) -> float:
    return 2.0 if km < 300 else 1.6 if km < 800 else 1.3


def estimate_trip_budget(
    origin: str,
    destination: str,
    num_days: int,
    num_people: int = 1,
    travel_mode: str = "auto",
    budget_limit: int = 0,
    budget_per_person: int = 0,   # ← FIX 2: new parameter
) -> str:
    # ── FIX 2: Normalise budget before caching ────────────────────────────────
    _bpp = int(budget_per_person or 0)
    _bl  = int(budget_limit or 0)
    _np  = max(1, int(num_people))
    if _bpp > 0 and _bl == 0:
        # User said "Rs X per person" — compute total for the group
        _bl = _bpp * _np

    def _compute():
        try:
            km, drive_min, _src = _route_distance(origin, destination)
            days   = max(1, int(num_days))
            people = _np
            mode   = travel_mode

            if mode == "auto":
                mode = "cab" if km <= 15 else "train" if km <= 800 else "flight"

            rate = (_train_per_km(km) if mode == "train"
                    else {"cab": 14, "bus": 2, "flight": 6}.get(mode, 5))

            # Cabs are shared (1 per ≤4 people); trains/flights charge per person
            if mode in {"train", "bus", "flight"}:
                travel_units = people
                travel_basis = f"{people} traveller(s)"
            else:
                travel_units = math.ceil(people / 4)
                travel_basis = f"{travel_units} cab(s)"

            travel = round(km * rate * travel_units * 2, -1)
            rooms  = math.ceil(people / 2)       # double occupancy
            tier   = _tier(destination)
            stay   = _HOTEL[tier] * days * rooms
            food   = _FOOD[tier]  * days * people
            misc   = _MISC[tier]  * days * people
            grand  = int(travel + stay + food + misc)

            # ── FIX 3: authoritative verdict ──────────────────────────────────
            if _bl > 0:
                diff = grand - _bl
                per_person_note = (
                    f" (Rs {_bl // people:,}/person × {people} people)"
                    if people > 1 else ""
                )
                if diff <= 0:
                    verdict = (f"\n  ✅ Within your Rs {_bl:,} budget"
                               f"{per_person_note} — spare Rs {abs(diff):,}")
                else:
                    verdict = (f"\n  ❌ Over your Rs {_bl:,} budget"
                               f"{per_person_note} by Rs {diff:,}")
            else:
                verdict = "\n  (No budget limit stated — showing estimate only.)"

            note   = f"Assuming {people} traveller" + ("s." if people > 1 else ".")
            drive  = f" (~{drive_min} min drive)" if drive_min else ""
            return (f"BUDGET ESTIMATE — tiered INR ({tier}), not real-time. {note}\n"
                    f"  Distance:            {km} km{drive}\n"
                    f"  Travel mode:         {mode}\n"
                    f"  Travel (round trip): Rs {int(travel):,}  "
                    f"({km} km × Rs {rate}/km × {travel_basis} × 2)\n"
                    f"  Stay  ({days}d × {rooms} room(s) @ Rs {_HOTEL[tier]}/night): "
                    f"Rs {stay:,}\n"
                    f"  Food  ({days}d × {people}p @ Rs {_FOOD[tier]}/day):   "
                    f"Rs {food:,}\n"
                    f"  Misc/local transport:                Rs {misc:,}\n"
                    f"  ──────────────────────────────────────────\n"
                    f"  ROUGH TOTAL:                         Rs {grand:,}"
                    f"{verdict}\n"
                    f"[source: {SOURCES['budget']}]")
        except Exception as e:
            return f"Budget estimate failed: {e}"

    return _cached("estimate_trip_budget",
                   {"origin": origin, "destination": destination,
                    "num_days": num_days, "num_people": _np,
                    "travel_mode": travel_mode, "budget_limit": _bl,
                    "budget_per_person": _bpp}, _compute)


# ── Dates ─────────────────────────────────────────────────────────────────────
def check_dates(reference: str = "this weekend") -> str:
    try:
        today = date.today()
        ref   = reference.strip().lower()

        def next_wd(start, wd, skip=False):
            d = wd - start.weekday()
            if d < 0 or (d == 0 and skip):
                d += 7
            return start + timedelta(days=d)

        if "today"        in ref: target = today
        elif "tomorrow"   in ref: target = today + timedelta(days=1)
        elif "next weekend" in ref: target = next_wd(today, 5) + timedelta(days=7)
        elif "weekend"    in ref: target = next_wd(today, 5)
        else:
            wds = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
            target = next((next_wd(today, i, skip=True)
                           for i, w in enumerate(wds) if w in ref), next_wd(today, 5))

        sat    = target if target.weekday() == 5 else next_wd(target, 5)
        sun    = sat + timedelta(days=1)
        away   = (target - today).days
        is_we  = target.weekday() >= 5
        return (f"'{reference}' → {target.strftime('%A, %d %B %Y')} ({away} day(s) away). "
                f"{'Weekend ✅' if is_we else 'Weekday — nearest weekend:'} "
                f"{sat.strftime('%a %d %b')} - {sun.strftime('%a %d %b %Y')}.\n"
                f"[source: {SOURCES['calendar']}]")
    except Exception as e:
        return f"Date check failed: {e}"


# ── Itinerary (proximity-grouped) ─────────────────────────────────────────────
def _parse_places(places_str: str):
    tokens = []
    for line in places_str.splitlines() or [places_str]:
        tokens.extend(re.split(r",\s*(?![^\[]*\])", line))
    items = []
    for raw in tokens:
        raw = raw.strip().lstrip("•-*").strip()
        if not raw:
            continue
        low = raw.lower()
        if low.startswith(("places near", "[source")) or raw[-1:] == ":":
            continue
        m     = re.search(r"\[(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\]", raw)
        clean = re.sub(r"\[.*?\]", "", raw).strip()
        disp  = re.sub(r"\s*\(.*?\)\s*$", "", clean).strip() or clean
        if not disp:
            continue
        if m:
            items.append({"name": disp, "lat": float(m.group(1)), "lon": float(m.group(2))})
        else:
            items.append({"name": disp, "lat": None, "lon": None})
    return items


def _greedy_route(items: list) -> list:
    with_c    = [p for p in items if p["lat"] is not None]
    without_c = [p for p in items if p["lat"] is None]
    if len(with_c) <= 1:
        return with_c + without_c
    unvisited = list(with_c)
    ordered   = [unvisited.pop(0)]
    while unvisited:
        last = ordered[-1]
        nxt  = min(unvisited,
                   key=lambda p: _haversine(last["lat"], last["lon"],
                                            p["lat"], p["lon"]))
        ordered.append(nxt)
        unvisited.remove(nxt)
    return ordered + without_c


def plan_daily_itinerary(places: str, num_days: int,
                         start_location: str = "") -> str:
    try:
        days  = max(1, int(num_days))
        items = _parse_places(places)
        if not items:
            return "No places provided for itinerary."
        items = _greedy_route(items)
        slots = [2 if (i == 0 or i == days - 1) else 3 for i in range(days)]
        names = [p["name"] for p in items]
        while len(names) < sum(slots):
            names += names
        names = names[:sum(slots)]
        lines = [f"ITINERARY ({days} days, proximity-grouped):"]
        idx   = 0
        full  = ["🌅 Morning", "🌞 Afternoon", "🌇 Evening"]
        short = ["🌅 Morning", "🌇 Evening"]
        for day, n_slots in enumerate(slots, start=1):
            first = day == 1
            last  = day == days
            if first and start_location:
                lines.append(f"\nDay {day} — Travel + Arrival")
                lines.append(f"  🚆 Morning:   Travel from {start_location}")
                lines.append(f"  🏨 Afternoon: Check in, settle")
                for j in range(n_slots):
                    lines.append(f"  📍 {short[j % 2]}: {names[idx]}"); idx += 1
            elif last and start_location:
                lines.append(f"\nDay {day} — Sightseeing + Return")
                for j in range(n_slots):
                    lines.append(f"  📍 {short[j % 2]}: {names[idx]}"); idx += 1
                lines.append(f"  🚆 Evening:   Return to {start_location}")
            else:
                lines.append(f"\nDay {day} — Full Sightseeing Day")
                for j in range(n_slots):
                    lines.append(f"  📍 {full[j % 3]}: {names[idx]}"); idx += 1
        lines.append(f"\n[source: {SOURCES['itinerary']}]")
        return "\n".join(lines)
    except Exception as e:
        return f"Itinerary planning failed: {e}"


# ── Currency ──────────────────────────────────────────────────────────────────
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    def _compute():
        try:
            frm = from_currency.upper().strip()
            to  = to_currency.upper().strip()
            amt = float(amount)
            if frm == to:
                return f"{amt:,.2f} {frm} = {amt:,.2f} {to} (same currency)."

            def _do():
                return _get("https://api.frankfurter.app/latest",
                            params={"amount": amt, "from": frm, "to": to}, timeout=15)

            data  = _retry(_do, label="frankfurter").json()
            rates = data.get("rates", {})
            if to not in rates:
                return (f"Couldn't convert {frm}→{to}. Frankfurter covers major "
                        f"currencies (USD, EUR, GBP, INR, JPY…); check the codes.")
            conv = rates[to]
            unit = conv / amt if amt else 0
            return (f"{amt:,.2f} {frm} = {conv:,.2f} {to}\n"
                    f"  Rate: 1 {frm} = {unit:,.4f} {to} "
                    f"(as of {data.get('date', 'latest')})\n"
                    f"[source: {SOURCES['currency']}]")
        except Exception as e:
            return f"Currency conversion failed: {e}"
    return _cached("convert_currency",
                   {"amount": amount, "from_currency": from_currency,
                    "to_currency": to_currency}, _compute)


# ── Units ─────────────────────────────────────────────────────────────────────
_UNITS = {
    "length": {"base": "m", "mm": 0.001, "cm": 0.01, "m": 1, "km": 1000,
               "in": 0.0254, "inch": 0.0254, "ft": 0.3048, "foot": 0.3048,
               "feet": 0.3048, "yd": 0.9144, "yard": 0.9144,
               "mi": 1609.344, "mile": 1609.344, "miles": 1609.344},
    "mass":   {"base": "g", "mg": 0.001, "g": 1, "gram": 1, "kg": 1000,
               "kilogram": 1000, "oz": 28.3495, "ounce": 28.3495,
               "lb": 453.592, "pound": 453.592, "lbs": 453.592,
               "ton": 1_000_000, "tonne": 1_000_000},
    "volume": {"base": "l", "ml": 0.001, "l": 1, "litre": 1, "liter": 1,
               "tsp": 0.00492892, "tbsp": 0.0147868, "cup": 0.24,
               "pint": 0.473176, "quart": 0.946353, "gallon": 3.78541,
               "fl_oz": 0.0295735},
    "speed":  {"base": "m/s", "m/s": 1, "km/h": 0.277778, "kmph": 0.277778,
               "kph": 0.277778, "mph": 0.44704, "knot": 0.514444,
               "knots": 0.514444},
}


def _find_category(unit: str):
    u = unit.strip().lower()
    for cat, table in _UNITS.items():
        if u in table:
            return cat, u
    return None, u


def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    try:
        val = float(value)
        fu, tu = from_unit.strip().lower(), to_unit.strip().lower()
        temp = {"c": "c", "celsius": "c", "°c": "c",
                "f": "f", "fahrenheit": "f", "°f": "f",
                "k": "k", "kelvin": "k"}
        if fu in temp and tu in temp:
            f, t = temp[fu], temp[tu]
            c   = {"c": val, "f": (val - 32) * 5 / 9, "k": val - 273.15}[f]
            out = {"c": c,   "f": c * 9 / 5 + 32,     "k": c + 273.15}[t]
            return f"{val:g} °{f.upper()} = {out:.2f} °{t.upper()}\n[source: {SOURCES['units']}]"
        cat_f, fu = _find_category(fu)
        cat_t, tu = _find_category(tu)
        if cat_f is None or cat_t is None:
            return (f"Unknown unit(s): {from_unit!r}/{to_unit!r}. Supported: "
                    f"length, mass, volume, speed, temperature.")
        if cat_f != cat_t:
            return f"Can't convert {cat_f} to {cat_t} ({from_unit} → {to_unit})."
        out = (val * _UNITS[cat_f][fu]) / _UNITS[cat_t][tu]
        return f"{val:g} {from_unit} = {out:.4g} {to_unit}\n[source: {SOURCES['units']}]"
    except Exception as e:
        return f"Unit conversion failed: {e}"


# ── Local time ────────────────────────────────────────────────────────────────
def get_local_time(city: str) -> str:
    def _compute():
        try:
            def _do():
                return _get("https://geocoding-api.open-meteo.com/v1/search",
                            params={"name": city, "count": 1}, timeout=15)

            results = _retry(_do, label="geo-time").json().get("results")
            if not results:
                return f"Couldn't locate {city!r} to find its timezone."
            r0    = results[0]
            tz    = r0.get("timezone")
            label = ", ".join(filter(None, [r0.get("name"), r0.get("country")]))
            if not tz:
                return f"No timezone available for {city!r}."
            now = datetime.now(ZoneInfo(tz))
            off = now.strftime("%z")
            return (f"LOCAL TIME — {label}\n"
                    f"  {now.strftime('%A, %d %B %Y, %I:%M %p')}\n"
                    f"  Timezone: {tz} (UTC{off[:3]}:{off[3:]})\n"
                    f"[source: {SOURCES['time']}]")
        except Exception as e:
            return f"Local time lookup failed: {e}"
    return _cached("get_local_time", {"city": city}, _compute)


# ── Calculator ────────────────────────────────────────────────────────────────
def calculator_calculate(expression: str) -> str:
    """Plain arithmetic only. NOT for trip budgets — use estimate_trip_budget."""
    try:
        safe   = re.sub(r"[^0-9+\-*/().\s]", "", expression)
        result = eval(safe, {"__builtins__": {}})
        return f"Result: {result}  [source: pure arithmetic]"
    except Exception as e:
        return f"Calculator error: {e}"


# ── arXiv ─────────────────────────────────────────────────────────────────────
def arxiv_search_papers(query: str, max_results: int = 2) -> str:
    import urllib.parse
    import xml.etree.ElementTree as ET
    capped = min(max(1, int(max_results)), 3)
    url = (f"https://export.arxiv.org/api/query?search_query=all:"
           f"{urllib.parse.quote(query)}&max_results={capped}&sortBy=relevance")
    try:
        r  = _retry(lambda: _get(url, timeout=30), label="arxiv")
        ns = {"a": "http://www.w3.org/2005/Atom"}
        root   = ET.fromstring(r.text)
        papers = []
        for entry in root.findall("a:entry", ns):
            title   = " ".join((entry.findtext("a:title",   "", ns) or "").split())
            summary = " ".join((entry.findtext("a:summary", "", ns) or "").split())[:400]
            link    = (entry.findtext("a:id", "", ns) or "").strip()
            authors = [a.findtext("a:name", "", ns)
                       for a in entry.findall("a:author", ns)][:3]
            papers.append(f"Title: {title}\nAuthors: {', '.join(authors)}\n"
                          f"Abstract: {summary}...\nURL: {link}")
        if not papers:
            return f"No arXiv papers found for: {query}"
        return "\n\n".join(papers) + f"\n\n[source: {SOURCES['arxiv']}]"
    except Exception as e:
        return f"arXiv error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# Tool registry + schemas
# ══════════════════════════════════════════════════════════════════════════════

TOOL_FUNCTIONS = {
    "get_distance_travel":  get_distance_travel,
    "get_weather_data":     get_weather_data,
    "get_weather_forecast": get_weather_forecast,
    "get_places_to_visit":  get_places_to_visit,
    "estimate_trip_budget": estimate_trip_budget,
    "plan_daily_itinerary": plan_daily_itinerary,
    "check_dates":          check_dates,
    "convert_currency":     convert_currency,
    "convert_units":        convert_units,
    "get_local_time":       get_local_time,
    "calculator_calculate": calculator_calculate,
    "arxiv_search_papers":  arxiv_search_papers,
}

DEPENDENCIES = {
    "plan_daily_itinerary": {
        "prereq":     "get_places_to_visit",
        "inject_arg": "places",
    },
}

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "get_distance_travel",
        "description": ("Real road distance and driving time between two places "
                        "(OSRM routing, straight-line fallback). Use for any "
                        "'how far / how long' question or to ground a trip."),
        "parameters": {"type": "object", "properties": {
            "origin":      {"type": "string"},
            "destination": {"type": "string"}},
            "required": ["origin", "destination"]}}},
    {"type": "function", "function": {
        "name": "get_weather_data",
        "description": ("Typical seasonal weather (temperature, rainfall) from last "
                        "year's records. Use for undated trips or trips >2 weeks away."),
        "parameters": {"type": "object", "properties": {
            "location":    {"type": "string"},
            "month_start": {"type": "integer", "description": "1-12"},
            "month_end":   {"type": "integer", "description": "1-12"}},
            "required": ["location", "month_start", "month_end"]}}},
    {"type": "function", "function": {
        "name": "get_weather_forecast",
        "description": ("Actual upcoming weather forecast (up to 16 days). Prefer this "
                        "over get_weather_data when the trip is soon — 'this weekend', "
                        "'next week', etc."),
        "parameters": {"type": "object", "properties": {
            "location": {"type": "string"},
            "days":     {"type": "integer", "description": "1-16, default 7"}},
            "required": ["location"]}}},
    {"type": "function", "function": {
        "name": "get_places_to_visit",
        "description": ("Real points of interest from OpenStreetMap. Output carries "
                        "[lat,lon] tags that feed straight into plan_daily_itinerary. "
                        "category: tourism | food | nature."),
        "parameters": {"type": "object", "properties": {
            "location": {"type": "string"},
            "category": {"type": "string", "description": "tourism | food | nature"}},
            "required": ["location"]}}},
    {"type": "function", "function": {
        "name": "estimate_trip_budget",
        "description": (
            "Itemised INR trip budget. Looks up the REAL distance itself — do not pass a "
            "distance number. For budget comparison:\n"
            "  • If user says 'Rs X per person': pass budget_per_person=X, num_people=N. "
            "    The tool multiplies automatically — total = X × N.\n"
            "  • If user gives a total for the whole group: pass budget_limit=total.\n"
            "  • If no budget mentioned: leave both at 0."
        ),
        "parameters": {"type": "object", "properties": {
            "origin":           {"type": "string"},
            "destination":      {"type": "string"},
            "num_days":         {"type": "integer"},
            "num_people":       {"type": "integer", "description": "travellers, default 1"},
            "travel_mode":      {"type": "string",
                                 "description": "auto | cab | train | bus | flight"},
            "budget_limit":     {"type": "integer",
                                 "description": "total group budget in Rs. 0 if not given."},
            "budget_per_person":{"type": "integer",
                                 "description": ("per-person budget in Rs. Use when user says "
                                                 "'per person'. Tool multiplies by num_people. "
                                                 "0 if not given.")}},
            "required": ["origin", "destination", "num_days"]}}},
    {"type": "function", "function": {
        "name": "plan_daily_itinerary",
        "description": ("Proximity-grouped day-by-day plan. Call AFTER get_places_to_visit "
                        "and pass its FULL output as `places` so the [lat,lon] tags are "
                        "preserved for grouping."),
        "parameters": {"type": "object", "properties": {
            "places":         {"type": "string",
                               "description": "full output from get_places_to_visit"},
            "num_days":       {"type": "integer"},
            "start_location": {"type": "string",
                               "description": "origin city for travel-day notes"}},
            "required": ["places", "num_days"]}}},
    {"type": "function", "function": {
        "name": "check_dates",
        "description": "Resolve a relative date like 'this weekend' or 'next Friday' to a real calendar date.",
        "parameters": {"type": "object", "properties": {
            "reference": {"type": "string", "description": "e.g. 'this weekend'"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "convert_currency",
        "description": "Convert an amount between major currencies at the latest exchange rate.",
        "parameters": {"type": "object", "properties": {
            "amount":        {"type": "number"},
            "from_currency": {"type": "string", "description": "ISO code, e.g. USD"},
            "to_currency":   {"type": "string", "description": "ISO code, e.g. INR"}},
            "required": ["amount", "from_currency", "to_currency"]}}},
    {"type": "function", "function": {
        "name": "convert_units",
        "description": "Convert length, mass, volume, speed, or temperature.",
        "parameters": {"type": "object", "properties": {
            "value":     {"type": "number"},
            "from_unit": {"type": "string"},
            "to_unit":   {"type": "string"}},
            "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {
        "name": "get_local_time",
        "description": "Current local time and timezone for a city anywhere in the world.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}},
            "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "calculator_calculate",
        "description": "Evaluate a plain arithmetic expression. NOT for trip budgets.",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string"}},
            "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "arxiv_search_papers",
        "description": "Search arXiv for research papers. Only for explicit research requests.",
        "parameters": {"type": "object", "properties": {
            "query":       {"type": "string"},
            "max_results": {"type": "integer", "description": "1-3, default 2"}},
            "required": ["query"]}}},
]


# ══════════════════════════════════════════════════════════════════════════════
# Execution: dependency-aware + parallel within a tier
# ══════════════════════════════════════════════════════════════════════════════

def _safe_json(s):
    try:
        return json.loads(s)
    except Exception:
        return {}


def _run_one(name, args):
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return f"Unknown tool: {name}"
    try:
        return fn(**args)
    except TypeError as e:
        return f"Bad arguments for {name}: {e}"
    except Exception as e:
        return f"Tool error in {name}: {e}"


def _execute_batch(calls: list) -> tuple:
    n      = len(calls)
    results = [None] * n
    norm   = []
    for c in calls:
        args = c["function"]["arguments"]
        norm.append({"name": c["function"]["name"],
                     "args": _safe_json(args) if isinstance(args, str) else dict(args)})

    names_in_batch = {c["name"] for c in norm}
    tier0 = [i for i, c in enumerate(norm)
              if c["name"] not in DEPENDENCIES
              or DEPENDENCIES[c["name"]]["prereq"] not in names_in_batch]
    tier1 = [i for i in range(n) if i not in tier0]

    if tier0:
        with ThreadPoolExecutor(max_workers=len(tier0)) as pool:
            futs = {pool.submit(_run_one, norm[i]["name"], norm[i]["args"]): i
                    for i in tier0}
            for f in as_completed(futs):
                results[futs[f]] = f.result()

    if tier1:
        prereq_output = {norm[i]["name"]: results[i] for i in tier0}
        for i in tier1:
            dep = DEPENDENCIES[norm[i]["name"]]
            inj = dep["inject_arg"]
            cur = str(norm[i]["args"].get(inj, "") or "")
            if len(cur) < 20 and dep["prereq"] in prereq_output:
                norm[i]["args"][inj] = prereq_output[dep["prereq"]]
        with ThreadPoolExecutor(max_workers=len(tier1)) as pool:
            futs = {pool.submit(_run_one, norm[i]["name"], norm[i]["args"]): i
                    for i in tier1}
            for f in as_completed(futs):
                results[futs[f]] = f.result()

    return results, [c["name"] for c in norm], [c["args"] for c in norm]


# ══════════════════════════════════════════════════════════════════════════════
# Planner → Execute → Synthesize with multi-turn memory
# ══════════════════════════════════════════════════════════════════════════════

def _plan_phase(messages: list, model: str) -> str:
    try:
        scratch = messages + [{"role": "user", "content": _planner_instruction()}]
        resp    = ollama.chat(model=model, messages=scratch)
        return (resp["message"].get("content") or "").strip()
    except Exception as e:
        log.debug("plan phase skipped: %s", e)
        return ""


def _surface_budget(final_answer: str, last_outputs: dict) -> str:
    """Always surface the full budget breakdown, even if the model compressed it."""
    budget = last_outputs.get("estimate_trip_budget")
    if not budget or "ROUGH TOTAL" not in budget:
        return final_answer
    if "ROUGH TOTAL" in final_answer or "rough total" in final_answer.lower():
        return final_answer
    return (final_answer.rstrip()
            + "\n\n💰 Full budget breakdown\n"
            + "\n".join("  " + ln for ln in budget.splitlines()))


def run_task(
    prompt: str,
    model:    str  | None = None,
    messages: list | None = None,
    max_turns: int = 12,
    plan: bool = True,
) -> dict:
    global _CURRENT_MODEL
    if model:
        _CURRENT_MODEL = model.replace("ollama/", "")

    start = time.time()
    if messages is None:
        _TOOL_CACHE.clear()
        _GEOCODE_CACHE.clear()   # fresh per conversation
        messages = [{"role": "system", "content": build_system_prompt()}]
    messages = list(messages)
    messages.append({"role": "user", "content": prompt})

    tool_calls_made, last_outputs = [], {}
    plan_text, final_answer, error = "", "", ""

    try:
        # PLAN
        if plan:
            plan_text = _plan_phase(messages, _CURRENT_MODEL)
            if plan_text:
                messages.append({"role": "assistant",
                                 "content": "PLAN:\n" + plan_text})

        # EXECUTE
        for turn in range(max_turns):
            response = ollama.chat(model=_CURRENT_MODEL,
                                   messages=messages, tools=TOOL_SCHEMAS)
            msg   = response["message"]
            messages.append(msg)
            calls = msg.get("tool_calls")
            if not calls:
                final_answer = msg.get("content", "") or ""
                break
            results, names, arglist = _execute_batch(calls)
            for name, args, result in zip(names, arglist, results):
                tool_calls_made.append({"tool": name, "arguments": args})
                last_outputs[name] = str(result)
                messages.append({"role": "tool", "tool_name": name,
                                 "content": str(result)})
            if turn == max_turns - 1:
                final_answer = msg.get("content", "") or "(reached max turns)"

        # SYNTHESIZE
        final_answer = _surface_budget(final_answer, last_outputs)
        if messages and messages[-1].get("role") != "assistant":
            messages.append({"role": "assistant", "content": final_answer})

    except Exception as e:
        error = str(e)

    return {
        "plan":         plan_text,
        "tool_calls":   tool_calls_made,
        "final_answer": final_answer,
        "messages":     messages,
        "time":         round(time.time() - start, 2),
        "error":        error,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Scoring / interactive / benchmark
# ══════════════════════════════════════════════════════════════════════════════

def score_task(tool_calls, expected_str, answer=None):
    expected = [t.strip() for t in expected_str.split(",") if t.strip()]
    called   = [c["tool"] for c in tool_calls]
    if not expected:
        return {"discovery_recall": None, "discovery_precision": None, "tool_score": None}
    matched   = sum(1 for e in expected if any(e in c or c in e for c in called))
    recall    = round(matched / len(expected), 3)
    precision = round(matched / len(called),   3) if called else 0.0
    success   = bool(answer and answer.strip() and "error" not in answer.lower()[:40])
    return {
        "discovery_recall":    recall,
        "discovery_precision": precision,
        "execution_success":   success,
        "tool_score":          0.0 if not success else round((recall + precision) / 2, 3),
    }


def interactive(model: str):
    print(f"\n{'=' * 70}")
    print(f"  Planner Agent v5 — {model}")
    print(f"  Tools: {list(TOOL_FUNCTIONS.keys())}")
    print("  Plan→Execute→Synthesize | Dependency-aware | Memory ON")
    print("  Type a question, 'reset' to clear memory, or 'quit'")
    print(f"{'=' * 70}\n")
    history = None
    while True:
        try:
            q = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not q or q.lower() == "quit":
            break
        if q.lower() == "reset":
            history = None
            print("(memory cleared)\n")
            continue
        print("Thinking...\n")
        result  = run_task(q, model, messages=history)
        history = result["messages"]
        if result["plan"]:
            print("PLAN:\n  " + result["plan"].replace("\n", "\n  ") + "\n")
        print(f"Tool calls: {len(result['tool_calls'])}  ({result['time']}s)")
        for c in result["tool_calls"]:
            print(f"  → {c['tool']}({json.dumps(c['arguments'])[:80]})")
        print(f"\n{result['final_answer'] or result.get('error', '(no answer)')}\n")
        print("─" * 70)


def bench(tasks, model, output):
    results, scores_all = [], []
    print(f"\nBenchmarking {len(tasks)} tasks — {model}\n")
    for i, task in enumerate(tasks, 1):
        tid    = task.get("TASK",   f"task_{i}")
        prompt = task.get("PROMPT", "")
        et     = task.get("ENABLED_TOOLS", "")
        print(f"[{i}/{len(tasks)}] {tid}\n  {prompt[:80]}")
        result = run_task(prompt, model)
        calls, answer = result["tool_calls"], result["final_answer"]
        sc = score_task(calls, et, answer)
        print(f"  → {len(calls)} call(s) | {result['time']}s | "
              f"recall={sc['discovery_recall']}")
        time.sleep(5)
        results.append({
            "TASK": tid, "PROMPT": prompt, "ENABLED_TOOLS": et,
            "plan": result["plan"], "script_model_response": answer,
            "trajectory": json.dumps(calls), "trajectory_time": result["time"],
            "agent_error": result["error"], **sc,
        })
        if sc["tool_score"] is not None:
            scores_all.append(sc["tool_score"])
    with open(output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    avg = round(sum(scores_all) / len(scores_all), 3) if scores_all else 0
    print(f"\n{'='*60}\n  Avg tool score: {avg*100:.1f}%\n"
          f"  Results: {output}\n{'='*60}")


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser(description="Planner Agent v5")
    p.add_argument("--model",     default=DEFAULT_MODEL)
    p.add_argument("--mode",      choices=["interactive", "bench"], default="interactive")
    p.add_argument("--input",     default="tasks.csv")
    p.add_argument("--output",    default=None)
    p.add_argument("--num-tasks", type=int, default=None)
    p.add_argument("--no-plan",   action="store_true", help="disable planning phase")
    args = p.parse_args()
    if args.mode == "interactive":
        interactive(args.model)
    else:
        tasks = load_csv(args.input)
        if args.num_tasks:
            tasks = tasks[:args.num_tasks]
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = (args.output or
               f"results_{args.model.replace('/','_').replace(':','_')}_{ts}.csv")
        bench(tasks, args.model, out)


if __name__ == "__main__":
    main()
