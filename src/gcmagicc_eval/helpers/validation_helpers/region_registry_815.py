"""Canonical region handling for 815 scenario exports."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Optional

import numpy as np

try:  # pragma: no cover - optional at import time
    import pycountry  # type: ignore
except Exception:  # pragma: no cover
    pycountry = None

try:  # pragma: no cover - optional at import time
    import regionmask  # type: ignore
except Exception:  # pragma: no cover
    regionmask = None


AR6_PREFIX = "AR6"
GLOBAL_REGION_ID = "global"

_SPACE_RE = re.compile(r"\s+")

_AR6_LABEL_OVERRIDES = {
    "Equatorial.Indic-Ocean": "Equatorial.Indian-Ocean",
    "S.Indic-Ocean": "S.Indian-Ocean",
}

_COUNTRY_CODE_ALIASES = {
    "DRC": "COD",
}

_COUNTRY_NAME_ALIASES = {
    "central african republic": "CAF",
    "democratic republic of the congo": "COD",
    "democratic republic of congo": "COD",
    "dr congo": "COD",
    "republic of the congo": "COG",
    "saudi arabia": "SAU",
    "south sudan": "SSD",
}

_COUNTRY_LABEL_OVERRIDES = {
    "CAF": "Central African Republic",
    "COD": "Democratic Republic of the Congo",
    "COG": "Republic of the Congo",
    "SAU": "Saudi Arabia",
    "SSD": "South Sudan",
}

_COUNTRY_MASK_NAME_ALIASES = {
    "COD": {"dem. rep. congo", "democratic republic of the congo"},
}


@dataclass(frozen=True)
class RegionSpec:
    canonical_id: str
    kind: str
    label: str
    storage_id: str
    mask_name: str | None = None
    ar6_abbrev: str | None = None


def _normalize_key(value: str) -> str:
    token = str(value or "").strip().replace("_", " ")
    token = _SPACE_RE.sub(" ", token)
    return token.casefold()


def _wrap_lon_to_180(lon: np.ndarray) -> np.ndarray:
    lon = np.asarray(lon, dtype=float)
    if np.nanmin(lon) >= 0.0 and np.nanmax(lon) > 180.0:
        return ((lon + 180.0) % 360.0) - 180.0
    return lon


def _get_country_regions():
    if regionmask is None:
        raise RuntimeError("regionmask is required for ISO3 masking")
    candidates = ["natural_earth_v5_0_0", "natural_earth_v4_1_0", "natural_earth"]
    for attr in candidates:
        regions = getattr(regionmask.defined_regions, attr, None)
        if regions is None:
            continue
        countries = getattr(regions, "countries_110", None)
        if countries is not None:
            return countries
    raise RuntimeError("No Natural Earth countries available via regionmask")


def _find_country_idx(countries, iso3: str) -> Optional[int]:
    iso3_u = str(iso3).upper()
    direct = [i for i, a in enumerate(countries.abbrevs) if str(a).upper() == iso3_u]
    if direct:
        return int(direct[0])
    if pycountry is None:
        return None
    country = pycountry.countries.get(alpha_3=iso3_u)
    if country is None:
        return None
    target_names = {str(country.name).lower()}
    if getattr(country, "official_name", None):
        target_names.add(str(country.official_name).lower())
    by_name = [i for i, name in enumerate(countries.names) if str(name).lower() in target_names]
    if by_name:
        return int(by_name[0])
    a2 = getattr(country, "alpha_2", None)
    if a2:
        by_a2 = [i for i, a in enumerate(countries.abbrevs) if str(a).upper() == str(a2).upper()]
        if by_a2:
            return int(by_a2[0])
    alias_names = _COUNTRY_MASK_NAME_ALIASES.get(iso3_u, set())
    if alias_names:
        by_alias = [i for i, name in enumerate(countries.names) if str(name).lower() in alias_names]
        if by_alias:
            return int(by_alias[0])
    return None


def _iso3_from_country_entry(name: str, abbrev: str) -> Optional[str]:
    code = str(abbrev).upper()
    if len(code) == 3 and code.isalpha():
        return _COUNTRY_CODE_ALIASES.get(code, code)
    if pycountry is None:
        return None
    country = None
    if len(str(abbrev)) == 2:
        country = pycountry.countries.get(alpha_2=str(abbrev).upper())
    if country is None:
        try:
            country = pycountry.countries.lookup(name)
        except Exception:
            country = None
    if country is not None:
        return _COUNTRY_CODE_ALIASES.get(str(country.alpha_3).upper(), str(country.alpha_3).upper())
    return None


def _country_label(iso3: str) -> str:
    code = str(iso3).upper()
    if code in _COUNTRY_LABEL_OVERRIDES:
        return _COUNTRY_LABEL_OVERRIDES[code]
    if pycountry is not None:
        country = pycountry.countries.get(alpha_3=code)
        if country is not None:
            for attr in ("common_name", "name", "official_name"):
                value = getattr(country, attr, None)
                if value:
                    return str(value)
    return code


def _pycountry_iso3(token: str) -> Optional[str]:
    if pycountry is None:
        return None
    code = str(token).strip().upper()
    country = pycountry.countries.get(alpha_3=code)
    if country is not None:
        return str(country.alpha_3).upper()
    try:
        looked_up = pycountry.countries.lookup(str(token).strip())
    except Exception:
        return None
    return str(looked_up.alpha_3).upper()


def _iso3_candidate(token: str) -> Optional[str]:
    stripped = str(token or "").strip()
    if not stripped:
        return None
    upper = stripped.upper()
    mapped = _COUNTRY_CODE_ALIASES.get(upper)
    if mapped:
        return mapped
    mapped = _COUNTRY_NAME_ALIASES.get(_normalize_key(stripped))
    if mapped:
        return mapped
    code = _pycountry_iso3(stripped)
    if code:
        return _COUNTRY_CODE_ALIASES.get(code, code)
    try:
        countries = _get_country_regions()
    except Exception:
        return None
    idx = _find_country_idx(countries, upper)
    if idx is None:
        return None
    return _iso3_from_country_entry(str(countries.names[idx]), str(countries.abbrevs[idx]))


def _build_ar6_registry() -> tuple[Dict[str, RegionSpec], Dict[str, str]]:
    if regionmask is None:
        raise RuntimeError("regionmask is required for AR6 masking")

    ar6 = regionmask.defined_regions.ar6.all
    by_id: Dict[str, RegionSpec] = {}
    aliases: Dict[str, str] = {}

    ambiguous_abbrevs: set[str] = set()
    for raw_abbrev in getattr(ar6, "abbrevs", []):
        abbrev = str(raw_abbrev).upper()
        if _iso3_candidate(abbrev):
            ambiguous_abbrevs.add(abbrev)

    for idx, raw_name in enumerate(ar6.names):
        mask_name = str(raw_name)
        abbrev = str(ar6.abbrevs[idx]).upper()
        canonical_id = f"{AR6_PREFIX}{abbrev}"
        label = _AR6_LABEL_OVERRIDES.get(mask_name, mask_name)
        spec = RegionSpec(
            canonical_id=canonical_id,
            kind="ar6",
            label=label,
            storage_id=canonical_id,
            mask_name=mask_name,
            ar6_abbrev=abbrev,
        )
        by_id[canonical_id] = spec
        for alias in {canonical_id, label, mask_name}:
            aliases[_normalize_key(alias)] = canonical_id
        if abbrev not in ambiguous_abbrevs:
            aliases[_normalize_key(abbrev)] = canonical_id

    return by_id, aliases


def _ar6_registry() -> tuple[Dict[str, RegionSpec], Dict[str, str]]:
    if not hasattr(_ar6_registry, "_cache"):
        setattr(_ar6_registry, "_cache", _build_ar6_registry())
    return getattr(_ar6_registry, "_cache")


def iter_ar6_specs() -> List[RegionSpec]:
    by_id, _ = _ar6_registry()
    return list(by_id.values())


def is_ar6_region_id(region: str) -> bool:
    token = str(region or "").strip().upper()
    by_id, _ = _ar6_registry()
    return token in by_id


def is_country_region_id(region: str) -> bool:
    token = str(region or "").strip().upper()
    return bool(token and not is_ar6_region_id(token) and _iso3_candidate(token) == token)


def canonical_region_id(region: str) -> str:
    raw = str(region or "").strip()
    if not raw:
        raise ValueError("Region token is empty")
    if raw.lower() == GLOBAL_REGION_ID:
        return GLOBAL_REGION_ID

    by_id, aliases = _ar6_registry()
    if raw.upper() in by_id:
        return raw.upper()

    ar6_match = aliases.get(_normalize_key(raw))
    if ar6_match:
        return ar6_match

    iso3 = _iso3_candidate(raw)
    if iso3:
        return iso3

    raise RuntimeError(f"Region '{region}' not recognized as canonical ISO3 or AR6")


def region_label(region: str) -> str:
    canonical = canonical_region_id(region)
    if canonical == GLOBAL_REGION_ID:
        return "Global"
    by_id, _ = _ar6_registry()
    spec = by_id.get(canonical)
    if spec is not None:
        return spec.label
    return _country_label(canonical)


def build_region_mask(region: str, lats, lons, min_pixels: int) -> np.ndarray:
    canonical = canonical_region_id(region)
    if canonical == GLOBAL_REGION_ID:
        return np.ones((len(lats), len(lons)), dtype=bool)

    if regionmask is None:
        raise RuntimeError("regionmask is required to build region masks")

    lon_wrapped = _wrap_lon_to_180(np.asarray(lons, dtype=float))
    by_id, _ = _ar6_registry()
    spec = by_id.get(canonical)
    if spec is not None:
        ar6 = regionmask.defined_regions.ar6.all
        matches = [i for i, ab in enumerate(ar6.abbrevs) if str(ab).upper() == str(spec.ar6_abbrev).upper()]
        if not matches:
            raise RuntimeError(f"AR6 region '{canonical}' is not available via regionmask")
        mask_da = ar6.mask(lon_wrapped, np.asarray(lats, dtype=float))
        mask = (mask_da == int(matches[0])).fillna(False).values
    else:
        countries = _get_country_regions()
        ridx = _find_country_idx(countries, canonical)
        if ridx is None:
            raise RuntimeError(f"ISO3 region '{canonical}' not found in Natural Earth countries")
        mask_da = countries.mask(lon_wrapped, np.asarray(lats, dtype=float))
        mask = (mask_da == int(ridx)).fillna(False).values

    if int(np.sum(mask)) < int(min_pixels):
        raise RuntimeError(f"Region '{canonical}' covers only {int(np.sum(mask))} grid cells (<{int(min_pixels)}).")
    return np.asarray(mask, dtype=bool)


def discover_regions(lats, lons, min_pixels: int) -> List[str]:
    regions: List[str] = [GLOBAL_REGION_ID]
    regions.extend(spec.canonical_id for spec in iter_ar6_specs())

    countries = _get_country_regions()
    lon_wrapped = _wrap_lon_to_180(np.asarray(lons, dtype=float))
    mask3d = countries.mask_3D(lon_wrapped, np.asarray(lats, dtype=float))
    region_coord = mask3d.coords.get("region")
    region_ids = np.asarray(region_coord.values if region_coord is not None else np.arange(mask3d.sizes.get("region", 0)))

    keep: List[str] = []
    for pos, rid in enumerate(region_ids):
        ridx = int(rid)
        if ridx < 0 or ridx >= len(countries.abbrevs):
            continue
        mask = mask3d.isel(region=pos).fillna(False).values
        if int(np.sum(mask)) < int(min_pixels):
            continue
        iso3 = _iso3_from_country_entry(str(countries.names[ridx]), str(countries.abbrevs[ridx]))
        if iso3:
            keep.append(iso3)

    regions.extend(sorted(dict.fromkeys(keep)))
    return regions
