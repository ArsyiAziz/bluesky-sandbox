"""Emit a design as readable Python source instead of a JSON document.

The structured half of a design is normally a spec dict. The helpers here turn
that dict into source fragments for generated task modules, so a generated task
reads as Python code rather than a serialized blob.
"""

from __future__ import annotations

from typing import Any

from bluesky_sandbox.interface.fields import queryables as _queryable_fields

from .spec import (
    DesignSpec,
    EnvSpec,
    FieldRef,
    extract_waypoint_field_dists,
    is_envelope_value,
    is_value_distribution,
    representative_value,
)

_FOOTPRINT_TYPES = {"box", "disk", "polygon", "sector", "annular_sector", "boolean"}


def _normalizer_import_line() -> str:
    """Emit the normalizer import, deriving the names by introspection.

    Every concrete ``Normalizer`` subclass is included, so a newly added
    normalizer is importable in generated code without editing this module.
    """
    import inspect

    from bluesky_sandbox.interface.wrappers.observations import normalizer as _norm

    names = sorted(
        name
        for name, obj in inspect.getmembers(_norm, inspect.isclass)
        if issubclass(obj, _norm.Normalizer)
        and obj is not _norm.Normalizer
        and obj.__module__ == _norm.__name__
        and not inspect.isabstract(obj)
    )
    return (
        "from bluesky_sandbox.interface.wrappers.observations.normalizer import (\n"
        f"    {', '.join(names)},\n)"
    )


class _Emitter:
    def __init__(
        self,
        package: str | None = None,
        regions: dict[str, Any] | None = None,
    ) -> None:
        self.scipy_names: set[str] = set()
        self.custom_imports: set[str] = set()  # bare module names referenced
        self.uses_envelope_sample = False
        self.package = package  # package-qualify custom imports when generating
        self.regions = regions or {}  # named bounds, for resolving {"ref": name}
        # Sampled footprint params encountered while emitting named regions:
        # {"<region>.<param path>": raw value dict}. The scenario template
        # emits these as _REGION_PARAM_DISTS and the region expressions
        # reference them as draw['<region>.<param path>'].
        self.sampled_region_params: dict[str, Any] = {}
        self._current_region: str | None = None

    # ---- scalar-or-distribution values ---------------------------------- #
    def value(self, v: Any) -> str:
        if isinstance(v, dict) and "type" in v:
            t = v["type"]
            if t == "range":
                return f"({self.num(v['low'])}, {self.num(v['high'])})"
            if t == "scipy":
                self.scipy_names.add(v["name"])
                parts = [self.num(a) for a in v.get("args", [])]
                parts += [f"{k}={self.num(w)}" for k, w in v.get("kwds", {}).items()]
                dist = f"{v['name']}({', '.join(parts)})"
                if "bounds" in v:
                    lo, hi = v["bounds"]
                    mode = v.get("mode", "truncate")
                    return f"Bounded({dist}, {self.num(lo)}, {self.num(hi)}, mode={mode!r})"
                return dist
            if t == "categorical":
                return f"Categorical({v['weights']!r})"
            if t == "envelope":
                self.uses_envelope_sample = True
                floor = v.get("alt_floor_ft")
                if floor is None:
                    return "EnvelopeSample()"
                return f"EnvelopeSample(alt_floor_ft={self.num(floor)})"
        if isinstance(v, dict):
            items = []
            for key, value in v.items():
                if key == "sample":
                    rendered = self.bounds_or_ref(value)
                elif is_envelope_value(value):
                    rendered = repr("envelope")
                else:
                    rendered = self.value(value)
                items.append(f"{key!r}: {rendered}")
            return "{" + ", ".join(items) + "}"
        if isinstance(v, list):
            return "[" + ", ".join(self.value(x) for x in v) + "]"
        if isinstance(v, str):
            return repr(v)
        if isinstance(v, bool) or v is None:
            return repr(v)
        return self.num(v)

    @staticmethod
    def num(x: Any) -> str:
        if isinstance(x, str):
            low = x.strip().lower()
            if low in ("inf", "+inf"):
                return "float('inf')"
            if low == "-inf":
                return "float('-inf')"
            return repr(x)
        return repr(x)

    # ---- geometry -------------------------------------------------------- #
    def latlon(self, d: Any) -> str:
        if isinstance(d, dict):
            return f"LatLon({self.num(d['lat_deg'])}, {self.num(d['lon_deg'])})"
        return f"LatLon({self.num(d[0])}, {self.num(d[1])})"

    def fparam(self, d: dict[str, Any], key: str, path: str) -> str:
        """Emit a footprint scalar param: literal, or a draw[...] lookup when sampled."""
        v = d[key]
        if not is_value_distribution(v):
            return self.num(v)
        if self._current_region is None:
            raise ValueError(
                f"sampled footprint param {key!r} is only supported on named "
                "regions (inline bounds cannot resample per episode)"
            )
        full = f"{self._current_region}.{path}{key}"
        self.sampled_region_params[full] = v
        return f"draw[{full!r}]"

    def footprint(self, d: dict[str, Any], path: str = "") -> str:
        t = d["type"]
        if t == "box":
            return (
                f"BoxFootprint({self.fparam(d, 'lat_min_deg', path)}, {self.fparam(d, 'lat_max_deg', path)}, "
                f"{self.fparam(d, 'lon_min_deg', path)}, {self.fparam(d, 'lon_max_deg', path)})"
            )
        if t == "disk":
            return f"DiskFootprint({self.latlon(d['center'])}, radius_nm={self.fparam(d, 'radius_nm', path)}, n_vertices={d.get('n_vertices', 72)})"
        if t == "polygon":
            coords = ", ".join(f"({self.num(a)}, {self.num(b)})" for a, b in d["coords"])
            return f"PolygonFootprint([{coords}])"
        if t == "sector":
            return (
                f"SectorFootprint({self.latlon(d['center'])}, radius_nm={self.fparam(d, 'radius_nm', path)}, "
                f"bearing_deg={self.fparam(d, 'bearing_deg', path)}, half_angle_deg={self.fparam(d, 'half_angle_deg', path)})"
            )
        if t == "annular_sector":
            return (
                f"AnnularSectorFootprint({self.latlon(d['center'])}, inner_radius_nm={self.fparam(d, 'inner_radius_nm', path)}, "
                f"outer_radius_nm={self.fparam(d, 'outer_radius_nm', path)}, bearing_deg={self.fparam(d, 'bearing_deg', path)}, "
                f"half_angle_deg={self.fparam(d, 'half_angle_deg', path)})"
            )
        if t == "boolean":
            ops = {"union": "|", "intersection": "&", "difference": "-"}
            return (
                f"({self.footprint(d['left'], path + 'left.')} {ops[d['op']]} "
                f"{self.footprint(d['right'], path + 'right.')})"
            )
        raise ValueError(f"cannot emit footprint {t!r}")

    def named_bounds(self, name: str, d: dict[str, Any]) -> str:
        """Emit a named region's bounds, allowing its params to be sampled."""
        self._current_region = name
        try:
            return self.bounds(d)
        finally:
            self._current_region = None

    def band(self, d: Any) -> str:
        if d is None:
            return "None"
        t = d["type"]
        if t == "constant":
            return f"ConstantAltitudeBand({self.num(d['min_ft'])}, {self.num(d['max_ft'])})"
        if t == "linear":
            return (
                f"LinearAltitudeBand({self.latlon(d['start'])}, {self.latlon(d['end'])}, "
                f"({self.num(d['start_band_ft'][0])}, {self.num(d['start_band_ft'][1])}), "
                f"({self.num(d['end_band_ft'][0])}, {self.num(d['end_band_ft'][1])}))"
            )
        if t == "radial":
            return (
                f"RadialAltitudeBand({self.latlon(d['center'])}, radius_nm={self.num(d['radius_nm'])}, "
                f"inner_band_ft=({self.num(d['inner_band_ft'][0])}, {self.num(d['inner_band_ft'][1])}), "
                f"outer_band_ft=({self.num(d['outer_band_ft'][0])}, {self.num(d['outer_band_ft'][1])}))"
            )
        if t == "vertex":
            verts = ", ".join(f"({self.num(a)}, {self.num(b)})" for a, b in d["vertices"])
            return f"VertexAltitudeBand([{verts}], {d['min_values_ft']!r}, {d['max_values_ft']!r})"
        raise ValueError(f"cannot emit altitude band {t!r}")

    def bounds(self, d: dict[str, Any]) -> str:
        # A statically-rotated bounds is baked to a rotated polygon for codegen
        # (reusing the same rotation as the runtime), so the emitted Python needs
        # no special rotation primitive.
        if d.get("rotation_deg"):
            from . import spec as _spec

            d = _spec.dump(_spec.load(d))
        return f"RegionBounds({self.footprint(d['footprint'])}, {self.band(d.get('altitude'))})"

    def bounds_or_ref(self, d: dict[str, Any]) -> str:
        """Emit a bounds, a ``REGIONS[name]`` reference, or a bare footprint."""
        if isinstance(d, dict) and set(d) == {"ref"}:
            return f"REGIONS[{d['ref']!r}]"
        if isinstance(d, dict) and d.get("type") in _FOOTPRINT_TYPES:
            return f"RegionBounds({self.footprint(d)}, None)"
        return self.bounds(d)

    def queryable(self, d: dict[str, Any]) -> str:
        if d["type"] == "query_region":
            return (
                f"QueryRegion({self.bounds_or_ref(d['bounds'])}, color={d.get('color', 'orange')!r}, "
                f"render_shape={d.get('render_shape', True)}, render_label={d.get('render_label', True)}, "
                f"track_temporal_state={d.get('track_temporal_state', False)})"
            )
        # waypoint
        args = []
        if d.get("waypoint") is not None:
            args.append(f"waypoint={d['waypoint']!r}")
        else:
            lat, lon = d.get("lat"), d.get("lon")
            # A sampled waypoint's static/support position is its region centre;
            # derive it when the dict didn't carry an explicit lat/lon. The
            # sample may be a footprint, a bounds, or a {"ref": name}.
            sample = d.get("sample")
            if (lat is None or lon is None) and sample:
                from . import spec as _spec

                region = self.regions[sample["ref"]] if "ref" in sample else sample
                (lat_min, lat_max), (lon_min, lon_max) = _spec.load(region).bounding_box
                lat = (lat_min + lat_max) / 2.0
                lon = (lon_min + lon_max) / 2.0
            args.append(f"lat={self.num(lat)}, lon={self.num(lon)}")
        # Numeric constraint/target fields may be distribution-valued; emit the
        # support scalar here (the per-episode draw lives in WAYPOINT_FIELDS).
        for k in (
            "alt_ft",
            "speed_kts",
            "alt_tolerance_ft",
            "speed_tolerance_kts",
            "speed_tolerance_mach",
        ):
            val = d.get(k)
            if val is None:
                continue
            if k in ("alt_ft", "speed_kts") and is_envelope_value(val):
                continue
            args.append(f"{k}={self.num(representative_value(val))}")
        if d.get("tsas_region") is not None:
            args.append(f"tsas_region={d['tsas_region']!r}")
        if "reach_radius_nm" in d:
            args.append(f"reach_radius_nm={self.num(representative_value(d['reach_radius_nm']))}")
        args.append(f"color={d.get('color', 'cyan')!r}")
        # Only when non-default, so this leaves every design that never touched
        # them byte-identical. Emitted at all because a waypoint the designer
        # marked invisible - e.g. one of several per-aircraft sampling
        # placeholders parked on the same point - was otherwise drawn anyway.
        for k in ("render_shape", "render_label"):
            if d.get(k) is False:
                args.append(f"{k}=False")
        args.append(f"track_temporal_state={d.get('track_temporal_state', False)!r}")
        return f"Waypoint({', '.join(args)})"

    def spawn_region(self, d: dict[str, Any]) -> str:
        params = ", ".join(f"{k!r}: {self.value(v)}" for k, v in d.get("params", {}).items())
        lines = [
            f"        bounds={self.bounds_or_ref(d['bounds'])},",
            f"        n_aircraft={self.value(d['n_aircraft'])},",
            f"        params={{{params}}},",
        ]
        if d.get("aircraft_type") is not None:
            lines.append(f"        aircraft_type={self.value(d['aircraft_type'])},")
        if d.get("callsign_prefixes") is not None:
            lines.append(f"        callsign_prefixes={self.value(d['callsign_prefixes'])},")
        if d.get("spawn_time"):
            lines.append(f"        spawn_time={self.value(d['spawn_time'])},")
        if d.get("route") is not None:
            lines.append(f"        route={self.value(d['route'])},")
        if d.get("name"):
            lines.append(f"        name={d['name']!r},")
        if d.get("maintain"):
            lines.append(f"        maintain={bool(d['maintain'])!r},")
        if d.get("controlled") is False:
            lines.append("        controlled=False,")
        if d.get("conflict_free_spawn") is not None:
            lines.append(
                f"        conflict_free_spawn={bool(d['conflict_free_spawn'])!r},"
            )
        if d.get("conflict_free_margin_nm") is not None:
            lines.append(
                f"        conflict_free_margin_nm={float(d['conflict_free_margin_nm'])!r},"
            )
        if d.get("conflict_free_margin_ft") is not None:
            lines.append(
                f"        conflict_free_margin_ft={float(d['conflict_free_margin_ft'])!r},"
            )
        if d.get("conflict_free_margin_s") is not None:
            lines.append(
                f"        conflict_free_margin_s={float(d['conflict_free_margin_s'])!r},"
            )
        body = "\n".join(lines)
        return f"SpawnRegion(\n{body}\n    )"

    def spawn(self, d: dict[str, Any]) -> str:
        regions_list = d.get("regions", [])
        if regions_list:
            regions = "[\n    " + ",\n    ".join(self.spawn_region(r) for r in regions_list) + ",\n]"
        else:
            regions = "[]"
        extra = ""
        if d.get("route") is not None:
            extra += f", route={self.value(d['route'])}"
        if d.get("routes"):
            # Per-route ``self.value(...)`` - not ``repr()`` - so each step's
            # ``"sample": {"ref": name}`` pointer resolves to the actual
            # ``REGIONS[name]`` object, exactly like the inline ``route=``
            # path above (line ~253). A bare repr() emits the raw ref dict as
            # a literal, which fails at runtime with "'dict' object has no
            # attribute 'sample_point'" the first time an episode samples it.
            routes_body = ", ".join(
                f"{name!r}: {self.value(route)}"
                for name, route in d["routes"].items()
            )
            extra += f", routes={{{routes_body}}}"
        if d.get("conflict_free_spawn"):
            extra += ", conflict_free_spawn=True"
        if d.get("conflict_free_margin_nm"):
            extra += f", conflict_free_margin_nm={float(d['conflict_free_margin_nm'])!r}"
        if d.get("conflict_free_margin_ft"):
            extra += f", conflict_free_margin_ft={float(d['conflict_free_margin_ft'])!r}"
        if d.get("conflict_free_margin_s"):
            extra += f", conflict_free_margin_s={float(d['conflict_free_margin_s'])!r}"
        return f"SpawnConfig(regions={regions}{extra})"

    # ---- fields ---------------------------------------------------------- #
    def field(self, ref: FieldRef, module_alias: str) -> str:
        name = ref.name
        if ":" in name:  # custom field by import path
            mod, cls = name.split(":", 1)
            self.custom_imports.add(mod)
            base = f"{mod}.{cls}"
        else:
            alias = (
                "qobs"
                if module_alias == "obs" and hasattr(_queryable_fields, name)
                else module_alias
            )
            base = f"{alias}.{name}"
        kwargs = ", ".join(f"{k}={self.field_kwarg(v)}" for k, v in ref.kwargs.items())
        expr = f"{base}({kwargs})"
        if ref.transform:
            tkw = ", ".join(f"{k}={self.field_kwarg(v)}" for k, v in ref.transform_kwargs.items())
            expr += f".{ref.transform}({tkw})"
        return expr

    def field_kwarg(self, value: Any) -> str:
        if isinstance(value, dict) and value.get("type") == "normalizer":
            kwargs = ", ".join(f"{k}={v!r}" for k, v in dict(value.get("kwargs", {})).items())
            return f"{value['name']}({kwargs})"
        return repr(value)

    def field_list(self, refs: list[FieldRef], module_alias: str) -> str:
        if not refs:
            return "[]"
        items = ", ".join(self.field(r, module_alias) for r in refs)
        return f"[{items}]"

    def field_tuple(self, refs: list[FieldRef], module_alias: str) -> str:
        if not refs:
            return "()"
        items = "\n".join(f"            {self.field(r, module_alias)}," for r in refs)
        return f"(\n{items}\n        )"


def emit_field_sources(env: EnvSpec, package: str | None = None) -> dict[str, str]:
    """Return imports and inline field/action expressions for a generated env.py."""
    em = _Emitter(package=package)
    obs = em.field_tuple(env.obs_fields, "obs")
    intr = (
        "None"
        if env.intruder_obs_fields is None
        else em.field_tuple(env.intruder_obs_fields, "obs")
    )
    critic_obs = (
        "None"
        if env.critic_obs_fields is None
        else em.field_tuple(env.critic_obs_fields, "obs")
    )
    critic_intr = (
        "None"
        if env.critic_intruder_obs_fields is None
        else em.field_tuple(env.critic_intruder_obs_fields, "obs")
    )
    actions = em.field_tuple(env.action_fields, "act")
    scipy_import = (
        f"from scipy.stats import {', '.join(sorted(em.scipy_names))}\n"
        if em.scipy_names
        else ""
    )
    if em.custom_imports:
        if package:
            custom_import = "".join(
                f"import {package}.{m} as {m}\n" for m in sorted(em.custom_imports)
            )
        else:
            custom_import = "".join(f"import {m}\n" for m in sorted(em.custom_imports))
    else:
        custom_import = ""
    imports = f'''{scipy_import}from bluesky_sandbox.interface.fields import actions as act
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.interface.fields import queryables as qobs
{_normalizer_import_line()}
{custom_import}'''
    return {
        "imports": imports.rstrip(),
        "obs_fields": obs,
        "intruder_obs_fields": intr,
        "critic_obs_fields": critic_obs,
        "critic_intruder_obs_fields": critic_intr,
        "action_fields": actions,
    }


def emit_env_sources(spec: DesignSpec, package: str | None = None) -> dict[str, str]:
    """Return imports, env constants, and field expressions for a generated env.py."""
    em = _Emitter(package=package)
    env: EnvSpec = spec.env

    obs = em.field_tuple(env.obs_fields, "obs")
    intr = (
        "None"
        if env.intruder_obs_fields is None
        else em.field_tuple(env.intruder_obs_fields, "obs")
    )
    critic_obs = (
        "None"
        if env.critic_obs_fields is None
        else em.field_tuple(env.critic_obs_fields, "obs")
    )
    critic_intr = (
        "None"
        if env.critic_intruder_obs_fields is None
        else em.field_tuple(env.critic_intruder_obs_fields, "obs")
    )
    actions = em.field_tuple(env.action_fields, "act")

    scipy_import = (
        f"from scipy.stats import {', '.join(sorted(em.scipy_names))}\n"
        if em.scipy_names
        else ""
    )
    if em.custom_imports:
        if package:
            custom_import = "".join(
                f"import {package}.{m} as {m}\n" for m in sorted(em.custom_imports)
            )
        else:
            custom_import = "".join(f"import {m}\n" for m in sorted(em.custom_imports))
    else:
        custom_import = ""

    imports = f'''{scipy_import}from bluesky_sandbox.interface.fields import actions as act
from bluesky_sandbox.interface.fields import observations as obs
from bluesky_sandbox.interface.fields import queryables as qobs
{_normalizer_import_line()}
{custom_import}'''
    return {
        "imports": imports.rstrip(),
        "allowed_aircraft": repr(env.allowed_aircraft),
        "dt": repr(env.dt),
        "simdt": repr(env.simdt),
        "cd_method": repr(env.cd_method),
        "reso_method": repr(env.reso_method),
        "pz_radius_nm": repr(env.pz_radius_nm),
        "pz_height_ft": repr(env.pz_height_ft),
        "lookahead_s": repr(env.lookahead_s),
        "performance_model": repr(env.performance_model),
        "wind_dir_deg": repr(env.wind_dir_deg),
        "wind_kts": repr(env.wind_kts),
        "turbulence_kts": repr(env.turbulence_kts),
        "gust_tau_s": repr(env.gust_tau_s),
        "obs_fields": obs,
        "intruder_obs_fields": intr,
        "critic_obs_fields": critic_obs,
        "critic_intruder_obs_fields": critic_intr,
        "action_fields": actions,
    }


def _emit_sampled_waypoints(em: _Emitter, spec: DesignSpec) -> str:
    """Emit the ``{name: footprint}`` body for sampled-position waypoints."""
    return ",\n    ".join(
        f"{name!r}: {em.bounds_or_ref(q['sample'])}"
        for name, q in spec.queryables.items()
        if isinstance(q, dict)
        and q.get("type") == "waypoint"
        and q.get("sample")
        and q.get("sample_per") != "aircraft"
    )


def _emit_waypoint_fields(em: _Emitter, spec: DesignSpec) -> str:
    """Emit ``{name: {field: value|dist}}`` for per-episode waypoint constraints."""
    rows = []
    for name, q in spec.queryables.items():
        if not (isinstance(q, dict) and q.get("type") == "waypoint"):
            continue
        _, dists = extract_waypoint_field_dists(q)
        if not dists:
            continue
        inner = ", ".join(f"{f!r}: {em.value(v)}" for f, v in dists.items())
        rows.append(f"{name!r}: {{{inner}}}")
    return ",\n    ".join(rows)


def _route_sampling_metadata(spec: DesignSpec) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, q in spec.queryables.items():
        if not (isinstance(q, dict) and q.get("type") == "waypoint"):
            continue
        if q.get("sample") is not None and q.get("sample_per") == "aircraft":
            out.setdefault(name, {})["sample"] = q["sample"]
        if is_envelope_value(q.get("alt_ft")):
            out.setdefault(name, {})["sample_alt_from_envelope"] = True
        if is_envelope_value(q.get("speed_kts")):
            out.setdefault(name, {})["sample_speed_from_envelope"] = True
        if "envelope_alt_floor_ft" in q and name in out:
            out[name]["envelope_alt_floor_ft"] = q["envelope_alt_floor_ft"]
        if q.get("reachable_from_spawn") and out.get(name, {}).get(
            "sample_alt_from_envelope"
        ):
            out[name]["reachable_from_spawn"] = True
            if q.get("reachable_vs_fraction") is not None:
                out[name]["reachable_vs_fraction"] = q["reachable_vs_fraction"]
    return out


def _spawn_with_route_sampling(spec: DesignSpec) -> dict[str, Any]:
    route_sampling = _route_sampling_metadata(spec)
    if not route_sampling:
        return spec.spawn

    def step_with_sampling(step):
        if isinstance(step, str):
            metadata = route_sampling.get(step)
            return {"waypoint": step, **metadata} if metadata else step
        if not isinstance(step, dict):
            return step
        if "waypoint" in step:
            metadata = route_sampling.get(step["waypoint"])
            return {**step, **metadata} if metadata else step
        if "choice" in step:
            return {
                **step,
                "choice": [
                    [step_with_sampling(s) for s in branch]
                    if isinstance(branch, list)
                    else step_with_sampling(branch)
                    for branch in step["choice"]
                ],
            }
        return step

    def route_with_sampling(route):
        if not isinstance(route, list):
            return route
        return [step_with_sampling(step) for step in route]

    spawn = dict(spec.spawn)
    spawn["route"] = route_with_sampling(spawn.get("route"))
    spawn["routes"] = {
        name: route_with_sampling(route)
        for name, route in spawn.get("routes", {}).items()
    }
    spawn["regions"] = [
        {**region, "route": route_with_sampling(region.get("route"))}
        if isinstance(region, dict)
        else region
        for region in spawn.get("regions", [])
    ]
    return spawn


def _emit_group(em: _Emitter, spec: DesignSpec, g: dict[str, Any]) -> str:
    """Emit one transform group as a runtime dict literal (matching the keys
    :func:`builder._parse_groups` produces: ``angle``/``translation``/``scale``).

    ``members`` are bounds names in the spec; they're expanded to the element ids
    the runtime transforms (matching :func:`builder._parse_groups`).
    """
    from .builder import expand_region_members

    angle_expr = em.value(g.get("angle_deg", 0.0))
    scale_expr = em.value(g.get("scale", 1.0))
    pivot = tuple(g["pivot"]) if g.get("pivot") else None
    members = expand_region_members(spec, list(g.get("members", [])))
    t = g.get("translation") or {}
    if t.get("east_nm") or t.get("north_nm"):
        translation = (
            f'{{"east": {em.value(t.get("east_nm", 0.0))}, '
            f'"north": {em.value(t.get("north_nm", 0.0))}}}'
        )
    else:
        translation = "None"
    return (
        f'{{"id": {g["id"]!r}, "angle": {angle_expr}, '
        f'"translation": {translation}, "scale": {scale_expr}, '
        f'"pivot": {pivot!r}, "members": {members!r}, "parent": {g.get("parent")!r}}}'
    )


def _emit_transform(em: _Emitter, spec: DesignSpec) -> str:
    """Emit the per-episode ``TRANSFORM`` (rotation groups, or legacy rotation)."""
    if spec.transform and spec.transform.get("groups"):
        groups = ", ".join(_emit_group(em, spec, g) for g in spec.transform["groups"])
        return f'{{"groups": [{groups}]}}'
    if spec.transform and spec.transform.get("rotation"):
        rot = spec.transform["rotation"]
        angle_expr = em.value(rot.get("angle_deg", 0.0))
        return f'{{"rotation": {{"angle_deg": {angle_expr}, "pivot": {rot.get("pivot")!r}}}}}'
    return "None"


def emit_scenario_sources(spec: DesignSpec) -> dict[str, str]:
    """Return imports and scenario expressions for a generated scenario.py."""
    em = _Emitter(regions=spec.regions)
    regions = ",\n    ".join(
        f"{name!r}: {em.named_bounds(name, b)}" for name, b in (spec.regions or {}).items()
    )
    airspace = em.bounds_or_ref(spec.airspace) if spec.airspace else "None"
    queryables = ",\n    ".join(
        f"{name!r}: {em.queryable(q)}" for name, q in spec.queryables.items()
    )
    sampled_waypoints = _emit_sampled_waypoints(em, spec)
    waypoint_fields = _emit_waypoint_fields(em, spec)
    spawn = em.spawn(_spawn_with_route_sampling(spec))

    transform = _emit_transform(em, spec)
    region_param_dists = ",\n        ".join(
        f"{key!r}: {em.value(v)}" for key, v in em.sampled_region_params.items()
    )

    scipy_import = (
        f"from scipy.stats import {', '.join(sorted(em.scipy_names))}\n"
        if em.scipy_names
        else ""
    )
    envelope_import = (
        "from bluesky_sandbox.sim.performance.envelope import EnvelopeSample\n"
        if em.uses_envelope_sample
        else ""
    )
    imports = f'''{scipy_import}from bluesky_sandbox.sim.bounds import (
    AnnularSectorFootprint, BooleanFootprint, BoxFootprint, ConstantAltitudeBand,
    DiskFootprint, LatLon, LinearAltitudeBand, PolygonFootprint, RadialAltitudeBand,
    RegionBounds, SectorFootprint, VertexAltitudeBand,
)
from bluesky_sandbox.sim.sampling.distributions import Bounded, Categorical
{envelope_import.rstrip()}
from bluesky_sandbox.sim.queryables import QueryRegion, Waypoint
from bluesky_sandbox.sim.spawn import SpawnConfig, SpawnRegion'''
    return {
        "imports": imports.rstrip(),
        "regions": "{\n            " + regions + "\n        }",
        "airspace": airspace,
        "queryables": "{\n            " + queryables + "\n        }",
        "sampled_waypoints": "{\n            " + sampled_waypoints + "\n        }",
        "waypoint_fields": "{\n            " + waypoint_fields + "\n        }",
        "spawn": spawn,
        "transform": transform,
        # Non-empty iff any named region has sampled footprint params; the
        # scenario template switches to the parametric-regions form then.
        "region_param_dists": (
            "{\n        " + region_param_dists + ",\n    }" if region_param_dists else ""
        ),
        # Non-empty iff the design declares waypoint stacks (shared merge
        # points). Also switches the scenario template to the parametric form,
        # since stacks need the per-episode geometry hook.
    }


def emit_design(spec: DesignSpec, package: str | None = None) -> str:
    """Return a legacy ``design.py`` source string constructing primitives.

    Defines module-level ``AIRSPACE``, ``QUERYABLES``, ``SPAWN``, ``TRANSFORM``,
    and the scalar env settings.
    """
    em = _Emitter(package=package, regions=spec.regions)
    env: EnvSpec = spec.env

    regions = ",\n    ".join(
        f"{name!r}: {em.bounds(b)}" for name, b in (spec.regions or {}).items()
    )
    airspace = em.bounds_or_ref(spec.airspace) if spec.airspace else "None"
    queryables = ",\n    ".join(
        f"{name!r}: {em.queryable(q)}" for name, q in spec.queryables.items()
    )
    sampled_waypoints = _emit_sampled_waypoints(em, spec)
    _emit_waypoint_fields(em, spec)
    spawn = em.spawn(_spawn_with_route_sampling(spec))

    transform = _emit_transform(em, spec)

    scipy_import = (
        f"from scipy.stats import {', '.join(sorted(em.scipy_names))}\n"
        if em.scipy_names
        else ""
    )
    envelope_import = (
        "from bluesky_sandbox.sim.performance.envelope import EnvelopeSample\n"
        if em.uses_envelope_sample
        else ""
    )

    return f'''"""Design definition - constructed in code (generated by the Environment Designer).

Edit the geometry/spawn here or back in the designer; ``scenario.py`` builds the
runtime Scenario and ``env.py`` builds the EnvConfig from these objects.
"""

from __future__ import annotations

{scipy_import}from bluesky_sandbox.sim.bounds import (
    AnnularSectorFootprint, BooleanFootprint, BoxFootprint, ConstantAltitudeBand,
    DiskFootprint, LatLon, LinearAltitudeBand, PolygonFootprint, RadialAltitudeBand,
    RegionBounds, SectorFootprint, VertexAltitudeBand,
)
from bluesky_sandbox.sim.sampling.distributions import Bounded, Categorical
{envelope_import.rstrip()}
from bluesky_sandbox.sim.queryables import QueryRegion, Waypoint
from bluesky_sandbox.sim.spawn import SpawnConfig, SpawnRegion

ALLOWED_AIRCRAFT = {env.allowed_aircraft!r}
DT = {env.dt!r}
SIMDT = {env.simdt!r}
CD_METHOD = {env.cd_method!r}
PERFORMANCE_MODEL = {env.performance_model!r}

# Named, reusable bounds referenced by REGIONS[name] below.
REGIONS = {{
    {regions}
}}

AIRSPACE = {airspace}

QUERYABLES = {{
    {queryables}
}}

# name -> region a waypoint's position (lat/lon, altitude) is drawn from each episode.
SAMPLED_WAYPOINTS = {{
    {sampled_waypoints}
}}

SPAWN = {spawn}

# Per-episode group transform (e.g. rotation sampled from a distribution), or None.
TRANSFORM = {transform}
'''
