"""Offline builder for near-discharge safety assessment snapshots."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import Any, Mapping, Sequence

from merl_env.core.sample import TaskSample
from merl_env.data._builder_utils import (
    build_sample_id,
    coerce_bool,
    coerce_float,
    coerce_int,
    empty_split_buckets,
    json_safe,
    parse_datetime,
    sort_split_buckets,
)
from merl_env.data.artifacts import TaskArtifactManifest, write_task_artifacts
from merl_env.data.mimic_source import MimicQuery, MimicSource
from merl_env.data.splits import SplitConfig, assign_subject_splits

DISCHARGE_TASK_NAME = "discharge"

DISCHARGE_COHORT_QUERY_NAME = "discharge_cohort"
DISCHARGE_ICU_QUERY_NAME = "discharge_icu_stays"
DISCHARGE_VITALS_QUERY_NAME = "discharge_vitals"
DISCHARGE_LABS_QUERY_NAME = "discharge_labs"
DISCHARGE_READMISSIONS_QUERY_NAME = "discharge_readmissions"
DISCHARGE_DIAGNOSES_QUERY_NAME = "discharge_diagnoses"
DISCHARGE_CHARLSON_QUERY_NAME = "discharge_charlson"
DISCHARGE_SOFA_QUERY_NAME = "discharge_sofa"

DEFAULT_REQUIRED_VITALS: tuple[str, ...] = (
    "heart_rate",
    "respiratory_rate",
    "spo2",
    "temperature_c",
)
DEFAULT_REQUIRED_LABS: tuple[str, ...] = (
    "Potassium",
    "Sodium",
    "Glucose",
    "Lactate",
    "Hemoglobin",
)


@dataclass(slots=True, kw_only=True)
class DischargeBuilderConfig:
    """Configuration for the discharge task builder."""

    hospital_dataset: str = "physionet-data.mimiciv_3_1_hosp"
    icu_dataset: str = "physionet-data.mimiciv_3_1_icu"
    derived_dataset: str = "physionet-data.mimiciv_3_1_derived"
    derived_vitals_table: str = "physionet-data.mimiciv_derived.vitalsign"
    snapshot_hours_before_discharge: int = 24
    vitals_window_hours: int = 6
    labs_window_hours: int = 24
    max_diagnoses_per_admission: int = 10
    deduplicate_latest_admission_per_subject: bool = True
    min_age: int = 18
    exclude_hospice: bool = False
    exclude_transfers: bool = False
    required_vitals: tuple[str, ...] = DEFAULT_REQUIRED_VITALS
    require_sbp_or_map: bool = True
    required_labs: tuple[str, ...] = DEFAULT_REQUIRED_LABS
    hb_sbp_lt: float = 90.0
    hb_map_lt: float = 65.0
    hb_spo2_lt: float = 90.0
    hb_rr_gt: float = 30.0
    hb_hr_gt: float = 130.0
    hb_temp_c_ge: float = 39.0
    hb_k_ge: float = 6.0
    hb_k_le: float = 3.0
    hb_na_ge: float = 160.0
    hb_na_le: float = 120.0
    hb_glu_ge: float = 400.0
    hb_glu_le: float = 50.0
    hb_lact_ge: float = 4.0
    hb_hgb_le: float = 7.0
    allowed_tools: tuple[str, ...] = ("lab_ranges",)


def build_discharge_cohort_query(config: DischargeBuilderConfig | None = None) -> MimicQuery:
    """Logical query for the base discharge cohort."""

    builder_config = config or DischargeBuilderConfig()
    hospice_filter = ""
    if builder_config.exclude_hospice:
        hospice_filter = "AND UPPER(a.discharge_location) NOT LIKE '%HOSPICE%'"

    transfer_filter = ""
    if builder_config.exclude_transfers:
        transfer_filter = (
            "AND UPPER(a.discharge_location) NOT LIKE '%TRANSFER%'\n"
            "AND UPPER(a.discharge_location) NOT LIKE '%HOSPITAL%'"
        )

    sql = f"""
    SELECT
      p.subject_id,
      a.hadm_id,
      a.admittime,
      a.dischtime,
      a.deathtime,
      p.dod,
      a.admission_type,
      a.admission_location,
      a.discharge_location,
      a.hospital_expire_flag,
      p.gender,
      p.anchor_age + DATETIME_DIFF(
        a.admittime,
        DATETIME(p.anchor_year, 1, 1, 0, 0, 0),
        YEAR
      ) AS age_at_admission
    FROM `{builder_config.hospital_dataset}.patients` AS p
    JOIN `{builder_config.hospital_dataset}.admissions` AS a
      ON p.subject_id = a.subject_id
    WHERE a.admittime IS NOT NULL
      AND a.dischtime IS NOT NULL
      AND a.hospital_expire_flag = 0
      AND (
        p.anchor_age + DATETIME_DIFF(
          a.admittime,
          DATETIME(p.anchor_year, 1, 1, 0, 0, 0),
          YEAR
        )
      ) >= {builder_config.min_age}
      {hospice_filter}
      {transfer_filter}
    """
    return MimicQuery(
        name=DISCHARGE_COHORT_QUERY_NAME,
        sql=sql.strip(),
        metadata={
            "task_name": DISCHARGE_TASK_NAME,
            "expected_columns": (
                "subject_id",
                "hadm_id",
                "admittime",
                "dischtime",
                "deathtime",
                "dod",
                "admission_type",
                "admission_location",
                "discharge_location",
                "gender",
                "age_at_admission",
            ),
        },
    )


def build_discharge_icu_query(
    hadm_ids: Sequence[int],
    config: DischargeBuilderConfig | None = None,
) -> MimicQuery:
    """Logical query for ICU stay intervals."""

    builder_config = config or DischargeBuilderConfig()
    sql = f"""
    SELECT
      hadm_id,
      stay_id,
      intime,
      outtime
    FROM `{builder_config.icu_dataset}.icustays`
    WHERE hadm_id IN UNNEST(@hadm_ids)
    """
    return MimicQuery(
        name=DISCHARGE_ICU_QUERY_NAME,
        sql=sql.strip(),
        parameters={"hadm_ids": list(hadm_ids)},
        metadata={"expected_columns": ("hadm_id", "stay_id", "intime", "outtime")},
    )


def build_discharge_vitals_query(
    hadm_ids: Sequence[int],
    config: DischargeBuilderConfig | None = None,
) -> MimicQuery:
    """Logical query for normalized vital-sign rows.

    Expected columns:
    - `hadm_id`
    - `charttime`
    - `vital_name`
    - `valuenum`
    - optional `source` with values like `derived` or `icu`
    """

    builder_config = config or DischargeBuilderConfig()
    sql = f"""
    -- This query is intentionally normalized to the columns the builder consumes.
    -- A production source should map both derived vitals and ICU fallback rows to:
    --   hadm_id, charttime, vital_name, valuenum, source
    SELECT hadm_id, charttime, vital_name, valuenum, source
    FROM `{builder_config.derived_vitals_table}`
    WHERE hadm_id IN UNNEST(@hadm_ids)
    """
    return MimicQuery(
        name=DISCHARGE_VITALS_QUERY_NAME,
        sql=sql.strip(),
        parameters={"hadm_ids": list(hadm_ids)},
        metadata={
            "expected_columns": ("hadm_id", "charttime", "vital_name", "valuenum", "source")
        },
    )


def build_discharge_labs_query(
    hadm_ids: Sequence[int],
    config: DischargeBuilderConfig | None = None,
) -> MimicQuery:
    """Logical query for normalized laboratory rows."""

    builder_config = config or DischargeBuilderConfig()
    sql = f"""
    -- Expected columns:
    --   hadm_id, charttime, lab_name, valuenum
    SELECT hadm_id, charttime, lab_name, valuenum
    FROM `{builder_config.hospital_dataset}.labevents`
    WHERE hadm_id IN UNNEST(@hadm_ids)
    """
    return MimicQuery(
        name=DISCHARGE_LABS_QUERY_NAME,
        sql=sql.strip(),
        parameters={"hadm_ids": list(hadm_ids)},
        metadata={"expected_columns": ("hadm_id", "charttime", "lab_name", "valuenum")},
    )


def build_discharge_readmissions_query(
    hadm_ids: Sequence[int],
    config: DischargeBuilderConfig | None = None,
) -> MimicQuery:
    """Logical query for readmission outcome flags."""

    builder_config = config or DischargeBuilderConfig()
    sql = f"""
    -- Expected columns:
    --   hadm_id, readmit_7d, readmit_72h, readmit_30d
    SELECT hadm_id, readmit_7d, readmit_72h, readmit_30d
    FROM `{builder_config.hospital_dataset}.admissions`
    WHERE hadm_id IN UNNEST(@hadm_ids)
    """
    return MimicQuery(
        name=DISCHARGE_READMISSIONS_QUERY_NAME,
        sql=sql.strip(),
        parameters={"hadm_ids": list(hadm_ids)},
        metadata={
            "expected_columns": ("hadm_id", "readmit_7d", "readmit_72h", "readmit_30d")
        },
    )


def build_discharge_diagnoses_query(
    hadm_ids: Sequence[int],
    config: DischargeBuilderConfig | None = None,
) -> MimicQuery:
    """Logical query for admission diagnoses."""

    builder_config = config or DischargeBuilderConfig()
    sql = f"""
    SELECT
      dx.hadm_id,
      dx.seq_num,
      dx.icd_code,
      dx.icd_version,
      d_dx.long_title
    FROM `{builder_config.hospital_dataset}.diagnoses_icd` AS dx
    LEFT JOIN `{builder_config.hospital_dataset}.d_icd_diagnoses` AS d_dx
      ON (dx.icd_version, dx.icd_code) = (d_dx.icd_version, d_dx.icd_code)
    WHERE dx.hadm_id IN UNNEST(@hadm_ids)
    ORDER BY dx.hadm_id, dx.seq_num
    """
    return MimicQuery(
        name=DISCHARGE_DIAGNOSES_QUERY_NAME,
        sql=sql.strip(),
        parameters={"hadm_ids": list(hadm_ids)},
        metadata={
            "expected_columns": ("hadm_id", "seq_num", "icd_code", "icd_version", "long_title")
        },
    )


def build_discharge_charlson_query(
    hadm_ids: Sequence[int],
    config: DischargeBuilderConfig | None = None,
) -> MimicQuery:
    """Logical query for Charlson comorbidity rows."""

    builder_config = config or DischargeBuilderConfig()
    sql = f"""
    SELECT *
    FROM `{builder_config.derived_dataset}.charlson`
    WHERE hadm_id IN UNNEST(@hadm_ids)
    """
    return MimicQuery(
        name=DISCHARGE_CHARLSON_QUERY_NAME,
        sql=sql.strip(),
        parameters={"hadm_ids": list(hadm_ids)},
        metadata={"expected_columns": ("hadm_id", "charlson_comorbidity_index")},
    )


def build_discharge_sofa_query(
    hadm_ids: Sequence[int],
    config: DischargeBuilderConfig | None = None,
) -> MimicQuery:
    """Logical query for time-indexed SOFA rows."""

    builder_config = config or DischargeBuilderConfig()
    sql = f"""
    SELECT *
    FROM `{builder_config.derived_dataset}.sofa`
    WHERE hadm_id IN UNNEST(@hadm_ids)
    ORDER BY hadm_id, starttime
    """
    return MimicQuery(
        name=DISCHARGE_SOFA_QUERY_NAME,
        sql=sql.strip(),
        parameters={"hadm_ids": list(hadm_ids)},
        metadata={"expected_columns": ("hadm_id", "starttime", "sofa_score")},
    )


def normalize_discharge_vitals(
    raw_vitals: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Normalize raw vital channels into stable protocol-level names."""

    normalized: dict[str, list[dict[str, Any]]] = {}

    def _merge_channel(target_name: str, source_names: Sequence[str]) -> None:
        merged: list[dict[str, Any]] = []
        for source_name in source_names:
            for point in raw_vitals.get(source_name, ()):
                value = coerce_float(point.get("value"))
                if value is None:
                    continue
                merged.append(
                    {
                        "value": value,
                        "time": str(point.get("time")),
                    }
                )
        if merged:
            normalized[target_name] = sorted(merged, key=lambda item: item["time"])

    _merge_channel("sbp", ("sbp", "nibp_sbp", "art_sbp"))
    _merge_channel("map", ("map", "nibp_map", "art_map"))

    for name in ("heart_rate", "respiratory_rate", "spo2"):
        if name not in raw_vitals:
            continue
        normalized[name] = [
            {
                "value": coerce_float(point.get("value")),
                "time": str(point.get("time")),
            }
            for point in raw_vitals[name]
            if coerce_float(point.get("value")) is not None
        ]

    temperature_points: list[dict[str, Any]] = []
    for source_name in ("temperature_c", "temperature", "temperature_f"):
        for point in raw_vitals.get(source_name, ()):
            value = coerce_float(point.get("value"))
            if value is None:
                continue
            if source_name == "temperature_f" or (source_name in {"temperature", "temperature_c"} and value > 80):
                value = (value - 32.0) * (5.0 / 9.0)
            temperature_points.append({"value": value, "time": str(point.get("time"))})
    if temperature_points:
        normalized["temperature_c"] = sorted(temperature_points, key=lambda item: item["time"])

    return normalized


def compute_hard_barrier(
    icu_t: bool,
    vitals: Mapping[str, Sequence[Mapping[str, Any]]],
    labs: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    config: DischargeBuilderConfig | None = None,
) -> bool:
    """Port the hard-barrier logic from the snapshot prototype."""

    if icu_t:
        return True

    builder_config = config or DischargeBuilderConfig()

    def _worst_min(series: Sequence[Mapping[str, Any]]) -> float | None:
        values = [
            value
            for item in series
            if (value := coerce_float(item.get("value"))) is not None
        ]
        return min(values) if values else None

    def _worst_max(series: Sequence[Mapping[str, Any]]) -> float | None:
        values = [
            value
            for item in series
            if (value := coerce_float(item.get("value"))) is not None
        ]
        return max(values) if values else None

    sbp = _worst_min(vitals.get("sbp", ()))
    map_value = _worst_min(vitals.get("map", ()))
    spo2 = _worst_min(vitals.get("spo2", ()))
    rr = _worst_max(vitals.get("respiratory_rate", ()))
    hr = _worst_max(vitals.get("heart_rate", ()))
    temperature_c = _worst_max(vitals.get("temperature_c", ()))

    if sbp is not None and sbp < builder_config.hb_sbp_lt:
        return True
    if map_value is not None and map_value < builder_config.hb_map_lt:
        return True
    if spo2 is not None and spo2 < builder_config.hb_spo2_lt:
        return True
    if rr is not None and rr > builder_config.hb_rr_gt:
        return True
    if hr is not None and hr > builder_config.hb_hr_gt:
        return True
    if temperature_c is not None and temperature_c >= builder_config.hb_temp_c_ge:
        return True

    def _all_values(series: Sequence[Mapping[str, Any]]) -> list[float]:
        return [
            value
            for item in series
            if (value := coerce_float(item.get("value"))) is not None
        ]

    potassium = _all_values(labs.get("Potassium", ()))
    sodium = _all_values(labs.get("Sodium", ()))
    glucose = _all_values(labs.get("Glucose", ()))
    lactate = _all_values(labs.get("Lactate", ()))
    hemoglobin = _all_values(labs.get("Hemoglobin", ()))

    if potassium and (
        max(potassium) >= builder_config.hb_k_ge or min(potassium) <= builder_config.hb_k_le
    ):
        return True
    if sodium and (
        max(sodium) >= builder_config.hb_na_ge or min(sodium) <= builder_config.hb_na_le
    ):
        return True
    if glucose and (
        max(glucose) >= builder_config.hb_glu_ge or min(glucose) <= builder_config.hb_glu_le
    ):
        return True
    if lactate and max(lactate) >= builder_config.hb_lact_ge:
        return True
    if hemoglobin and min(hemoglobin) <= builder_config.hb_hgb_le:
        return True
    return False


def deduplicate_latest_admissions(
    cohort_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep one admission per subject, matching the prototype's latest-admission rule."""

    latest_by_subject: dict[str, dict[str, Any]] = {}
    for row in cohort_rows:
        subject_key = str(row.get("subject_id"))
        current = latest_by_subject.get(subject_key)
        candidate = dict(row)
        if current is None or _admission_sort_key(candidate) > _admission_sort_key(current):
            latest_by_subject[subject_key] = candidate
    return sorted(
        latest_by_subject.values(),
        key=lambda item: (
            coerce_int(item.get("subject_id")) or 0,
            parse_datetime(item.get("admittime")) or parse_datetime(item.get("dischtime")),
            coerce_int(item.get("hadm_id")) or 0,
        ),
    )


def build_discharge_samples(
    source: MimicSource,
    *,
    config: DischargeBuilderConfig | None = None,
    split_config: SplitConfig | None = None,
) -> dict[str, list[TaskSample]]:
    """Build normalized discharge-snapshot samples grouped by split."""

    builder_config = config or DischargeBuilderConfig()
    cohort_rows = source.fetch_rows(build_discharge_cohort_query(builder_config))
    if builder_config.deduplicate_latest_admission_per_subject:
        cohort_rows = deduplicate_latest_admissions(cohort_rows)

    if not cohort_rows:
        return empty_split_buckets()

    hadm_ids = [
        hadm_id
        for row in cohort_rows
        if (hadm_id := coerce_int(row.get("hadm_id"))) is not None
    ]
    icu_rows = source.fetch_rows(build_discharge_icu_query(hadm_ids, builder_config))
    vital_rows = _apply_vital_source_precedence(
        source.fetch_rows(build_discharge_vitals_query(hadm_ids, builder_config))
    )
    lab_rows = source.fetch_rows(build_discharge_labs_query(hadm_ids, builder_config))
    readmission_rows = source.fetch_rows(
        build_discharge_readmissions_query(hadm_ids, builder_config)
    )
    diagnosis_rows = source.fetch_rows(build_discharge_diagnoses_query(hadm_ids, builder_config))
    charlson_rows = source.fetch_rows(build_discharge_charlson_query(hadm_ids, builder_config))
    sofa_rows = source.fetch_rows(build_discharge_sofa_query(hadm_ids, builder_config))

    icu_index = _index_icu_rows(icu_rows)
    vital_index = _index_measurement_rows(vital_rows, name_key="vital_name")
    lab_index = _index_measurement_rows(lab_rows, name_key="lab_name")
    readmission_index = _index_by_hadm(readmission_rows)
    diagnosis_index = _index_diagnoses(
        diagnosis_rows,
        max_diagnoses=builder_config.max_diagnoses_per_admission,
    )
    charlson_index = _index_single_row_payload(charlson_rows)
    sofa_index = _index_time_ordered_rows(sofa_rows, time_key="starttime")

    assignments = assign_subject_splits(
        [row["subject_id"] for row in cohort_rows],
        config=split_config,
    )
    samples_by_split = empty_split_buckets()

    for admission_row in cohort_rows:
        subject_id = admission_row.get("subject_id")
        hadm_id = coerce_int(admission_row.get("hadm_id"))
        admittime = parse_datetime(admission_row.get("admittime"))
        dischtime = parse_datetime(admission_row.get("dischtime"))
        if hadm_id is None or dischtime is None:
            continue

        split = assignments[subject_id]
        snapshot_time = dischtime - timedelta(hours=builder_config.snapshot_hours_before_discharge)
        icu_t = _icu_at_time(icu_index, hadm_id, snapshot_time)

        raw_vitals = _window_measurements(
            vital_index.get(hadm_id, ()),
            snapshot_time,
            hours=builder_config.vitals_window_hours,
        )
        vitals = normalize_discharge_vitals(raw_vitals)
        labs = _window_measurements(
            lab_index.get(hadm_id, ()),
            snapshot_time,
            hours=builder_config.labs_window_hours,
        )
        vitals, labs = _ensure_required_measurement_keys(vitals, labs, builder_config)
        core_fields_complete = _has_required_fields(vitals, labs, builder_config)

        adverse_7d, adverse_72h, readmit_30d = _compute_adverse_flags(
            admission_row,
            readmission_index.get(hadm_id),
            dischtime,
        )
        discharged_in_window = snapshot_time < dischtime <= (
            snapshot_time + timedelta(hours=builder_config.snapshot_hours_before_discharge)
        )
        safe_for_discharge = bool(discharged_in_window and not adverse_7d)
        has_hard_barrier = compute_hard_barrier(
            icu_t,
            vitals,
            labs,
            config=builder_config,
        )

        sample = TaskSample(
            sample_id=build_sample_id(
                DISCHARGE_TASK_NAME,
                hadm_id,
                snapshot_time.strftime("%Y%m%d%H%M%S"),
            ),
            task_name=DISCHARGE_TASK_NAME,
            split=split,
            input_payload=json_safe(
                {
                    "subject_id": subject_id,
                    "hadm_id": hadm_id,
                    "snapshot_time": snapshot_time,
                    "admittime": admittime,
                    "dischtime": dischtime,
                    "age_at_admission": coerce_int(admission_row.get("age_at_admission")),
                    "gender": admission_row.get("gender"),
                    "admission_type": admission_row.get("admission_type"),
                    "admission_location": admission_row.get("admission_location"),
                    "discharge_location": admission_row.get("discharge_location"),
                    "icu_t": icu_t,
                    "vitals": vitals,
                    "labs": labs,
                    "diagnoses": diagnosis_index.get(hadm_id, []),
                    "charlson": charlson_index.get(hadm_id),
                    "sofa": _latest_row_before(sofa_index.get(hadm_id, ()), snapshot_time, "starttime"),
                }
            ),
            reference={
                "safe_for_discharge_24h": safe_for_discharge,
                "has_hard_barrier": has_hard_barrier,
            },
            metadata=json_safe(
                {
                    "subject_id": subject_id,
                    "hadm_id": hadm_id,
                    "core_fields_complete": core_fields_complete,
                    "adverse_7d": adverse_7d,
                    "adverse_72h": adverse_72h,
                    "readmit_30d": readmit_30d,
                }
            ),
            allowed_tools=builder_config.allowed_tools,
        )
        samples_by_split[split].append(sample)

    return sort_split_buckets(samples_by_split)


def build_discharge_artifacts(
    out_dir: str,
    source: MimicSource,
    *,
    config: DischargeBuilderConfig | None = None,
    split_config: SplitConfig | None = None,
) -> TaskArtifactManifest:
    """Build discharge samples and write them to canonical local artifacts."""

    builder_config = config or DischargeBuilderConfig()
    samples_by_split = build_discharge_samples(
        source,
        config=builder_config,
        split_config=split_config,
    )
    return write_task_artifacts(
        out_dir,
        DISCHARGE_TASK_NAME,
        samples_by_split,
        source={
            "query_names": (
                DISCHARGE_COHORT_QUERY_NAME,
                DISCHARGE_ICU_QUERY_NAME,
                DISCHARGE_VITALS_QUERY_NAME,
                DISCHARGE_LABS_QUERY_NAME,
                DISCHARGE_READMISSIONS_QUERY_NAME,
                DISCHARGE_DIAGNOSES_QUERY_NAME,
                DISCHARGE_CHARLSON_QUERY_NAME,
                DISCHARGE_SOFA_QUERY_NAME,
            ),
        },
        metadata={"builder_config": json_safe(asdict(builder_config))},
    )


def _admission_sort_key(row: Mapping[str, Any]) -> tuple[Any, Any, int]:
    return (
        parse_datetime(row.get("dischtime")) or parse_datetime(row.get("admittime")),
        parse_datetime(row.get("admittime")) or parse_datetime(row.get("dischtime")),
        coerce_int(row.get("hadm_id")) or 0,
    )


def _apply_vital_source_precedence(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    passthrough: list[dict[str, Any]] = []

    for row in rows:
        hadm_id = coerce_int(row.get("hadm_id"))
        copied = dict(row)
        if hadm_id is None:
            passthrough.append(copied)
            continue
        grouped[hadm_id].append(copied)

    filtered: list[dict[str, Any]] = list(passthrough)
    for hadm_id, group in grouped.items():
        del hadm_id
        has_derived = any(str(item.get("source") or "").lower() == "derived" for item in group)
        if has_derived:
            filtered.extend(
                item for item in group if str(item.get("source") or "").lower() == "derived"
            )
        else:
            filtered.extend(group)
    return filtered


def _index_icu_rows(rows: Sequence[Mapping[str, Any]]) -> dict[int, list[tuple[Any, Any]]]:
    indexed: dict[int, list[tuple[Any, Any]]] = defaultdict(list)
    for row in rows:
        hadm_id = coerce_int(row.get("hadm_id"))
        if hadm_id is None:
            continue
        intime = parse_datetime(row.get("intime"))
        outtime = parse_datetime(row.get("outtime"))
        if intime is None or outtime is None:
            continue
        indexed[hadm_id].append((intime, outtime))
    return indexed


def _index_measurement_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    name_key: str,
) -> dict[int, list[dict[str, Any]]]:
    indexed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        hadm_id = coerce_int(row.get("hadm_id"))
        charttime = parse_datetime(row.get("charttime"))
        name = str(row.get(name_key) or "").strip()
        value = coerce_float(row.get("valuenum"))
        if hadm_id is None or charttime is None or not name or value is None:
            continue
        indexed[hadm_id].append(
            {
                "name": name,
                "value": value,
                "time": charttime,
            }
        )
    for measurements in indexed.values():
        measurements.sort(key=lambda item: item["time"])
    return indexed


def _index_by_hadm(rows: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        hadm_id = coerce_int(row.get("hadm_id"))
        if hadm_id is None:
            continue
        indexed[hadm_id] = dict(row)
    return indexed


def _index_diagnoses(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_diagnoses: int,
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        hadm_id = coerce_int(row.get("hadm_id"))
        if hadm_id is None:
            continue
        grouped[hadm_id].append(dict(row))

    indexed: dict[int, list[dict[str, Any]]] = {}
    for hadm_id, group in grouped.items():
        group.sort(key=lambda row: coerce_int(row.get("seq_num")) or 10**9)
        indexed[hadm_id] = json_safe(
            [
                {
                    "seq_num": coerce_int(row.get("seq_num")),
                    "icd_code": str(row.get("icd_code") or "").strip() or None,
                    "icd_version": coerce_int(row.get("icd_version")),
                    "long_title": str(row.get("long_title") or "").strip() or None,
                }
                for row in group[:max_diagnoses]
            ]
        )
    return indexed


def _index_single_row_payload(rows: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        hadm_id = coerce_int(row.get("hadm_id"))
        if hadm_id is None:
            continue
        indexed[hadm_id] = json_safe(
            {
                key: value
                for key, value in row.items()
                if key != "hadm_id"
            }
        )
    return indexed


def _index_time_ordered_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    time_key: str,
) -> dict[int, list[dict[str, Any]]]:
    indexed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        hadm_id = coerce_int(row.get("hadm_id"))
        event_time = parse_datetime(row.get(time_key))
        if hadm_id is None or event_time is None:
            continue
        copied = dict(row)
        copied[time_key] = event_time
        indexed[hadm_id].append(copied)
    for group in indexed.values():
        group.sort(key=lambda item: item[time_key])
    return indexed


def _icu_at_time(
    icu_index: Mapping[int, Sequence[tuple[Any, Any]]],
    hadm_id: int,
    snapshot_time,
) -> bool:
    for intime, outtime in icu_index.get(hadm_id, ()):
        if intime <= snapshot_time <= outtime:
            return True
    return False


def _window_measurements(
    rows: Sequence[Mapping[str, Any]],
    snapshot_time,
    *,
    hours: int,
) -> dict[str, list[dict[str, Any]]]:
    window_start = snapshot_time - timedelta(hours=hours)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        charttime = row["time"]
        if not (window_start < charttime <= snapshot_time):
            continue
        grouped[row["name"]].append(
            {
                "value": row["value"],
                "time": charttime.isoformat(),
            }
        )
    return dict(grouped)


def _ensure_required_measurement_keys(
    vitals: Mapping[str, Sequence[Mapping[str, Any]]],
    labs: Mapping[str, Sequence[Mapping[str, Any]]],
    config: DischargeBuilderConfig,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    vitals_out = {key: [dict(point) for point in value] for key, value in vitals.items()}
    labs_out = {key: [dict(point) for point in value] for key, value in labs.items()}

    for name in config.required_vitals:
        vitals_out.setdefault(name, [])
    vitals_out.setdefault("sbp", [])
    vitals_out.setdefault("map", [])
    for name in config.required_labs:
        labs_out.setdefault(name, [])
    return vitals_out, labs_out


def _has_required_fields(
    vitals: Mapping[str, Sequence[Mapping[str, Any]]],
    labs: Mapping[str, Sequence[Mapping[str, Any]]],
    config: DischargeBuilderConfig,
) -> bool:
    for name in config.required_vitals:
        if not vitals.get(name):
            return False
    if config.require_sbp_or_map and not (vitals.get("sbp") or vitals.get("map")):
        return False
    for name in config.required_labs:
        if not labs.get(name):
            return False
    return True


def _compute_adverse_flags(
    admission_row: Mapping[str, Any],
    readmission_row: Mapping[str, Any] | None,
    dischtime,
) -> tuple[bool, bool, bool]:
    readmit_7d = coerce_bool(readmission_row.get("readmit_7d")) if readmission_row else False
    readmit_72h = coerce_bool(readmission_row.get("readmit_72h")) if readmission_row else False
    readmit_30d = coerce_bool(readmission_row.get("readmit_30d")) if readmission_row else False

    death_7d = False
    death_72h = False
    dod = parse_datetime(admission_row.get("dod"))
    if dod is not None:
        day_delta = (dod.date() - dischtime.date()).days
        death_7d = 0 <= day_delta <= 7
        death_72h = 0 <= day_delta <= 3
    else:
        deathtime = parse_datetime(admission_row.get("deathtime"))
        if deathtime is not None:
            hours_after = (deathtime - dischtime).total_seconds() / 3600.0
            death_7d = 0 <= hours_after <= 24 * 7
            death_72h = 0 <= hours_after <= 72

    return (readmit_7d or death_7d), (readmit_72h or death_72h), readmit_30d


def _latest_row_before(
    rows: Sequence[Mapping[str, Any]],
    snapshot_time,
    time_key: str,
) -> dict[str, Any] | None:
    latest: Mapping[str, Any] | None = None
    for row in rows:
        event_time = row[time_key]
        if event_time <= snapshot_time:
            latest = row
        else:
            break
    if latest is None:
        return {"sofa_score": None}
    return json_safe({key: value for key, value in latest.items() if key != "hadm_id"})


__all__ = [
    "DISCHARGE_CHARLSON_QUERY_NAME",
    "DISCHARGE_COHORT_QUERY_NAME",
    "DISCHARGE_DIAGNOSES_QUERY_NAME",
    "DISCHARGE_ICU_QUERY_NAME",
    "DISCHARGE_LABS_QUERY_NAME",
    "DISCHARGE_READMISSIONS_QUERY_NAME",
    "DISCHARGE_SOFA_QUERY_NAME",
    "DISCHARGE_TASK_NAME",
    "DISCHARGE_VITALS_QUERY_NAME",
    "DEFAULT_REQUIRED_LABS",
    "DEFAULT_REQUIRED_VITALS",
    "DischargeBuilderConfig",
    "build_discharge_artifacts",
    "build_discharge_charlson_query",
    "build_discharge_cohort_query",
    "build_discharge_diagnoses_query",
    "build_discharge_icu_query",
    "build_discharge_labs_query",
    "build_discharge_readmissions_query",
    "build_discharge_samples",
    "build_discharge_sofa_query",
    "build_discharge_vitals_query",
    "compute_hard_barrier",
    "deduplicate_latest_admissions",
    "normalize_discharge_vitals",
]
