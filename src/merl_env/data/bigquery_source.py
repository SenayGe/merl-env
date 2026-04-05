"""BigQuery-backed source for building local MIMIC task artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from merl_env.data.diagnosis_builder import DIAGNOSIS_QUERY_NAME, DiagnosisBuilderConfig
from merl_env.data.discharge_builder import (
    DISCHARGE_CHARLSON_QUERY_NAME,
    DISCHARGE_COHORT_QUERY_NAME,
    DISCHARGE_DIAGNOSES_QUERY_NAME,
    DISCHARGE_ICU_QUERY_NAME,
    DISCHARGE_LABS_QUERY_NAME,
    DISCHARGE_READMISSIONS_QUERY_NAME,
    DISCHARGE_SOFA_QUERY_NAME,
    DISCHARGE_VITALS_QUERY_NAME,
    DischargeBuilderConfig,
)
from merl_env.data.icd_builder import ICD_QUERY_NAME
from merl_env.data.mimic_source import MimicQuery, MimicSource

_DEFAULT_DISCHARGE_PULL_DAYS = 7

_VITAL_ITEMIDS: dict[str, tuple[int, ...]] = {
    "heart_rate": (220045,),
    "respiratory_rate": (220210,),
    "spo2": (220277,),
    "temperature_c": (223762,),
    "temperature_f": (223761,),
    "nibp_sbp": (220179,),
    "nibp_map": (220181,),
    "art_sbp": (220050,),
    "art_map": (220052,),
}
_ITEMID_TO_VITAL = {itemid: name for name, itemids in _VITAL_ITEMIDS.items() for itemid in itemids}

_LAB_ITEMIDS: dict[str, tuple[int, ...]] = {
    "Sodium": (50983,),
    "Potassium": (50971,),
    "Glucose": (50931,),
    "Lactate": (50813,),
    "Hemoglobin": (51222,),
}
_ITEMID_TO_LAB = {itemid: name for name, itemids in _LAB_ITEMIDS.items() for itemid in itemids}

_CHARLSON_COMPONENTS: tuple[str, ...] = (
    "myocardial_infarct",
    "congestive_heart_failure",
    "peripheral_vascular_disease",
    "cerebrovascular_disease",
    "dementia",
    "chronic_pulmonary_disease",
    "rheumatic_disease",
    "peptic_ulcer_disease",
    "mild_liver_disease",
    "diabetes_without_cc",
    "diabetes_with_cc",
    "paraplegia",
    "renal_disease",
    "malignant_cancer",
    "severe_liver_disease",
    "metastatic_solid_tumor",
    "aids",
)

_DERIVED_VITAL_COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "heart_rate": ("heart_rate", "hr"),
    "respiratory_rate": ("resp_rate", "respiratory_rate", "rr"),
    "spo2": ("spo2", "o2sat", "sao2"),
    "temperature_c": ("temperature_c", "temperature", "temp", "temperature_f"),
    "sbp": ("sbp", "systolic_bp"),
    "map": ("mbp", "map", "mean_bp", "mean_arterial_pressure"),
}


@dataclass(slots=True, kw_only=True)
class BigQueryMimicSourceConfig:
    """Configuration for a BigQuery-backed MIMIC source."""

    project_id: str
    location: str = "US"

    def __post_init__(self) -> None:
        self.project_id = self.project_id.strip()
        self.location = self.location.strip() or "US"
        if not self.project_id:
            raise ValueError("project_id must be a non-empty string")


class BigQueryMimicSource(MimicSource):
    """Fetch MIMIC-derived rows from BigQuery using ADC credentials."""

    def __init__(
        self,
        config: BigQueryMimicSourceConfig,
        *,
        client: Any | None = None,
        bigquery_module: Any | None = None,
    ) -> None:
        self._config = config
        self._bigquery = bigquery_module or _load_bigquery_module()
        self._client = client or self._bigquery.Client(project=config.project_id)
        self._diagnosis_config = DiagnosisBuilderConfig()
        self._discharge_config = DischargeBuilderConfig()

    @property
    def config(self) -> BigQueryMimicSourceConfig:
        return self._config

    def fetch_rows(self, query: MimicQuery) -> list[dict[str, Any]]:
        if query.name == DIAGNOSIS_QUERY_NAME:
            return self._run_query(query.sql, parameters=query.parameters)
        if query.name == ICD_QUERY_NAME:
            raise NotImplementedError(
                "BigQueryMimicSource does not yet support the 'icd' task"
            )

        discharge_handlers = {
            DISCHARGE_COHORT_QUERY_NAME: self._fetch_discharge_cohort,
            DISCHARGE_ICU_QUERY_NAME: self._fetch_discharge_icu_stays,
            DISCHARGE_VITALS_QUERY_NAME: self._fetch_discharge_vitals,
            DISCHARGE_LABS_QUERY_NAME: self._fetch_discharge_labs,
            DISCHARGE_READMISSIONS_QUERY_NAME: self._fetch_discharge_readmissions,
            DISCHARGE_DIAGNOSES_QUERY_NAME: self._fetch_discharge_diagnoses,
            DISCHARGE_CHARLSON_QUERY_NAME: self._fetch_discharge_charlson,
            DISCHARGE_SOFA_QUERY_NAME: self._fetch_discharge_sofa,
        }
        handler = discharge_handlers.get(query.name)
        if handler is not None:
            return handler(query)

        raise NotImplementedError(
            f"BigQueryMimicSource received unsupported query {query.name!r}"
        )

    def _run_query(
        self,
        sql: str,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        job_config = self._build_job_config(parameters or {})
        job = self._client.query(
            sql,
            job_config=job_config,
            location=self._config.location,
        )
        return [_row_to_dict(row) for row in job.result()]

    def _build_job_config(self, parameters: dict[str, Any]) -> Any | None:
        if not parameters:
            return None
        query_parameters = [
            self._convert_query_parameter(name, value)
            for name, value in parameters.items()
        ]
        return self._bigquery.QueryJobConfig(query_parameters=query_parameters)

    def _convert_query_parameter(self, name: str, value: Any) -> Any:
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, list):
            element_type = _infer_bigquery_type_from_sequence(value)
            return self._bigquery.ArrayQueryParameter(name, element_type, value)
        return self._bigquery.ScalarQueryParameter(name, _infer_bigquery_type(value), value)

    def _fetch_discharge_cohort(self, query: MimicQuery) -> list[dict[str, Any]]:
        del query
        cfg = self._discharge_config
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
        FROM `{cfg.hospital_dataset}.patients` AS p
        JOIN `{cfg.hospital_dataset}.admissions` AS a
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
          ) >= {cfg.min_age}
        """
        return self._run_query(sql)

    def _fetch_discharge_icu_stays(self, query: MimicQuery) -> list[dict[str, Any]]:
        cfg = self._discharge_config
        sql = f"""
        SELECT
          hadm_id,
          stay_id,
          intime,
          outtime
        FROM `{cfg.icu_dataset}.icustays`
        WHERE hadm_id IN UNNEST(@hadm_ids)
        """
        return self._run_query(sql, parameters={"hadm_ids": _int_list(query, "hadm_ids")})

    def _fetch_discharge_vitals(self, query: MimicQuery) -> list[dict[str, Any]]:
        hadm_ids = _int_list(query, "hadm_ids")
        return [
            *self._fetch_derived_vitals(hadm_ids),
            *self._fetch_icu_vitals(hadm_ids),
        ]

    def _fetch_derived_vitals(self, hadm_ids: Sequence[int]) -> list[dict[str, Any]]:
        cfg = self._discharge_config
        sql = f"""
        WITH coh AS (
          SELECT subject_id, hadm_id, admittime, dischtime
          FROM `{cfg.hospital_dataset}.admissions`
          WHERE hadm_id IN UNNEST(@hadm_ids)
        )
        SELECT
          coh.hadm_id,
          vs.*
        FROM `{cfg.derived_vitals_table}` AS vs
        JOIN coh
          ON vs.subject_id = coh.subject_id
         AND vs.charttime BETWEEN coh.admittime AND coh.dischtime
         AND vs.charttime BETWEEN DATETIME_SUB(
           coh.dischtime,
           INTERVAL {_DEFAULT_DISCHARGE_PULL_DAYS} DAY
         ) AND coh.dischtime
        WHERE vs.charttime IS NOT NULL
        """
        rows = self._run_query(sql, parameters={"hadm_ids": list(hadm_ids)})
        normalized: list[dict[str, Any]] = []
        for row in rows:
            hadm_id = _coerce_int(row.get("hadm_id"))
            charttime = row.get("charttime")
            if hadm_id is None or charttime is None:
                continue
            for vital_name, source_columns in _DERIVED_VITAL_COLUMN_CANDIDATES.items():
                source_column = _first_present_key(row, source_columns)
                if source_column is None:
                    continue
                value = _coerce_float(row.get(source_column))
                if value is None:
                    continue
                if vital_name == "temperature_c" and (source_column == "temperature_f" or value > 80.0):
                    value = (value - 32.0) * (5.0 / 9.0)
                normalized.append(
                    {
                        "hadm_id": hadm_id,
                        "charttime": charttime,
                        "vital_name": vital_name,
                        "valuenum": value,
                        "source": "derived",
                    }
                )
        return normalized

    def _fetch_icu_vitals(self, hadm_ids: Sequence[int]) -> list[dict[str, Any]]:
        cfg = self._discharge_config
        itemids = sorted(_ITEMID_TO_VITAL)
        sql = f"""
        WITH coh AS (
          SELECT hadm_id, dischtime
          FROM `{cfg.hospital_dataset}.admissions`
          WHERE hadm_id IN UNNEST(@hadm_ids)
        )
        SELECT
          icu.hadm_id,
          ce.charttime,
          ce.itemid,
          ce.valuenum
        FROM `{cfg.icu_dataset}.chartevents` AS ce
        JOIN `{cfg.icu_dataset}.icustays` AS icu
          ON ce.stay_id = icu.stay_id
        JOIN coh
          ON icu.hadm_id = coh.hadm_id
        WHERE icu.hadm_id IN UNNEST(@hadm_ids)
          AND ce.itemid IN UNNEST(@itemids)
          AND ce.valuenum IS NOT NULL
          AND ce.charttime IS NOT NULL
          AND ce.charttime BETWEEN DATETIME_SUB(
            coh.dischtime,
            INTERVAL {_DEFAULT_DISCHARGE_PULL_DAYS} DAY
          ) AND coh.dischtime
        """
        rows = self._run_query(
            sql,
            parameters={"hadm_ids": list(hadm_ids), "itemids": itemids},
        )
        normalized: list[dict[str, Any]] = []
        for row in rows:
            hadm_id = _coerce_int(row.get("hadm_id"))
            itemid = _coerce_int(row.get("itemid"))
            value = _coerce_float(row.get("valuenum"))
            charttime = row.get("charttime")
            vital_name = _ITEMID_TO_VITAL.get(itemid or -1)
            if hadm_id is None or charttime is None or value is None or vital_name is None:
                continue
            if vital_name == "temperature_f":
                vital_name = "temperature_c"
                value = (value - 32.0) * (5.0 / 9.0)
            normalized.append(
                {
                    "hadm_id": hadm_id,
                    "charttime": charttime,
                    "vital_name": vital_name,
                    "valuenum": value,
                    "source": "icu",
                }
            )
        return normalized

    def _fetch_discharge_labs(self, query: MimicQuery) -> list[dict[str, Any]]:
        hadm_ids = _int_list(query, "hadm_ids")
        cfg = self._discharge_config
        itemids = sorted(_ITEMID_TO_LAB)
        sql = f"""
        WITH coh AS (
          SELECT hadm_id, dischtime
          FROM `{cfg.hospital_dataset}.admissions`
          WHERE hadm_id IN UNNEST(@hadm_ids)
        )
        SELECT
          le.hadm_id,
          le.charttime,
          le.itemid,
          le.valuenum
        FROM `{cfg.hospital_dataset}.labevents` AS le
        JOIN coh
          ON le.hadm_id = coh.hadm_id
        WHERE le.hadm_id IN UNNEST(@hadm_ids)
          AND le.itemid IN UNNEST(@itemids)
          AND le.valuenum IS NOT NULL
          AND le.charttime IS NOT NULL
          AND le.charttime BETWEEN DATETIME_SUB(
            coh.dischtime,
            INTERVAL {_DEFAULT_DISCHARGE_PULL_DAYS} DAY
          ) AND coh.dischtime
        """
        rows = self._run_query(
            sql,
            parameters={"hadm_ids": hadm_ids, "itemids": itemids},
        )
        normalized: list[dict[str, Any]] = []
        for row in rows:
            hadm_id = _coerce_int(row.get("hadm_id"))
            itemid = _coerce_int(row.get("itemid"))
            value = _coerce_float(row.get("valuenum"))
            charttime = row.get("charttime")
            lab_name = _ITEMID_TO_LAB.get(itemid or -1)
            if hadm_id is None or charttime is None or value is None or lab_name is None:
                continue
            normalized.append(
                {
                    "hadm_id": hadm_id,
                    "charttime": charttime,
                    "lab_name": lab_name,
                    "valuenum": value,
                }
            )
        return normalized

    def _fetch_discharge_readmissions(self, query: MimicQuery) -> list[dict[str, Any]]:
        hadm_ids = _int_list(query, "hadm_ids")
        cfg = self._discharge_config
        sql = f"""
        WITH idx AS (
          SELECT subject_id, hadm_id, dischtime
          FROM `{cfg.hospital_dataset}.admissions`
          WHERE hadm_id IN UNNEST(@hadm_ids)
        ),
        pairs AS (
          SELECT
            a1.hadm_id AS index_hadm_id,
            a2.hadm_id AS readmit_hadm_id,
            DATETIME_DIFF(a2.admittime, a1.dischtime, HOUR) AS hours_to_readmit,
            ROW_NUMBER() OVER (PARTITION BY a1.hadm_id ORDER BY a2.admittime) AS rn
          FROM idx AS a1
          JOIN `{cfg.hospital_dataset}.admissions` AS a2
            ON a1.subject_id = a2.subject_id
           AND a2.admittime > a1.dischtime
        )
        SELECT
          index_hadm_id AS hadm_id,
          readmit_hadm_id,
          hours_to_readmit,
          CASE WHEN hours_to_readmit <= 24 * 7 THEN 1 ELSE 0 END AS readmit_7d,
          CASE WHEN hours_to_readmit <= 72 THEN 1 ELSE 0 END AS readmit_72h,
          CASE WHEN hours_to_readmit <= 24 * 30 THEN 1 ELSE 0 END AS readmit_30d
        FROM pairs
        WHERE rn = 1
        """
        return self._run_query(sql, parameters={"hadm_ids": hadm_ids})

    def _fetch_discharge_diagnoses(self, query: MimicQuery) -> list[dict[str, Any]]:
        hadm_ids = _int_list(query, "hadm_ids")
        cfg = self._discharge_config
        sql = f"""
        SELECT
          dx.hadm_id,
          dx.seq_num,
          dx.icd_code,
          dx.icd_version,
          d_dx.long_title
        FROM `{cfg.hospital_dataset}.diagnoses_icd` AS dx
        LEFT JOIN `{cfg.hospital_dataset}.d_icd_diagnoses` AS d_dx
          ON (dx.icd_version, dx.icd_code) = (d_dx.icd_version, d_dx.icd_code)
        WHERE dx.hadm_id IN UNNEST(@hadm_ids)
        ORDER BY dx.hadm_id, dx.seq_num
        """
        return self._run_query(sql, parameters={"hadm_ids": hadm_ids})

    def _fetch_discharge_charlson(self, query: MimicQuery) -> list[dict[str, Any]]:
        hadm_ids = _int_list(query, "hadm_ids")
        cfg = self._discharge_config
        component_columns = ",\n          ".join(_CHARLSON_COMPONENTS)
        sql = f"""
        SELECT
          hadm_id,
          charlson_comorbidity_index,
          {component_columns}
        FROM `{cfg.derived_dataset}.charlson`
        WHERE hadm_id IN UNNEST(@hadm_ids)
        """
        return self._run_query(sql, parameters={"hadm_ids": hadm_ids})

    def _fetch_discharge_sofa(self, query: MimicQuery) -> list[dict[str, Any]]:
        hadm_ids = _int_list(query, "hadm_ids")
        cfg = self._discharge_config
        sql = f"""
        SELECT
          icu.hadm_id,
          s.stay_id,
          s.starttime,
          s.endtime,
          s.sofa_24hours AS sofa_score,
          s.respiration_24hours AS sofa_resp,
          s.coagulation_24hours AS sofa_coag,
          s.liver_24hours AS sofa_liver,
          s.cardiovascular_24hours AS sofa_cardio,
          s.cns_24hours AS sofa_cns,
          s.renal_24hours AS sofa_renal
        FROM `{cfg.derived_dataset}.sofa` AS s
        JOIN `{cfg.icu_dataset}.icustays` AS icu
          ON s.stay_id = icu.stay_id
        WHERE icu.hadm_id IN UNNEST(@hadm_ids)
        """
        return self._run_query(sql, parameters={"hadm_ids": hadm_ids})


def _row_to_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return dict(row)
    if hasattr(row, "items"):
        return dict(row.items())
    return dict(row)


def _load_bigquery_module() -> Any:
    try:
        from google.cloud import bigquery
    except ImportError as exc:  # pragma: no cover - exercised only when dependency is absent
        raise ImportError(
            "google-cloud-bigquery is required for BigQueryMimicSource. "
            "Install merl-env[bigquery]."
        ) from exc
    return bigquery


def _infer_bigquery_type(value: Any) -> str:
    if isinstance(value, bool):
        return "BOOL"
    if isinstance(value, int):
        return "INT64"
    if isinstance(value, float):
        return "FLOAT64"
    return "STRING"


def _infer_bigquery_type_from_sequence(values: Sequence[Any]) -> str:
    for value in values:
        if value is None:
            continue
        return _infer_bigquery_type(value)
    return "STRING"


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_list(query: MimicQuery, parameter_name: str) -> list[int]:
    raw_values = query.parameters.get(parameter_name, [])
    if isinstance(raw_values, tuple):
        raw_values = list(raw_values)
    if not isinstance(raw_values, list):
        raise ValueError(f"Expected query parameter {parameter_name!r} to be a list")
    values = [value for item in raw_values if (value := _coerce_int(item)) is not None]
    return values


def _first_present_key(row: dict[str, Any], candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        if candidate in row:
            return candidate
    return None


__all__ = ["BigQueryMimicSource", "BigQueryMimicSourceConfig"]
