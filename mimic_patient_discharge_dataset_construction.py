"""%% Cell google auth"""
from google.colab import auth, userdata

auth.authenticate_user()
print("✅ Google Cloud authentication complete!")



"""%% Cell: dataset construction"""
"""
MIMIC-IV snapshot dataset builder aligned with the evaluation protocol.

- Cohort:
  - Adult admissions (>=18)
  - Exclude in-hospital death (hospital_expire_flag = 0)
  - Optional exclude hospice/transfers (ablations)
  - Deterministic de-dup by subject_id (keep latest discharge admission)
- Evaluation timepoints:
  - Primary: t = dischtime - 24h (one snapshot per admission)
  - Optional temporal mode: every 12–24h over final 3–5 days
- Feature windowing:
  - Vitals: worst value in (t-6h, t]
  - Labs: most recent value in (t-24h, t]
  - ICU(t): true if ICU stay interval contains t
  - Vitals source precedence:
    1) hospital-wide derived vitals table
    2) ICU chartevents fallback when derived vitals are absent for an admission
- Clinical features (merged from extraction pipeline):
  - Diagnoses: ICD codes with titles (top-N by seq_num per admission)
  - Charlson comorbidity index + individual comorbidity flags
  - SOFA scores: most recent score at each snapshot timepoint
- Labels:
  - HARD_BARRIER(t) = ICU(t) OR critical vitals OR critical labs
  - SAFE_D24(t) = DISCHARGED_IN_24H(t) AND NOT ADVERSE_7D
    where ADVERSE_7D = readmit within 7d (within-system) OR death within 7d
    also produces ADVERSE_72H for sensitivity analysis
- Missingness:
  - Snapshots are retained when core fields are missing.
  - Missing values are explicit nulls (Python None in dicts / JSON null in output).
  - `core_fields_complete` indicates whether required core fields are present.

How to run:
  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/creds.json
  python build_snapshots.py --project_id YOUR_GCP_PROJECT --dataset physionet-data.mimiciv_3_1 --out_dir ./out

  # Filter to cardiac patients only:
  python build_snapshots.py --project_id YOUR_GCP_PROJECT --out_dir ./out --filter-dx-categories heart_failure ischemic_heart_disease

  # List available diagnosis categories:
  python build_snapshots.py --list-dx-categories

Outputs:
  - out/snapshots.parquet
  - out/snapshots.jsonl
  - out/run_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from google.cloud import bigquery
except ImportError:  # pragma: no cover - exercised only when dependency is absent
    bigquery = None


# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    project_id: str
    dataset: str  # e.g., "physionet-data.mimiciv_3_1"
    derived_vitals_table: str = "physionet-data.mimiciv_derived.vitalsign"
    derived_dataset: str = "physionet-data.mimiciv_3_1_derived"

    # Cohort
    min_age: int = 18
    exclude_hospice: bool = False
    exclude_transfers: bool = False
    max_admissions: Optional[int] = None

    # Timepoints
    near_discharge_hours: int = 24
    use_temporal_mode: bool = False
    temporal_days: int = 5
    temporal_step_hours: int = 24

    # Windows
    delta_v_hours: int = 6
    delta_l_hours: int = 24

    # Required fields
    required_vitals: Tuple[str, ...] = ("heart_rate", "respiratory_rate", "spo2", "temperature_c")
    require_sbp_or_map: bool = True
    required_labs: Tuple[str, ...] = ("Potassium", "Sodium", "Glucose", "Lactate", "Hemoglobin")

    # HARD_BARRIER thresholds
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

    # Diagnoses
    max_diagnoses_per_admission: int = 10  # Top-N ICD codes by seq_num
    filter_dx_categories: Optional[List[str]] = None  # e.g. ["heart_failure", "sepsis"]
    filter_dx_match_any_seq: bool = True  # Match any diagnosis position (vs primary only)
    require_known_dx_category: bool = False  # Drop admissions with no matched clinical category

    # Splits
    seed: int = 7
    train_frac: float = 0.7
    val_frac: float = 0.1
    test_frac: float = 0.2

    # Extraction bounds
    pull_days_before_discharge: int = 7


# -----------------------------
# Item mappings
# -----------------------------
# ICU chartevents itemids (commonly used in MIMIC-IV)
VITAL_ITEMIDS = {
    "heart_rate": [220045],
    "respiratory_rate": [220210],
    "spo2": [220277],
    "temperature_c": [223762],
    "temperature_f": [223761],
    "nibp_sbp": [220179],
    "nibp_map": [220181],
    "art_sbp": [220050],
    "art_map": [220052],
}

# Labs (hospital labevents itemids)
LAB_ITEMIDS = {
    "Sodium": [50983],
    "Potassium": [50971],
    "Glucose": [50931],
    "Lactate": [50813],
    "Hemoglobin": [51222],
    # --- Additional labs from extraction pipeline ---
    "Creatinine": [50912],
    "BUN": [51006],
    "Hematocrit": [51221],
    "Platelet_Count": [51265],
    "WBC": [51301],
    "Bicarbonate": [50882],
}

# Worst-direction for vitals
WORST_IS_MIN = {"spo2", "sbp", "map"}
WORST_IS_MAX = {"heart_rate", "respiratory_rate", "temperature_c"}
ITEMID_TO_VITAL = {iid: name for name, ids in VITAL_ITEMIDS.items() for iid in ids}

# Charlson comorbidity component columns
CHARLSON_COMPONENTS = [
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
]

# ---------------------------------------------------------------------------
# Diagnosis category filter – ICD-10 prefix matching
# ---------------------------------------------------------------------------
DIAGNOSIS_CATEGORIES = {
    # 1. Cardiovascular
    "hypertension": ["I10", "I11", "I12", "I13", "I15"],
    "heart_failure": ["I50"],
    "ischemic_heart_disease": ["I20", "I21", "I22", "I24", "I25"],
    "valvular_heart_disease": ["I34", "I35", "I36", "I37"],
    "arrhythmia": ["I47", "I48", "I49"],
    "stroke": ["I60", "I61", "I62", "I63", "I69"],
    # 2. Infectious / Sepsis
    "sepsis": ["A40", "A41", "R65.2"],
    "pneumonia": ["J12", "J13", "J14", "J15", "J16", "J17", "J18", "J69"],
    "uti": ["N39.0", "N30"],
    "skin_infection": ["L03", "L00", "L01", "L02", "L08"],
    # 3. Respiratory
    "respiratory_failure": ["J96"],
    "copd_asthma": ["J44", "J45"],
    # 4. Metabolic / Renal / GI
    "diabetes": ["E08", "E09", "E10", "E11", "E13"],
    "kidney_failure": ["N17", "N18", "N19"],
    "fluid_electrolyte": ["E86", "E87"],
    "gi_bleed": ["K92", "I85"],
    "liver_disease": ["K70", "K71", "K72", "K73", "K74"],
    # 5. Trauma / Other
    "intracranial_injury": ["S06"],
    "fracture": ["S02", "S12", "S22", "S32", "S42", "S52", "S62", "S72", "S82"],
    "substance_abuse": ["F10", "F11", "F12", "F13", "F14", "F15", "F16", "F19"],
    # 6. Obstetrics
    "obstetric": ["O"],
}

# Pre-built reverse index: ICD prefix → set of category names
_PREFIX_TO_CATEGORIES: Dict[str, set] = {}
for _cat, _prefixes in DIAGNOSIS_CATEGORIES.items():
    for _pfx in _prefixes:
        _PREFIX_TO_CATEGORIES.setdefault(_pfx, set()).add(_cat)


def icd_code_matches_categories(
    icd_code: Optional[str],
    target_categories: Optional[set] = None,
    target_prefixes: Optional[List[str]] = None,
) -> bool:
    """
    Check whether an ICD code belongs to any of the requested diagnosis categories.

    Args:
        icd_code: Raw ICD code string (e.g. "I501", "J189", "O80")
        target_categories: Set of category names from DIAGNOSIS_CATEGORIES to accept.
                           If None, no category-based filtering is applied.
        target_prefixes: Flattened list of ICD prefixes derived from target_categories.
                         Pass this for performance when calling in a loop.

    Returns:
        True if the code matches at least one target category.
    """
    if not icd_code:
        return False
    code = str(icd_code).strip().upper().replace(".", "")
    if not code:
        return False

    # Build prefixes on the fly if not pre-computed
    if target_prefixes is None:
        if target_categories is None:
            return True  # no filter
        target_prefixes = []
        for cat in target_categories:
            target_prefixes.extend(DIAGNOSIS_CATEGORIES.get(cat, []))

    # Normalize prefixes (remove dots for matching)
    for pfx in target_prefixes:
        norm_pfx = pfx.replace(".", "").upper()
        if code.startswith(norm_pfx):
            return True
    return False


def resolve_dx_prefixes(category_names: List[str]) -> List[str]:
    """Expand category names to a flat list of ICD prefixes for matching."""
    prefixes = []
    for cat in category_names:
        cat_lower = cat.strip().lower()
        if cat_lower in DIAGNOSIS_CATEGORIES:
            prefixes.extend(DIAGNOSIS_CATEGORIES[cat_lower])
        else:
            # Treat as a raw ICD prefix (allows e.g. --filter-dx-categories I50 J18)
            prefixes.append(cat.strip())
    return prefixes


def get_matching_category_names(icd_code: Optional[str]) -> List[str]:
    """Return all diagnosis category names that match a given ICD code."""
    if not icd_code:
        return []
    code = str(icd_code).strip().upper().replace(".", "")
    matches = []
    for pfx, cats in _PREFIX_TO_CATEGORIES.items():
        norm_pfx = pfx.replace(".", "").upper()
        if code.startswith(norm_pfx):
            matches.extend(cats)
    return sorted(set(matches))


def filter_cohort_by_diagnosis_category(
    cohort: pd.DataFrame,
    diagnoses: pd.DataFrame,
    target_categories: List[str],
    match_any_seq: bool = True,
) -> pd.DataFrame:
    """
    Filter the cohort to only include admissions that have at least one
    diagnosis matching the requested categories.

    Args:
        cohort: DataFrame with hadm_id column.
        diagnoses: DataFrame with hadm_id, icd_code columns (from extract_diagnoses).
        target_categories: List of category names and/or raw ICD prefixes.
        match_any_seq: If True, match on any diagnosis position.
                       If False, match only on primary (seq_num=1).

    Returns:
        Filtered cohort DataFrame.
    """
    if not target_categories:
        return cohort

    prefixes = resolve_dx_prefixes(target_categories)
    if not prefixes:
        return cohort

    if diagnoses is None or diagnoses.empty:
        print("   ⚠️  No diagnoses available for filtering — returning empty cohort")
        return cohort.iloc[:0].copy()

    dx = diagnoses.copy()
    if not match_any_seq:
        dx = dx[dx["seq_num"] == 1]

    # Vectorized prefix matching
    dx["_code_norm"] = dx["icd_code"].astype(str).str.strip().str.upper().str.replace(".", "", regex=False)
    mask = pd.Series(False, index=dx.index)
    for pfx in prefixes:
        norm_pfx = pfx.replace(".", "").upper()
        mask = mask | dx["_code_norm"].str.startswith(norm_pfx)

    matched_hadm_ids = set(dx.loc[mask, "hadm_id"].astype(int).unique())
    filtered = cohort[cohort["hadm_id"].astype(int).isin(matched_hadm_ids)].copy()
    return filtered


# -----------------------------
# Utilities
# -----------------------------
def dt_to_ts(dt: pd.Timestamp) -> pd.Timestamp:
    if isinstance(dt, pd.Timestamp):
        return dt
    return pd.to_datetime(dt)


def safe_float(x) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def make_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def deduplicate_latest_admission_per_subject(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Keep exactly one admission per subject_id:
    latest dischtime, then latest admittime, then largest hadm_id.
    """
    if cohort is None or cohort.empty:
        return cohort.copy()

    ordered = cohort.sort_values(
        ["subject_id", "dischtime", "admittime", "hadm_id"],
        ascending=[True, True, True, True],
    )
    dedup = ordered.groupby("subject_id", as_index=False).tail(1)
    dedup = dedup.sort_values(["subject_id", "admittime", "hadm_id"]).reset_index(drop=True)
    return dedup


def _pick_first_existing_column(columns: List[str], candidates: Tuple[str, ...]) -> Optional[str]:
    cols = set(columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def melt_derived_vitals_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a wide derived-vitals dataframe to:
      hadm_id, charttime, vital_name, valuenum
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["hadm_id", "charttime", "vital_name", "valuenum"])

    col_map = {
        "heart_rate": _pick_first_existing_column(df.columns.tolist(), ("heart_rate", "hr")),
        "respiratory_rate": _pick_first_existing_column(df.columns.tolist(), ("resp_rate", "respiratory_rate", "rr")),
        "spo2": _pick_first_existing_column(df.columns.tolist(), ("spo2", "o2sat", "sao2")),
        "temperature_c": _pick_first_existing_column(df.columns.tolist(), ("temperature_c", "temperature", "temp")),
        "sbp": _pick_first_existing_column(df.columns.tolist(), ("sbp", "systolic_bp")),
        "map": _pick_first_existing_column(
            df.columns.tolist(), ("mbp", "map", "mean_bp", "mean_arterial_pressure")
        ),
    }

    parts: List[pd.DataFrame] = []
    for vname, src_col in col_map.items():
        if src_col is None:
            continue
        part = df[["hadm_id", "charttime", src_col]].copy()
        part = part.rename(columns={src_col: "valuenum"})
        part["vital_name"] = vname
        part["valuenum"] = pd.to_numeric(part["valuenum"], errors="coerce")
        part = part.dropna(subset=["valuenum"])
        if not part.empty:
            parts.append(part[["hadm_id", "charttime", "vital_name", "valuenum"]])

    if not parts:
        return pd.DataFrame(columns=["hadm_id", "charttime", "vital_name", "valuenum"])

    out = pd.concat(parts, axis=0, ignore_index=True)
    out["hadm_id"] = out["hadm_id"].astype(int)
    out["charttime"] = pd.to_datetime(out["charttime"])
    out["valuenum"] = out["valuenum"].astype(float)
    return out


def merge_vitals_primary_with_fallback(primary_vitals: pd.DataFrame, fallback_vitals: pd.DataFrame) -> pd.DataFrame:
    """
    Use primary vitals when an admission has any primary rows in pull window.
    Use fallback rows only for admissions absent from primary rows.
    """
    empty = pd.DataFrame(columns=["hadm_id", "charttime", "vital_name", "valuenum"])
    if (primary_vitals is None or primary_vitals.empty) and (fallback_vitals is None or fallback_vitals.empty):
        return empty
    if primary_vitals is None or primary_vitals.empty:
        return fallback_vitals.copy()
    if fallback_vitals is None or fallback_vitals.empty:
        return primary_vitals.copy()

    covered_hadm = set(primary_vitals["hadm_id"].astype(int).unique().tolist())
    fb = fallback_vitals[~fallback_vitals["hadm_id"].astype(int).isin(covered_hadm)].copy()
    if fb.empty:
        return primary_vitals.copy()
    return pd.concat([primary_vitals, fb], axis=0, ignore_index=True)


# -----------------------------
# BigQuery extraction
# -----------------------------
class MIMICExtractor:
    def __init__(self, cfg: Config):
        if bigquery is None:
            raise ImportError("google-cloud-bigquery is required to run this builder.")
        self.cfg = cfg
        self.client = bigquery.Client(project=cfg.project_id)

    def _run(self, sql: str, job_config: Optional[bigquery.QueryJobConfig] = None) -> pd.DataFrame:
        job = self.client.query(sql, job_config=job_config)
        return job.result().to_dataframe(create_bqstorage_client=True)

    # -----------------------------------------------------------------
    # Cohort
    # -----------------------------------------------------------------
    def extract_cohort(self) -> pd.DataFrame:
        hospice_filter = ""
        if self.cfg.exclude_hospice:
            hospice_filter = "AND UPPER(a.discharge_location) NOT LIKE '%HOSPICE%'"

        transfer_filter = ""
        if self.cfg.exclude_transfers:
            transfer_filter = """
            AND UPPER(a.discharge_location) NOT LIKE '%TRANSFER%'
            AND UPPER(a.discharge_location) NOT LIKE '%HOSPITAL%'
            """

        limit_clause = f"LIMIT {self.cfg.max_admissions}" if self.cfg.max_admissions else ""

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
        FROM `{self.cfg.dataset}_hosp.patients` p
        JOIN `{self.cfg.dataset}_hosp.admissions` a
          ON p.subject_id = a.subject_id
        WHERE a.admittime IS NOT NULL
          AND a.dischtime IS NOT NULL
          AND a.hospital_expire_flag = 0
          AND (p.anchor_age + DATETIME_DIFF(
                a.admittime,
                DATETIME(p.anchor_year, 1, 1, 0, 0, 0),
                YEAR
              )) >= {self.cfg.min_age}
          {hospice_filter}
          {transfer_filter}
        ORDER BY p.subject_id, a.admittime
        {limit_clause}
        """
        df = self._run(sql)
        df["admittime"] = pd.to_datetime(df["admittime"])
        df["dischtime"] = pd.to_datetime(df["dischtime"])
        df["deathtime"] = pd.to_datetime(df["deathtime"], errors="coerce")
        df["dod"] = pd.to_datetime(df["dod"], errors="coerce")
        return df

    # -----------------------------------------------------------------
    # ICU stays
    # -----------------------------------------------------------------
    def extract_icu_stays_for_cohort(self, cohort: pd.DataFrame) -> pd.DataFrame:
        hadm_ids = cohort["hadm_id"].astype(int).unique().tolist()
        if not hadm_ids:
            return pd.DataFrame()

        sql = f"""
        SELECT
            subject_id, hadm_id, stay_id, first_careunit, last_careunit, intime, outtime
        FROM `{self.cfg.dataset}_icu.icustays`
        WHERE hadm_id IN UNNEST(@hadm_ids)
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("hadm_ids", "INT64", hadm_ids)]
        )
        df = self._run(sql, job_config=job_config)
        if df.empty:
            return df
        df["intime"] = pd.to_datetime(df["intime"])
        df["outtime"] = pd.to_datetime(df["outtime"])
        return df

    # -----------------------------------------------------------------
    # Derived vitals (hospital-wide)
    # -----------------------------------------------------------------
    def extract_vitals_derived(self, cohort: pd.DataFrame) -> pd.DataFrame:
        hadm_ids = cohort["hadm_id"].astype(int).unique().tolist()
        if not hadm_ids:
            return pd.DataFrame(columns=["hadm_id", "charttime", "vital_name", "valuenum"])

        pull_days = int(self.cfg.pull_days_before_discharge)

        sql = f"""
        WITH coh AS (
          SELECT subject_id, hadm_id, admittime, dischtime
          FROM `{self.cfg.dataset}_hosp.admissions`
          WHERE hadm_id IN UNNEST(@hadm_ids)
        )
        SELECT
          coh.hadm_id,
          vs.*
        FROM `{self.cfg.derived_vitals_table}` vs
        JOIN coh
          ON vs.subject_id = coh.subject_id
         AND vs.charttime BETWEEN coh.admittime AND coh.dischtime
         AND vs.charttime BETWEEN DATETIME_SUB(coh.dischtime, INTERVAL {pull_days} DAY) AND coh.dischtime
        WHERE vs.charttime IS NOT NULL
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("hadm_ids", "INT64", hadm_ids)]
        )
        df = self._run(sql, job_config=job_config)
        if df.empty:
            return pd.DataFrame(columns=["hadm_id", "charttime", "vital_name", "valuenum"])

        if "hadm_id_1" in df.columns and "hadm_id" in df.columns:
            df["hadm_id"] = df["hadm_id"]
        elif "hadm_id" not in df.columns:
            candidate = _pick_first_existing_column(df.columns.tolist(), ("coh_hadm_id", "hadm_id_1"))
            if candidate is not None:
                df["hadm_id"] = df[candidate]

        if "charttime" not in df.columns:
            return pd.DataFrame(columns=["hadm_id", "charttime", "vital_name", "valuenum"])

        df["charttime"] = pd.to_datetime(df["charttime"])
        df["hadm_id"] = pd.to_numeric(df["hadm_id"], errors="coerce")
        df = df[df["hadm_id"].notna()].copy()
        if df.empty:
            return pd.DataFrame(columns=["hadm_id", "charttime", "vital_name", "valuenum"])

        df["hadm_id"] = df["hadm_id"].astype(int)
        return melt_derived_vitals_to_long(df)

    # -----------------------------------------------------------------
    # ICU chartevents vitals (fallback)
    # -----------------------------------------------------------------
    def extract_vitals_chartevents(self, icu_stays: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
        if icu_stays.empty:
            return pd.DataFrame(columns=["hadm_id", "charttime", "vital_name", "valuenum"])

        stay_ids = icu_stays["stay_id"].astype(int).unique().tolist()
        if not stay_ids:
            return pd.DataFrame(columns=["hadm_id", "charttime", "vital_name", "valuenum"])

        itemids = sorted({iid for lst in VITAL_ITEMIDS.values() for iid in lst})
        pull_days = int(self.cfg.pull_days_before_discharge)

        sql = f"""
        WITH coh AS (
          SELECT hadm_id, dischtime
          FROM `{self.cfg.dataset}_hosp.admissions`
          WHERE hadm_id IN UNNEST(@hadm_ids)
        )
        SELECT
          icu.hadm_id,
          ce.charttime,
          ce.itemid,
          ce.valuenum
        FROM `{self.cfg.dataset}_icu.chartevents` ce
        JOIN `{self.cfg.dataset}_icu.icustays` icu
          ON ce.stay_id = icu.stay_id
        JOIN coh
          ON icu.hadm_id = coh.hadm_id
        WHERE ce.stay_id IN UNNEST(@stay_ids)
          AND ce.itemid IN UNNEST(@itemids)
          AND ce.valuenum IS NOT NULL
          AND ce.charttime IS NOT NULL
          AND ce.charttime BETWEEN DATETIME_SUB(coh.dischtime, INTERVAL {pull_days} DAY) AND coh.dischtime
        """
        hadm_ids = cohort["hadm_id"].astype(int).unique().tolist()
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("stay_ids", "INT64", stay_ids),
                bigquery.ArrayQueryParameter("itemids", "INT64", itemids),
                bigquery.ArrayQueryParameter("hadm_ids", "INT64", hadm_ids),
            ]
        )
        df = self._run(sql, job_config=job_config)
        if df.empty:
            return pd.DataFrame(columns=["hadm_id", "charttime", "vital_name", "valuenum"])

        df["charttime"] = pd.to_datetime(df["charttime"])
        df["hadm_id"] = pd.to_numeric(df["hadm_id"], errors="coerce")
        df["itemid"] = pd.to_numeric(df["itemid"], errors="coerce")
        df["valuenum"] = pd.to_numeric(df["valuenum"], errors="coerce")
        df = df.dropna(subset=["hadm_id", "itemid", "valuenum", "charttime"]).copy()
        if df.empty:
            return pd.DataFrame(columns=["hadm_id", "charttime", "vital_name", "valuenum"])

        df["hadm_id"] = df["hadm_id"].astype(int)
        df["itemid"] = df["itemid"].astype(int)
        df["vital_name"] = df["itemid"].map(lambda x: ITEMID_TO_VITAL.get(int(x)))
        df = df[df["vital_name"].notna()].copy()
        if df.empty:
            return pd.DataFrame(columns=["hadm_id", "charttime", "vital_name", "valuenum"])

        return df[["hadm_id", "charttime", "vital_name", "valuenum"]]

    # -----------------------------------------------------------------
    # Labs
    # -----------------------------------------------------------------
    def extract_labs_labevents(self, cohort: pd.DataFrame) -> pd.DataFrame:
        hadm_ids = cohort["hadm_id"].astype(int).unique().tolist()
        if not hadm_ids:
            return pd.DataFrame()

        itemids = sorted({iid for lst in LAB_ITEMIDS.values() for iid in lst})
        pull_days = int(self.cfg.pull_days_before_discharge)

        sql = f"""
        WITH coh AS (
          SELECT hadm_id, dischtime
          FROM `{self.cfg.dataset}_hosp.admissions`
          WHERE hadm_id IN UNNEST(@hadm_ids)
        )
        SELECT
          le.hadm_id,
          le.charttime,
          le.itemid,
          le.valuenum
        FROM `{self.cfg.dataset}_hosp.labevents` le
        JOIN coh
          ON le.hadm_id = coh.hadm_id
        WHERE le.hadm_id IN UNNEST(@hadm_ids)
          AND le.itemid IN UNNEST(@itemids)
          AND le.valuenum IS NOT NULL
          AND le.charttime IS NOT NULL
          AND le.charttime BETWEEN DATETIME_SUB(coh.dischtime, INTERVAL {pull_days} DAY) AND coh.dischtime
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("hadm_ids", "INT64", hadm_ids),
                bigquery.ArrayQueryParameter("itemids", "INT64", itemids),
            ]
        )
        df = self._run(sql, job_config=job_config)
        if df.empty:
            return df
        df["charttime"] = pd.to_datetime(df["charttime"])
        df["hadm_id"] = df["hadm_id"].astype(int)
        df["itemid"] = df["itemid"].astype(int)
        return df

    # -----------------------------------------------------------------
    # Diagnoses (NEW - from extraction pipeline)
    # -----------------------------------------------------------------
    def extract_diagnoses(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """
        Extract ICD diagnoses for the cohort, joined with d_icd_diagnoses
        for human-readable titles.

        Output columns: hadm_id, seq_num, icd_code, icd_version, long_title
        """
        hadm_ids = cohort["hadm_id"].astype(int).unique().tolist()
        if not hadm_ids:
            return pd.DataFrame(
                columns=["hadm_id", "seq_num", "icd_code", "icd_version", "long_title"]
            )

        sql = f"""
        SELECT
            d.hadm_id,
            d.seq_num,
            d.icd_code,
            d.icd_version,
            di.long_title
        FROM `{self.cfg.dataset}_hosp.diagnoses_icd` d
        LEFT JOIN `{self.cfg.dataset}_hosp.d_icd_diagnoses` di
            ON d.icd_code = di.icd_code AND d.icd_version = di.icd_version
        WHERE d.hadm_id IN UNNEST(@hadm_ids)
        ORDER BY d.hadm_id, d.seq_num
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("hadm_ids", "INT64", hadm_ids)]
        )
        df = self._run(sql, job_config=job_config)
        if df.empty:
            return pd.DataFrame(
                columns=["hadm_id", "seq_num", "icd_code", "icd_version", "long_title"]
            )
        df["hadm_id"] = df["hadm_id"].astype(int)
        df["seq_num"] = pd.to_numeric(df["seq_num"], errors="coerce")
        return df

    # -----------------------------------------------------------------
    # Charlson comorbidity (NEW - from extraction pipeline)
    # -----------------------------------------------------------------
    def extract_charlson_comorbidity(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """
        Extract Charlson comorbidity index and individual component flags
        from the MIMIC-IV derived.charlson table.

        Output: one row per hadm_id with charlson_comorbidity_index + 17 component columns.
        """
        hadm_ids = cohort["hadm_id"].astype(int).unique().tolist()
        if not hadm_ids:
            cols = ["hadm_id", "charlson_comorbidity_index"] + CHARLSON_COMPONENTS
            return pd.DataFrame(columns=cols)

        component_cols = ",\n            ".join(CHARLSON_COMPONENTS)

        sql = f"""
        SELECT
            hadm_id,
            charlson_comorbidity_index,
            {component_cols}
        FROM `{self.cfg.derived_dataset}.charlson`
        WHERE hadm_id IN UNNEST(@hadm_ids)
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("hadm_ids", "INT64", hadm_ids)]
        )
        df = self._run(sql, job_config=job_config)
        if df.empty:
            cols = ["hadm_id", "charlson_comorbidity_index"] + CHARLSON_COMPONENTS
            return pd.DataFrame(columns=cols)
        df["hadm_id"] = df["hadm_id"].astype(int)
        return df

    # -----------------------------------------------------------------
    # SOFA scores (NEW - from extraction pipeline)
    # -----------------------------------------------------------------
    def extract_sofa_scores(self, icu_stays: pd.DataFrame) -> pd.DataFrame:
        """
        Extract SOFA scores from the derived.sofa table for all ICU stays.

        Output columns: stay_id, hadm_id, starttime, endtime,
                        sofa_score, sofa_resp, sofa_coag, sofa_liver,
                        sofa_cardio, sofa_cns, sofa_renal
        """
        if icu_stays is None or icu_stays.empty:
            return pd.DataFrame(columns=[
                "stay_id", "hadm_id", "starttime", "endtime",
                "sofa_score", "sofa_resp", "sofa_coag", "sofa_liver",
                "sofa_cardio", "sofa_cns", "sofa_renal",
            ])

        stay_ids = icu_stays["stay_id"].astype(int).unique().tolist()
        if not stay_ids:
            return pd.DataFrame(columns=[
                "stay_id", "hadm_id", "starttime", "endtime",
                "sofa_score", "sofa_resp", "sofa_coag", "sofa_liver",
                "sofa_cardio", "sofa_cns", "sofa_renal",
            ])

        sql = f"""
        SELECT
            s.stay_id,
            icu.hadm_id,
            s.starttime,
            s.endtime,
            s.sofa_24hours AS sofa_score,
            s.respiration_24hours AS sofa_resp,
            s.coagulation_24hours AS sofa_coag,
            s.liver_24hours AS sofa_liver,
            s.cardiovascular_24hours AS sofa_cardio,
            s.cns_24hours AS sofa_cns,
            s.renal_24hours AS sofa_renal
        FROM `{self.cfg.derived_dataset}.sofa` s
        JOIN `{self.cfg.dataset}_icu.icustays` icu
          ON s.stay_id = icu.stay_id
        WHERE s.stay_id IN UNNEST(@stay_ids)
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("stay_ids", "INT64", stay_ids)]
        )
        df = self._run(sql, job_config=job_config)
        if df.empty:
            return pd.DataFrame(columns=[
                "stay_id", "hadm_id", "starttime", "endtime",
                "sofa_score", "sofa_resp", "sofa_coag", "sofa_liver",
                "sofa_cardio", "sofa_cns", "sofa_renal",
            ])
        df["starttime"] = pd.to_datetime(df["starttime"])
        df["endtime"] = pd.to_datetime(df["endtime"])
        df["hadm_id"] = df["hadm_id"].astype(int)
        df["stay_id"] = df["stay_id"].astype(int)
        return df

    # -----------------------------------------------------------------
    # Readmissions
    # -----------------------------------------------------------------
    def extract_readmissions(self, cohort: pd.DataFrame) -> pd.DataFrame:
        subject_ids = cohort["subject_id"].astype(int).unique().tolist()
        if not subject_ids:
            return pd.DataFrame()

        sql = f"""
        WITH idx AS (
          SELECT subject_id, hadm_id, dischtime
          FROM `{self.cfg.dataset}_hosp.admissions`
          WHERE subject_id IN UNNEST(@subject_ids)
        ),
        pairs AS (
          SELECT
            a1.hadm_id AS index_hadm_id,
            a1.subject_id,
            a1.dischtime AS index_dischtime,
            a2.hadm_id AS readmit_hadm_id,
            a2.admittime AS readmit_admittime,
            DATETIME_DIFF(a2.admittime, a1.dischtime, HOUR) AS hours_to_readmit,
            ROW_NUMBER() OVER (PARTITION BY a1.hadm_id ORDER BY a2.admittime) AS rn
          FROM idx a1
          JOIN `{self.cfg.dataset}_hosp.admissions` a2
            ON a1.subject_id = a2.subject_id
           AND a2.admittime > a1.dischtime
        )
        SELECT
          index_hadm_id AS hadm_id,
          readmit_hadm_id,
          hours_to_readmit,
          CASE WHEN hours_to_readmit <= 24*7 THEN 1 ELSE 0 END AS readmit_7d,
          CASE WHEN hours_to_readmit <= 72 THEN 1 ELSE 0 END AS readmit_72h,
          CASE WHEN hours_to_readmit <= 24*30 THEN 1 ELSE 0 END AS readmit_30d
        FROM pairs
        WHERE rn = 1
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("subject_ids", "INT64", subject_ids)]
        )
        df = self._run(sql, job_config=job_config)
        if df.empty:
            return df
        df["hadm_id"] = df["hadm_id"].astype(int)
        df = df.set_index("hadm_id", drop=True)
        return df


# -----------------------------
# Snapshot building
# -----------------------------
class SnapshotBuilder:
    def __init__(
        self,
        cfg: Config,
        cohort: pd.DataFrame,
        icu_stays: pd.DataFrame,
        vitals: pd.DataFrame,
        labs: pd.DataFrame,
        readmit: pd.DataFrame,
        diagnoses: pd.DataFrame,
        charlson: pd.DataFrame,
        sofa: pd.DataFrame,
    ):
        self.cfg = cfg
        self.cohort = cohort.copy()

        self.icu_intervals_by_hadm: Dict[int, List[Tuple[pd.Timestamp, pd.Timestamp, int]]] = {}
        self._index_icu_intervals(icu_stays)

        self.vitals_by_hadm = vitals.groupby("hadm_id") if not vitals.empty else None
        self.labs_by_hadm = labs.groupby("hadm_id") if not labs.empty else None
        self.readmit = readmit if readmit is not None else pd.DataFrame()

        self.itemid_to_lab = self._build_itemid_to_lab()

        # --- New clinical data indexes ---
        self.diagnoses_by_hadm = self._index_diagnoses(diagnoses)
        self.charlson_by_hadm = self._index_charlson(charlson)
        self.sofa_by_hadm = self._index_sofa(sofa)

    def _index_icu_intervals(self, icu_stays: pd.DataFrame) -> None:
        by_hadm: Dict[int, List[Tuple[pd.Timestamp, pd.Timestamp, int]]] = {}
        if icu_stays is None or icu_stays.empty:
            self.icu_intervals_by_hadm = by_hadm
            return
        for _, r in icu_stays.iterrows():
            hadm_id = int(r["hadm_id"])
            intime = dt_to_ts(r["intime"])
            outtime = dt_to_ts(r["outtime"])
            stay_id = int(r["stay_id"])
            by_hadm.setdefault(hadm_id, []).append((intime, outtime, stay_id))
        self.icu_intervals_by_hadm = by_hadm

    def _build_itemid_to_lab(self) -> Dict[int, str]:
        m = {}
        for name, ids in LAB_ITEMIDS.items():
            for iid in ids:
                m[iid] = name
        return m

    def _index_diagnoses(self, diagnoses: pd.DataFrame) -> Dict[int, List[Dict[str, Any]]]:
        """Group diagnoses by hadm_id, keeping top-N by seq_num."""
        out: Dict[int, List[Dict[str, Any]]] = {}
        if diagnoses is None or diagnoses.empty:
            return out
        for hadm_id, grp in diagnoses.groupby("hadm_id"):
            sorted_grp = grp.sort_values("seq_num").head(self.cfg.max_diagnoses_per_admission)
            dx_list = []
            for _, row in sorted_grp.iterrows():
                dx_list.append({
                    "seq_num": int(row["seq_num"]) if pd.notna(row["seq_num"]) else None,
                    "icd_code": str(row["icd_code"]) if pd.notna(row.get("icd_code")) else None,
                    "icd_version": int(row["icd_version"]) if pd.notna(row.get("icd_version")) else None,
                    "long_title": str(row["long_title"]) if pd.notna(row.get("long_title")) else None,
                })
            out[int(hadm_id)] = dx_list
        return out

    def _index_charlson(self, charlson: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        """Index Charlson data by hadm_id as a dict of {component: value}."""
        out: Dict[int, Dict[str, Any]] = {}
        if charlson is None or charlson.empty:
            return out
        for _, row in charlson.iterrows():
            hadm_id = int(row["hadm_id"])
            entry: Dict[str, Any] = {
                "charlson_comorbidity_index": safe_float(row.get("charlson_comorbidity_index")),
            }
            for comp in CHARLSON_COMPONENTS:
                val = row.get(comp)
                entry[comp] = int(val) if pd.notna(val) else None
            out[hadm_id] = entry
        return out

    def _index_sofa(self, sofa: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """Group SOFA rows by hadm_id for time-based lookup."""
        out: Dict[int, pd.DataFrame] = {}
        if sofa is None or sofa.empty:
            return out
        for hadm_id, grp in sofa.groupby("hadm_id"):
            out[int(hadm_id)] = grp.sort_values("starttime").copy()
        return out

    # -----------------------------------------------------------------
    # Time-based lookups
    # -----------------------------------------------------------------
    def icu_at_time(self, hadm_id: int, t: pd.Timestamp) -> Tuple[bool, Optional[int]]:
        intervals = self.icu_intervals_by_hadm.get(int(hadm_id), [])
        for intime, outtime, stay_id in intervals:
            if pd.notna(intime) and pd.notna(outtime) and intime <= t <= outtime:
                return True, stay_id
        return False, None

    def _get_labs_window(self, hadm_id: int, t: pd.Timestamp) -> Dict[str, List[Dict[str, Any]]]:
        """Return all lab measurements in (t - delta_l, t] grouped by lab name, sorted by time."""
        out: Dict[str, List[Dict[str, Any]]] = {}
        if self.labs_by_hadm is None:
            return out
        if hadm_id not in self.labs_by_hadm.groups:
            return out

        df = self.labs_by_hadm.get_group(hadm_id)
        w0 = t - timedelta(hours=self.cfg.delta_l_hours)
        w = df[(df["charttime"] > w0) & (df["charttime"] <= t)]
        if w.empty:
            return out

        w = w.sort_values("charttime")
        for _, row in w.iterrows():
            lab_name = self.itemid_to_lab.get(int(row["itemid"]))
            if not lab_name:
                continue
            val = safe_float(row["valuenum"])
            if val is None:
                continue
            out.setdefault(lab_name, []).append({
                "value": val,
                "time": row["charttime"].isoformat(),
            })
        return out

    def _get_vitals_window(self, hadm_id: int, t: pd.Timestamp) -> Dict[str, List[Dict[str, Any]]]:
        """Return all vital measurements in (t - delta_v, t] grouped by raw vital name, sorted by time."""
        out: Dict[str, List[Dict[str, Any]]] = {}
        if self.vitals_by_hadm is None:
            return out
        if hadm_id not in self.vitals_by_hadm.groups:
            return out

        df = self.vitals_by_hadm.get_group(hadm_id)
        w0 = t - timedelta(hours=self.cfg.delta_v_hours)
        w = df[(df["charttime"] > w0) & (df["charttime"] <= t)]
        if w.empty:
            return out

        w = w.copy()
        w = w[w["vital_name"].notna()]
        if w.empty:
            return out

        w = w.sort_values("charttime")
        for _, row in w.iterrows():
            vname = row["vital_name"]
            val = safe_float(row["valuenum"])
            if val is None:
                continue
            out.setdefault(vname, []).append({
                "value": val,
                "time": row["charttime"].isoformat(),
            })
        return out

    def _get_sofa_at_time(self, hadm_id: int, t: pd.Timestamp) -> Dict[str, Optional[float]]:
        """Get most recent SOFA score at or before time t."""
        sofa_df = self.sofa_by_hadm.get(int(hadm_id))
        if sofa_df is None or sofa_df.empty:
            return {
                "sofa_score": None,
                "sofa_resp": None,
                "sofa_coag": None,
                "sofa_liver": None,
                "sofa_cardio": None,
                "sofa_cns": None,
                "sofa_renal": None,
            }

        # Most recent SOFA window that started at or before t
        eligible = sofa_df[sofa_df["starttime"] <= t]
        if eligible.empty:
            return {
                "sofa_score": None,
                "sofa_resp": None,
                "sofa_coag": None,
                "sofa_liver": None,
                "sofa_cardio": None,
                "sofa_cns": None,
                "sofa_renal": None,
            }

        latest = eligible.iloc[-1]
        return {
            "sofa_score": safe_float(latest.get("sofa_score")),
            "sofa_resp": safe_float(latest.get("sofa_resp")),
            "sofa_coag": safe_float(latest.get("sofa_coag")),
            "sofa_liver": safe_float(latest.get("sofa_liver")),
            "sofa_cardio": safe_float(latest.get("sofa_cardio")),
            "sofa_cns": safe_float(latest.get("sofa_cns")),
            "sofa_renal": safe_float(latest.get("sofa_renal")),
        }

    # -----------------------------------------------------------------
    # Normalization & checks
    # -----------------------------------------------------------------
    def _normalize_vitals_window(
        self, raw: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Merge raw vital channels into protocol-level names, each as a time series.
        - sbp: merge sbp + nibp_sbp + art_sbp
        - map: merge map + nibp_map + art_map
        - temperature_c: merge temperature_c + converted temperature_f
        - heart_rate, respiratory_rate, spo2: pass through
        All channels sorted by time.
        """
        out: Dict[str, List[Dict[str, Any]]] = {}

        # SBP: merge all systolic sources
        sbp_points: List[Dict[str, Any]] = []
        for key in ("sbp", "nibp_sbp", "art_sbp"):
            sbp_points.extend(raw.get(key, []))
        if sbp_points:
            out["sbp"] = sorted(sbp_points, key=lambda p: p["time"])

        # MAP: merge all mean pressure sources
        map_points: List[Dict[str, Any]] = []
        for key in ("map", "nibp_map", "art_map"):
            map_points.extend(raw.get(key, []))
        if map_points:
            out["map"] = sorted(map_points, key=lambda p: p["time"])

        # Direct pass-through
        for key in ("heart_rate", "respiratory_rate", "spo2"):
            if key in raw and raw[key]:
                out[key] = raw[key]  # already sorted by time from _get_vitals_window

        # Temperature: merge temperature_c + converted temperature_f
        temp_points: List[Dict[str, Any]] = []
        for pt in raw.get("temperature_c", []):
            val = pt["value"]
            # Sanity: if value > 80, it's likely Fahrenheit mis-labeled
            if val is not None and val > 80:
                val = (val - 32.0) * (5.0 / 9.0)
            temp_points.append({"value": val, "time": pt["time"]})
        for pt in raw.get("temperature", []):
            val = pt["value"]
            if val is not None and val > 80:
                val = (val - 32.0) * (5.0 / 9.0)
            temp_points.append({"value": val, "time": pt["time"]})
        for pt in raw.get("temperature_f", []):
            val = pt["value"]
            if val is not None:
                val = (val - 32.0) * (5.0 / 9.0)
            temp_points.append({"value": val, "time": pt["time"]})
        if temp_points:
            out["temperature_c"] = sorted(temp_points, key=lambda p: p["time"])

        return out

    def _ensure_core_keys(
        self,
        vitals: Dict[str, List[Dict[str, Any]]],
        labs: Dict[str, List[Dict[str, Any]]],
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
        """Ensure all required keys exist (as empty lists if missing)."""
        out_vitals = dict(vitals)
        out_labs = dict(labs)

        for v in self.cfg.required_vitals:
            out_vitals.setdefault(v, [])
        out_vitals.setdefault("sbp", [])
        out_vitals.setdefault("map", [])

        for lab in self.cfg.required_labs:
            out_labs.setdefault(lab, [])
        return out_vitals, out_labs

    def _has_required_fields(
        self,
        vitals: Dict[str, List[Dict[str, Any]]],
        labs: Dict[str, List[Dict[str, Any]]],
    ) -> bool:
        """Check whether all required vitals and labs have at least one measurement."""
        for v in self.cfg.required_vitals:
            if not vitals.get(v):
                return False
        if self.cfg.require_sbp_or_map:
            if not vitals.get("sbp") and not vitals.get("map"):
                return False
        for lab in self.cfg.required_labs:
            if not labs.get(lab):
                return False
        return True

    # -----------------------------------------------------------------
    # Label computation
    # -----------------------------------------------------------------
    def _compute_adverse_flags(self, admission_row: pd.Series) -> Tuple[bool, bool, bool]:
        """Returns (adverse_7d, adverse_72h, readmit_30d)."""
        hadm_id = int(admission_row["hadm_id"])
        dischtime = dt_to_ts(admission_row["dischtime"])

        readmit_7d = False
        readmit_72h = False
        readmit_30d = False
        if self.readmit is not None and not self.readmit.empty and hadm_id in self.readmit.index:
            r = self.readmit.loc[hadm_id]
            readmit_7d = bool(int(r.get("readmit_7d", 0)) == 1)
            readmit_72h = bool(int(r.get("readmit_72h", 0)) == 1)
            readmit_30d = bool(int(r.get("readmit_30d", 0)) == 1)

        death_7d = False
        death_72h = False
        dod = admission_row.get("dod", pd.NaT)
        if pd.notna(dod):
            dod_dt = dt_to_ts(dod)
            discharge_date = dischtime.normalize()
            dod_date = dod_dt.normalize()
            day_delta = (dod_date - discharge_date).days
            if 0 <= day_delta <= 7:
                death_7d = True
            if 0 <= day_delta <= 3:
                death_72h = True
        else:
            deathtime = admission_row.get("deathtime", pd.NaT)
            if pd.notna(deathtime):
                dt = dt_to_ts(deathtime)
                hours_after = (dt - dischtime).total_seconds() / 3600.0
                if 0 <= hours_after <= 24 * 7:
                    death_7d = True
                if 0 <= hours_after <= 72:
                    death_72h = True

        return (readmit_7d or death_7d), (readmit_72h or death_72h), readmit_30d

    def _compute_hard_barrier(
        self,
        icu_t: bool,
        vitals: Dict[str, List[Dict[str, Any]]],
        labs: Dict[str, List[Dict[str, Any]]],
    ) -> bool:
        if icu_t:
            return True

        def _worst_min(series: List[Dict[str, Any]]) -> Optional[float]:
            """Worst = minimum (for sbp, map, spo2)."""
            vals = [p["value"] for p in series if p.get("value") is not None]
            return min(vals) if vals else None

        def _worst_max(series: List[Dict[str, Any]]) -> Optional[float]:
            """Worst = maximum (for hr, rr, temp)."""
            vals = [p["value"] for p in series if p.get("value") is not None]
            return max(vals) if vals else None

        sbp = _worst_min(vitals.get("sbp", []))
        mapv = _worst_min(vitals.get("map", []))
        spo2 = _worst_min(vitals.get("spo2", []))
        rr = _worst_max(vitals.get("respiratory_rate", []))
        hr = _worst_max(vitals.get("heart_rate", []))
        temp_c = _worst_max(vitals.get("temperature_c", []))

        crit_vitals = False
        if sbp is not None and sbp < self.cfg.hb_sbp_lt:
            crit_vitals = True
        if mapv is not None and mapv < self.cfg.hb_map_lt:
            crit_vitals = True
        if spo2 is not None and spo2 < self.cfg.hb_spo2_lt:
            crit_vitals = True
        if rr is not None and rr > self.cfg.hb_rr_gt:
            crit_vitals = True
        if hr is not None and hr > self.cfg.hb_hr_gt:
            crit_vitals = True
        if temp_c is not None and temp_c >= self.cfg.hb_temp_c_ge:
            crit_vitals = True

        # For labs, check the most extreme value in the window
        def _any_val(series: List[Dict[str, Any]]) -> List[float]:
            return [p["value"] for p in series if p.get("value") is not None]

        k_vals = _any_val(labs.get("Potassium", []))
        na_vals = _any_val(labs.get("Sodium", []))
        glu_vals = _any_val(labs.get("Glucose", []))
        lact_vals = _any_val(labs.get("Lactate", []))
        hgb_vals = _any_val(labs.get("Hemoglobin", []))

        crit_labs = False
        if k_vals and (max(k_vals) >= self.cfg.hb_k_ge or min(k_vals) <= self.cfg.hb_k_le):
            crit_labs = True
        if na_vals and (max(na_vals) >= self.cfg.hb_na_ge or min(na_vals) <= self.cfg.hb_na_le):
            crit_labs = True
        if glu_vals and (max(glu_vals) >= self.cfg.hb_glu_ge or min(glu_vals) <= self.cfg.hb_glu_le):
            crit_labs = True
        if lact_vals and max(lact_vals) >= self.cfg.hb_lact_ge:
            crit_labs = True
        if hgb_vals and min(hgb_vals) <= self.cfg.hb_hgb_le:
            crit_labs = True

        return bool(crit_vitals or crit_labs)

    def _timepoints_for_admission(self, dischtime: pd.Timestamp) -> List[pd.Timestamp]:
        t_primary = dischtime - timedelta(hours=self.cfg.near_discharge_hours)
        tps = [t_primary]

        if self.cfg.use_temporal_mode:
            start = dischtime - timedelta(days=self.cfg.temporal_days)
            step = timedelta(hours=self.cfg.temporal_step_hours)
            t = start
            while t < dischtime:
                tps.append(t)
                t += step

        tps = sorted(set(tps))
        tps = [t for t in tps if t <= dischtime]
        return tps

    # -----------------------------------------------------------------
    # Main build loop
    # -----------------------------------------------------------------
    def build(self) -> Tuple[pd.DataFrame, Dict]:
        rows: List[Dict] = []

        missing_core = 0
        built = 0

        for _, adm in tqdm(self.cohort.iterrows(), total=len(self.cohort), desc="Building snapshots"):
            hadm_id = int(adm["hadm_id"])
            subject_id = int(adm["subject_id"])
            dischtime = dt_to_ts(adm["dischtime"])

            tps = self._timepoints_for_admission(dischtime)

            # Admission-level features (same across all timepoints)
            diagnoses_list = self.diagnoses_by_hadm.get(hadm_id, [])
            charlson_dict = self.charlson_by_hadm.get(hadm_id, None)

            # Compute matched diagnosis categories for this admission
            matched_dx_cats: List[str] = []
            for dx in diagnoses_list:
                matched_dx_cats.extend(get_matching_category_names(dx.get("icd_code")))
            matched_dx_cats = sorted(set(matched_dx_cats))

            for t in tps:
                icu_t, _ = self.icu_at_time(hadm_id, t)

                # Labs (full time series in window)
                labs = self._get_labs_window(hadm_id, t)

                # Vitals (full time series in window, then normalize channels)
                vitals_raw = self._get_vitals_window(hadm_id, t)
                vitals = self._normalize_vitals_window(vitals_raw)
                vitals, labs = self._ensure_core_keys(vitals, labs)

                core_fields_complete = self._has_required_fields(vitals, labs)
                if not core_fields_complete:
                    missing_core += 1

                adverse_7d, adverse_72h, readmit_30d = self._compute_adverse_flags(adm)

                discharged_in_24h = (dischtime > t) and (dischtime <= (t + timedelta(hours=24)))
                safe_d24 = bool(discharged_in_24h and (not adverse_7d))

                hard_barrier = self._compute_hard_barrier(icu_t, vitals, labs)

                # SOFA at time t
                sofa_at_t = self._get_sofa_at_time(hadm_id, t)

                row = {
                    "subject_id": subject_id,
                    "hadm_id": hadm_id,
                    "t": t.isoformat(),
                    "dischtime": dischtime.isoformat(),
                    "admittime": dt_to_ts(adm["admittime"]).isoformat(),
                    "age_at_admission": int(adm["age_at_admission"]),
                    "gender": str(adm.get("gender", "")),
                    "admission_type": str(adm.get("admission_type", "")),
                    "admission_location": str(adm.get("admission_location", "")),
                    "discharge_location": str(adm.get("discharge_location", "")),
                    "icu_t": bool(icu_t),

                    # Labels
                    "hard_barrier": bool(hard_barrier),
                    "safe_d24": bool(safe_d24),
                    "adverse_7d": bool(adverse_7d),
                    "adverse_72h": bool(adverse_72h),
                    "readmit_30d": bool(readmit_30d),
                    "core_fields_complete": bool(core_fields_complete),

                    # Features: vitals & labs
                    "vitals": vitals,
                    "labs": labs,

                    # Features: diagnoses (admission-level, list of dicts)
                    "diagnoses": diagnoses_list,

                    # Features: Charlson comorbidity (admission-level, dict)
                    "charlson": charlson_dict,

                    # Features: SOFA at snapshot time (dict)
                    "sofa": sofa_at_t,

                    # Diagnosis category annotations
                    "matched_dx_categories": matched_dx_cats,
                }

                rows.append(row)
                built += 1

        df = pd.DataFrame(rows)
        summary = {
            "base_admissions": int(len(self.cohort)),
            "snapshots_built": int(built),
            "snapshots_with_missing_core": int(missing_core),
            "core_fields_complete_rate": float(df["core_fields_complete"].mean()) if not df.empty else None,
            "hb_rate": float(df["hard_barrier"].mean()) if not df.empty else None,
            "safe_d24_rate": float(df["safe_d24"].mean()) if not df.empty else None,
            "icu_t_rate": float(df["icu_t"].mean()) if not df.empty else None,
            "readmit_30d_rate": float(df["readmit_30d"].mean()) if not df.empty else None,
            "diagnoses_coverage": float(
                df["diagnoses"].apply(lambda x: len(x) > 0 if isinstance(x, list) else False).mean()
            ) if not df.empty else None,
            "charlson_coverage": float(
                df["charlson"].apply(lambda x: x is not None).mean()
            ) if not df.empty else None,
            "sofa_coverage": float(
                df["sofa"].apply(lambda x: x.get("sofa_score") is not None if isinstance(x, dict) else False).mean()
            ) if not df.empty else None,
        }
        return df, summary


# -----------------------------
# Splits
# -----------------------------
def assign_patient_splits(cfg: Config, snapshots: pd.DataFrame) -> pd.DataFrame:
    if snapshots.empty:
        snapshots["split"] = []
        return snapshots

    rng = np.random.default_rng(cfg.seed)
    subjects = snapshots["subject_id"].astype(int).unique()
    rng.shuffle(subjects)

    n = len(subjects)
    n_train = int(round(cfg.train_frac * n))
    n_val = int(round(cfg.val_frac * n))

    train_set = set(subjects[:n_train])
    val_set = set(subjects[n_train:n_train + n_val])

    def which(sid: int) -> str:
        if sid in train_set:
            return "train"
        if sid in val_set:
            return "val"
        return "test"

    snapshots = snapshots.copy()
    snapshots["split"] = snapshots["subject_id"].astype(int).map(which)
    return snapshots


# -----------------------------
# Serialization helpers
# -----------------------------
def save_outputs(out_dir: str, df: pd.DataFrame, summary: Dict) -> None:
    make_out_dir(out_dir)

    # Save JSONL
    jsonl_path = os.path.join(out_dir, "snapshots.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            obj = r.to_dict()
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # Save Parquet
    parquet_path = os.path.join(out_dir, "snapshots.parquet")
    df.to_parquet(parquet_path, index=False)

    # Save summary
    summary_path = os.path.join(out_dir, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Wrote:\n- {parquet_path}\n- {jsonl_path}\n- {summary_path}\n")


# -----------------------------
# Main
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build MIMIC-IV snapshots with derived-vitals primary extraction, "
            "diagnoses, Charlson comorbidity, SOFA scores, and explicit null "
            "propagation for missing core fields."
        )
    )
    p.add_argument("--project_id", default="smartwatch-release", help="GCP project id for BigQuery billing")
    p.add_argument("--dataset", default="physionet-data.mimiciv_3_1", help="MIMIC-IV dataset prefix")
    p.add_argument("--derived_vitals_table", default="physionet-data.mimiciv_3_1_derived.vitalsign")
    p.add_argument("--derived_dataset", default="physionet-data.mimiciv_3_1_derived",
                    help="Derived tables dataset (for charlson, sofa)")
    p.add_argument("--out_dir", default="./out")
    p.add_argument("--max_admissions", type=int, default=None)
    p.add_argument("--max_diagnoses", type=int, default=10, help="Top-N diagnoses per admission")
    p.add_argument(
        "--filter-dx-categories",
        nargs="*",
        default=None,
        metavar="CAT",
        help=(
            "Optional: filter cohort to admissions matching these diagnosis categories. "
            "Accepts category names (e.g. heart_failure sepsis diabetes) and/or raw ICD "
            "prefixes (e.g. I50 J18). Use --list-dx-categories to see all options."
        ),
    )
    p.add_argument(
        "--filter-dx-primary-only",
        action="store_true",
        help="When filtering by diagnosis, match only the primary diagnosis (seq_num=1).",
    )
    p.add_argument(
        "--require-known-dx-category",
        action="store_true",
        help="Exclude admissions that don't match any of the 22 predefined clinical categories.",
    )
    p.add_argument(
        "--list-dx-categories",
        action="store_true",
        help="Print available diagnosis category names and exit.",
    )
    p.add_argument("--exclude_hospice", action="store_true")
    p.add_argument("--exclude_transfers", action="store_true")
    p.add_argument("--use_temporal_mode", action="store_true")
    p.add_argument("--temporal_days", type=int, default=5)
    p.add_argument("--temporal_step_hours", type=int, default=24)
    p.add_argument("--seed", type=int, default=7)

    args, _unknown = p.parse_known_args()
    return args


def main() -> None:
    args = parse_args()

    # Handle --list-dx-categories
    if args.list_dx_categories:
        print("\nAvailable diagnosis categories (use with --filter-dx-categories):\n")
        for cat, prefixes in sorted(DIAGNOSIS_CATEGORIES.items()):
            print(f"  {cat:30s}  ICD prefixes: {', '.join(prefixes)}")
        print(f"\n  Total: {len(DIAGNOSIS_CATEGORIES)} categories")
        print("\nYou can also pass raw ICD prefixes directly, e.g.: --filter-dx-categories I50 J18 E11")
        return

    cfg = Config(
        project_id=args.project_id,
        dataset=args.dataset,
        derived_vitals_table=args.derived_vitals_table,
        derived_dataset=args.derived_dataset,
        max_admissions=None,
        max_diagnoses_per_admission=10,
        filter_dx_categories=None,
        filter_dx_match_any_seq= True, #not args.filter_dx_primary_only,
        require_known_dx_category= True,  #args.require_known_dx_category,
        exclude_hospice= True, #args.exclude_hospice,
        exclude_transfers=True, #args.exclude_transfers,
        use_temporal_mode=True,
        temporal_days=5,
        temporal_step_hours=24,
        delta_v_hours=6,
        delta_l_hours=24,
        seed=7
    )

    print("\n🔧 Config:")
    print(json.dumps(asdict(cfg), indent=2, default=str))

    ex = MIMICExtractor(cfg)

    print("\n1) Extracting cohort...")
    cohort = ex.extract_cohort()
    cohort_before_dedup = int(len(cohort))
    print(f"   cohort admissions (pre-dedup): {cohort_before_dedup:,}")

    cohort = deduplicate_latest_admission_per_subject(cohort)
    cohort_after_dedup = int(len(cohort))
    print(f"   cohort admissions (post-dedup): {cohort_after_dedup:,}")

    # --- Extract diagnoses early (needed for optional filtering) ---
    print("\n2) Extracting diagnoses (ICD codes)...")
    diagnoses = ex.extract_diagnoses(cohort)
    print(f"   diagnoses rows: {len(diagnoses):,}")
    if not diagnoses.empty:
        n_hadm_with_dx = diagnoses["hadm_id"].nunique()
        print(f"   admissions with diagnoses: {n_hadm_with_dx:,}")

    # --- Optional: filter cohort by diagnosis category ---
    cohort_before_dx_filter = int(len(cohort))
    if cfg.filter_dx_categories:
        print(f"\n2b) Filtering cohort by diagnosis categories: {cfg.filter_dx_categories}")
        prefixes = resolve_dx_prefixes(cfg.filter_dx_categories)
        print(f"    Resolved ICD prefixes: {prefixes}")
        cohort = filter_cohort_by_diagnosis_category(
            cohort, diagnoses, cfg.filter_dx_categories,
            match_any_seq=cfg.filter_dx_match_any_seq,
        )
        print(f"    Cohort after dx filter: {len(cohort):,} "
              f"(removed {cohort_before_dx_filter - len(cohort):,})")
        if cohort.empty:
            print("    ❌ No admissions match the requested diagnosis categories!")
            return
        # Re-filter diagnoses to only matched hadm_ids
        kept_hadm = set(cohort["hadm_id"].astype(int).unique())
        diagnoses = diagnoses[diagnoses["hadm_id"].astype(int).isin(kept_hadm)].copy()

    # --- Optional: drop admissions with no recognized clinical category ---
    if cfg.require_known_dx_category:
        cohort_before_known = int(len(cohort))
        if diagnoses.empty:
            print(f"\n2c) Requiring known clinical category but no diagnoses extracted!")
            print("    ❌ All admissions would be dropped. Aborting.")
            return

        all_category_prefixes = resolve_dx_prefixes(list(DIAGNOSIS_CATEGORIES.keys()))

        # Find hadm_ids that have at least one ICD code matching any category
        dx_tmp = diagnoses.copy()
        dx_tmp["_code_norm"] = (
            dx_tmp["icd_code"].astype(str).str.strip().str.upper().str.replace(".", "", regex=False)
        )
        has_known = pd.Series(False, index=dx_tmp.index)
        for pfx in all_category_prefixes:
            norm_pfx = pfx.replace(".", "").upper()
            has_known = has_known | dx_tmp["_code_norm"].str.startswith(norm_pfx)
        known_hadm_ids = set(dx_tmp.loc[has_known, "hadm_id"].astype(int).unique())

        cohort = cohort[cohort["hadm_id"].astype(int).isin(known_hadm_ids)].copy()
        print(f"\n2c) Requiring known clinical category: {len(cohort):,} "
              f"(dropped {cohort_before_known - len(cohort):,} uncategorized)")
        if cohort.empty:
            print("    ❌ No admissions have a recognized clinical category!")
            return
        kept_hadm = set(cohort["hadm_id"].astype(int).unique())
        diagnoses = diagnoses[diagnoses["hadm_id"].astype(int).isin(kept_hadm)].copy()

    print("\n3) Extracting ICU stays...")
    icu_stays = ex.extract_icu_stays_for_cohort(cohort)
    print(f"   ICU stays rows: {len(icu_stays):,}")

    print("\n4) Extracting vitals (derived primary + ICU fallback)...")
    vitals_derived = ex.extract_vitals_derived(cohort)
    print(f"   derived vitals rows: {len(vitals_derived):,}")
    vitals_icu = ex.extract_vitals_chartevents(icu_stays, cohort)
    print(f"   ICU vitals rows: {len(vitals_icu):,}")
    vitals = merge_vitals_primary_with_fallback(vitals_derived, vitals_icu)
    print(f"   merged vitals rows: {len(vitals):,}")

    derived_hadm_ids = set(vitals_derived["hadm_id"].astype(int).unique().tolist()) if not vitals_derived.empty else set()
    fallback_hadm_ids = (
        set(vitals_icu["hadm_id"].astype(int).unique().tolist()) - derived_hadm_ids if not vitals_icu.empty else set()
    )
    print(f"   admissions using ICU fallback: {len(fallback_hadm_ids):,}")

    print("\n5) Extracting labs (hosp labevents)...")
    labs = ex.extract_labs_labevents(cohort)
    print(f"   labs rows: {len(labs):,}")

    print("\n6) Extracting readmissions...")
    readmit = ex.extract_readmissions(cohort)
    print(f"   readmission index size: {len(readmit):,}")

    print("\n7) Extracting Charlson comorbidity...")
    charlson = ex.extract_charlson_comorbidity(cohort)
    print(f"   Charlson rows: {len(charlson):,}")

    print("\n8) Extracting SOFA scores...")
    sofa = ex.extract_sofa_scores(icu_stays)
    print(f"   SOFA rows: {len(sofa):,}")

    print("\n9) Building snapshots...")
    builder = SnapshotBuilder(cfg, cohort, icu_stays, vitals, labs, readmit, diagnoses, charlson, sofa)
    snapshots, run_summary = builder.build()

    print("\n10) Assigning patient-level splits...")
    snapshots = assign_patient_splits(cfg, snapshots)

    # Add split stats
    split_counts = snapshots["split"].value_counts(dropna=False).to_dict() if not snapshots.empty else {}
    run_summary["split_counts"] = {k: int(v) for k, v in split_counts.items()}
    run_summary["final_snapshots"] = int(len(snapshots))
    run_summary["cohort_before_dedup"] = int(cohort_before_dedup)
    run_summary["cohort_after_dedup"] = int(cohort_after_dedup)
    run_summary["cohort_before_dx_filter"] = int(cohort_before_dx_filter)
    run_summary["cohort_after_dx_filter"] = int(len(cohort))
    run_summary["filter_dx_categories"] = cfg.filter_dx_categories
    run_summary["derived_vitals_hadm_covered"] = int(len(derived_hadm_ids))
    run_summary["icu_fallback_hadm_covered"] = int(len(fallback_hadm_ids))

    print("\n📌 Summary:")
    print(json.dumps(run_summary, indent=2))

    print("\n11) Saving outputs...")
    save_outputs(args.out_dir, snapshots, run_summary)


if __name__ == "__main__":
    main()
