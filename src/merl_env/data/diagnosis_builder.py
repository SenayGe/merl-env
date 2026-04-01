"""Offline builder for the primary-diagnosis task."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Mapping, Sequence

from merl_env.core.sample import TaskSample
from merl_env.data._builder_utils import (
    build_sample_id,
    coerce_int,
    coerce_str_list,
    empty_split_buckets,
    json_safe,
    sort_split_buckets,
)
from merl_env.data.artifacts import TaskArtifactManifest, write_task_artifacts
from merl_env.data.mimic_source import MimicQuery, MimicSource
from merl_env.data.splits import SplitConfig, assign_subject_splits

DIAGNOSIS_TASK_NAME = "diagnosis"
DIAGNOSIS_QUERY_NAME = "diagnosis_cases"

DIAGNOSIS_SECTION_ORDER: tuple[str, ...] = ("HPI", "PMH", "PE", "RESULTS")
DIAGNOSIS_SECTION_PATTERNS: dict[str, str] = {
    "HPI": r"^\s*(History of Present Illness|Present Illness|HPI)\s*(?:[:\-]|$)",
    "PMH": r"^\s*(Past Medical History|PMH)\s*(?:[:\-]|$)",
    "PE": r"^\s*(Physical Examination?|PE|Phys\.? Exam)\s*(?:[:\-]|$)",
    "RESULTS": r"^\s*(Pertinent Results|Laboratory Data|Labs?|Imaging Studies?)\s*(?:[:\-]|$)",
}
DIAGNOSIS_BOUNDARY_HEADERS: tuple[str, ...] = (
    r"^\s*(Brief\s+)?Hospital Course",
    r"^\s*Admission Diagnosis",
    r"^\s*Reason For Admission",
    r"^\s*Chief Complaint",
    r"^\s*Discharge Diagnoses?",
    r"^\s*Final Diagnoses?",
    r"^\s*Problem List",
    r"^\s*Active Problems?",
    r"^\s*Assessment\s*$",
    r"^\s*Impression\s*$",
    r"^\s*ICD ?Codes?",
    r"^\s*DRG",
    r"^\s*Discharge Labs",
    r"^\s*Interim Labs",
    r"^\s*Discharge Medications?",
    r"^\s*Discharge Instructions?",
    r"^\s*Follow[- ]?Up",
)

DEFAULT_DISEASE_MAPPING: dict[str, tuple[str, ...]] = {
    "Sepsis / septic shock": ("A419", "0389"),
    "NSTEMI (non-ST-elevation MI)": ("I214", "41071"),
    "Acute on chronic heart failure": ("I5023", "I5033", "42823", "42833"),
    "Pulmonary embolism": ("I269", "41519"),
    "Ischemic stroke": ("I630", "I639", "43491"),
    "Hypertensive emergency": ("I161",),
    "Acute kidney injury (AKI)": ("N179", "5849"),
    "Pneumonia": ("J189", "486"),
    "Gastrointestinal haemorrhage": ("5789",),
    "Diabetic ketoacidosis (DKA)": ("E1010", "E1110"),
    "Hypertensive heart disease": ("I110",),
}

_KEEP_RE = {
    name: re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    for name, pattern in DIAGNOSIS_SECTION_PATTERNS.items()
}
_BOUNDARY_RE = re.compile("|".join(DIAGNOSIS_BOUNDARY_HEADERS), re.IGNORECASE | re.MULTILINE)


@dataclass(slots=True, kw_only=True)
class DiagnosisBuilderConfig:
    """Configuration for the diagnosis task builder."""

    hospital_dataset: str = "physionet-data.mimiciv_3_1_hosp"
    note_table: str = "physionet-data.mimiciv_note.discharge"
    min_note_length: int = 500
    min_required_sections: int = 2
    section_order: tuple[str, ...] = DIAGNOSIS_SECTION_ORDER
    allowed_tools: tuple[str, ...] = ()
    disease_mapping: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(DEFAULT_DISEASE_MAPPING)
    )


def extract_diagnosis_sections(note_text: str) -> dict[str, str | None]:
    """Return the clinically useful diagnosis-note sections from a discharge note."""

    note = note_text or ""
    sections: dict[str, str | None] = {name: None for name in DIAGNOSIS_SECTION_ORDER}

    markers: list[tuple[int, int, str | None, bool]] = []
    for name, pattern in _KEEP_RE.items():
        for match in pattern.finditer(note):
            markers.append((match.start(), match.end(), name, True))
    for match in _BOUNDARY_RE.finditer(note):
        markers.append((match.start(), match.end(), None, False))
    markers.sort(key=lambda item: item[0])

    for current, following in zip(markers, markers[1:] + [(len(note), None, None, False)]):
        start, header_end, section_name, keep = current
        del start
        content_end = following[0]
        if not keep or section_name is None or content_end <= header_end:
            continue
        normalized = re.sub(r"\s+", " ", note[header_end:content_end]).strip()
        if not normalized:
            continue
        existing = sections[section_name]
        sections[section_name] = f"{existing}\n{normalized}" if existing else normalized
    return sections


def build_diagnosis_query(config: DiagnosisBuilderConfig | None = None) -> MimicQuery:
    """Build the logical BigQuery used for diagnosis-task extraction."""

    builder_config = config or DiagnosisBuilderConfig()
    disease_rows: list[str] = []
    for label, codes in builder_config.disease_mapping.items():
        for code in codes:
            icd_version = 10 if code[:1].isalpha() else 9
            disease_rows.append(
                f"SELECT '{label}' AS label, {icd_version} AS icd_version, '{code}' AS icd_code"
            )
    disease_map_sql = "\nUNION ALL\n".join(disease_rows)

    sql = f"""
    WITH disease_map AS (
      {disease_map_sql}
    ),
    principal_dx AS (
      SELECT
        dx.hadm_id,
        dx.subject_id,
        dm.label AS primary_label,
        dx.icd_code AS primary_icd,
        dx.icd_version AS primary_icd_version,
        d_dx.long_title AS primary_long_title
      FROM `{builder_config.hospital_dataset}.diagnoses_icd` AS dx
      JOIN disease_map AS dm
        ON (dx.icd_version, dx.icd_code) = (dm.icd_version, dm.icd_code)
      JOIN `{builder_config.hospital_dataset}.d_icd_diagnoses` AS d_dx
        ON (dx.icd_version, dx.icd_code) = (d_dx.icd_version, d_dx.icd_code)
      WHERE dx.seq_num = 1
    ),
    all_dx AS (
      SELECT
        pd.hadm_id,
        dm.label
      FROM principal_dx AS pd
      JOIN `{builder_config.hospital_dataset}.diagnoses_icd` AS dx
        ON dx.hadm_id = pd.hadm_id
      JOIN disease_map AS dm
        ON (dx.icd_version, dx.icd_code) = (dm.icd_version, dm.icd_code)
    )
    SELECT
      pd.subject_id,
      pd.hadm_id,
      n.note_id,
      n.text AS clinical_note,
      pd.primary_label,
      pd.primary_icd,
      pd.primary_icd_version,
      pd.primary_long_title,
      ARRAY_AGG(DISTINCT ad.label) AS all_mapped_labels
    FROM principal_dx AS pd
    JOIN all_dx AS ad
      ON ad.hadm_id = pd.hadm_id
    JOIN `{builder_config.note_table}` AS n
      ON n.hadm_id = pd.hadm_id
    WHERE n.text IS NOT NULL
      AND LENGTH(n.text) >= {builder_config.min_note_length}
    GROUP BY
      pd.subject_id,
      pd.hadm_id,
      n.note_id,
      n.text,
      pd.primary_label,
      pd.primary_icd,
      pd.primary_icd_version,
      pd.primary_long_title
    """

    return MimicQuery(
        name=DIAGNOSIS_QUERY_NAME,
        sql=sql.strip(),
        metadata={
            "task_name": DIAGNOSIS_TASK_NAME,
            "expected_columns": (
                "subject_id",
                "hadm_id",
                "note_id",
                "clinical_note",
                "primary_label",
                "primary_icd",
                "primary_icd_version",
                "primary_long_title",
                "all_mapped_labels",
            ),
        },
    )


def build_diagnosis_samples(
    source: MimicSource,
    *,
    config: DiagnosisBuilderConfig | None = None,
    split_config: SplitConfig | None = None,
) -> dict[str, list[TaskSample]]:
    """Build normalized diagnosis samples grouped by split."""

    builder_config = config or DiagnosisBuilderConfig()
    rows = source.fetch_rows(build_diagnosis_query(builder_config))

    prepared_rows: list[dict[str, Any]] = []
    for row in rows:
        note_text = str(row.get("clinical_note") or "")
        sections = extract_diagnosis_sections(note_text)
        captured_sections = [
            name for name in builder_config.section_order if sections.get(name)
        ]
        if len(captured_sections) < builder_config.min_required_sections:
            continue

        summary_parts = [
            f"### {name}\n{sections[name]}"
            for name in builder_config.section_order
            if sections.get(name)
        ]
        prepared_rows.append(
            {
                "subject_id": row.get("subject_id"),
                "hadm_id": row.get("hadm_id"),
                "note_id": row.get("note_id"),
                "sections": sections,
                "clinical_summary": "\n\n".join(summary_parts),
                "primary_diagnosis": str(row.get("primary_long_title") or "").strip(),
                "primary_label": str(row.get("primary_label") or "").strip() or None,
                "primary_icd_code": str(row.get("primary_icd") or "").strip() or None,
                "primary_icd_version": coerce_int(row.get("primary_icd_version")),
                "all_mapped_labels": coerce_str_list(row.get("all_mapped_labels")),
                "source_note_length": len(note_text),
                "captured_sections": captured_sections,
            }
        )

    if not prepared_rows:
        return empty_split_buckets()

    assignments = assign_subject_splits(
        [row["subject_id"] for row in prepared_rows],
        config=split_config,
    )
    samples_by_split = empty_split_buckets()

    for row in prepared_rows:
        split = assignments[row["subject_id"]]
        sample = TaskSample(
            sample_id=build_sample_id(
                DIAGNOSIS_TASK_NAME,
                row["hadm_id"],
                row["note_id"],
            ),
            task_name=DIAGNOSIS_TASK_NAME,
            split=split,
            input_payload=json_safe(
                {
                    "subject_id": row["subject_id"],
                    "hadm_id": row["hadm_id"],
                    "note_id": row["note_id"],
                    "sections": row["sections"],
                    "clinical_summary": row["clinical_summary"],
                }
            ),
            reference={"primary_diagnosis": row["primary_diagnosis"]},
            metadata=json_safe(
                {
                    "subject_id": row["subject_id"],
                    "hadm_id": row["hadm_id"],
                    "note_id": row["note_id"],
                    "primary_label": row["primary_label"],
                    "primary_icd_code": row["primary_icd_code"],
                    "primary_icd_version": row["primary_icd_version"],
                    "all_mapped_labels": row["all_mapped_labels"],
                    "captured_sections": row["captured_sections"],
                    "source_note_length": row["source_note_length"],
                }
            ),
            allowed_tools=builder_config.allowed_tools,
        )
        samples_by_split[split].append(sample)

    return sort_split_buckets(samples_by_split)


def build_diagnosis_artifacts(
    out_dir: str,
    source: MimicSource,
    *,
    config: DiagnosisBuilderConfig | None = None,
    split_config: SplitConfig | None = None,
) -> TaskArtifactManifest:
    """Build diagnosis samples and write them to canonical local artifacts."""

    builder_config = config or DiagnosisBuilderConfig()
    samples_by_split = build_diagnosis_samples(
        source,
        config=builder_config,
        split_config=split_config,
    )
    return write_task_artifacts(
        out_dir,
        DIAGNOSIS_TASK_NAME,
        samples_by_split,
        source={
            "query_name": DIAGNOSIS_QUERY_NAME,
            "query": build_diagnosis_query(builder_config).sql,
        },
        metadata={"builder_config": json_safe(asdict(builder_config))},
    )


__all__ = [
    "DEFAULT_DISEASE_MAPPING",
    "DIAGNOSIS_QUERY_NAME",
    "DIAGNOSIS_SECTION_ORDER",
    "DIAGNOSIS_TASK_NAME",
    "DiagnosisBuilderConfig",
    "build_diagnosis_artifacts",
    "build_diagnosis_query",
    "build_diagnosis_samples",
    "extract_diagnosis_sections",
]
