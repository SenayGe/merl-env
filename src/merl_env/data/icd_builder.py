"""Offline builder for the ICD multi-label coding task."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from merl_env.core.sample import TaskSample
from merl_env.data._builder_utils import (
    build_sample_id,
    coerce_int,
    empty_split_buckets,
    json_safe,
    sort_split_buckets,
)
from merl_env.data.artifacts import TaskArtifactManifest, write_task_artifacts
from merl_env.data.mimic_source import MimicQuery, MimicSource
from merl_env.data.splits import SplitConfig, assign_subject_splits

ICD_TASK_NAME = "icd"
ICD_QUERY_NAME = "icd_cases"


@dataclass(slots=True, kw_only=True)
class IcdBuilderConfig:
    """Configuration for the ICD task builder."""

    hospital_dataset: str = "physionet-data.mimiciv_3_1_hosp"
    note_table: str = "physionet-data.mimiciv_note.discharge"
    min_note_length: int = 200
    max_codes: int = 10
    allowed_tools: tuple[str, ...] = ("icd_lookup",)


def build_icd_query(config: IcdBuilderConfig | None = None) -> MimicQuery:
    """Build the logical BigQuery used for ICD task extraction."""

    builder_config = config or IcdBuilderConfig()
    sql = f"""
    SELECT
      n.subject_id,
      n.hadm_id,
      n.note_id,
      n.text AS clinical_note,
      dx.seq_num,
      dx.icd_code,
      dx.icd_version,
      d_dx.long_title
    FROM `{builder_config.note_table}` AS n
    JOIN `{builder_config.hospital_dataset}.diagnoses_icd` AS dx
      ON dx.hadm_id = n.hadm_id
    LEFT JOIN `{builder_config.hospital_dataset}.d_icd_diagnoses` AS d_dx
      ON (dx.icd_version, dx.icd_code) = (d_dx.icd_version, d_dx.icd_code)
    WHERE n.text IS NOT NULL
      AND LENGTH(n.text) >= {builder_config.min_note_length}
    ORDER BY n.hadm_id, n.note_id, dx.seq_num
    """
    return MimicQuery(
        name=ICD_QUERY_NAME,
        sql=sql.strip(),
        metadata={
            "task_name": ICD_TASK_NAME,
            "expected_columns": (
                "subject_id",
                "hadm_id",
                "note_id",
                "clinical_note",
                "seq_num",
                "icd_code",
                "icd_version",
                "long_title",
            ),
        },
    )


def collect_top_icd_codes(
    diagnosis_rows: list[dict[str, Any]],
    *,
    max_codes: int = 10,
) -> list[dict[str, Any]]:
    """Keep the first `max_codes` unique ICD labels ordered by `seq_num`."""

    def _sort_key(row: dict[str, Any]) -> tuple[int, str]:
        seq_num = coerce_int(row.get("seq_num"))
        return (seq_num if seq_num is not None else 10**9, str(row.get("icd_code") or ""))

    seen_codes: set[str] = set()
    top_codes: list[dict[str, Any]] = []
    for row in sorted(diagnosis_rows, key=_sort_key):
        code = str(row.get("icd_code") or "").strip()
        if not code or code in seen_codes:
            continue
        seen_codes.add(code)
        top_codes.append(
            {
                "seq_num": coerce_int(row.get("seq_num")),
                "icd_code": code,
                "icd_version": coerce_int(row.get("icd_version")),
                "long_title": str(row.get("long_title") or "").strip() or None,
            }
        )
        if len(top_codes) >= max_codes:
            break
    return top_codes


def build_icd_samples(
    source: MimicSource,
    *,
    config: IcdBuilderConfig | None = None,
    split_config: SplitConfig | None = None,
) -> dict[str, list[TaskSample]]:
    """Build normalized ICD samples grouped by split."""

    builder_config = config or IcdBuilderConfig()
    rows = source.fetch_rows(build_icd_query(builder_config))

    grouped_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        hadm_id = row.get("hadm_id")
        note_id = row.get("note_id")
        key = (str(hadm_id), str(note_id))
        group = grouped_rows.setdefault(
            key,
            {
                "subject_id": row.get("subject_id"),
                "hadm_id": hadm_id,
                "note_id": note_id,
                "clinical_note": str(row.get("clinical_note") or ""),
                "codes": [],
            },
        )
        group["codes"].append(dict(row))

    if not grouped_rows:
        return empty_split_buckets()

    prepared_rows: list[dict[str, Any]] = []
    for group in grouped_rows.values():
        top_codes = collect_top_icd_codes(group["codes"], max_codes=builder_config.max_codes)
        if not top_codes:
            continue
        prepared_rows.append(
            {
                "subject_id": group["subject_id"],
                "hadm_id": group["hadm_id"],
                "note_id": group["note_id"],
                "clinical_note": group["clinical_note"],
                "top_codes": top_codes,
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
            sample_id=build_sample_id(ICD_TASK_NAME, row["hadm_id"], row["note_id"]),
            task_name=ICD_TASK_NAME,
            split=split,
            input_payload=json_safe(
                {
                    "subject_id": row["subject_id"],
                    "hadm_id": row["hadm_id"],
                    "note_id": row["note_id"],
                    "note_text": row["clinical_note"],
                }
            ),
            reference={"icd_codes": [code["icd_code"] for code in row["top_codes"]]},
            metadata=json_safe(
                {
                    "subject_id": row["subject_id"],
                    "hadm_id": row["hadm_id"],
                    "note_id": row["note_id"],
                    "note_length": len(row["clinical_note"]),
                    "icd_labels": row["top_codes"],
                }
            ),
            allowed_tools=builder_config.allowed_tools,
        )
        samples_by_split[split].append(sample)

    return sort_split_buckets(samples_by_split)


def build_icd_artifacts(
    out_dir: str,
    source: MimicSource,
    *,
    config: IcdBuilderConfig | None = None,
    split_config: SplitConfig | None = None,
) -> TaskArtifactManifest:
    """Build ICD samples and write them to canonical local artifacts."""

    builder_config = config or IcdBuilderConfig()
    samples_by_split = build_icd_samples(
        source,
        config=builder_config,
        split_config=split_config,
    )
    return write_task_artifacts(
        out_dir,
        ICD_TASK_NAME,
        samples_by_split,
        source={
            "query_name": ICD_QUERY_NAME,
            "query": build_icd_query(builder_config).sql,
        },
        metadata={"builder_config": json_safe(asdict(builder_config))},
    )


__all__ = [
    "ICD_QUERY_NAME",
    "ICD_TASK_NAME",
    "IcdBuilderConfig",
    "build_icd_artifacts",
    "build_icd_query",
    "build_icd_samples",
    "collect_top_icd_codes",
]
