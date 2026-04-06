"""
Mimic diagnosis eval data
"""

import os
import random
import re

import pandas as pd
from datasets import Dataset, DatasetDict
from google.cloud import bigquery

KEEP_HEADERS = {
    "HPI": r"^\s*(History of Present Illness|Present Illness|HPI)\s*(?:[:\-]|$)",
    "PMH": r"^\s*(Past Medical History|PMH)\s*(?:[:\-]|$)",
    "PE": r"^\s*(Physical Examination?|PE|Phys\.? Exam)\s*(?:[:\-]|$)",
    "RESULTS": r"^\s*(Pertinent Results|Laboratory Data|Labs?|Imaging Studies?)\s*(?:[:\-]|$)",
    # ✱ Drop Assessment/Plan if you don’t want leakage
}

# 2) Add a SEPARATE “boundary” regex that matches *any* header
#    you DON’T want, so we can stop the capture cleanly.
BOUNDARY_HEADERS = [
    r"^(\s*Brief )?Hospital Course",  # Hospital Course family
    r"^Admission Diagnosis",
    r"^Reason For Admission",
    r"^Chief Complaint",
    r"^Discharge Diagnoses?",
    r"^Final Diagnoses?",
    r"^Problem List",
    r"^Active Problems?",
    r"^Assessment\s*$",
    r"^Impression\s*$",
    r"^ICD ?Codes?",
    r"^DRG",
    r"^Discharge Labs",
    r"^Interim Labs",
    r"^Discharge Medications?",
    r"^Discharge Instructions?",
    r"^Follow[- ]?Up",
]

# 3) Compile everything once
KEEP_RE = {k: re.compile(v, re.I | re.M) for k, v in KEEP_HEADERS.items()}
BOUNDARY_RE = re.compile("|".join(BOUNDARY_HEADERS), re.I | re.M)

GCP_PROJECT = None

def extract_sections(note):
    """Return only the allowed sections, drop anything past a boundary."""
    out = {k: None for k in KEEP_HEADERS}
    # ── Find starts of *all* headers (keepers + boundaries) ──────────
    markers = []
    for lbl, pat in KEEP_RE.items():
        markers += [
            (m.start(), m.end(), lbl, True)  # is_keep = True
            for m in pat.finditer(note)
        ]
    for m in BOUNDARY_RE.finditer(note):
        markers.append((m.start(), m.end(), None, False))  # boundary
    markers.sort(key=lambda x: x[0])

    # ── Slice by consecutive markers ────────────────────────────────
    for (s0, e0, lbl, keep), next_marker in zip(
        markers, markers[1:] + [(len(note), None, None, False)]
    ):
        s1 = e0  # start of content
        e1 = next_marker[0]  # stop at next header
        if keep and e1 > s1:
            txt = re.sub(r"\s+", " ", note[s1:e1]).strip()
            out[lbl] = out[lbl] + "\n" + txt if out[lbl] else txt
    return out


DISEASE_MAPPING = {
    # ──────────────── High diagnosability ────────────────
    "Sepsis / septic shock": [
        "A419",  # Sepsis, unspecified organism  (ICD-10)
        "0389",  # Unspecified septicemia        (ICD-9)
        # "R6520",  # Severe sepsis without septic shock                (ICD-10)
        # "R6521",  # Severe sepsis with septic shock                (ICD-10)
        # "78552"   # Septic shock                  (ICD-9)
    ],
    "NSTEMI (non-ST-elevation MI)": [
        "I214",  # NSTEMI                        (ICD-10)
        "41071",  # Subendocardial infarction (initial/subseq) (ICD-9)
    ],
    "Acute on chronic heart failure": [
        "I5023",
        "I5033",  # Acute-on-chronic systolic / diastolic HF (ICD-10)
        "42823",
        "42833",  # Acute-on-chronic systolic / diastolic HF (ICD-9)
    ],
    "Pulmonary embolism": [
        "I269",  # Pulmonary embolism w/o cor pulmonale     (ICD-10)
        "41519",  # Iatrogenic / other PE                    (ICD-9)
    ],
    "Ischemic stroke": [
        "I630",
        "I639",  # Cerebral infarction (thromb/unspec)      (ICD-10)
        "43491",  # Cerebral artery occlusion, unspec        (ICD-9)
        # "436"                    # Acute cerebrovascular accident           (ICD-9)
    ],
    "Hypertensive emergency": [
        "I161",  # Hypertensive emergency                   (ICD-10)
        # "4010", "4019"           # Malignant / unspecified essential HTN    (ICD-9)
    ],
    # ──────────────── Medium diagnosability ──────────────
    "Acute kidney injury (AKI)": [
        "N179",  # Acute kidney failure, unspecified        (ICD-10)
        "5849",  # Acute renal failure, unspecified         (ICD-9)
    ],
    "Pneumonia": [
        "J189",  # Pneumonia, organism unspecified          (ICD-10)
        "486",
        # "481",  # Pneumonia, unspec / pneumococcal         (ICD-9)
    ],
    "Gastrointestinal haemorrhage": [
        "5789",
        #     "K922",
        #     "K920",  # GI hemorrhage (melena / unspecified)     (ICD-10)
        #     "I8501",  # Esophageal varices with bleeding         (ICD-10)
        #     "4560",  # GI hemorrhage, unspec / varices bleed    (ICD-9)
    ],
    "Diabetic ketoacidosis (DKA)": [
        "E1010",
        "E1110",  # Type 1 / Type 2 DM with DKA, w/o coma    (ICD-10)
        # "25010",
        # "25011",  # DM w/ ketoacidosis (not/uncontrolled)    (ICD-9)
    ],
    "Hypertensive heart disease": ["I110"],
}

SECTION_PATTERNS = {
    "HPI": re.compile(
        r"^\s*(History of Present Illness|Present Illness|HPI)\s*(?:[:\-]|$)",
        re.IGNORECASE | re.MULTILINE,
    ),
    "PMH": re.compile(
        r"^\s*(Past Medical History|PMH)\s*:", re.IGNORECASE | re.MULTILINE
    ),
    # "MEDS_ADM": re.compile(r"^\s*(Medications on Admission|Admission Medications)\s*:", re.IGNORECASE | re.MULTILINE),
    "PE": re.compile(
        r"^\s*(Physical Exam|Physical Examination|Exam|PE)\s*(?:[:\-]|$)",
        re.IGNORECASE | re.MULTILINE,
    ),
    "RESULTS": re.compile(
        r"^\s*(Pertinent Results|Labs|Imaging|Studies)\s*(?:[:\-]|$)",
        re.IGNORECASE | re.MULTILINE,
    ),
    # "HOSP_COURSE": re.compile(r"^\s*(Hospital Course|Hospital Day \d+|Brief Hospital Course)\s*(?:[:\-]|$)", re.IGNORECASE | re.MULTILINE),
    "ASSESSMENT_PLAN": re.compile(
        r"^\s*(Assessment and Plan|A/P|Assessment|Plan)\s*(?:[:\-]|$)",
        re.IGNORECASE | re.MULTILINE,
    ),
}
END_NARRATIVE_PATTERNS = [
    re.compile(r"^\s*DISCHARGE LABS\s*:", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*INTERIM LABS\s*:", re.IGNORECASE | re.MULTILINE),
    re.compile(
        r"^\s*(Discharge Diagnoses|Final Diagnoses|Discharge Medications|Discharge Instructions|Follow Up)\s*:",
        re.IGNORECASE | re.MULTILINE,
    ),
]

SYSTEM_PROMPT = """
You are a helpful medical AI assistant expert in medical diagnosis.
You will be provided with a structured clinical health information about a patient, the discharge summary, which contains History, Exam, Hospital Course, etc., and you will determine the primary diagnosis for the patient.
You first think step by step about the medical case inside  <reasoning>...</reasoning> tags as an internal monologue and then provide the user with the final diagnosis inside the <answer>...</answer> tags.
Give your output response in following format:
<reasoning>
...
</reasoning>
<answer>
....
</answer>
"""


# ── Sampling constants ────────────────────────────────────────────────
# N_DISEASE_SAMPLE = 220  # total notes we want per label
# N_DISEASE_SFT = 160  # → supervised-finetune slice
# N_DISEASE_GRPO = 50  # → GRPO-train slice
# N_DISEASE_EVAL = 10  # → held-out eval

N_DISEASE_SAMPLE = 480  # total notes we want per label
N_DISEASE_SFT = 150  # → supervised-finetune slice
N_DISEASE_GRPO = 300  # → GRPO-train slice
N_DISEASE_EVAL = 30  # → held-out eval
assert N_DISEASE_SAMPLE == N_DISEASE_SFT + N_DISEASE_GRPO + N_DISEASE_EVAL


# ── Main data-prep function ───────────────────────────────────────────
def prepare_mimic_data(
    GCP_PROJECT_ID= GCP_PROJECT,
    LOCATION="US",
    OUTPUT_DIR="/content/drive/MyDrive/mimic_diagnosis",
    RANDOM_SEED=42,
):
    """
    Build a one-label-per-admission corpus from MIMIC-IV discharge notes.
    The label is the *principal* (seq_num = 1) diagnosis ICD code,
    restricted to DISEASE_MAPPING. The 'answer' field is the official
    long title of that diagnosis.
    """
    import os
    import random

    import pandas as pd
    from datasets import Dataset, DatasetDict
    from google.cloud import bigquery

    print("\n── Data prep for primary-diagnosis task ──")

    # 1) Build disease_map (as before)
    disease_rows = []
    for lbl, codes in DISEASE_MAPPING.items():
        for code in codes:
            icd_version = 10 if code[0].isalpha() else 9
            disease_rows.append(
                f"SELECT '{lbl}' AS label, {icd_version} AS icd_version, '{code}' AS icd_code"
            )
    disease_map_sql = "\nUNION ALL\n".join(disease_rows)

    try:
        from google.colab import auth as colab_auth

        colab_auth.authenticate_user()
    except ImportError:
        pass
    client = bigquery.Client(project=GCP_PROJECT_ID)

    # ---------------------------------------------------------------
    # 2) ▼▼▼ EDIT: Query now joins d_icd_diagnoses to get the long title ▼▼▼
    QUERY = f"""
    WITH disease_map AS (
      {disease_map_sql}
    ),

    -- A) principal_dx → admissions whose *primary* (seq_num=1) code is in the map
    principal_dx AS (
      SELECT
        dx.hadm_id,
        dx.subject_id,
        dm.label AS primary_label,
        dx.icd_code AS primary_icd,
        d_dx.long_title AS primary_long_title -- <<< CHANGE: Fetch the long title
      FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` AS dx
      JOIN disease_map dm
           ON (dx.icd_version, dx.icd_code) = (dm.icd_version, dm.icd_code)
      -- <<< CHANGE: Join the dictionary table to get the official title
      JOIN `physionet-data.mimiciv_3_1_hosp.d_icd_diagnoses` AS d_dx
           ON (dx.icd_version, dx.icd_code) = (d_dx.icd_version, d_dx.icd_code)
      WHERE dx.seq_num = 1
    ),

    -- B) all target codes for those admissions (no changes here)
    all_dx AS (
      SELECT
        pd.hadm_id,
        dm.label
      FROM principal_dx AS pd
      JOIN `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` AS dx
           ON dx.hadm_id = pd.hadm_id
      JOIN disease_map AS dm
           ON (dx.icd_version, dx.icd_code) = (dm.icd_version, dm.icd_code)
    ),

    -- C) aggregate labels + bring in the note
    final AS (
      SELECT
        pd.primary_label,
        pd.primary_icd,
        pd.primary_long_title, -- <<< CHANGE: Pass the long title through
        ARRAY_AGG(DISTINCT ad.label) AS all_mapped_labels, -- Renamed for clarity
        n.subject_id,
        n.hadm_id,
        n.note_id,
        n.text AS clinical_note
      FROM principal_dx AS pd
      JOIN all_dx AS ad USING (hadm_id)
      JOIN `physionet-data.mimiciv_note.discharge` AS n USING (hadm_id)
      WHERE n.text IS NOT NULL
        AND LENGTH(n.text) > 500
      GROUP BY pd.primary_label, pd.primary_icd, pd.primary_long_title, n.subject_id, n.hadm_id, n.note_id, n.text
    )
    SELECT * FROM final;
    """

    # 3) Run query
    print("Fetching data from BigQuery...")
    df = client.query(QUERY).result().to_dataframe()
    print(f"Fetched {len(df)} rows.")

    # 4) Balance on the *primary* label
    rng = random.Random(RANDOM_SEED)
    sampled = []
    for lbl, grp in df.groupby("primary_label", group_keys=False):
        take = (
            grp
            if len(grp) < N_DISEASE_SAMPLE
            else grp.sample(N_DISEASE_SAMPLE, random_state=RANDOM_SEED)
        )
        sampled.append(take)
    df_sampled = pd.concat(sampled).reset_index(drop=True)

    # 5) Split + prompt/answer
    splits = {"sft": [], "train": [], "eval": []}
    section_order = ["HPI", "PMH", "PE", "RESULTS"]
    essential_sections = set(section_order)

    for lbl, grp in df_sampled.groupby("primary_label", sort=False):
        grp = grp.sample(frac=1, random_state=RANDOM_SEED)
        sft_rows = grp.iloc[:N_DISEASE_SFT]
        train_rows = grp.iloc[N_DISEASE_SFT : N_DISEASE_SFT + N_DISEASE_GRPO]
        eval_rows = grp.iloc[-N_DISEASE_EVAL:]

        for split_name, rows in [
            ("sft", sft_rows),
            ("train", train_rows),
            ("eval", eval_rows),
        ]:
            for _, row in rows.iterrows():
                sections = extract_sections(row["clinical_note"])
                if sum(1 for s in essential_sections if sections.get(s)) < 2:
                    continue

                prompt_text = "\n\n".join(
                    f"### {sec}\n{sections[sec]}"
                    for sec in section_order
                    if sections.get(sec)
                ).strip()

                # ▼▼▼ EDIT: The answer now comes from the 'primary_long_title' column ▼▼▼
                answer_text = row["primary_long_title"]

                sample = {
                    "id": row["hadm_id"],
                    "icd_code": row["primary_icd"],
                    # This field can be useful for analysis if needed
                    "all_mapped_labels": row["all_mapped_labels"],
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                "What is the primary diagnosis for the patient "
                                "based on this clinical summary:\n" + prompt_text
                            ),
                        },
                    ],
                    # Use the official diagnosis title as the answer
                    "answer": answer_text,
                }
                # ▼▼▼ EDIT: Simplified the data structure for clarity ▼▼▼
                splits[split_name].append(sample)

    # 6) Shuffle & save (unchanged)
    rng.shuffle(splits["sft"])
    rng.shuffle(splits["train"])
    rng.shuffle(splits["eval"])

    # 7) HuggingFace datasets → disk
    ds = DatasetDict(
        {
            "sft": Dataset.from_list(splits["sft"]),
            "train": Dataset.from_list(splits["train"]),
            "eval": Dataset.from_list(splits["eval"]),
        }
    )

    # os.makedirs(OUTPUT_DIR, exist_ok=True)
    # ds.save_to_disk(OUTPUT_DIR)
    # print(
    #     f"✅  Saved to {OUTPUT_DIR} — "
    #     f"sft={len(ds['sft'])}, train={len(ds['train'])}, eval={len(ds['eval'])}"
    # )
    return ds
