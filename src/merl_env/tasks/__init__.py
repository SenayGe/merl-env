"""Task definitions package."""

from merl_env.tasks.diagnosis import DiagnosisTaskSpec
from merl_env.tasks.discharge import DischargeTaskSpec
from merl_env.tasks.icd import IcdTaskSpec

__all__ = ["DiagnosisTaskSpec", "DischargeTaskSpec", "IcdTaskSpec"]
