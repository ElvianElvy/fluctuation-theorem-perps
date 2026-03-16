"""3-State Thermodynamic Regime Classifier — Data-Adaptive.

Maps physics signals to POSITION SIZING multipliers, not binary trade/no-trade.
The key insight: funding arb is always profitable on average. Physics tells
us WHEN to size up (equilibrium) and WHEN to size down (NESS).

Thresholds are computed from rolling training windows (percentiles),
not hardcoded — automatically adapts to different assets and β scales.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import numpy as np


class RegimeState(Enum):
    EQUILIBRIUM = "EQUILIBRIUM"
    WARM = "WARM"
    NESS = "NESS"


@dataclass
class RegimeConfig:
    """Per-regime position sizing."""
    position_multiplier: float   # multiplier on base position size
    label: str


REGIME_CONFIGS = {
    RegimeState.EQUILIBRIUM: RegimeConfig(position_multiplier=1.0, label="Full exposure"),
    RegimeState.WARM: RegimeConfig(position_multiplier=0.60, label="Reduced exposure"),
    RegimeState.NESS: RegimeConfig(position_multiplier=0.25, label="Minimal exposure"),
}


@dataclass
class RegimeClassification:
    state: RegimeState
    config: RegimeConfig
    confidence: float
    relaxation_ratio: float
    temperature_zscore: float
    entropy_rate: float
    je_health: float


class ThermodynamicRegimeClassifier:
    """Data-adaptive regime classifier.

    Uses PERCENTILES from training history for entropy and JE thresholds,
    ensuring automatic adaptation across different β scales and assets.

    Classification:
      EQUILIBRIUM: τ_relax < τ_funding AND temperature stable AND entropy healthy
      NESS: τ_relax >> τ_funding OR entropy extremely low OR JE extremely broken
      WARM: everything else
    """

    def __init__(
        self,
        r_safe: float = 1.05,       # slightly above 1.0 to be less restrictive
        r_danger: float = 1.4,       # only flag NESS for clearly slow relaxation
        entropy_ness_pct: float = 10, # bottom 10% of entropy → NESS
        entropy_equil_pct: float = 40, # above 40th percentile → EQUILIBRIUM candidate
        je_danger_pct: float = 90,    # top 10% of JE deviation → NESS
        temp_equil_threshold: float = -0.5,  # not dramatically hotter than average
    ):
        self.r_safe = r_safe
        self.r_danger = r_danger
        self.entropy_ness_pct = entropy_ness_pct
        self.entropy_equil_pct = entropy_equil_pct
        self.je_danger_pct = je_danger_pct
        self.temp_equil = temp_equil_threshold

    def classify_with_history(
        self,
        relaxation_ratio: float,
        temperature_zscore: float,
        entropy_rate: float,
        je_health: float,
        entropy_history: np.ndarray,
        je_history: np.ndarray,
    ) -> RegimeClassification:
        """Classify using data-adaptive thresholds from training history."""

        # Compute adaptive thresholds from history
        if len(entropy_history) > 10:
            ent_ness_thresh = np.percentile(entropy_history, self.entropy_ness_pct)
            ent_equil_thresh = np.percentile(entropy_history, self.entropy_equil_pct)
        else:
            ent_ness_thresh = -1e10
            ent_equil_thresh = 0.0

        if len(je_history) > 10:
            je_danger_thresh = np.percentile(je_history, self.je_danger_pct)
        else:
            je_danger_thresh = 1e10

        # NESS: any danger condition
        is_ness = (
            relaxation_ratio > self.r_danger
            or entropy_rate < ent_ness_thresh
            or je_health > je_danger_thresh
        )

        # EQUILIBRIUM: all favorable
        is_equil = (
            relaxation_ratio < self.r_safe
            and temperature_zscore > self.temp_equil
            and entropy_rate > ent_equil_thresh
            and not is_ness
        )

        if is_ness:
            state = RegimeState.NESS
            confidence = 0.8
        elif is_equil:
            state = RegimeState.EQUILIBRIUM
            confidence = 0.9
        else:
            state = RegimeState.WARM
            confidence = 0.5

        return RegimeClassification(
            state=state, config=REGIME_CONFIGS[state],
            confidence=confidence,
            relaxation_ratio=relaxation_ratio,
            temperature_zscore=temperature_zscore,
            entropy_rate=entropy_rate,
            je_health=je_health,
        )

    def classify_series(
        self, signals_df, train_window: int = 360,
    ) -> list[RegimeClassification]:
        """Classify every point using rolling training history."""
        ent = signals_df["entropy_rate"].values.astype(float)
        jeh = signals_df["je_health"].values.astype(float)
        results = []
        for i in range(len(signals_df)):
            row = signals_df.iloc[i]
            # Use all PAST entropy/JE values for adaptive thresholds
            ent_hist = ent[max(0, i - train_window):i]
            je_hist = jeh[max(0, i - train_window):i]
            rc = self.classify_with_history(
                row["relaxation_ratio"], row["temperature_zscore"],
                row["entropy_rate"], row["je_health"],
                ent_hist, je_hist,
            )
            results.append(rc)
        return results
