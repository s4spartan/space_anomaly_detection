"""
risk_scorer.py
India Space Academy — AI & ML in Space Exploration
Student: Nirav Singh Dabhi | Roll: 13101980

Multi-factor Risk Scoring System for Deep Space Mission Anomalies.

Formula:
  Risk = w1*AnomalySeverity + w2*SubsystemCriticality
       + w3*RULUrgency + w4*CommunicationWindowFactor
       + w5*EnvironmentalFactor

Weights derived from spacecraft operations literature and
Chandrayaan-3 / Aditya-L1 subsystem criticality profiles.
"""

import numpy as np
from dataclasses import dataclass

# Subsystem criticality (0–1): higher = more critical
SUBSYSTEM_CRITICALITY = {
    "propulsion":    0.95,
    "power":         0.90,
    "attitude":      0.85,
    "thermal":       0.80,
    "communication": 0.70,
    "instruments":   0.55,
    "solar":         0.50,
}

# Risk level thresholds
RISK_LEVELS = {
    "CRITICAL": 0.80,
    "HIGH":     0.60,
    "MEDIUM":   0.40,
    "LOW":      0.20,
    "NOMINAL":  0.00,
}

# Risk formula weights — must sum to 1.0
WEIGHTS = {
    "anomaly_severity":     0.30,
    "subsystem_criticality":0.25,
    "rul_urgency":          0.20,
    "comm_window":          0.15,
    "environmental":        0.10,
}


@dataclass
class RiskProfile:
    total_score: float          # 0.0 – 1.0
    level: str                  # NOMINAL / LOW / MEDIUM / HIGH / CRITICAL
    components: dict
    recommendation_priority: str


class RiskScorer:
    """
    Computes a composite risk score for any anomaly event.

    Component definitions:

    1. anomaly_severity (0-1): raw anomaly score from detection model,
       normalised to [0,1]. Directly reflects how far readings deviate
       from nominal distributions.

    2. subsystem_criticality (0-1): fixed weight by subsystem type.
       Propulsion and power have highest criticality — failure = mission loss.

    3. rul_urgency (0-1): urgency increases non-linearly as RUL approaches 0.
       Formula: 1 - (rul / MAX_RUL)^0.5 — square root gives early warning.

    4. communication_window_factor (0-1): higher risk when next contact
       window is far away. If next window < 30 min, factor = 0.2 (low
       urgency — can wait for ground). If > 8 hours: factor = 0.9.

    5. environmental_factor (0-1): elevated during CME events, solar flares,
       or lunar night (Chandrayaan context). Environmental anomalies scored
       lower than fault anomalies by design.
    """

    MAX_RUL = 130  # max cycle horizon for urgency normalisation

    def compute(
        self,
        anomaly_severity: float,
        subsystem: str,
        rul_cycles: float = None,
        minutes_to_next_comm_window: float = 60.0,
        solar_event_active: bool = False,
        anomaly_type: str = "unknown"
    ) -> RiskProfile:

        # 1. Anomaly severity — direct input from model
        s1 = float(np.clip(anomaly_severity, 0, 1))

        # 2. Subsystem criticality
        s2 = SUBSYSTEM_CRITICALITY.get(subsystem, 0.65)

        # 3. RUL urgency — non-linear (square root)
        if rul_cycles is not None:
            rul_norm = np.clip(rul_cycles / self.MAX_RUL, 0, 1)
            s3 = float(1 - rul_norm ** 0.5)
        else:
            s3 = 0.4  # default moderate urgency when RUL unknown

        # 4. Communication window factor
        if minutes_to_next_comm_window < 30:
            s4 = 0.20   # can pass to ground soon
        elif minutes_to_next_comm_window < 120:
            s4 = 0.45
        elif minutes_to_next_comm_window < 480:
            s4 = 0.70
        else:
            s4 = 0.90   # > 8 hours — must act autonomously

        # 5. Environmental factor
        if solar_event_active and anomaly_type == "cme_event":
            s5 = 0.30   # environment, not spacecraft fault — lower weight
        elif solar_event_active:
            s5 = 0.60   # solar background elevates all risks
        else:
            s5 = 0.10

        # Weighted composite
        total = (WEIGHTS["anomaly_severity"]      * s1
               + WEIGHTS["subsystem_criticality"] * s2
               + WEIGHTS["rul_urgency"]           * s3
               + WEIGHTS["comm_window"]           * s4
               + WEIGHTS["environmental"]         * s5)

        total = float(np.clip(total, 0, 1))

        # Determine level
        level = "NOMINAL"
        for lvl, threshold in sorted(RISK_LEVELS.items(),
                                      key=lambda x: -x[1]):
            if total >= threshold:
                level = lvl
                break

        # Priority recommendation
        if level == "CRITICAL":
            priority = "IMMEDIATE AUTONOMOUS ACTION REQUIRED"
        elif level == "HIGH":
            priority = "Autonomous action recommended within 5 minutes"
        elif level == "MEDIUM":
            priority = "Monitor closely — prepare contingency"
        else:
            priority = "Log event — no immediate action required"

        return RiskProfile(
            total_score=round(total, 4),
            level=level,
            components={
                "anomaly_severity":      round(s1, 3),
                "subsystem_criticality": round(s2, 3),
                "rul_urgency":           round(s3, 3),
                "comm_window_factor":    round(s4, 3),
                "environmental_factor":  round(s5, 3),
            },
            recommendation_priority=priority
        )

    def score_batch(self, events: list) -> list:
        """Score a list of anomaly event dicts. Returns list of RiskProfiles."""
        return [self.compute(**e) for e in events]

    @staticmethod
    def level_color(level: str) -> str:
        """Streamlit/HTML colour for dashboard display."""
        return {
            "CRITICAL": "#CC0000",
            "HIGH":     "#FF6600",
            "MEDIUM":   "#FFB300",
            "LOW":      "#4CAF50",
            "NOMINAL":  "#2196F3",
        }.get(level, "#888888")
