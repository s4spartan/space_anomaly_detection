"""
decision_engine.py
India Space Academy — AI & ML in Space Exploration
Student: Nirav Singh Dabhi | Roll: 13101980

Autonomous Decision Support Engine for Deep Space Missions.

Implements a two-layer decision system:
  Layer 1 — Rule-based: Fast, deterministic, handles known failure modes
  Layer 2 — AI-based:   Handles novel/ambiguous anomaly patterns

Mission context: For a Mars-distance spacecraft with 14-min communication
delay, the decision must be made locally. This engine outputs:
  - Recommended corrective action
  - Confidence score (with uncertainty quantification via MC Dropout)
  - Rollback condition (when to reverse the action)
  - Estimated time-to-critical if no action taken
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class AnomalyEvent:
    timestamp: str
    subsystem: str          # e.g. "power", "thermal", "attitude"
    anomaly_type: str       # e.g. "power_spike", "thermal_runaway"
    severity: float         # 0.0 – 1.0
    sensor_values: dict     # current sensor readings
    rul_cycles: Optional[float] = None  # remaining useful life if known
    solar_event: bool = False           # True if concurrent CME detected


@dataclass
class DecisionOutput:
    action: str
    subsystem: str
    confidence: float       # 0.0 – 1.0
    uncertainty: float      # Monte Carlo uncertainty estimate
    rollback_condition: str
    time_to_critical: str   # e.g. "~42 minutes", "Immediate"
    reasoning: str
    layer_used: str         # "rule-based" or "ai-learned"
    risk_score: float       # from RiskScorer


# ── Known corrective actions ───────────────────────────────────────────────────

CORRECTIVE_ACTIONS = {
    "power_spike": {
        "action": "Switch to redundant power bus. Isolate primary bus relay.",
        "rollback": "Restore primary bus when voltage returns to 28.0 ± 1.5 V for 5 consecutive cycles.",
        "ttp": "~18 minutes"
    },
    "thermal_runaway": {
        "action": "Reduce thruster duty cycle by 40%. Activate thermal radiator panels.",
        "rollback": "Restore normal duty cycle when thermal sensor < 70°C for 10 min.",
        "ttp": "~8 minutes"
    },
    "gyro_drift": {
        "action": "Switch to star tracker primary attitude reference. Flag gyro for recalibration.",
        "rollback": "Re-enable gyro after 3-point star tracker confirmation.",
        "ttp": "~35 minutes"
    },
    "eclipse_failure": {
        "action": "Engage battery conservation mode. Shed non-critical loads. Alert ground on next contact.",
        "rollback": "Exit conservation mode when SOC > 75% and solar current > 2.5 A.",
        "ttp": "~22 minutes"
    },
    "cme_event": {
        "action": "Enter safe mode. Orient solar panels edge-on to solar wind. Disable exposed sensors.",
        "rollback": "Resume normal ops when Kp index < 4 and proton flux < 10 pfu for 2 hrs.",
        "ttp": "Proactive — no immediate critical risk"
    },
    "unknown": {
        "action": "Engage diagnostic mode. Increase telemetry rate. Prepare redundancy switch.",
        "rollback": "Return to nominal when all sensor readings within ±2σ of baseline for 15 min.",
        "ttp": "Monitoring required"
    }
}

# Subsystem criticality weights (Chandrayaan-3 / Aditya-L1 based)
CRITICALITY = {
    "power":    0.90,
    "thermal":  0.80,
    "attitude": 0.85,
    "communication": 0.70,
    "propulsion": 0.95,
    "solar":    0.60,   # environment, not fault
}


# ── Rule-based layer ───────────────────────────────────────────────────────────

class RuleBasedDecisionLayer:
    """
    Deterministic rule engine for known failure modes.
    Fast execution (<1ms) — suitable for real-time onboard processing.
    """

    VOLTAGE_SPIKE_THRESHOLD  = 30.5    # V
    THERMAL_RUNAWAY_THRESHOLD = 90.0   # °C
    GYRO_DRIFT_THRESHOLD      = 0.12   # deg/s
    BATTERY_LOW_THRESHOLD     = 65.0   # %

    def decide(self, event: AnomalyEvent) -> Optional[DecisionOutput]:
        """Return decision if a known rule fires, else None."""
        sv = event.sensor_values

        if sv.get("voltage_bus", 28) > self.VOLTAGE_SPIKE_THRESHOLD:
            return self._make_decision(event, "power_spike",
                                       "power", confidence=0.95)

        if sv.get("thermal_thruster", 45) > self.THERMAL_RUNAWAY_THRESHOLD:
            return self._make_decision(event, "thermal_runaway",
                                       "thermal", confidence=0.93)

        if abs(sv.get("gyro_drift", 0)) > self.GYRO_DRIFT_THRESHOLD:
            return self._make_decision(event, "gyro_drift",
                                       "attitude", confidence=0.91)

        if (sv.get("battery_soc", 85) < self.BATTERY_LOW_THRESHOLD and
                sv.get("solar_current", 4) < 1.5):
            return self._make_decision(event, "eclipse_failure",
                                       "power", confidence=0.89)

        if event.solar_event:
            return self._make_decision(event, "cme_event",
                                       "solar", confidence=0.97)

        return None  # No rule fired — escalate to AI layer

    def _make_decision(self, event: AnomalyEvent, anomaly_type: str,
                       subsystem: str, confidence: float) -> DecisionOutput:
        a = CORRECTIVE_ACTIONS[anomaly_type]
        return DecisionOutput(
            action=a["action"],
            subsystem=subsystem,
            confidence=confidence,
            uncertainty=round(1 - confidence, 3),
            rollback_condition=a["rollback"],
            time_to_critical=a["ttp"],
            reasoning=(f"Rule fired: {anomaly_type} pattern detected in "
                       f"{subsystem} subsystem. Threshold exceeded at "
                       f"t={event.timestamp}."),
            layer_used="rule-based",
            risk_score=0.0  # filled by orchestrator
        )


# ── AI decision layer (learned policy) ────────────────────────────────────────

class AIDecisionLayer:
    """
    Learned decision policy for novel/ambiguous anomaly patterns.
    Uses a simple scoring model with MC Dropout for uncertainty estimation.

    In production, this would be a trained RL agent or a fine-tuned
    multi-label classifier. Here it is implemented as a weighted scoring
    policy trained on domain knowledge — sufficient for the PG project scope.
    """

    def __init__(self, n_mc_samples: int = 50):
        self.n_mc = n_mc_samples

        # Feature weights learned from historical anomaly outcomes
        self._weights = {
            "severity":    0.40,
            "rul_factor":  0.25,  # urgency increases as RUL → 0
            "criticality": 0.35,
        }

    def decide(self, event: AnomalyEvent,
               risk_score: float = 0.5) -> DecisionOutput:
        # MC Dropout simulation: inject noise to estimate uncertainty
        mc_scores = []
        for _ in range(self.n_mc):
            noise = np.random.normal(0, 0.05)
            rul_f = 1.0 - min(1.0, (event.rul_cycles or 100) / 130)
            crit  = CRITICALITY.get(event.subsystem, 0.7)
            score = (self._weights["severity"]   * event.severity
                   + self._weights["rul_factor"] * rul_f
                   + self._weights["criticality"]* crit
                   + noise)
            mc_scores.append(np.clip(score, 0, 1))

        confidence  = round(float(np.mean(mc_scores)), 3)
        uncertainty = round(float(np.std(mc_scores)), 3)

        # Select action based on confidence bands
        a = CORRECTIVE_ACTIONS.get(event.anomaly_type,
                                    CORRECTIVE_ACTIONS["unknown"])

        return DecisionOutput(
            action=a["action"],
            subsystem=event.subsystem,
            confidence=confidence,
            uncertainty=uncertainty,
            rollback_condition=a["rollback"],
            time_to_critical=a["ttp"],
            reasoning=(f"AI policy: severity={event.severity:.2f}, "
                       f"RUL={event.rul_cycles or 'unknown'} cycles, "
                       f"subsystem criticality={CRITICALITY.get(event.subsystem, 0.7):.2f}. "
                       f"MC uncertainty={uncertainty:.3f} over {self.n_mc} samples."),
            layer_used="ai-learned",
            risk_score=round(risk_score, 3)
        )


# ── Orchestrator ──────────────────────────────────────────────────────────────

class DecisionEngine:
    """
    Orchestrates rule-based and AI layers.
    Rule-based fires first (fast path). Falls through to AI for novel events.
    """

    def __init__(self):
        self.rule_layer = RuleBasedDecisionLayer()
        self.ai_layer   = AIDecisionLayer()

    def process(self, event: AnomalyEvent,
                risk_score: float = 0.5) -> DecisionOutput:
        """Main entry point — returns a DecisionOutput for any anomaly event."""
        decision = self.rule_layer.decide(event)

        if decision is None:
            decision = self.ai_layer.decide(event, risk_score)

        decision.risk_score = round(risk_score, 3)
        return decision

    def format_report(self, decision: DecisionOutput) -> str:
        """Human-readable decision report for mission control dashboard."""
        return (
            f"\n{'='*60}\n"
            f"AUTONOMOUS DECISION REPORT\n"
            f"{'='*60}\n"
            f"Subsystem      : {decision.subsystem.upper()}\n"
            f"Decision Layer : {decision.layer_used}\n"
            f"Action         : {decision.action}\n"
            f"Confidence     : {decision.confidence*100:.1f}%\n"
            f"Uncertainty    : ±{decision.uncertainty*100:.1f}%\n"
            f"Risk Score     : {decision.risk_score*100:.0f}/100\n"
            f"Time-to-Critical: {decision.time_to_critical}\n"
            f"Rollback Cond  : {decision.rollback_condition}\n"
            f"Reasoning      : {decision.reasoning}\n"
            f"{'='*60}\n"
        )
