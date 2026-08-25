"""The scorer must not reward a model for failing quickly.

Latency and success rate come from the same telemetry rows. Before the success
term existed, a provider that errored out in half a second scored better on the
latency axis than one that did real work for thirty seconds, and the routing
table drifted towards whichever provider failed fastest.
"""

from __future__ import annotations

import unittest

from ai_orchestrator.models import ModelConfig, TaskSpec
from ai_orchestrator.routing import MIN_OBSERVATION_SAMPLES, score_route_candidate


def _model(identifier: str, *, quality: float = 0.80) -> ModelConfig:
    return ModelConfig(
        identifier,
        f"family-{identifier}",
        f"provider-{identifier}",
        "codex_cli",
        identifier,
        True,
        "available",
        ("tests",),
        "internal",
        1,
        quality_score=quality,
    )


def _task() -> TaskSpec:
    # Built directly rather than through from_mapping: the scorer only reads
    # max_context_tokens, and from_mapping would require a repository on disk.
    return TaskSpec(
        task_id="t",
        repo=".",
        task_type="tests",
        sensitivity="internal",
        max_context_tokens=32000,
    )


def _score(model: ModelConfig, observation: dict | None) -> float:
    return score_route_candidate(
        model, _task(), role_priority=0, role_size=1, observation=observation
    )


class SuccessRateTests(unittest.TestCase):
    def test_a_fast_failing_model_scores_below_a_slow_working_one(self) -> None:
        model = _model("m")
        fast_and_broken = _score(
            model,
            {"latency_ms": 530.0, "success_rate": 0.0, "sample_count": 12},
        )
        slow_but_working = _score(
            model,
            {"latency_ms": 240_000.0, "success_rate": 1.0, "sample_count": 12},
        )
        self.assertGreater(slow_but_working, fast_and_broken)

    def test_success_rate_moves_the_score(self) -> None:
        model = _model("m")
        healthy = _score(model, {"success_rate": 1.0, "sample_count": 20})
        failing = _score(model, {"success_rate": 0.0, "sample_count": 20})
        self.assertAlmostEqual(0.08, healthy - failing, places=6)


class ObservationSampleFloorTests(unittest.TestCase):
    def test_a_thin_sample_is_ignored_entirely(self) -> None:
        model = _model("m")
        thin = _score(
            model,
            {
                "latency_ms": 530.0,
                "success_rate": 0.0,
                "sample_count": MIN_OBSERVATION_SAMPLES - 1,
            },
        )
        self.assertEqual(_score(model, None), thin)

    def test_the_floor_lets_a_real_sample_through(self) -> None:
        model = _model("m")
        trusted = _score(
            model,
            {
                "latency_ms": 530.0,
                "success_rate": 0.0,
                "sample_count": MIN_OBSERVATION_SAMPLES,
            },
        )
        self.assertNotEqual(_score(model, None), trusted)

    def test_a_model_that_never_ran_is_not_punished_for_it(self) -> None:
        # Absence of evidence must stay neutral, or nothing new is ever chosen
        # and the incumbent keeps the role for good.
        never_ran = _score(_model("new"), None)
        proven = _score(_model("old"), {"success_rate": 1.0, "sample_count": 50})
        self.assertGreater(never_ran, proven - 0.05)


class WeightTests(unittest.TestCase):
    def test_a_perfect_candidate_scores_exactly_one(self) -> None:
        # Guards the weight table: any future term that does not come out of
        # the existing budget breaks this.
        model = ModelConfig(
            "perfect",
            "family",
            "provider",
            "codex_cli",
            "perfect",
            True,
            "available",
            ("tests",),
            "internal",
            1,
            quality_score=1.0,
            included_quota=True,
            marginal_cost_per_million=0.0,
            context_window=10_000_000,
            cache_supported=True,
            observed_latency_ms=0,
            cost_tier="free",
        )
        score = score_route_candidate(
            model,
            _task(),
            role_priority=0,
            role_size=1,
            observation={
                "success_rate": 1.0,
                "sample_count": 100,
                "latency_ms": 0.0,
                "cache_hit_rate": 1.0,
            },
        )
        self.assertAlmostEqual(1.0, score, places=6)


if __name__ == "__main__":
    unittest.main()
