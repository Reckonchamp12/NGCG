"""
tests/test_metrics.py
=====================
Unit tests for NGCG metrics.
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ngcg.metrics import (
    constancy_score,
    trajectory_diversity,
    discovery_rate,
    false_discovery_rate,
    f1_score_ngcg,
    true_law_constancy,
)


def _make_harmonic_traj(n=20, T=100, omega=1.0, seed=0):
    """Harmonic oscillator trajectories with different energies."""
    rng = np.random.default_rng(seed)
    E   = rng.uniform(0.5, 2.0, n)          # different energies
    t   = np.linspace(0, 10, T)
    q   = np.sqrt(2*E)[:, None] * np.cos(omega * t)[None, :]
    p   = -np.sqrt(2*E)[:, None] * np.sin(omega * t)[None, :]
    return np.stack([q, p], axis=-1).astype(np.float32)  # (N, T, 2)


class TestConstancyScore:
    def test_true_energy_is_constant(self):
        traj = _make_harmonic_traj()
        # H = p^2/2 + q^2/2
        sc = constancy_score("p**2/2 + q**2/2", ["q","p"], traj)
        assert sc < 1e-6, f"Energy should be constant, got {sc}"

    def test_non_conserved_has_high_score(self):
        traj = _make_harmonic_traj()
        # p alone is not conserved
        sc = constancy_score("p", ["q","p"], traj)
        assert sc > 0.1, f"p alone should not be constant, got {sc}"

    def test_empty_expr_returns_999(self):
        traj = _make_harmonic_traj()
        assert constancy_score("", ["q","p"], traj) == 999.0

    def test_invalid_expr_returns_999(self):
        traj = _make_harmonic_traj()
        assert constancy_score("not_a_valid_expression!!!@#", ["q","p"], traj) == 999.0


class TestTrajectoryDiversity:
    def test_genuine_invariant_has_high_diversity(self):
        traj = _make_harmonic_traj()
        ratio = trajectory_diversity("p**2/2 + q**2/2", ["q","p"], traj)
        assert ratio > 10.0, f"Energy diversity ratio should be >> 10, got {ratio}"

    def test_global_constant_has_low_diversity(self):
        traj = _make_harmonic_traj()
        # "1" is a global constant — same value for all trajectories
        ratio = trajectory_diversity("1", ["q","p"], traj)
        assert ratio < 1.0, f"Global constant should have near-zero diversity, got {ratio}"


class TestDiscoveryRate:
    def test_finds_law(self):
        cands = [("expr1", 0.001), ("expr2", 0.5)]
        DR, DR_strict = discovery_rate(cands, has_true_law=True, tol_loose=0.05)
        assert DR == 1.0

    def test_misses_law(self):
        cands = [("expr1", 0.3), ("expr2", 0.5)]
        DR, _ = discovery_rate(cands, has_true_law=True, tol_loose=0.05)
        assert DR == 0.0

    def test_no_law_always_zero(self):
        cands = [("expr1", 0.001)]
        DR, _ = discovery_rate(cands, has_true_law=False)
        assert DR == 0.0

    def test_empty_candidates(self):
        DR, DR_strict = discovery_rate([], has_true_law=True)
        assert DR == 0.0 and DR_strict == 0.0


class TestFalseDiscoveryRate:
    def test_all_true_positives(self):
        accepted = [("e1", 0.001), ("e2", 0.002)]
        fdr = false_discovery_rate(accepted, has_true_law=True, tol_loose=0.05)
        assert fdr == 0.0

    def test_all_false_positives(self):
        accepted = [("e1", 0.1), ("e2", 0.2)]
        fdr = false_discovery_rate(accepted, has_true_law=True, tol_loose=0.05)
        assert fdr == 1.0

    def test_no_law_any_accepted_is_fp(self):
        accepted = [("e1", 0.001)]
        fdr = false_discovery_rate(accepted, has_true_law=False)
        assert fdr == 1.0

    def test_empty_accepted(self):
        fdr = false_discovery_rate([], has_true_law=True)
        assert fdr == 0.0


class TestF1Score:
    def test_perfect(self):
        assert f1_score_ngcg(1.0, 0.0) == 1.0

    def test_zero_dr(self):
        assert f1_score_ngcg(0.0, 0.0) == 0.0

    def test_high_fdr(self):
        f1 = f1_score_ngcg(1.0, 1.0)
        assert f1 == 0.0


class TestTrueLawConstancy:
    def test_harmonic_energy_near_zero(self):
        traj = _make_harmonic_traj()
        tlc  = true_law_constancy("p**2/2 + q**2/2", ["q","p"], traj)
        assert tlc < 1e-5, f"True law constancy should be ≈0, got {tlc}"

    def test_no_law_returns_zero(self):
        traj = _make_harmonic_traj()
        tlc  = true_law_constancy(None, ["q","p"], traj)
        assert tlc == 0.0
