import itertools
import math
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from adaptive_gp import (
    aggregate,
    bind_study,
    coordinate_maximum,
    engine_identity,
    exploration_probability,
    fantasy_variance,
    fixed_baseline_point,
    inducing_basis,
    UCI_OPTION,
    pending_configurations,
    validate_opening_budget,
)
from logistic_gp import ELO_PER_LOGIT, LogisticGP, MixedSpace


class MixedAcquisitionTest(unittest.TestCase):
    def setUp(self):
        self.space = MixedSpace({
            "parameters": [
                {"name": "X", "type": "integer", "min": 0, "max": 100,
                 "default": 50, "scale": 20},
                {"name": "Y", "type": "integer", "min": 0, "max": 20,
                 "default": 10, "scale": 5},
            ],
            "max_candidates": 16,
            "max_grid": 100,
        })

    def test_dense_domain_is_separate_from_design(self):
        target = (37, 13)

        def score(points):
            values = np.asarray(points)
            return -np.sum((values - target) ** 2, axis=1)

        self.assertEqual(len(self.space.candidates), 16)
        self.assertNotIn(target, self.space.candidates)
        self.assertEqual(
            coordinate_maximum(self.space, self.space.candidates, score, set(), None),
            target,
        )

    def test_halton_design_handles_full_tuning_space(self):
        parameters = [
            {"name": f"X{i}", "type": "discrete", "values": [0, 1, 2], "default": 1}
            for i in range(25)
        ]
        space = MixedSpace({"parameters": parameters, "max_candidates": 64})
        self.assertEqual(len(space.candidates), 64)
        self.assertIn(space.default, space.candidates)

    def test_coordinate_search_matches_exhaustive_gp_ucb(self):
        domain = list(itertools.product(*self.space.coordinate_values))
        observed = [domain[index] for index in (0, 211, 702, 1050, 1537, 2120)]
        model = LogisticGP(self.space.prior_mean, self.space.kernel).fit(
            observed, [2, 7, 4, 8, 3, 6], [10] * len(observed))

        def score(points):
            mean, variance = model.predict(points)
            return mean + np.sqrt(variance)

        exact = domain[int(np.argmax(score(domain)))]
        found = coordinate_maximum(
            self.space, [*self.space.candidates, *observed], score, set(), None)
        self.assertEqual(found, exact)

    def test_coordinate_search_rejects_observations_outside_new_domain(self):
        def score(points):
            return -np.asarray(points)[:, 0]

        found = coordinate_maximum(
            self.space, [(-10, 10), *self.space.candidates], score, set(), None)
        self.assertEqual(found[0], 0)

    def test_structural_constraint_is_preserved(self):
        space = MixedSpace({
            "parameters": [
                {"name": "X", "type": "integer", "min": 0, "max": 10,
                 "default": 5},
                {"name": "MODE", "type": "categorical", "values": ["on", "off"],
                 "default": "on", "off_values": ["off"]},
            ],
        })

        def score(points):
            values = np.asarray(points)
            return -(values[:, 0] - 5) ** 2 - values[:, 1]

        ordinary = coordinate_maximum(space, space.candidates, score, set(), False)
        structural = coordinate_maximum(space, space.candidates, score, set(), True)
        self.assertEqual(space.knobs(ordinary), {"X": 5, "MODE": "on"})
        self.assertEqual(space.knobs(structural), {"X": 5, "MODE": "off"})

    def test_inactive_parameters_are_canonicalized(self):
        space = MixedSpace({
            "parameters": [
                {"name": "MARGIN", "type": "integer", "min": 0, "max": 2,
                 "default": 1},
                {"name": "MODE", "type": "discrete", "values": [0, 99],
                 "default": 0},
            ],
            "conditions": [{"when": {"MODE": [99]}, "reset": ["MARGIN"]}],
        })
        off = space.canonical({"MARGIN": 2, "MODE": 99})
        self.assertEqual(off, space.canonical({"MARGIN": 0, "MODE": 99}))
        self.assertEqual(space.knobs(off), {"MARGIN": 1, "MODE": 99})

        def score(points):
            values = np.asarray(points)
            return values[:, 1] * 10 + values[:, 0]

        found = coordinate_maximum(space, space.candidates, score, set(), None)
        self.assertEqual(found, off)

    def test_inactive_parameters_can_have_explicit_values(self):
        space = MixedSpace({
            "parameters": [
                {"name": "MARGIN", "type": "integer", "min": 0, "max": 2,
                 "default": 1},
                {"name": "MODE", "type": "discrete", "values": [0, 99],
                 "default": 0},
            ],
            "conditions": [{"when": {"MODE": [99]}, "set": {"MARGIN": 0},
                            "reset": []}],
        })
        off = space.canonical({"MARGIN": 2, "MODE": 99})
        self.assertEqual(space.knobs(off), {"MARGIN": 0, "MODE": 99})

    def test_one_won_pair_does_not_collapse_uncertainty(self):
        point = [(0,) * 9]
        model = LogisticGP().fit(point, [1], [1])
        mean, variance = model.predict(point)
        self.assertLess(mean[0] * ELO_PER_LOGIT, 20)
        self.assertGreater(math.sqrt(variance[0]) * ELO_PER_LOGIT, 60)

        model = LogisticGP().fit(point, [0], [1])
        mean, variance = model.predict(point)
        self.assertGreater(mean[0] * ELO_PER_LOGIT, -20)
        self.assertGreater(math.sqrt(variance[0]) * ELO_PER_LOGIT, 60)

    def test_color_swapped_pair_is_one_bounded_observation(self):
        point = self.space.default
        batch = {
            "knobs": self.space.knobs(point),
            "wins": 2,
            "draws": 0,
            "losses": 0,
        }
        points, design, success, trials = aggregate([batch], 0.5, self.space)
        self.assertEqual(points, [point])
        np.testing.assert_array_equal(design, [[1]])
        self.assertEqual(success, [1])
        self.assertEqual(trials, [1])

    def test_mixed_space_has_no_hidden_nondefault_penalty(self):
        spec = {
            "parameters": [{
                "name": "MODE", "type": "categorical",
                "values": ["current", "simpler"], "default": "current",
            }],
        }
        neutral = MixedSpace(spec)
        np.testing.assert_array_equal(neutral.prior_mean(neutral.candidates), [0, 0])
        penalized = MixedSpace(spec | {"clause_logit_prior": -0.2})
        np.testing.assert_array_equal(penalized.prior_mean(penalized.candidates), [0, -0.2])

    def test_additive_kernel_transfers_evidence_axis_by_axis(self):
        product = self.space.kernel([(0, 0)], [(0, 20)])[0, 0]
        additive = MixedSpace({
            "parameters": self.space.parameters,
            "interaction_fraction": 0,
        }).kernel([(0, 0)], [(0, 20)])[0, 0]
        self.assertGreater(additive, 100 * product)

    def test_fixed_baseline_conditions_its_value_to_zero(self):
        self.space.condition(self.space.default)
        points = [self.space.default, (0, 0), (100, 20)]
        np.testing.assert_allclose(self.space.kernel([self.space.default], points), 0, atol=1e-15)
        self.assertAlmostEqual(self.space.kernel_diagonal([self.space.default])[0], 0)

    def test_large_noisy_comparison_graph_converges(self):
        rng = np.random.default_rng(194)
        points = list(itertools.product(range(8), repeat=2))
        design = np.zeros((800, len(points)))
        left = rng.integers(len(points), size=len(design))
        right = rng.integers(len(points), size=len(design))
        right[right == left] = (right[right == left] + 1) % len(points)
        design[np.arange(len(design)), left] = 1
        design[np.arange(len(design)), right] = -1
        success = rng.integers(3, size=len(design)) / 2
        model = LogisticGP(self.space.prior_mean, self.space.kernel).fit_comparisons(
            points, design, success, np.ones(len(design)))
        mean, variance = model.predict(points)
        self.assertTrue(np.all(np.isfinite(mean)) and np.all(variance > 0))

    def test_exploration_never_decays_away(self):
        probabilities = [exploration_probability(n, 0.5, 0.2, 40) for n in range(10000)]
        self.assertEqual(probabilities[0], 0.5)
        self.assertTrue(all(probability > 0.2 for probability in probabilities))
        self.assertLess(probabilities[-1], 0.22)

    def test_pair_at_a_time_fills_every_lane(self):
        self.assertEqual(pending_configurations(10, 1), 10)
        self.assertEqual(pending_configurations(10, 3), 4)
        self.assertEqual(pending_configurations(10, 10), 1)

    def test_opening_budget_rejects_silent_fallback(self):
        with tempfile.NamedTemporaryFile("w") as openings:
            openings.write("one\ntwo\nthree\n")
            openings.flush()
            validate_opening_budget(openings.name, 2, 2, 1)
            with self.assertRaisesRegex(ValueError, "needs opening 4"):
                validate_opening_budget(openings.name, 2, 3, 1)

    def test_pending_pairs_retain_noise_and_multiplicity(self):
        point = self.space.default
        points = [point]
        prior = np.diag(self.space.kernel(points, points))
        one = fantasy_variance(None, self.space, [(point, None)], points, prior, 1)
        ten = fantasy_variance(None, self.space, [(point, None)] * 10, points, prior, 1)
        batch = fantasy_variance(None, self.space, [(point, None)], points, prior, 10)
        self.assertTrue(np.all(0 < ten) and np.all(ten < one) and np.all(one < prior))
        np.testing.assert_allclose(ten, batch)

    def test_sparse_basis_keeps_heavily_tested_endpoints(self):
        points = [(0, 10), (50, 10), (100, 10)]
        design = np.eye(3)
        trials = np.array([1, 1, 500])
        basis = inducing_basis(points, design, trials, self.space, 2)
        self.assertIn(self.space.default, basis)
        self.assertIn((100, 10), basis)

    def test_uci_option_parser_keeps_multiword_names(self):
        line = "option name Null threat margin type spin default 200"
        self.assertEqual(UCI_OPTION.match(line).group(1), "Null threat margin")

    def test_engine_identity_hashes_binary_and_argument_files(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = pathlib.Path(directory, "engine")
            tables = pathlib.Path(directory, "tables")
            binary.write_bytes(b"first")
            tables.write_bytes(b"weights")
            first = engine_identity(str(binary), str(tables), {"X": 1})
            binary.write_bytes(b"second")
            second = engine_identity(str(binary), str(tables), {"X": 1})
        self.assertNotEqual(first, second)
        self.assertEqual(len(first["files"]), 2)

    def test_state_refuses_a_different_study(self):
        state = {"next_opening": 1, "batches": []}
        bind_study(state, {"version": 1, "tc": "3+0.1"})
        bind_study(state, {"version": 1, "tc": "3+0.1"})
        with self.assertRaisesRegex(RuntimeError, "tc changed"):
            bind_study(state, {"version": 1, "tc": "5+0.1"})

    def test_identical_parameterized_baseline_is_not_an_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = pathlib.Path(directory, "engine")
            binary.write_bytes(b"engine")
            space = MixedSpace({
                "parameters": [{
                    "name": "X", "type": "integer",
                    "min": 0, "max": 2, "default": 1,
                }],
            })
            args = SimpleNamespace(
                engine=str(binary), engine_args="",
                baseline_engine=str(binary), baseline_args="",
                baseline_options={"X": 1},
            )
            self.assertEqual(fixed_baseline_point(args, space), space.default)
            args.baseline_options = {"X": 2}
            self.assertIsNone(fixed_baseline_point(args, space))

    def test_unidentified_observations_are_not_adopted(self):
        state = {"next_opening": 2, "batches": [{"wins": 1}]}
        with self.assertRaisesRegex(RuntimeError, "no study identity"):
            bind_study(state, {"version": 1})


if __name__ == "__main__":
    unittest.main()
