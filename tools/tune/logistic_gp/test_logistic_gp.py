import argparse
from collections import Counter
import itertools
import json
import math
import pathlib
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import sunfish

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import sunfish_gate

from adaptive_gp import (
    aggregate,
    bind_study,
    checkpoint_state,
    choose,
    choose_opponent,
    commit_selection,
    compatible_seed_batches,
    coordinate_maximum,
    design_variance,
    duration,
    empirical_mean,
    engine_identity,
    exploration_probability,
    exploitation,
    fantasy_variance,
    fixed_baseline_point,
    gate_policy,
    import_seed_batches,
    inducing_basis,
    load_state,
    OpeningSchedule,
    selection_state,
    save_state,
    state_file_identity,
    UCI_OPTION,
    pending_configurations,
    validate_opening_budget,
)
from logistic_gp import ELO_PER_LOGIT, LogisticGP, MixedSpace
from report_gp import report_domain


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

    def test_axis_design_moves_one_kernel_length_at_a_time(self):
        self.assertEqual(self.space.axis_design(), [
            (50, 10), (30, 10), (70, 10), (50, 5), (50, 15),
        ])

    def test_length_scale_widens_axis_design(self):
        self.space.length_scale = 2
        self.assertEqual(self.space.axis_design(), [
            (50, 10), (10, 10), (90, 10), (50, 0), (50, 20),
        ])

    def test_axis_initial_design_changes_only_one_coordinate(self):
        state = {"batches": [{
            "knobs": self.space.knobs(self.space.default),
            "wins": 0, "draws": 2, "losses": 0,
        }]}
        args = SimpleNamespace(
            pair_weight=1, inducing=0, initial_design=5, axis_design=True,
            explore_start=0, explore_floor=0, explore_half_life=1,
            exploration=0, explore_optimism=0, pairs=1,
            gate_all=False, acquisition_restarts=4,
        )
        candidates = sorted(set(self.space.candidates + self.space.axis_design()))
        vector, diagnostics = choose(
            state, self.space.prior_mean, candidates, [], args, self.space)
        self.assertEqual(diagnostics["mode"], "design")
        self.assertEqual(sum(a != b for a, b in zip(vector, self.space.default)), 1)

    def test_local_acquisition_changes_one_coordinate_from_observed(self):
        observed = (30, 5)
        state = {"batches": [{
            "knobs": self.space.knobs(observed),
            "wins": 2, "draws": 0, "losses": 0,
        }], "selections": 5}
        args = SimpleNamespace(
            pair_weight=1, inducing=0, initial_design=1, axis_design=False,
            local_acquisition=True, explore_start=0, explore_floor=0,
            explore_half_life=1, exploration=0, explore_optimism=0,
            pairs=1, gate_all=False, gate_design=False,
            acquisition_restarts=4,
        )
        vector, diagnostics = choose(
            state, self.space.prior_mean, self.space.candidates, [], args, self.space)
        self.assertEqual(diagnostics["mode"], "ucb")
        self.assertLessEqual(sum(a != b for a, b in zip(vector, observed)), 1)

        other, _ = choose(
            state, self.space.prior_mean, self.space.candidates,
            [(vector, None)], args, self.space)
        self.assertNotEqual(other, vector)

    def test_local_acquisition_expands_only_from_supported_points(self):
        unsupported = (30, 5)
        state = {"batches": [
            {"knobs": self.space.knobs(self.space.default),
             "wins": 0, "draws": 2, "losses": 0},
            {"knobs": self.space.knobs(unsupported),
             "wins": 2, "draws": 0, "losses": 0},
        ], "selections": 5}
        args = SimpleNamespace(
            pair_weight=1, inducing=0, initial_design=1, axis_design=False,
            local_acquisition=True, local_support=2,
            explore_start=0, explore_floor=0, explore_half_life=1,
            exploration=0, explore_optimism=0, pairs=1,
            gate_all=False, gate_design=False, acquisition_restarts=4,
        )
        vector, _ = choose(
            state, self.space.prior_mean, self.space.candidates, [], args, self.space)
        self.assertLessEqual(
            sum(a != b for a, b in zip(vector, self.space.default)), 1)

    def test_bounded_coordinate_ascent_stays_one_step_from_a_seed(self):
        target = (37, 13)

        def score(points):
            values = np.asarray(points)
            return -np.sum((values - target) ** 2, axis=1)

        point = coordinate_maximum(
            self.space, [self.space.default], score, set(), None, steps=1)
        self.assertNotEqual(point, target)
        self.assertEqual(sum(a != b for a, b in zip(point, self.space.default)), 1)

    def test_local_coordinate_ascent_stays_inside_one_kernel_radius(self):
        target = (100, 20)

        def score(points):
            values = np.asarray(points)
            return -np.sum((values - target) ** 2, axis=1)

        point = coordinate_maximum(
            self.space, [self.space.default], score, set(), None, steps=1, local=True)
        self.assertEqual(sum(a != b for a, b in zip(point, self.space.default)), 1)
        self.assertLessEqual(abs(point[0] - self.space.default[0]), 20)
        self.assertLessEqual(abs(point[1] - self.space.default[1]), 5)

    def test_gate_all_keeps_acquisition_in_the_validated_design(self):
        target = np.array((37, 13))

        class Model:
            @staticmethod
            def predict(points):
                mean = -np.sum((np.asarray(points) - target) ** 2, axis=1)
                return mean, np.ones(len(points))

        state = {
            "batches": [{
                "knobs": self.space.knobs(self.space.default),
                "wins": 1, "draws": 0, "losses": 1,
            }],
            "selections": 1,
        }
        args = SimpleNamespace(
            pair_weight=.5, inducing=0, initial_design=1,
            explore_start=0, explore_floor=0, explore_half_life=1,
            exploration=1, explore_optimism=0, pairs=1,
            gate_all=True, acquisition_restarts=4,
        )
        candidates = [point for point in self.space.candidates
                      if point != self.space.default]
        vector, _ = choose(
            state, self.space.prior_mean, candidates, [], args, self.space, Model())
        self.assertIn(vector, candidates)
        self.assertNotEqual(vector, tuple(target))

    def test_gate_design_retry_stays_in_the_validated_set(self):
        target = np.array((37, 13))

        class Model:
            @staticmethod
            def predict(points):
                mean = -np.sum((np.asarray(points) - target) ** 2, axis=1)
                return mean, np.ones(len(points))

        state = {
            "batches": [{
                "knobs": self.space.knobs(self.space.default),
                "wins": 1, "draws": 0, "losses": 1,
            }],
            "selections": 1,
        }
        args = SimpleNamespace(
            pair_weight=.5, inducing=0, initial_design=1,
            explore_start=0, explore_floor=0, explore_half_life=1,
            exploration=1, explore_optimism=0, pairs=1,
            gate_all=False, acquisition_restarts=4,
        )
        candidates = [point for point in self.space.candidates
                      if point != self.space.default]
        vector, _ = choose(
            state, self.space.prior_mean, candidates, [], args, self.space,
            Model(), validated={tuple(target)}, validated_only=True)
        self.assertEqual(vector, tuple(target))

    def test_exploration_can_validate_an_unseen_design_point(self):
        class Model:
            @staticmethod
            def predict(points):
                values = np.asarray(points)
                return np.zeros(len(points)), np.sum(values * values, axis=1) + 1

        safe = self.space.canonical({"X": 0, "Y": 10})
        state = {
            "batches": [{
                "knobs": self.space.knobs(self.space.default),
                "wins": 1, "draws": 0, "losses": 1,
            }],
        }
        args = SimpleNamespace(
            pair_weight=.5, inducing=0, initial_design=1,
            explore_start=1, explore_floor=1, explore_half_life=1,
            exploration=1, explore_optimism=0, pairs=1,
            gate_all=False, acquisition_restarts=4,
        )
        candidates = [point for point in self.space.candidates
                      if point != self.space.default]
        vector, diagnostics = choose(
            state, self.space.prior_mean, candidates, [], args, self.space,
            Model(), validated={safe})
        self.assertEqual(diagnostics["mode"], "explore")
        self.assertIn(vector, candidates)
        self.assertNotEqual(vector, safe)

    def test_exploration_does_not_require_a_validated_challenger(self):
        class Model:
            @staticmethod
            def predict(points):
                return np.zeros(len(points)), np.ones(len(points))

        default = self.space.default
        state = {"batches": [], "selections": 1}
        args = SimpleNamespace(
            pair_weight=.5, inducing=0, initial_design=1,
            explore_start=1, explore_floor=1, explore_half_life=1,
            exploration=1, explore_optimism=0, pairs=1,
            gate_all=False, acquisition_restarts=4,
        )
        vector, diagnostics = choose(
            state, self.space.prior_mean, self.space.candidates, [], args,
            self.space, Model(), forbidden={default}, validated={default},
            observation_counts=Counter({default: 1}))
        self.assertEqual(diagnostics["mode"], "explore")
        self.assertNotEqual(vector, default)
        self.assertEqual(state["exploration_credit"], 0)

    def test_exploration_prefers_unseen_points(self):
        observed, unseen = self.space.candidates[:2]

        class Model:
            @staticmethod
            def predict(points):
                variance = np.array([10 if point == observed else 1 for point in points])
                return np.zeros(len(points)), variance

        state = {"batches": [], "selections": 1}
        args = SimpleNamespace(
            pair_weight=.5, inducing=0, initial_design=1,
            explore_start=1, explore_floor=1, explore_half_life=1,
            exploration=1, explore_optimism=0, pairs=1,
            gate_all=False, acquisition_restarts=4,
        )
        vector, diagnostics = choose(
            state, self.space.prior_mean, [observed, unseen], [], args,
            self.space, Model(), observation_counts=Counter({observed: 1}))
        self.assertEqual(diagnostics["mode"], "explore")
        self.assertEqual(vector, unseen)

    def test_exploration_avoids_pending_points(self):
        pending, unseen = self.space.candidates[:2]

        class Model:
            @staticmethod
            def predict(points):
                variance = np.array([10 if point == pending else 1 for point in points])
                return np.zeros(len(points)), variance

            @staticmethod
            def predict_covariance(points):
                return np.zeros(len(points)), np.eye(len(points))

            @staticmethod
            def predict_cross_covariance(left, right):
                return np.zeros((len(left), len(right)))

        state = {"batches": [], "selections": 1}
        args = SimpleNamespace(
            pair_weight=.5, inducing=0, initial_design=1,
            explore_start=1, explore_floor=1, explore_half_life=1,
            exploration=1, explore_optimism=0, pairs=1,
            gate_all=False, acquisition_restarts=4,
        )
        vector, diagnostics = choose(
            state, self.space.prior_mean, [pending, unseen], [(pending, None)], args,
            self.space, Model(), observation_counts=Counter())
        self.assertEqual(diagnostics["mode"], "explore")
        self.assertEqual(vector, unseen)

    def test_exploration_drops_supportedly_dominated_points(self):
        weak = self.space.canonical({"X": 100, "Y": 20})
        plausible = self.space.canonical({"X": 0, "Y": 10})

        class Model:
            @staticmethod
            def predict(points):
                mean = np.array([-10 if point == weak else 0 for point in points])
                variance = np.array([4 if point == weak else 1 for point in points])
                return mean, variance

        state = {"batches": [], "selections": 1}
        args = SimpleNamespace(
            pair_weight=.5, inducing=0, initial_design=1,
            explore_start=1, explore_floor=1, explore_half_life=1,
            exploration=1, explore_optimism=0, explore_confidence=1.96, pairs=1,
            gate_all=False, acquisition_restarts=4,
        )
        vector, diagnostics = choose(
            state, self.space.prior_mean, [weak, plausible], [], args,
            self.space, Model(), validated={weak, plausible},
            observation_counts=Counter({weak: 1, plausible: 1}))
        self.assertEqual(diagnostics["mode"], "explore")
        self.assertEqual(vector, plausible)

    def test_acquisition_uses_incremental_observation_counts(self):
        class Model:
            @staticmethod
            def predict(points):
                return np.zeros(len(points)), np.ones(len(points))

        observed = self.space.candidates[0]
        state = {"batches": None, "selections": 1}
        args = SimpleNamespace(
            pair_weight=.5, inducing=0, initial_design=1,
            explore_start=0, explore_floor=0, explore_half_life=1,
            exploration=1, explore_optimism=0, pairs=1,
            gate_all=True, acquisition_restarts=4,
        )
        candidates = [point for point in self.space.candidates if point != observed]
        _, diagnostics = choose(
            state, self.space.prior_mean, candidates, [], args, self.space,
            Model(), observation_counts=Counter({observed: 7}))
        self.assertEqual(diagnostics["unique"], 1)
        self.assertEqual(diagnostics["coverage"], 0)

    def test_seeded_new_axis_is_designed_after_a_mature_clock(self):
        class Model:
            @staticmethod
            def predict(points):
                mean = np.array([100 if point[0] != 50 else 0 for point in points])
                return mean, np.ones(len(points))

        state = {
            "batches": [{
                "knobs": self.space.knobs(self.space.default),
                "wins": 1, "draws": 0, "losses": 1,
            }],
            "new_axes": ["Y"],
            "selections": 1000,
        }
        args = SimpleNamespace(
            pair_weight=.5, inducing=0, initial_design=1,
            explore_start=0, explore_floor=0, explore_half_life=1,
            exploration=1, explore_optimism=0, pairs=1,
            gate_all=True, acquisition_restarts=4,
        )
        candidates = [point for point in self.space.candidates
                      if point != self.space.default]
        vector, diagnostics = choose(
            state, self.space.prior_mean, candidates, [], args, self.space, Model())
        self.assertEqual(diagnostics["mode"], "design")
        self.assertEqual(vector[0], self.space.default[0])
        self.assertNotEqual(vector[1], self.space.default[1])

    def test_full_axis_design_covers_values_seen_only_as_neither_endpoint(self):
        space = MixedSpace({"parameters": [{
            "name": "X", "type": "discrete", "values": [0, 1, 2], "default": 1,
        }]})

        class Model:
            @staticmethod
            def predict(points):
                return np.zeros(len(points)), np.ones(len(points))

        state = {
            "batches": [{
                "knobs": {"X": 1}, "opponent_knobs": {"X": 0},
                "wins": 1, "draws": 0, "losses": 1,
            }],
            "selections": 1000,
        }
        args = SimpleNamespace(
            pair_weight=.5, inducing=0, initial_design=1,
            explore_start=0, explore_floor=0, explore_half_life=1,
            exploration=1, explore_optimism=0, pairs=1,
            gate_all=True, acquisition_restarts=4, full_axis_design=True,
        )
        candidates = [point for point in space.candidates if point != space.default]
        vector, diagnostics = choose(
            state, space.prior_mean, candidates, [], args, space, Model())
        self.assertEqual((vector, diagnostics["mode"]), ((2,), "design"))

    def test_new_axis_design_uses_its_closest_feasible_frontier(self):
        frontier = self.space.canonical({"X": 0, "Y": 0})
        distractor = self.space.canonical({"X": 0, "Y": 10})

        class Model:
            @staticmethod
            def predict(points):
                mean = np.array([100 if point == distractor else 0 for point in points])
                return mean, np.ones(len(points))

        state = {
            "batches": [{
                "knobs": self.space.knobs(self.space.default),
                "wins": 1, "draws": 0, "losses": 1,
            }],
            "new_axes": ["Y"],
            "selections": 1000,
        }
        args = SimpleNamespace(
            pair_weight=.5, inducing=0, initial_design=1,
            explore_start=0, explore_floor=0, explore_half_life=1,
            exploration=1, explore_optimism=0, pairs=1,
            gate_all=True, acquisition_restarts=4,
        )
        vector, diagnostics = choose(
            state, self.space.prior_mean, [frontier, distractor], [], args,
            self.space, Model())
        self.assertEqual(diagnostics["mode"], "design")
        self.assertEqual(vector, frontier)
        state["batches"].append({
            "knobs": self.space.knobs(frontier),
            "wins": 1, "draws": 0, "losses": 1,
        })
        _, diagnostics = choose(
            state, self.space.prior_mean, [frontier, distractor], [], args,
            self.space, Model())
        self.assertEqual(diagnostics["mode"], "ucb")

    def test_gate_all_report_excludes_seeded_points_outside_design(self):
        historical = (37, 13)
        self.assertNotIn(historical, self.space.candidates)
        points, tested = report_domain(self.space, [historical], True)
        self.assertNotIn(historical, points)
        self.assertNotIn(historical, tested)
        self.assertNotIn(self.space.default, tested)
        self.assertIn(historical, report_domain(self.space, [historical], False)[0])
        self.assertNotIn((1000, 13), report_domain(self.space, [(1000, 13)], False)[1])

    def test_report_domain_excludes_known_rejections(self):
        rejected = self.space.candidates[-1]
        points, tested = report_domain(
            self.space, [rejected, self.space.candidates[0]], False, {rejected})
        self.assertNotIn(rejected, points)
        self.assertNotIn(rejected, tested)

    def test_halton_design_handles_full_tuning_space(self):
        parameters = [
            {"name": f"X{i}", "type": "discrete", "values": [0, 1, 2], "default": 1}
            for i in range(32)
        ]
        space = MixedSpace({"parameters": parameters, "max_candidates": 96})
        self.assertEqual(len(space.candidates), 96)
        self.assertIn(space.default, space.candidates)

    def test_local_design_reserves_pairwise_combinations(self):
        parameters = [
            {"name": name, "type": "discrete", "values": [0, 1, 2], "default": 0}
            for name in "XYZ"
        ]
        space = MixedSpace({
            "parameters": parameters,
            "max_candidates": 10,
            "local_interactions": 10,
        })
        distances = [sum(value != 0 for value in point) for point in space.candidates]
        self.assertEqual(distances.count(0), 1)
        self.assertEqual(distances.count(1), 6)
        self.assertEqual(distances.count(2), 3)
        self.assertNotIn(3, distances)

    def test_local_design_samples_wide_integer_domains_deterministically(self):
        parameters = [
            {"name": name, "type": "integer", "min": 0, "max": 10000,
             "default": 5000, "scale": 1000}
            for name in "XYZ"
        ]
        spec = {
            "parameters": parameters,
            "max_candidates": 32,
            "local_interactions": 16,
            "design_oversample": 2,
        }
        first, second = MixedSpace(spec), MixedSpace(spec)
        self.assertEqual(len(first.candidates), 32)
        self.assertEqual(first.candidates, second.candidates)

    def test_joint_eval_domain_preserves_mate_band_invariants(self):
        path = pathlib.Path(__file__).with_name("all_parameters.json")
        parameters = {parameter["name"]: parameter for parameter in json.loads(
            path.read_text())["parameters"]}
        for parameter in parameters.values():
            parameter["values"] = MixedSpace.parameter_values(parameter)
        value_names = ["VALUE_N", "VALUE_B", "VALUE_R", "VALUE_Q"]
        scale_names = [f"PST_{piece}" for piece in "PNBRQK"] + ["PST_KE"]
        # Each inequality is monotone per coordinate, so the box's corners
        # cover every discrete interior value as well.
        values = [[parameter["values"][0], parameter["values"][-1]]
                  for name in value_names for parameter in [parameters[name]]]
        scales = [[parameter["values"][0], parameter["values"][-1]]
                  for name in scale_names for parameter in [parameters[name]]]
        squares = [rank * 10 + file for rank in range(2, 10) for file in range(1, 9)]

        def scaled(value, percent):
            product = value * percent
            return int((product + (50 if product >= 0 else -50)) / 100)

        for coordinates in itertools.product(*values, *scales):
            knobs = dict(zip(value_names + scale_names, coordinates))
            piece = sunfish.piece | dict(zip("NBRQ", coordinates))
            tables = {
                name: [piece[name] + scaled(
                    sunfish.pst[name][square] - sunfish.piece[name],
                    knobs[f"PST_{name}"])
                    for square in squares]
                for name in "PNBRQK"
            }
            king_end = [piece["K"] + scaled(
                sunfish.K_END[square] - sunfish.piece["K"], knobs["PST_KE"])
                for square in squares]
            kings = tables["K"] + king_end
            nonkings = sum((tables[name] for name in "PNBRQ"), [])
            mate_lower = piece["K"] - 13 * piece["Q"]
            army = (9 * max(tables["Q"]) + 2 * max(tables["R"])
                    + 2 * max(tables["B"]) + 2 * max(tables["N"]))
            drop = max(max(tables[name]) - min(tables[name]) for name in "PNBRQK")
            promotion = min(tables[name][square] - tables["P"][square]
                            for name in "NBRQ" for square in range(8))
            self.assertGreaterEqual(min(nonkings), 0)
            self.assertLess(max(kings) - min(kings) + 15 * max(nonkings), mate_lower)
            self.assertLessEqual(mate_lower, min(kings) - army)
            self.assertLess(mate_lower + drop, min(kings))
            self.assertGreaterEqual(promotion, 0)

    def test_joint_space_tunes_the_fuel_amount(self):
        path = pathlib.Path(__file__).with_name("all_parameters.json")
        spec = json.loads(path.read_text())
        fuel = next(parameter for parameter in spec["parameters"]
                    if parameter["name"] == "FUEL_NULL")
        self.assertEqual(fuel["values"], [0, 1, 2])
        self.assertEqual(fuel["off_values"], [0])
        disabled = next(condition for condition in spec["conditions"]
                        if condition["when"] == {"FUEL_MIN_DEPTH": [99]})
        self.assertIn("FUEL_NULL", disabled["reset"])
        no_probe = next(condition for condition in spec["conditions"]
                        if condition["when"] == {"FUEL_NULL": [0]})
        self.assertEqual(no_probe["reset"], ["NULL_MARGIN", "NULL_RED"])
        space = MixedSpace({
            "parameters": [
                {"name": "FUEL_NULL", "type": "discrete", "default": 1,
                 "values": [0, 1, 2]},
                {"name": "NULL_MARGIN", "type": "discrete", "default": -200,
                 "values": [-200, 800]},
                {"name": "NULL_RED", "type": "discrete", "default": 7,
                 "values": [3, 7]},
            ],
            "conditions": [no_probe],
        })
        lmr_only = space.canonical({"FUEL_NULL": 0, "NULL_MARGIN": 800, "NULL_RED": 3})
        self.assertEqual(space.knobs(lmr_only)["NULL_MARGIN"], -200)
        self.assertEqual(space.knobs(lmr_only)["NULL_RED"], 7)
        cap = next(parameter for parameter in spec["parameters"]
                   if parameter["name"] == "FUT_CAP")
        self.assertEqual(cap["values"], [0, 1, 2])
        depth = next(parameter for parameter in spec["parameters"]
                     if parameter["name"] == "FUT_CAP_DEPTH")
        self.assertEqual((depth["min"], depth["max"]), (2, 6))
        self.assertNotIn("MATE_DIST", {
            parameter["name"] for parameter in spec["parameters"]})

    def test_mate_gate_rejects_flat_mate_policies_before_running_engine(self):
        gate = pathlib.Path(__file__).with_name("sunfish_gate.py")
        self.assertEqual(sunfish_gate.SUITES,
            (("mate1.fen", 1, 8, 8),
             ("mate2_eventual.fen", 2, 5, 5),
             ("mate3_eventual.fen", 3, 2, 2)))
        for options in ({"MATE_DIST": 0}, {"EVAL_ROUGHNESS": 0}):
            request = json.dumps({
                "engine": "/does/not/exist",
                "engine_args": "",
                "options": options,
            })
            result = subprocess.run(
                [sys.executable, gate], input=request, text=True,
                capture_output=True)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stdout.strip(), "mate-distance:disabled")

    def test_mate_gate_prices_each_search_policy(self):
        def depths(options):
            return tuple(sunfish_gate.mate_depth(options, moves)
                         for _, moves, _, _ in sunfish_gate.SUITES)

        self.assertEqual(depths({}), (4, 10, 16))
        self.assertEqual(depths({"NULL_LIMIT": 0, "LMR": -70000}), (4, 6, 8))
        self.assertEqual(depths({
            "FUEL_NULL": 2,
            "FUEL_MIN_DEPTH": 12,
            "FUT_CAP_DEPTH": 6,
        }), (7, 16, 24))
        with self.assertRaisesRegex(ValueError, "unbounded-classical-null"):
            sunfish_gate.mate_depth({"FUEL_NULL": 0}, 3)
        with self.assertRaisesRegex(ValueError, "unbounded-classical-null"):
            sunfish_gate.mate_depth({"FUEL_MIN_DEPTH": 99}, 2)

    def test_horizon_gate_does_not_run_the_engine(self):
        gate = pathlib.Path(__file__).with_name("sunfish_gate.py")
        request = json.dumps({
            "engine": "/does/not/exist",
            "engine_args": "",
            "options": {},
        })
        result = subprocess.run(
            [sys.executable, gate, "--horizon-only"], input=request, text=True,
            capture_output=True)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(),
            "mate1.fen:depth=4 mate2_eventual.fen:depth=10 mate3_eventual.fen:depth=16")

    def test_joint_space_anchors_master_and_covers_search_ranges(self):
        path = pathlib.Path(__file__).with_name("all_parameters.json")
        spec = json.loads(path.read_text())
        parameters = {parameter["name"]: parameter for parameter in spec["parameters"]}

        def values(name):
            return MixedSpace.parameter_values(parameters[name])

        limit = parameters["NULL_LIMIT"]["default"]
        self.assertEqual(limit, 750)
        root = path.parents[3]
        self.assertIn(f"static int NULL_LIMIT = {limit};",
                      (root / "tools/ctwin/sunfish.c").read_text())
        self.assertIn(f"calm = abs(pos.score) < {limit}",
                      (root / "sunfish.py").read_text())
        self.assertIn(500, values("NULL_LIMIT"))
        self.assertEqual((min(values("QS")), max(values("QS"))), (0, 300))
        self.assertEqual((min(values("QS_A")), max(values("QS_A"))), (20, 300))
        self.assertEqual(max(values("EVAL_ROUGHNESS")), 50)
        self.assertGreater(min(values("EVAL_ROUGHNESS")), 0)
        self.assertLessEqual(min(value for value in values("LMR") if value > -1000), -200)
        self.assertEqual(max(values("LMR")), 200)
        self.assertLessEqual(min(values("NULL_MARGIN")), -300)
        self.assertGreaterEqual(max(values("NULL_MARGIN")), 800)
        self.assertLessEqual(min(values("VALUE_R")), 400)

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

    def test_full_axis_design_contains_every_single_parameter_value(self):
        design = set(self.space.full_axis_design())
        defaults = self.space.knobs(self.space.default)
        expected = {
            self.space.canonical(defaults | {parameter["name"]: value})
            for parameter in self.space.parameters for value in parameter["values"]
        }
        self.assertEqual(design, expected)

    def test_explicit_required_points_remain_named(self):
        space = MixedSpace({
            "parameters": [{
                "name": "X", "type": "discrete", "values": [0, 1], "default": 0,
            }],
            "required": [{"X": 1}],
        })
        self.assertEqual(space.required, [(1,)])

    def test_seed_drops_observations_from_removed_nondefault_knobs(self):
        seed = {
            "study": {"baseline": {"options": {"X": 50, "Y": 10, "OLD": 0}}},
            "batches": [
                {"knobs": {"X": 0, "Y": 10, "OLD": 0}, "opponent_knobs": None},
                {"knobs": {"X": 0, "Y": 10, "OLD": 1}, "opponent_knobs": None},
                {"knobs": {"X": 0, "Y": 10, "OLD": 0},
                 "opponent_knobs": {"X": 50, "Y": 10, "OLD": 1}},
            ],
        }
        self.assertEqual(compatible_seed_batches(seed, self.space), seed["batches"][:1])

    def test_seed_copy_preserves_the_old_coupled_parameter(self):
        space = MixedSpace({
            "parameters": [
                {"name": "X", "type": "integer", "min": 0, "max": 20,
                 "default": 10},
                {"name": "Y", "type": "integer", "min": 0, "max": 20,
                 "default": 10},
            ],
            "seed_copies": {"Y": "X"},
        })
        seed = {
            "study": {"baseline": {"options": {"X": 10}}},
            "batches": [
                {"knobs": {"X": 5}, "opponent_knobs": None},
                {"knobs": {"X": 15}, "opponent_knobs": {"X": 20}},
            ],
        }
        imported = import_seed_batches(seed, space)
        self.assertEqual(imported[0]["knobs"], {"X": 5, "Y": 5})
        self.assertEqual(imported[0]["opponent_knobs"], {"X": 10, "Y": 10})
        self.assertEqual(imported[1]["knobs"], {"X": 15, "Y": 15})
        self.assertEqual(imported[1]["opponent_knobs"], {"X": 20, "Y": 20})

    def test_pairwise_only_seed_needs_no_fixed_baseline_identity(self):
        seed = {
            "study": {"allocation": {}},
            "batches": [{
                "knobs": {"X": 0, "Y": 10},
                "opponent_knobs": {"X": 50, "Y": 20},
            }],
        }
        self.assertEqual(import_seed_batches(seed, self.space), seed["batches"])

    def test_seed_drops_values_outside_a_narrowed_axis(self):
        seed = {
            "study": {"baseline": {"options": {"X": 50, "Y": 10}}},
            "batches": [
                {"knobs": {"X": 0, "Y": 10}, "opponent_knobs": None},
                {"knobs": {"X": -10, "Y": 10}, "opponent_knobs": None},
                {"knobs": {"X": 0, "Y": 10},
                 "opponent_knobs": {"X": 50, "Y": 30}},
            ],
        }
        self.assertEqual(import_seed_batches(seed, self.space), [{
            "knobs": {"X": 0, "Y": 10},
            "opponent_knobs": {"X": 50, "Y": 10},
        }])

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

    def test_empirical_mean_waits_for_the_initial_design(self):
        batch = {
            "knobs": self.space.knobs(self.space.default),
            "wins": 0, "draws": 0, "losses": 2,
        }
        prior = self.space.prior_mean
        self.assertIs(empirical_mean(prior, [batch], 2, self.space), prior)
        mean = empirical_mean(prior, [batch, batch], 2, self.space)
        self.assertLess(mean([self.space.default])[0], -6)

    def test_empirical_mean_ignores_unanchored_duels(self):
        batch = {
            "knobs": self.space.knobs(self.space.default),
            "opponent_knobs": self.space.knobs(self.space.candidates[-1]),
            "wins": 0, "draws": 0, "losses": 2,
        }
        prior = self.space.prior_mean
        self.assertIs(empirical_mean(prior, [batch], 1, self.space), prior)

    def test_empirical_mean_is_frozen_after_the_design(self):
        design = {
            "knobs": self.space.knobs(self.space.default),
            "wins": 0, "draws": 0, "losses": 2, "allocation": "design",
        }
        adaptive = design | {
            "wins": 2, "losses": 0, "allocation": "ucb",
        }
        prior = self.space.prior_mean
        first = empirical_mean(prior, [design], 1, self.space)
        later = empirical_mean(prior, [design, adaptive], 1, self.space)
        self.assertAlmostEqual(first([self.space.default])[0],
                               later([self.space.default])[0])

    def test_sparse_comparisons_match_the_dense_posterior(self):
        batches = [
            {"knobs": {"X": 0, "Y": 0}, "wins": 2, "draws": 0, "losses": 0},
            {"knobs": {"X": 100, "Y": 20}, "opponent_knobs": {"X": 0, "Y": 0},
             "wins": 0, "draws": 1, "losses": 1},
        ]
        points, dense, success, trials = aggregate(batches, 0.5, self.space)
        _, sparse, _, _ = aggregate(batches, 0.5, self.space, sparse=True)
        arguments = self.space.prior_mean, self.space.kernel, self.space.kernel_diagonal
        dense_model = LogisticGP(*arguments).fit_comparisons(points, dense, success, trials)
        sparse_model = LogisticGP(*arguments).fit_comparisons(points, sparse, success, trials)
        dense_mean, dense_variance = dense_model.predict(self.space.candidates)
        sparse_mean, sparse_variance = sparse_model.predict(self.space.candidates)
        np.testing.assert_allclose(sparse_mean, dense_mean)
        np.testing.assert_allclose(sparse_variance, dense_variance)

    def test_online_updates_track_a_full_sparse_fit(self):
        points = [(0, 0), (50, 10), (100, 20)]
        basis = self.space.inducing_points(8)
        arguments = self.space.prior_mean, self.space.kernel, self.space.kernel_diagonal
        full = LogisticGP(*arguments, basis).fit_comparisons(
            points, (np.array([0, 1, 2]), np.array([-1, 0, 1])),
            [0.5, 1, 0], [1, 1, 1])
        online = LogisticGP(*arguments, basis).fit_comparisons(
            points[:1], (np.array([0]), np.array([-1])), [0.5], [1])
        online.update_comparisons(
            points, (np.array([1, 2]), np.array([0, 1])), [1, 0], [1, 1])
        full_mean, _ = full.predict(self.space.candidates)
        online_mean, _ = online.predict(self.space.candidates)
        np.testing.assert_allclose(online_mean, full_mean, atol=2e-3)

    def test_aggregate_rejects_observations_outside_new_domain(self):
        valid = {"knobs": {"X": 50, "Y": 10}, "wins": 1, "draws": 1, "losses": 0}
        invalid = valid | {"knobs": {"X": -10, "Y": 10}}
        invalid_opponent = valid | {"opponent_knobs": {"X": 50, "Y": 30}}
        points, design, success, trials = aggregate(
            [valid, invalid, invalid_opponent], 0.5, self.space)
        self.assertEqual(points, [self.space.default])
        np.testing.assert_array_equal(design, [[1]])
        self.assertEqual(success, [0.75])
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

    def test_expected_improvement_values_uncertainty_below_baseline(self):
        mean = np.array([-1.0, -1.0])
        variance = np.array([0.1, 1.0])
        score = exploitation(mean, variance, SimpleNamespace(acquisition="ei"))
        self.assertTrue(np.all(score > 0))
        self.assertGreater(score[1], score[0])

    def test_design_variance_needs_only_the_kernel_diagonal(self):
        candidates = self.space.candidates
        sites = candidates[:3]
        covariance = self.space.kernel(candidates, candidates)
        cross = self.space.kernel(sites, candidates)
        site_covariance = self.space.kernel(sites, sites) + np.eye(len(sites)) * 1e-6
        expected = np.diag(covariance) - np.sum(
            cross * np.linalg.solve(site_covariance, cross), axis=0)
        np.testing.assert_allclose(design_variance(sites, candidates, self.space), expected)

    def test_pair_at_a_time_fills_every_lane(self):
        self.assertEqual(pending_configurations(10, 1), 10)
        self.assertEqual(pending_configurations(10, 3), 4)
        self.assertEqual(pending_configurations(10, 10), 1)

    def test_human_durations_are_seconds(self):
        self.assertEqual(duration("3d"), 259200)
        self.assertEqual(duration("1.5h"), 5400)
        with self.assertRaises(argparse.ArgumentTypeError):
            duration("later")

    def test_seed_identity_includes_the_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            state = pathlib.Path(directory, "study.json")
            state.write_text("{}")
            journal = state.with_suffix(".jsonl")
            journal.write_text("first\n")
            before = state_file_identity(state)
            journal.write_text("second\n")
            self.assertNotEqual(before, state_file_identity(state))

    def test_wall_time_drains_the_reserved_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            engine = root / "engine"
            manager = root / "fastchess"
            engine.write_text(
                f"#!{sys.executable}\n"
                "print('option name X type spin default 0 min 0 max 1')\n"
                "print('uciok')\n")
            manager.write_text(
                f"#!{sys.executable}\n"
                "import time\n"
                "time.sleep(.6)\n"
                "print('Score of candidate vs baseline: 1 - 0 - 1  [0.750] 2')\n")
            engine.chmod(0o755)
            manager.chmod(0o755)
            space = root / "space.json"
            space.write_text(json.dumps({
                "parameters": [{
                    "name": "X", "type": "integer", "min": 0, "max": 1,
                    "default": 0,
                }],
            }))
            openings = root / "openings.fen"
            openings.write_text("startpos\n")
            state = root / "state.json"
            command = [
                sys.executable, str(pathlib.Path(__file__).with_name("adaptive_gp.py")),
                "--fastchess", str(manager), "--engine", str(engine),
                "--baseline-options", "default", "--space", str(space),
                "--openings", str(openings), "--cycle-openings",
                "--slots", "1", "--queue-batches", "1", "--refill-batches", "1",
                "--initial-design", "1", "--wall-time", "0.5s", "--batches", "100",
                "--state", str(state), "--logs", str(root / "logs"),
            ]
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
            self.assertEqual(len(load_state(state, 1)["batches"]), 1)

    def test_policy_gates_do_not_block_queued_games(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            starts, intervals = root / "starts", root / "gates"
            engine, manager, gate = root / "engine", root / "fastchess", root / "gate"
            engine.write_text(
                f"#!{sys.executable}\n"
                "print('option name X type spin default 0 min 0 max 9')\n"
                "print('uciok')\n")
            manager.write_text(
                f"#!{sys.executable}\n"
                "import pathlib,time\n"
                f"p=pathlib.Path({str(starts)!r})\n"
                "with p.open('a') as f:f.write(f'{time.time()}\\n')\n"
                "time.sleep(.08)\n"
                "print('Score of candidate vs baseline: 1 - 0 - 1  [0.750] 2')\n")
            gate.write_text(
                f"#!{sys.executable}\n"
                "import json,pathlib,sys,time\n"
                "start=time.time();time.sleep(.1+.2*json.load(sys.stdin)['options']['X'])\n"
                f"p=pathlib.Path({str(intervals)!r})\n"
                "with p.open('a') as f:f.write(f'{start} {time.time()}\\n')\n")
            for program in (engine, manager, gate):
                program.chmod(0o755)
            space = root / "space.json"
            space.write_text(json.dumps({
                "parameters": [{
                    "name": "X", "type": "integer", "min": 0, "max": 9,
                    "default": 0,
                }],
            }))
            openings = root / "openings.fen"
            openings.write_text("startpos\n")
            subprocess.run([
                sys.executable, str(pathlib.Path(__file__).with_name("adaptive_gp.py")),
                "--fastchess", str(manager), "--engine", str(engine),
                "--baseline-options", "default", "--space", str(space),
                "--openings", str(openings), "--cycle-openings",
                "--gate", str(gate), "--gate-workers", "3", "--slots", "1",
                "--queue-batches", "3", "--refill-batches", "1",
                "--initial-design", "9", "--batches", "3",
                "--state", str(root / "state.json"), "--logs", str(root / "logs"),
            ], check=True, stdout=subprocess.DEVNULL)
            games = [float(value) for value in starts.read_text().splitlines()]
            gates = sorted(tuple(map(float, line.split()))
                           for line in intervals.read_text().splitlines())
            self.assertLess(games[0], max(end for _, end in gates))
            self.assertLess(games[1], max(end for _, end in gates))

    def test_gate_all_restricts_the_candidate_space(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            engine, manager, gate = root / "engine", root / "fastchess", root / "gate"
            calls = root / "calls"
            engine.write_text(
                f"#!{sys.executable}\n"
                "print('option name X type spin default 3 min 0 max 5')\n"
                "print('uciok')\n")
            manager.write_text(
                f"#!{sys.executable}\n"
                "print('Score of candidate vs baseline: 1 - 0 - 1  [0.750] 2')\n")
            gate.write_text(
                f"#!{sys.executable}\n"
                "import json,pathlib,sys\n"
                f"p=pathlib.Path({str(calls)!r})\n"
                "with p.open('a') as f:f.write('x')\n"
                "sys.exit(json.load(sys.stdin)['options']['X'] < 3)\n")
            for program in (engine, manager, gate):
                program.chmod(0o755)
            space = root / "space.json"
            space.write_text(json.dumps({
                "parameters": [{
                    "name": "X", "type": "integer", "min": 0, "max": 5,
                    "default": 3,
                }],
            }))
            openings = root / "openings.fen"
            openings.write_text("startpos\n")
            state = root / "state.json"
            subprocess.run([
                sys.executable, str(pathlib.Path(__file__).with_name("adaptive_gp.py")),
                "--fastchess", str(manager), "--engine", str(engine),
                "--baseline-options", "default", "--space", str(space),
                "--openings", str(openings), "--cycle-openings",
                "--gate", str(gate), "--gate-all", "--gate-workers", "3",
                "--slots", "1", "--queue-batches", "2", "--refill-batches", "1",
                "--initial-design", "2", "--batches", "3",
                "--state", str(state), "--logs", str(root / "logs"),
            ], check=True, stdout=subprocess.DEVNULL)
            result = load_state(state, 1)
            self.assertEqual(len(result["gates"]), 5)
            self.assertNotIn('{"X":3}', result["gates"])
            self.assertTrue(all(batch["knobs"]["X"] >= 3 for batch in result["batches"]))
            result["batches"].append(result["batches"][-1])
            save_state(state, result)
            with state.with_suffix(".jsonl").open("ab") as journal:
                journal.write(b'{"batches":')
            resumed = root / "resumed.json"
            resume = [
                sys.executable, str(pathlib.Path(__file__).with_name("adaptive_gp.py")),
                "--fastchess", str(manager), "--engine", str(engine),
                "--baseline-options", "default", "--space", str(space),
                "--openings", str(openings), "--cycle-openings",
                "--gate", str(gate), "--gate-all", "--gate-workers", "3",
                "--slots", "1", "--queue-batches", "1", "--refill-batches", "1",
                "--initial-design", "2", "--batches", "1", "--start", "5",
                "--seed-state", str(state),
                "--state", str(resumed), "--logs", str(root / "resumed-logs"),
            ]
            subprocess.run(resume, check=True, stdout=subprocess.DEVNULL)
            self.assertEqual(calls.read_text(), "x" * 5)
            result = load_state(resumed, 1)
            self.assertEqual(len(result["batches"]), 5)
            self.assertEqual(result["selections"], 4)
            subprocess.run(resume, check=True, stdout=subprocess.DEVNULL)

    def test_duels_keep_a_directly_anchored_opponent(self):
        anchored = self.space.canonical({"X": 0, "Y": 10})
        challenger = self.space.canonical({"X": 100, "Y": 10})
        state = {
            "batches": [{
                "knobs": self.space.knobs(anchored), "opponent_knobs": None,
                "wins": 1, "draws": 0, "losses": 1,
            }],
        }
        args = SimpleNamespace(duel_fraction=0.3, pair_weight=0.5, inducing=0)
        opponents = [
            choose_opponent(state, self.space.prior_mean, challenger, args, self.space)
            for _ in range(10)
        ]
        self.assertEqual(opponents.count(anchored), 3)
        self.assertEqual(opponents.count(None), 7)

    def test_opponent_information_needs_only_a_covariance_row(self):
        challenger = self.space.canonical({"X": 100, "Y": 10})
        anchored = {
            self.space.canonical({"X": 0, "Y": 10}),
            self.space.canonical({"X": 50, "Y": 0}),
        }

        class Model:
            @staticmethod
            def predict(points):
                return np.arange(len(points)), np.ones(len(points))

            @staticmethod
            def predict_cross_covariance(_left, _right):
                return np.asarray([[.9, 0]])

            @staticmethod
            def predict_covariance(_points):
                raise AssertionError("full covariance should not be constructed")

        args = SimpleNamespace(duel_fraction=1, pair_weight=.5, inducing=0)
        found = choose_opponent(
            {}, self.space.prior_mean, challenger, args, self.space, Model(), anchored)
        self.assertEqual(found, sorted(anchored)[1])

    def test_explicit_default_opponents_are_anchored(self):
        anchored = self.space.canonical({"X": 0, "Y": 10})
        challenger = self.space.canonical({"X": 100, "Y": 10})
        state = {
            "batches": [{
                "knobs": self.space.knobs(anchored),
                "opponent_knobs": self.space.knobs(self.space.default),
                "wins": 1, "draws": 0, "losses": 1,
            }],
        }
        args = SimpleNamespace(duel_fraction=1, pair_weight=0.5, inducing=0)
        opponent = choose_opponent(
            state, self.space.prior_mean, challenger, args, self.space)
        self.assertEqual(opponent, anchored)

    def test_opening_epochs_are_balanced_and_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            book = pathlib.Path(directory, "book.epd")
            book.write_text("a\nb\nc\n")
            first = OpeningSchedule(book, seed=7, cycle=True)
            second = OpeningSchedule(book, seed=7, cycle=True)
            sequence = [first.opening(index) for index in range(1, 7)]
            self.assertEqual(sequence, [second.opening(index) for index in range(1, 7)])
            self.assertEqual(sorted(sequence[:3]), [1, 2, 3])
            self.assertEqual(sorted(sequence[3:]), [1, 2, 3])
            sequential = OpeningSchedule(book, cycle=True, order="sequential")
            self.assertEqual([sequential.opening(index) for index in range(1, 7)],
                             [1, 2, 3, 1, 2, 3])
            with self.assertRaises(ValueError):
                OpeningSchedule(book).opening(4)

    def test_policy_gate_is_cached_without_advancing_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            gate = pathlib.Path(directory, "gate.py")
            calls = pathlib.Path(directory, "calls")
            gate.write_text(
                "import json,pathlib,sys\n"
                f"p=pathlib.Path({str(calls)!r});p.write_text(p.read_text()+'x' if p.exists() else 'x')\n"
                "sys.exit(json.load(sys.stdin)['options']['X'] == 0)\n")
            args = SimpleNamespace(
                gate=f"{sys.executable} {gate}", gate_timeout=5,
                engine="engine", engine_args="")
            state = {"batches": [], "selections": 4, "allocations": {"ucb": 4}}
            point = self.space.canonical({"X": 0, "Y": 10})
            self.assertFalse(gate_policy(args, state, self.space, point))
            self.assertFalse(gate_policy(args, state, self.space, point))
            self.assertEqual(calls.read_text(), "x")
            trial = selection_state(state)
            trial["selections"] += 1
            trial["allocations"]["ucb"] += 1
            self.assertEqual(state["selections"], 4)
            commit_selection(state, trial)
            self.assertEqual(state["selections"], 5)

    def test_policy_gate_timeout_is_a_cached_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            gate = pathlib.Path(directory, "gate.py")
            gate.write_text("import time\ntime.sleep(10)\n")
            args = SimpleNamespace(
                gate=f"{sys.executable} {gate}", gate_timeout=0.01,
                engine="engine", engine_args="")
            state = {"batches": []}
            self.assertFalse(gate_policy(args, state, self.space, self.space.default))
            record = next(iter(state["gates"].values()))
            self.assertIn("timeout", record["output"])

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
        sparse = inducing_basis(
            points, (np.arange(3), np.full(3, -1)), trials, self.space, 2)
        self.assertEqual(set(sparse), set(basis))

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

    def test_state_journal_replays_incrementally_across_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory, "state.json")
            state = {"next_opening": 1, "batches": [], "allocations": {}}
            save_state(path, state)
            state["next_opening"] = 2
            state["allocations"]["explore"] = 1
            state["batches"].append({"wins": 1})
            state["gates"] = {"a": {"accepted": True}}
            save_state(path, state)
            self.assertEqual(load_state(path, 1), state)
            checkpoint_state(path, state)
            state["batches"].append({"draws": 2})
            save_state(path, state)
            self.assertEqual(load_state(path, 1), state)

    def test_state_journal_ignores_a_partial_last_event(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory, "state.json")
            state = {"next_opening": 1, "batches": []}
            save_state(path, state)
            journal = path.with_suffix(".jsonl")
            journal.write_text('{"batches":[')
            self.assertEqual(load_state(path, 1), state)
            self.assertEqual(journal.read_text(), "")
            state["batches"].append({"wins": 1})
            save_state(path, state)
            self.assertEqual(load_state(path, 1), state)

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
