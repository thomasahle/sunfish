import argparse
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

from adaptive_gp import (
    aggregate,
    bind_study,
    checkpoint_state,
    choose_opponent,
    commit_selection,
    coordinate_maximum,
    design_variance,
    duration,
    engine_identity,
    exploration_probability,
    fantasy_variance,
    fixed_baseline_point,
    gate_policy,
    inducing_basis,
    load_state,
    OpeningSchedule,
    selection_state,
    save_state,
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
            self.assertEqual(len(result["gates"]), 6)
            self.assertTrue(all(batch["knobs"]["X"] >= 3 for batch in result["batches"]))
            resumed = root / "resumed.json"
            subprocess.run([
                sys.executable, str(pathlib.Path(__file__).with_name("adaptive_gp.py")),
                "--fastchess", str(manager), "--engine", str(engine),
                "--baseline-options", "default", "--space", str(space),
                "--openings", str(openings), "--cycle-openings",
                "--gate", str(gate), "--gate-all", "--gate-workers", "3",
                "--slots", "1", "--queue-batches", "1", "--refill-batches", "1",
                "--initial-design", "2", "--batches", "1", "--start", "4",
                "--seed-state", str(state), "--seed-selections", "3",
                "--state", str(resumed), "--logs", str(root / "resumed-logs"),
            ], check=True, stdout=subprocess.DEVNULL)
            self.assertEqual(calls.read_text(), "x" * 6)

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
