import argparse
import hashlib
import importlib.util
import json
import math
import pathlib
import runpy
import shlex
import sys
import tempfile
import types
import unittest
from unittest import mock

import pytest

# tools/tune modules import numpy at module level; the PyPy CI env has no numpy,
# and an ImportError here aborts collection of the ENTIRE suite. Skip loudly instead.
np = pytest.importorskip("numpy")


ROOT = pathlib.Path(__file__).parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ctt = load("ctt_fastchess_shim", "tools/tune/chess_tuning_tools/fastchess_shim.py")
ctt_config = load("ctt_make_config", "tools/tune/chess_tuning_tools/make_config.py")
clop = load("clop_fastchess", "tools/tune/clop/clop_fastchess.py")
gating = load("gating", "tools/tune/gating.py")
pentanomial = load("pentanomial", "tools/tune/pentanomial.py")
locking = load("locking", "tools/tune/locking.py")
plot_recovery = load("plot_recovery", "tools/tune/plot_recovery.py")
recovery = load("recovery_starts", "tools/tune/recovery_starts.py")
recovery_decision = load("recovery_decision", "tools/tune/recovery_decision.py")
recommend = load("recommend", "tools/tune/recommend.py")
freeze_recommendations = load(
    "freeze_recommendations", "tools/tune/freeze_recommendations.py")
rbfopt_stub = types.ModuleType("rbfopt")
rbfopt_stub.RbfoptAlgorithm = type("RbfoptAlgorithm", (), {})
rbfopt_stub.RbfoptBlackBox = type("RbfoptBlackBox", (), {})
rbfopt_stub.RbfoptSettings = type("RbfoptSettings", (), {})
with mock.patch.dict(sys.modules, {"rbfopt": rbfopt_stub}):
    rbfopt = load("rbfopt_runner", "tools/tune/rbfopt/run_rbfopt.py")
spsa = load("spsa", "tools/tune/spsa/spsa.py")
spsa_candidates = load("spsa_candidates", "tools/tune/spsa/candidates.py")
spsa_pool = load("spsa_pool", "tools/tune/spsa/pool.py")
validate = load("validate", "tools/tune/validate.py")
verify_recovery = load("verify_recovery", "tools/tune/verify_recovery.py")


class TuningToolsTest(unittest.TestCase):
    def test_frozen_recovery_manifest_is_self_consistent(self):
        manifest, space = verify_recovery.audit(root=ROOT)
        self.assertEqual(manifest["budget"]["games"], 1000)
        self.assertEqual(len(manifest["starts"]), 3)
        self.assertNotIn("FUT_MAX", {item["name"] for item in space["parameters"]})
        source = manifest["artifacts"]["engine_source"]
        self.assertEqual(source["commit"], "c01915f2349849598e617d24149b74d2fc65ef2a")
        self.assertEqual(source["compiler"]["sha256"],
                         "f679a0ba1bddf27acd9523a1df45909b8e681f1f84f2d0f1cc87f5e115a6ec26")
        self.assertIn("CFLAGS=-O3 -march=native -Wall -Wextra", source["command"])
        with tempfile.TemporaryDirectory() as directory:
            verify_recovery.materialize(manifest, space, directory)
            generated = json.loads(pathlib.Path(directory, "start-05.json").read_text())
            self.assertEqual(generated["parameters"][4]["default"], 21)

    def test_rbfopt_recovers_only_an_exact_complete_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory, "evaluation.log")
            identity = "a" * 64
            path.write_text(
                f"rbfopt-match-identity {identity}\n"
                "Finished game 1 (candidate vs baseline): 1-0\n"
                "Finished game 2 (baseline vs candidate): 1/2-1/2\n"
                "Score of candidate vs baseline: 1 - 0 - 1\n")
            counts, wdl = rbfopt.recover_evaluation(path, identity, 1)
            self.assertEqual((counts, wdl), ([0, 1, 0, 0, 0], (1, 0, 1)))
            self.assertIsNone(rbfopt.recover_evaluation(path, "b" * 64, 1))

    def test_rbfopt_model_replacement_is_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory, "model.pkl")
            path.write_text("old")

            class Optimizer:
                @staticmethod
                def save_to_file(target):
                    pathlib.Path(target).write_text("new")

            rbfopt.save_model(Optimizer(), path)
            self.assertEqual(path.read_text(), "new")
            self.assertFalse(path.with_suffix(".pkl.tmp").exists())

    def test_rbfopt_reuses_a_match_completed_before_model_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            for name in ("engine", "fastchess"):
                (directory / name).write_text(name)
            openings = directory / "openings.epd"
            openings.write_text("opening\n")
            space_path = directory / "space.json"
            space_path.write_text(json.dumps({"parameters": [{
                "name": "X", "type": "integer", "min": 0, "default": 1, "max": 2,
            }]}))
            logs = directory / "logs"
            logs.mkdir()
            args = argparse.Namespace(
                fastchess=str(directory / "fastchess"), engine=str(directory / "engine"),
                baseline_engine=None, engine_args="", baseline_args=None,
                fixed_option=[], baseline_option=[], space=str(space_path),
                openings=str(openings), state=str(directory / "state.json"),
                logs=str(logs), tc="3+0.1", games=1000, noisy_pairs=1,
                accurate_pairs=1, slots=1, start=1, seed=2026, gate=None, gate_timeout=60,
            )
            calls = 0

            def run(*command, stdout, **kwargs):
                nonlocal calls
                calls += 1
                stdout.write(
                    b"Finished game 1 (candidate vs baseline): 1-0\n"
                    b"Finished game 2 (baseline vs candidate): 1/2-1/2\n"
                    b"Score of candidate vs baseline: 1 - 0 - 1\n")
                return types.SimpleNamespace(returncode=0)

            space = rbfopt.Space(space_path)
            with mock.patch.object(rbfopt.subprocess, "run", side_effect=run):
                first = rbfopt.ChessBox(args, space).play(space.start, 1, "noisy")
                second = rbfopt.ChessBox(args, space).play(space.start, 1, "noisy")
            self.assertEqual(first, second)
            self.assertEqual(calls, 1)

    def test_spsa_candidates_decode_and_average_in_optimizer_space(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            space = directory / "space.json"
            space.write_text(json.dumps({"parameters": [{
                "name": "X", "type": "integer", "min": 0,
                "default": 4, "max": 10, "step": 2,
            }, {
                "name": "MODE", "type": "discrete", "default": 20,
                "values": [0, 20, 99], "ordered_values": [99, 20, 0],
            }]}))
            run = directory / "run"
            run.mkdir()
            state = run / "spsa.json"
            state.write_text(json.dumps({"results": [
                {"theta": {"X": 5, "MODE": 0}},
                {"theta": {"X": 9, "MODE": 2}},
            ]}))
            records = spsa_candidates.extract([state], space, [2])
            self.assertEqual(records[0]["options"], {"X": 8, "MODE": 0})
            self.assertEqual(records[1]["options"], {"X": 8, "MODE": 20})
            self.assertEqual(records[1]["trained_games"], 4)
            records = spsa_candidates.extract([state], space, [2], 1)
            self.assertEqual(records[0]["options"], {"X": 4, "MODE": 99})
            self.assertEqual(records[0]["trained_games"], 2)

    def test_spsa_pool_preserves_pairwise_orientation(self):
        with tempfile.TemporaryDirectory() as directory:
            state = pathlib.Path(directory, "spsa.json")
            state.write_text(json.dumps({
                "results": [{
                    "plus": {"X": 2}, "minus": {"X": 1},
                    "wins": 2, "draws": 0, "losses": 0, "opening": 7,
                }],
                "gates": {"plus": {"accepted": True, "knobs": {"X": 2}}},
            }))
            pooled = spsa_pool.pool([state])
            self.assertEqual(pooled["batches"], [{
                "knobs": {"X": 2}, "opponent_knobs": {"X": 1},
                "wins": 2, "draws": 0, "losses": 0, "opening": 7,
                "allocation": "spsa-perturbation",
            }])
            self.assertEqual(pooled["gates"], {
                "plus": {"accepted": True, "knobs": {"X": 2}},
            })
            pooled = spsa_pool.pool([state], 1)
            self.assertEqual(pooled["study"]["sources"][0]["used_results"], 1)

    def test_policy_gate_caches_acceptance_and_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            counter = directory / "calls"
            gate = directory / "gate.py"
            gate.write_text(
                "import json, pathlib, sys\n"
                "p = pathlib.Path(sys.argv[1])\n"
                "p.write_text(p.read_text() + 'x' if p.exists() else 'x')\n"
                "sys.exit(json.load(sys.stdin)['options']['X'])\n")
            command = shlex.join([sys.executable, str(gate), str(counter)])
            cache = {}
            accepted = {"engine": "engine", "engine_args": "", "options": {"X": 0}}
            rejected = accepted | {"options": {"X": 1}}
            self.assertTrue(gating.policy(command, 1, accepted, cache))
            self.assertTrue(gating.policy(command, 1, accepted, cache))
            self.assertFalse(gating.policy(command, 1, rejected, cache))
            self.assertEqual(counter.read_text(), "xx")
            self.assertEqual(len(cache), 2)

    def test_policy_gate_timeout_is_a_cached_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            gate = pathlib.Path(directory, "gate.py")
            gate.write_text("import time\ntime.sleep(10)\n")
            payload = {"engine": "engine", "engine_args": "", "options": {"X": 0}}
            cache = {}
            self.assertFalse(gating.policy(
                shlex.join([sys.executable, str(gate)]), .01, payload, cache))
            self.assertIn("timeout", next(iter(cache.values()))["output"])

    def test_tuner_lock_rejects_a_duplicate_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            state = pathlib.Path(directory, "state.json")
            first = locking.exclusive(state)
            with self.assertRaisesRegex(RuntimeError, "another process"):
                locking.exclusive(state)
            first.close()
            locking.exclusive(state).close()

    def test_recovery_chart_combines_match_and_start_uncertainty(self):
        records = [
            {"method": "gp", "checkpoint": 100, "trained_games": 100,
             "elo": -100, "error": 10},
            {"method": "gp", "checkpoint": 100, "trained_games": 98,
             "elo": -50, "error": 10},
            {"method": "spsa", "checkpoint": 100, "trained_games": 100,
             "elo": 5, "error": 12},
        ]
        summary = plot_recovery.summarize(records)
        gp = next(row for row in summary if row["method"] == "gp")
        spsa_row = next(row for row in summary if row["method"] == "spsa")
        self.assertEqual((gp["trained_games"], gp["starts"], gp["elo"]), (99, 2, -75))
        self.assertGreater(gp["error"], 49)
        self.assertAlmostEqual(spsa_row["error"], 12)

    def test_recovery_gain_uses_shared_opening_pairs(self):
        records = [{
            "method": "gp", "start": 1, "checkpoint": 0, "validation": "start",
            "elo": 0, "error": 60, "pair_scores": [.25, .5, .75],
        }, {
            "method": "gp", "start": 1, "checkpoint": 100, "validation": "same",
            "elo": 0, "error": 60, "pair_scores": [.25, .5, .75],
        }]
        plot_recovery.add_gains(records)
        self.assertEqual((records[1]["gain"], records[1]["gain_error"]), (0, 0))

    def test_recovery_method_comparison_uses_shared_stratified_bootstrap(self):
        records = []
        candidates = {
            ("a", 5): [.75, .5, .75, .5], ("b", 5): [.5, .5, .5, .5],
            ("a", 15): [.5, .75, .5, .75], ("b", 15): [.5, .5, .5, .5],
        }
        starts = {5: [.25, .5, .25, .5], 15: [.5, .25, .5, .25]}
        identity = {
            "validation_start": 220001, "validation_pairs": 4,
            "validation_openings": "book", "validation_protocol": "protocol",
            "benchmark_protocol": "frozen-study",
        }
        for method in ("a", "b"):
            for start in (5, 15):
                records += [
                    {"method": method, "start": start, "checkpoint": 0,
                     "pair_scores": starts[start], "validation": f"start-{start}"} | identity,
                    {"method": method, "start": start, "checkpoint": 1000,
                     "pair_scores": candidates[method, start],
                     "validation": f"{method}-{start}"} | identity,
                ]
        first = plot_recovery.paired_comparisons(
            records, replicates=2000, seed=7, expected_starts=(5, 15))
        second = plot_recovery.paired_comparisons(
            records, replicates=2000, seed=7, expected_starts=(5, 15))
        self.assertEqual(first, second)
        self.assertEqual((first[0]["method_a"], first[0]["method_b"]), ("a", "b"))
        self.assertGreater(first[0]["ci_low"], 0)
        self.assertAlmostEqual(first[0]["score_difference"], .125)

    def test_recovery_method_comparison_rejects_misalignment(self):
        def record(method, start, checkpoint, scores):
            return {
                "method": method, "start": start, "checkpoint": checkpoint,
                "pair_scores": scores,
                "validation": f"start-{start}" if checkpoint == 0 else f"{method}-{start}",
                "validation_start": 220001, "validation_pairs": len(scores),
                "validation_openings": "book", "validation_protocol": "protocol",
                "benchmark_protocol": "frozen-study",
            }

        records = [
            record("a", 5, 0, [.5, .5]), record("a", 5, 1000, [.5, .5]),
            record("b", 15, 0, [.5, .5]), record("b", 15, 1000, [.5, .5]),
        ]
        with self.assertRaisesRegex(ValueError, "misaligned starts"):
            plot_recovery.paired_comparisons(records, replicates=10, expected_starts=(5, 15))
        records[2]["start"] = records[3]["start"] = 5
        records[2]["validation"] = "start-5"
        records[3]["pair_scores"] = [.5]
        records[3]["validation_pairs"] = 1
        with self.assertRaisesRegex(ValueError, "pair counts"):
            plot_recovery.paired_comparisons(records, replicates=10, expected_starts=(5,))
        records[3]["pair_scores"] = records[3]["pair_scores"] * 2
        records[3]["validation_pairs"] = len(records[3]["pair_scores"])
        records[3]["benchmark_protocol"] = "different-study"
        with self.assertRaisesRegex(ValueError, "benchmark protocols"):
            plot_recovery.paired_comparisons(records, replicates=10, expected_starts=(5,))

    def test_recovery_area_uses_the_shared_checkpoint_budget(self):
        summary = [
            {"method": "a", "checkpoint": 0, "gain": 0},
            {"method": "a", "checkpoint": 100, "gain": 20},
            {"method": "a", "checkpoint": 200, "gain": 30},
            {"method": "b", "checkpoint": 0, "gain": 0},
            {"method": "b", "checkpoint": 100, "gain": 10},
            {"method": "b", "checkpoint": 200, "gain": 10},
        ]
        self.assertEqual(plot_recovery.areas(summary, 200), [
            {"method": "a", "horizon": 200, "recovery_auc": 17.5},
            {"method": "b", "horizon": 200, "recovery_auc": 7.5},
        ])
        with self.assertRaisesRegex(ValueError, "does not span"):
            plot_recovery.areas(summary[:-1], 200)

    def test_recommendation_freeze_requires_and_normalizes_the_full_grid(self):
        benchmark = json.loads((ROOT / "tools/tune/recovery_benchmark.json").read_text())
        analysis = ROOT / "tools/tune/recovery_analysis.json"
        methods = json.loads(analysis.read_text())["method_labels"].values()
        checkpoints = benchmark["budget"]["checkpoints"]
        starts = {item["source_index"]: item["options"] for item in benchmark["starts"]}
        with tempfile.TemporaryDirectory() as directory:
            sources = []
            for method in methods:
                for start, options in starts.items():
                    source = pathlib.Path(directory, f"{method}-{start}.json")
                    source.write_text(json.dumps([{
                        "method": method, "start": start, "checkpoint": checkpoint,
                        "trained_games": checkpoint, "recommendation_games": checkpoint,
                        "options": options if checkpoint == 0 else benchmark["reference"]["options"],
                    } for checkpoint in checkpoints]))
                    sources.append(source)
            first = freeze_recommendations.freeze(
                ROOT / "tools/tune/recovery_benchmark.json", analysis, sources)
            second = freeze_recommendations.freeze(
                ROOT / "tools/tune/recovery_benchmark.json", analysis, reversed(sources))
        self.assertEqual(first, second)
        aliases, audit = first
        self.assertEqual((len(aliases), audit["aliases"], audit["unique_configurations"]),
                         (90, 90, 4))
        self.assertEqual(set(aliases[0]["options"]), set(benchmark["reference"]["options"]))
        self.assertEqual(len(audit["recommendation_artifacts"]), 15)

    def test_primary_selection_retains_methods_tied_with_the_runner_up(self):
        recoveries = {"a": 30, "b": 20, "c": 15, "d": 0}
        comparisons = [
            {"method_a": "b", "method_b": "c", "ci_low": -4, "ci_high": 3},
            {"method_a": "b", "method_b": "d", "ci_low": 10, "ci_high": 25},
        ]
        ranking, leader, runner, finalists = recovery_decision.select_finalists(
            recoveries, comparisons)
        self.assertEqual((ranking, leader, runner), (["a", "b", "c", "d"], "a", "b"))
        self.assertEqual(finalists, ["a", "b", "c"])

    def test_exact_sign_flip_and_holm_are_deterministic(self):
        self.assertAlmostEqual(float(recovery_decision.exact_sign_flip([.25, .25])), .25)
        self.assertAlmostEqual(float(recovery_decision.exact_sign_flip([.25, -.25])), .75)
        hypotheses = recovery_decision.holm([
            {"name": "a", "_p": np.longdouble(.01)},
            {"name": "b", "_p": np.longdouble(.03)},
            {"name": "c", "_p": np.longdouble(.04)},
        ])
        self.assertEqual([round(item["adjusted_p_value"], 2) for item in hypotheses],
                         [.03, .06, .06])

    def test_confirmation_tests_every_finalist_against_zero_and_every_rival(self):
        scores = {
            "a": np.asarray([.75] * 12),
            "b": np.asarray([.5] * 12),
        }
        hypotheses = recovery_decision.confirmation_tests(scores)
        self.assertEqual({item["name"] for item in hypotheses},
                         {"a>zero", "b>zero", "a>b", "b>a"})
        accepted = {
            method for method in scores
            if all(item["score_difference"] > 0 and item["adjusted_p_value"] <= .05
                   for item in hypotheses if item["method"] == method)
        }
        self.assertEqual(accepted, {"a"})

    def test_confirmation_sign_flips_shared_openings_not_start_aliases(self):
        repeated = np.asarray([[.75] * 4] * 3)
        hypotheses = recovery_decision.confirmation_tests({
            "a": repeated,
            "b": np.asarray([[.5] * 4] * 3),
        })
        a_zero = next(item for item in hypotheses if item["name"] == "a>zero")
        self.assertEqual(a_zero["p_value"], 1 / 16)
        self.assertEqual(a_zero["score_difference"], .25)
        with self.assertRaisesRegex(ValueError, "start/opening grid"):
            recovery_decision.confirmation_tests({"a": repeated, "b": np.asarray([[.5] * 4])})

    def test_recovery_protocol_rejects_identity_and_opening_drift(self):
        benchmark = {
            "budget": {"time_control": "3+0.1"},
            "artifacts": {
                "engine": "engine", "tables": "tables", "fastchess": "fastchess",
                "heldout_book": "book",
            },
            "reference": {"options": {"X": 1}},
        }
        phase = {"pairs": 10, "start": 20, "slice_sha256": "slice"}
        identity = {
            "arguments": "tables", "files": [{"sha256": "engine"}, {"sha256": "tables"}],
        }
        protocol = {
            "tc": "3+0.1", "pairs": 10, "start": 20,
            "opening_slice_sha256": "slice", "opening_format": "epd",
            "openings": {"sha256": "book"}, "fastchess": {"sha256": "fastchess"},
            "recommendations": {"sha256": "recommendations"},
            "candidate": identity, "baseline": identity, "baseline_options": {"X": "1"},
        }
        recovery_decision.verify_protocol(protocol, benchmark, phase, "recommendations")
        protocol["openings"]["sha256"] = "other"
        with self.assertRaisesRegex(ValueError, "opening book"):
            recovery_decision.verify_protocol(protocol, benchmark, phase, "recommendations")

    def test_recovery_validation_requires_the_exact_record_and_match_grid(self):
        parameters = [{"name": "X", "type": "integer", "min": 0, "default": 1, "max": 2}]
        reference, starts = {"X": 1}, {5: {"X": 0}}
        fingerprint = {"study": "frozen"}
        benchmark = {
            "budget": {"time_control": "3+0.1"},
            "artifacts": {
                "engine": "engine", "tables": "tables", "fastchess": "fastchess",
                "heldout_book": "book",
            },
            "reference": {"options": reference},
        }
        phase = {"pairs": 2, "start": 20, "slice_sha256": "slice"}
        identity = {
            "arguments": "tables", "files": [{"sha256": "engine"}, {"sha256": "tables"}],
        }
        protocol = {
            "tc": "3+0.1", "pairs": 2, "start": 20,
            "opening_slice_sha256": "slice", "opening_format": "epd",
            "openings": {"sha256": "book"}, "fastchess": {"sha256": "fastchess"},
            "recommendations": {"sha256": "recommendations"},
            "candidate": identity, "baseline": identity, "baseline_options": {"X": "1"},
        }
        records, matches = [], {}
        for checkpoint, options, scores in ((0, {"X": 0}, [.5, .5]),
                                             (1000, {"X": 2}, [.5, .75])):
            config = validate.identifier(options)
            records.append({
                "method": "a", "start": 5, "checkpoint": checkpoint,
                "configuration": config, "validation": config,
                "benchmark_protocol": fingerprint, "options": options,
            })
            wins, losses, draws = ((0, 0, 4) if checkpoint == 0 else (2, 1, 1))
            matches[config] = {
                "pair_scores": scores,
                "pentanomial": ([0, 0, 2, 0, 0] if checkpoint == 0 else [0, 1, 1, 0, 0]),
                "wins": wins, "losses": losses, "draws": draws,
            }
        with tempfile.TemporaryDirectory() as directory:
            validation = pathlib.Path(directory, "validation.json")
            validation.write_text(json.dumps({
                "protocol": protocol, "records": records, "matches": matches,
            }))
            recovery_decision.verify_validation(
                validation, benchmark, phase, parameters, reference, starts, fingerprint,
                ["a"], (0, 1000), "recommendations")
            records.pop()
            validation.write_text(json.dumps({
                "protocol": protocol, "records": records, "matches": matches,
            }))
            with self.assertRaisesRegex(ValueError, "unrelated matches"):
                recovery_decision.verify_validation(
                    validation, benchmark, phase, parameters, reference, starts, fingerprint,
                    ["a"], (0,), "recommendations")

    def test_validation_opening_slice_hash_is_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            openings = pathlib.Path(directory, "openings.epd")
            openings.write_bytes(b"one\n\ntwo\nthree\n")
            expected = hashlib.sha256(b"two\nthree\n").hexdigest()
            self.assertEqual(validate.slice_digest(openings, 2, 2), expected)
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                validate.slice_digest(openings, 3, 2)

    def test_validation_parser_keeps_paired_statistics(self):
        output = b"""Finished game 2 (baseline vs candidate): 1-0
Score of candidate vs baseline: 1 - 0 - 0
Finished game 1 (candidate vs baseline): 1-0
Score of candidate vs baseline: 1 - 1 - 0
Elo difference: +0.00 +/- 12.50
"""
        result = validate.parse(output, 1)
        self.assertEqual(result["pentanomial"], [0, 0, 1, 0, 0])
        self.assertEqual((result["wins"], result["losses"], result["draws"]), (1, 1, 0))
        self.assertEqual((result["elo"], result["error"]), (0, 12.5))
        self.assertEqual(validate.identifier({"A": 1, "B": 2}),
                         validate.identifier({"B": 2, "A": 1}))
        result = validate.parse(output.replace(b"12.50", b"-nan"), 1)
        self.assertTrue(math.isfinite(result["error"]))
        self.assertIsNone(result["fastchess_error"])

    def test_game_result_tools_reject_engine_failures(self):
        failures = (
            "Finished game 1 (a vs b): 1-0 {Black disconnects}",
            "Finished game 1 (a vs b): 1-0 {Black's connection stalls}",
            "Finished game 1 (a vs b): 1-0 {Black makes an illegal move}",
            "Finished game 1 (a vs b): 1-0 {Black loses on time (2ms overrun)}",
            "Warning; Engine candidate didn't respond to uci.",
            "Engine candidate did not respond to uciok in time.",
            "Engine candidate is not responsive", "Engine crashed (stdout)",
            "Warning; Illegal move a1a1 played by candidate", "Timeouts: 2",
            "Crashed: 1", "termination: time forfeit",
        )
        for output in failures:
            with self.subTest(output=output), self.assertRaises(RuntimeError):
                pentanomial.game_results(output)

    def test_game_result_failure_matching_ignores_benign_substrings(self):
        for output in (
                "installation complete", "Score of Stallion vs CrashTest: 1 - 0 - 1",
                "Crashed: 0", "Timeouts: 0", "info string engine is crash-resistant"):
            with self.subTest(output=output):
                pentanomial.reject_failures(output)

    def test_validation_recovers_complete_pairs_across_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            log = pathlib.Path(directory, "configuration-test.log")
            log.write_text("""Finished game 1 (candidate vs baseline): 1-0
Finished game 2 (baseline vs candidate): 1/2-1/2
Finished game 3 (candidate vs baseline): 0-1
""")
            log.with_name("configuration-test.part-02.log").write_text(
                """Finished game 1 (candidate vs baseline): 0-1
Finished game 2 (baseline vs candidate): 1-0
""")
            results, _ = validate.recover(log)
            self.assertEqual(len(results), 4)
            self.assertEqual(validate.result(results)["pentanomial"], [0, 1, 0, 0, 1])

    def test_validation_recovers_after_an_empty_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            log = pathlib.Path(directory, "configuration-test.log")
            log.write_text("")
            log.with_name("configuration-test.part-01.log").write_text(
                "validation-match-identity interrupted\n")
            log.with_name("configuration-test.part-02.log").write_text(
                """Finished game 1 (candidate vs baseline): 1-0
Finished game 2 (baseline vs candidate): 1/2-1/2
""")
            results, _ = validate.recover(log)
            self.assertEqual(len(results), 2)

    def test_validation_does_not_recover_past_an_engine_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            log = pathlib.Path(directory, "configuration-test.log")
            log.write_text(
                "Finished game 1 (candidate vs baseline): 0-1 {White disconnects}\n")
            with self.assertRaisesRegex(RuntimeError, "disconnect"):
                validate.recover(log)

    def test_validation_checks_uci_options_before_games(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = pathlib.Path(directory, "engine")
            engine.write_text(
                "#!/usr/bin/env python3\n"
                "print('option name Present type spin default 1 min 0 max 2')\n"
                "print('uciok')\n")
            engine.chmod(0o755)
            validate.validate_options(str(engine), "", ["Present"])
            with self.assertRaisesRegex(RuntimeError, "Missing"):
                validate.validate_options(str(engine), "", ["Missing"])

    def test_recommendation_records_training_and_point_vintages(self):
        records = recommend.at_checkpoints(
            "example", [(20, {"X": 1})], [0, 10, 20, 30], {"X": 0},
            range(2, 25, 2))
        self.assertEqual([row["checkpoint"] for row in records], [0, 10, 20])
        self.assertEqual([row["trained_games"] for row in records], [0, 10, 20])
        self.assertEqual([row["recommendation_games"] for row in records], [0, 0, 20])
        self.assertEqual([row["options"]["X"] for row in records], [0, 0, 1])
        self.assertEqual(recommend.metadata(["start=11", "label=bad"]),
                         {"start": 11, "label": "bad"})

    def test_recommendation_reports_prior_when_first_update_overshoots_budget(self):
        records = recommend.at_checkpoints(
            "example", [(20, {"X": 1})], [0, 10, 20, 30], {"X": 0}, [20])
        self.assertEqual([row["checkpoint"] for row in records], [0, 10, 20])
        self.assertEqual([row["trained_games"] for row in records], [0, 0, 20])
        self.assertEqual([row["options"]["X"] for row in records], [0, 0, 1])

    def test_clop_recommendations_use_only_complete_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            space = directory / "space.json"
            space.write_text(json.dumps({"parameters": [{
                "name": "X", "type": "integer", "min": 0, "default": 0, "max": 5,
            }]}))
            config = directory / "study.clop"
            config.write_text("Name study\nReplications 2\n")
            (directory / "study.dat").write_text(
                "R 0 0\nR 2 0\nR 3 0\nR 1 0\n")
            replay = types.SimpleNamespace(
                returncode=0, stderr="",
                stdout="1 0 0 .5 1\n2 0 0 .5 2\n3 0 0 .5 3\n4 0 0 .5 4\n")
            args = argparse.Namespace(
                space=str(space), config=str(config), console="console", method="clop")
            with mock.patch.object(recommend.subprocess, "run", return_value=replay):
                records = recommend.clop(args, [0, 2, 3, 4, 6])
            self.assertEqual([row["checkpoint"] for row in records], [0, 2, 3, 4])
            self.assertEqual([row["options"]["X"] for row in records], [0, 0, 3, 4])

    def test_clop_recommendations_require_the_adapter_replication_count(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            space = directory / "space.json"
            space.write_text('{"parameters": []}')
            config = directory / "study.clop"
            config.write_text("Name study\nReplications 4\n")
            args = argparse.Namespace(
                space=str(space), config=str(config), console="console", method="clop")
            with self.assertRaisesRegex(RuntimeError, "Replications 2"):
                recommend.clop(args, [0])

    def test_gp_recommendation_accepts_pairwise_only_studies(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            space = directory / "space.json"
            space.write_text(json.dumps({"parameters": [{
                "name": "X", "type": "integer", "default": 0,
                "min": 0, "max": 1,
            }]}))
            state = directory / "state.json"
            state.write_text(json.dumps({
                "study": {"allocation": {}},
                "batches": [{
                    "knobs": {"X": 1}, "opponent_knobs": {"X": 0},
                    "wins": 2, "draws": 0, "losses": 0,
                }],
            }))
            args = argparse.Namespace(
                state=str(state), space=str(space), method="gp",
                pair_weight=.5, inducing=0)
            with mock.patch.object(recommend, "gp_recommend", return_value={"X": 1}):
                records = recommend.gp(args, [2])
            self.assertEqual(records[0]["options"], {"X": 1})

    def test_pentanomial_parser_restores_game_order(self):
        output = """Finished game 2 (baseline vs candidate): 1-0
Score of candidate vs baseline: 1 - 0 - 0
Finished game 1 (candidate vs baseline): 1-0
Score of candidate vs baseline: 1 - 1 - 0
"""
        counts, wdl = pentanomial.parse(output)
        self.assertEqual(counts, [0, 0, 1, 0, 0])
        self.assertEqual(wdl, (1, 1, 0))
        self.assertEqual(pentanomial.pair_scores(output), [.5])

    def test_pentanomial_parser_does_not_require_per_game_score_snapshots(self):
        output = """Finished game 1 (candidate vs baseline): 1-0
Finished game 2 (baseline vs candidate): 1/2-1/2
Finished game 3 (candidate vs baseline): 0-1
Finished game 4 (baseline vs candidate): 0-1
Score of candidate vs baseline: 2 - 1 - 1
"""
        counts, wdl = pentanomial.parse(output)
        self.assertEqual(counts, [0, 1, 1, 0, 0])
        self.assertEqual(wdl, (2, 1, 1))

    def test_pentanomial_parser_recovers_complete_pairs_from_partial_match(self):
        output = """Finished game 1 (candidate vs baseline): 1-0
Finished game 2 (baseline vs candidate): 1/2-1/2
Score of candidate vs baseline: 1 - 0 - 1
Finished game 3 (candidate vs baseline): 0-1
"""
        results, wdl = pentanomial.game_results(output, partial=True)
        self.assertEqual(results, [(1, 0, 0), (0, 0, 1)])
        self.assertEqual(wdl, (1, 0, 1))

    def test_recovery_space_is_symmetric_and_contains_reference(self):
        spec = {"parameters": [
            {"name": "integer", "type": "integer", "min": 0, "default": 5, "max": 12},
            {"name": "real", "type": "real", "min": -2, "default": 1, "max": 4},
        ]}
        point = {"integer": 7, "real": 2}
        result = recovery.centered(spec, point)
        for original, parameter in zip(spec["parameters"], result["parameters"]):
            self.assertEqual(parameter["default"], point[parameter["name"]])
            self.assertEqual(parameter["default"] - parameter["min"],
                             parameter["max"] - parameter["default"])
            self.assertLessEqual(parameter["min"], original["default"])
            self.assertGreaterEqual(parameter["max"], original["default"])

    def test_ctt_opening_checkpoint_commits_on_the_next_persisted_iteration(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            book = directory / "openings.epd"
            book.write_text("one\ntwo\nthree\nfour\nfive\n")
            state = directory / "next-opening"
            command = [
                "fastchess", "-openings", f"file={book}", "format=epd", "order=random",
                "-rounds", "3",
            ]
            with mock.patch.dict("os.environ", {
                    "CTT_OPENING_STATE": str(state), "CTT_OPENING_START": "1",
                    "CTT_STUDY_ID": "study", "CTT_ITERATION": "0"}):
                sequenced, transaction = ctt.sequence_openings(command.copy())
                self.assertIn("start=1", sequenced)
                first = json.loads(state.read_text())
                self.assertEqual((first["games"], first["next_opening"]), (0, 1))
                self.assertEqual(first["pending"]["iteration"], 0)
                self.assertEqual(transaction[1], first["pending"]["identity"])
                sequenced, _ = ctt.sequence_openings(command.copy())
                self.assertIn("start=1", sequenced)
            with mock.patch.dict("os.environ", {
                    "CTT_OPENING_STATE": str(state), "CTT_OPENING_START": "1",
                    "CTT_STUDY_ID": "study", "CTT_ITERATION": "1"}):
                sequenced, _ = ctt.sequence_openings(command.copy())
                self.assertIn("start=4", sequenced)
                second = json.loads(state.read_text())
                self.assertEqual((second["games"], second["next_opening"]), (6, 4))
                self.assertEqual(second["pending"]["iteration"], 1)
                ctt.commit_openings(2)
                final = json.loads(state.read_text())
                self.assertEqual((final["games"], final["next_opening"]), (12, 2))
                self.assertNotIn("pending", final)

    def test_ctt_fastchess_state_does_not_overwrite_tuner_config(self):
        command = ctt.translate([], {})
        self.assertIn("outname=fastchess-state.json", command)

    def test_ctt_recovers_only_an_exact_complete_iteration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory, "match.log")
            identity = "a" * 64
            path.write_text(
                f"ctt-match-identity {identity}\n"
                "Finished game 1 (candidate vs baseline): 1-0\n"
                "Finished game 2 (baseline vs candidate): 1/2-1/2\n"
                "Score of candidate vs baseline: 1 - 0 - 1\n")
            self.assertIn("Finished game 2", ctt.complete_log(path, identity, 1))
            self.assertIsNone(ctt.complete_log(path, "b" * 64, 1))

    def test_ctt_reuses_a_match_completed_before_data_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            book = directory / "openings.epd"
            book.write_text("opening\n")
            state = directory / "openings.json"
            command = [
                "fastchess", "-openings", f"file={book}", "format=epd", "order=random",
                "-rounds", "1",
            ]
            calls = 0

            def popen(*args, **kwargs):
                nonlocal calls
                calls += 1
                return types.SimpleNamespace(
                    stdout=[
                        "Finished game 1 (candidate vs baseline): 1-0\n",
                        "Finished game 2 (baseline vs candidate): 1/2-1/2\n",
                        "Score of candidate vs baseline: 1 - 0 - 1\n",
                    ], wait=lambda: 0)

            environment = {
                "CTT_OPENING_STATE": str(state), "CTT_STUDY_ID": "study",
                "CTT_ITERATION": "0", "CTT_MATCH_DIR": str(directory / "logs"),
            }
            with mock.patch.dict("os.environ", environment), \
                    mock.patch.object(sys, "argv", ["fastchess_shim.py"]), \
                    mock.patch.object(ctt, "engines", return_value={}), \
                    mock.patch.object(ctt, "translate", side_effect=lambda *_: command.copy()), \
                    mock.patch.object(ctt.subprocess, "Popen", side_effect=popen):
                self.assertEqual(ctt.main(), 0)
                self.assertEqual(ctt.main(), 0)
            self.assertEqual(calls, 1)

    def test_ctt_does_not_advance_after_a_recovered_engine_failure(self):
        process = types.SimpleNamespace(stdout=["Engine crashed (stdout)\n"], wait=lambda: 0)
        with mock.patch.object(ctt, "engines", return_value={}), \
                mock.patch.object(ctt, "translate", return_value=["fastchess"]), \
                mock.patch.object(ctt, "sequence_openings",
                                  return_value=(["fastchess"], None)), \
                mock.patch.object(ctt.subprocess, "Popen", return_value=process) as popen:
            self.assertEqual(ctt.main(), 1)
        self.assertEqual(popen.call_args.kwargs["errors"], "replace")

    def test_clop_adapter_rejects_a_recovered_time_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            openings = pathlib.Path(directory, "openings.epd")
            openings.write_text("opening\n")
            def run(*command, stdout, **kwargs):
                stdout.write(
                    "Finished game 1 (candidate vs baseline): 0-1 "
                    "{White loses on time (1ms overrun)}\n"
                    "Score of candidate vs baseline: 0 - 1 - 0\n")
                return types.SimpleNamespace(returncode=0)

            argv = [
                "clop_fastchess.py", "--fastchess", "fastchess", "--engine", "engine",
                "--openings", str(openings), "--opening-count", "1",
                "--cache", str(pathlib.Path(directory, "cache")), "local", "0",
            ]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(clop.subprocess, "run", side_effect=run), \
                    self.assertRaisesRegex(RuntimeError, "loses on time"):
                clop.main()

    def test_clop_reuses_only_the_exact_completed_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            openings = directory / "openings.epd"
            openings.write_text("opening\n")
            engine = directory / "engine"
            engine.write_text("engine\n")
            fastchess = directory / "fastchess"
            fastchess.write_text("manager\n")
            calls = 0

            def run(*command, stdout, **kwargs):
                nonlocal calls
                calls += 1
                stdout.write("Score of candidate vs baseline: 1 - 0 - 0\n")
                return types.SimpleNamespace(returncode=0)

            base = [
                "clop_fastchess.py", "--fastchess", str(fastchess), "--engine", str(engine),
                "--openings", str(openings), "--opening-count", "1",
                "--cache", str(directory / "cache"), "local", "0",
            ]
            with mock.patch.object(clop.subprocess, "run", side_effect=run):
                with mock.patch.object(sys, "argv", base):
                    self.assertEqual(clop.main(), 0)
                with mock.patch.object(sys, "argv", base):
                    self.assertEqual(clop.main(), 0)
                with mock.patch.object(sys, "argv", [*base, "X", "1"]):
                    self.assertEqual(clop.main(), 0)
            self.assertEqual(calls, 2)

    def test_spsa_adapter_rejects_a_recovered_time_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            openings = directory / "openings.epd"
            openings.write_text("opening\n")
            logs = directory / "logs"
            logs.mkdir()
            def run(*command, stdout, **kwargs):
                stdout.write(
                    b"Finished game 1 (plus vs minus): 0-1 "
                    b"{White loses on time (1ms overrun)}\n"
                    b"Score of plus vs minus: 0 - 1 - 0\n")
                return types.SimpleNamespace(returncode=0)

            args = argparse.Namespace(
                fastchess="fastchess", engine="engine", engine_args="", tc="3+0.1",
                openings=str(openings), slots=1, logs=str(logs))
            with mock.patch.object(spsa.subprocess, "run", side_effect=run), \
                    self.assertRaisesRegex(RuntimeError, "loses on time"):
                spsa.play(args, {"study": 1}, 0, 1, 1, {}, {})

    def test_spsa_reuses_a_complete_identity_bound_step(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            openings = directory / "openings.epd"
            openings.write_text("opening\n")
            logs = directory / "logs"
            logs.mkdir()
            calls = 0

            def run(*command, stdout, **kwargs):
                nonlocal calls
                calls += 1
                stdout.write(b"Score of plus vs minus: 1 - 0 - 1\n")
                return types.SimpleNamespace(returncode=0)

            args = argparse.Namespace(
                fastchess="fastchess", engine="engine", engine_args="", tc="3+0.1",
                openings=str(openings), slots=1, logs=str(logs))
            with mock.patch.object(spsa.subprocess, "run", side_effect=run):
                first = spsa.play(args, {"study": 1}, 0, 1, 1, {"X": 1}, {"X": 0})
                second = spsa.play(args, {"study": 1}, 0, 1, 1, {"X": 1}, {"X": 0})
                changed = spsa.play(args, {"study": 2}, 0, 1, 1, {"X": 1}, {"X": 0})
            self.assertEqual((first, second, changed), ((1, 0, 1),) * 3)
            self.assertEqual(calls, 2)

    def test_ctt_config_carries_its_declared_start_point(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            space = directory / "space.json"
            space.write_text(json.dumps({"parameters": [{
                "name": "X", "type": "integer", "min": 0, "default": 3, "max": 4,
            }]}))
            config = directory / "tuner.json"
            start = directory / "start.csv"
            argv = ["make_config.py", "--space", str(space), "--output", str(config),
                    "--start-output", str(start), "--candidate", "candidate",
                    "--baseline", "baseline", "--openings", "openings.epd",
                    "--rounds", "1", "--iterations", "500"]
            with mock.patch.object(sys, "argv", argv):
                ctt_config.main()
            settings = json.loads(config.read_text())
            self.assertEqual(settings["evaluate_points"], str(start))
            self.assertEqual(settings["max_iterations"] * settings["rounds"] * 2, 1000)
            self.assertEqual(start.read_text(), "3,1\n")

    def test_ctt_recommendation_rejects_missing_optimum_history(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            space = directory / "space.json"
            space.write_text(json.dumps({"parameters": [{
                "name": "X", "type": "integer", "min": 0, "default": 1, "max": 2,
            }]}))
            config = directory / "tuner.json"
            config.write_text(json.dumps({"rounds": 1, "n_initial_points": 2}))
            data = directory / "data.npz"
            import numpy as np
            np.savez(data, [[1], [2]], [0, 0], [1, 1], [], [], 2)
            args = argparse.Namespace(space=str(space), config=str(config), data=str(data),
                                      method="ctt-mes")
            with self.assertRaisesRegex(RuntimeError, "no current optimum"):
                recommend.ctt(args, [0, 2, 4])

    def test_ctt_finite_run_persists_its_final_incumbent(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            package = directory / "tune.py"
            package.write_text("")
            (directory / "local.py").write_text(
                "score = float(prob_to_elo(dist.mean().dot(scores), k=score_scale))\n"
                "        return best_point, estimated_elo, float(best_std * 100)\n")
            (directory / "cli.py").write_text(
                "import logging\n"
                "    extra_points = load_points_to_evaluate(\n"
                "        space=opt.space,\n"
                "        csv_file=evaluate_points,\n"
                "    while True:\n"
                "        used_extra_point = False\n"
                "        for output_line in run_match(**match_settings):\n"
                "        with AtomicWriter(model_path, mode=\"wb\", overwrite=True).open() as f:\n"
                "            dill.dump(opt, f)\n")
            with (directory / "local.py").open("a") as source:
                source.write("            if opt.space == old_opt.space:\n")
            (directory / "utils.py").write_text('''    raw_bounds = getattr(res.space, "bounds", None)
    minimize_bounds = None

    if raw_bounds is not None:
        minimize_bounds = []
        for bound in raw_bounds:
            lower, upper = bound
            lower_cast = None if lower is None else float(np.float64(lower))
            upper_cast = None if upper is None else float(np.float64(upper))
            minimize_bounds.append((lower_cast, upper_cast))
        minimize_bounds = tuple(minimize_bounds)
''')
            fake = types.SimpleNamespace(__file__=str(package))
            with mock.patch.dict(sys.modules, {"tune": fake}):
                runpy.run_path(ROOT / "tools/tune/chess_tuning_tools/patch_ctt.py")
            source = (directory / "cli.py").read_text()
            self.assertIn("while iteration <= max_iterations", source)
            self.assertLess(source.index("if iteration == max_iterations"),
                            source.index("for output_line in run_match"))
            self.assertLess(source.index("np.savez_compressed"), source.index("break"))
            self.assertIn('None if len(X) else settings.get("evaluate_points"', source)
            self.assertIn('os.environ["CTT_ITERATION"] = str(iteration)', source)
            self.assertIn('cutechess-cli", "--commit-iteration"', source)
            self.assertIn("list(old_opt.Xi) == X", (directory / "local.py").read_text())

    def test_spsa_uses_pairs_as_its_iteration_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            executable = directory / "engine"
            executable.write_text("placeholder")
            openings = directory / "openings.epd"
            openings.write_text("\n".join(f"opening {i}" for i in range(8)) + "\n")
            space = directory / "space.json"
            space.write_text(json.dumps({"parameters": [{
                "name": "X", "type": "real", "min": 0, "default": 2, "max": 10,
            }]}))
            state = directory / "state.json"
            calls = []

            def play(args, study, number, opening, pairs, plus, minus):
                calls.append((number, opening, pairs))
                return ((2 * pairs, 0, 0) if plus["X"] > minus["X"]
                        else (0, 2 * pairs, 0))

            args = argparse.Namespace(
                fastchess=str(executable), engine=str(executable), engine_args="",
                space=str(space), openings=str(openings), state=str(state),
                logs=str(directory / "logs"), tc="3+0.1", iterations=5,
                slots=2, pairs_per_step=2, start=2, seed=2026, fixed_option=[],
                a_ratio=.1, alpha=.602, gamma=.101, c_ratio=1 / 6,
                r_end=None, draw_ratio=.2, precision=.5,
                gate=None, gate_timeout=1, gate_attempts=10,
            )
            with mock.patch.object(spsa, "validate_options"), mock.patch.object(spsa, "play", play):
                spsa.optimize(args)
            saved = json.loads(state.read_text())
            self.assertEqual(calls, [(0, 2, 2), (1, 4, 2), (2, 6, 1)])
            self.assertEqual([item["iteration"] for item in saved["results"]], [0, 2, 4])
            self.assertEqual([item["pairs"] for item in saved["results"]], [2, 2, 1])
            self.assertGreater(saved["parameters"][0]["theta"], 2)

    def test_spsa_tunes_ordered_choices_in_index_space(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            space = directory / "space.json"
            space.write_text(json.dumps({"parameters": [{
                "name": "MODE", "type": "discrete", "default": 20,
                "values": [0, 20, 99], "ordered_values": [99, 20],
            }]}))
            args = argparse.Namespace(
                precision=.5, draw_ratio=.2, c_ratio=1 / 6, r_end=.02,
                gamma=.101, alpha=.602, a_ratio=.1,
                initial_option=["MODE=99"],
            )
            parameters = spsa.load_parameters(space, 100, args)
            self.assertEqual(
                (parameters[0]["min"], parameters[0]["theta"], parameters[0]["max"]),
                (0, 0, 1),
            )
            self.assertEqual(spsa.render(parameters, {"MODE": 1}, 1), {"MODE": 20})

            state = directory / "state.json"
            state.write_text(json.dumps({"results": [{
                "pairs": 1, "theta": {"MODE": 0.4},
            }]}))
            records = recommend.spsa(
                argparse.Namespace(state=str(state), space=str(space), method="spsa"), [2])
            self.assertEqual(records[0]["options"], {"MODE": 99})

    def test_spsa_resamples_until_both_perturbations_pass_the_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            executable = directory / "engine"
            executable.write_text("placeholder")
            openings = directory / "openings.epd"
            openings.write_text("opening\n")
            space = directory / "space.json"
            space.write_text(json.dumps({"parameters": [{
                "name": "X", "type": "integer", "min": 0,
                "default": 5, "max": 10,
            }]}))
            args = argparse.Namespace(
                fastchess=str(executable), engine=str(executable), engine_args="",
                space=str(space), openings=str(openings),
                state=str(directory / "state.json"), logs=str(directory / "logs"),
                tc="3+0.1", iterations=1, slots=1, pairs_per_step=1,
                start=1, seed=2026, fixed_option=[], a_ratio=.1,
                alpha=.602, gamma=.101, c_ratio=1 / 6, r_end=.02,
                draw_ratio=.2, precision=.5, gate="gate",
                gate_timeout=1, gate_attempts=2,
            )
            with mock.patch.object(spsa, "validate_options"), \
                    mock.patch.object(spsa, "play", return_value=(0, 0, 2)) as play, \
                    mock.patch.object(spsa.gating, "policy",
                                      side_effect=[False, True, True]) as gate:
                spsa.optimize(args)
            self.assertEqual(gate.call_count, 3)
            play.assert_called_once()


if __name__ == "__main__":
    unittest.main()
