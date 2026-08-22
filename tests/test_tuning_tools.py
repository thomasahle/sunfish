import argparse
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
pytest.importorskip("numpy")


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
recommend = load("recommend", "tools/tune/recommend.py")
spsa = load("spsa", "tools/tune/spsa/spsa.py")
spsa_candidates = load("spsa_candidates", "tools/tune/spsa/candidates.py")
spsa_pool = load("spsa_pool", "tools/tune/spsa/pool.py")
validate = load("validate", "tools/tune/validate.py")


class TuningToolsTest(unittest.TestCase):
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

    def test_ctt_opening_checkpoint_commits_only_after_match(self):
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
                    "CTT_OPENING_STATE": str(state), "CTT_OPENING_START": "1"}):
                sequenced, checkpoint = ctt.sequence_openings(command.copy())
                self.assertIn("start=1", sequenced)
                self.assertFalse(state.exists())
                ctt.advance_openings(checkpoint)
                self.assertEqual(json.loads(state.read_text()),
                                 {"games": 6, "next_opening": 4})
                sequenced, _ = ctt.sequence_openings(command.copy())
                self.assertIn("start=4", sequenced)

    def test_ctt_fastchess_state_does_not_overwrite_tuner_config(self):
        command = ctt.translate([], {})
        self.assertIn("outname=fastchess-state.json", command)

    def test_ctt_does_not_advance_after_a_recovered_engine_failure(self):
        process = types.SimpleNamespace(stdout=["Engine crashed (stdout)\n"], wait=lambda: 0)
        with mock.patch.object(ctt, "engines", return_value={}), \
                mock.patch.object(ctt, "translate", return_value=["fastchess"]), \
                mock.patch.object(ctt, "sequence_openings",
                                  return_value=(["fastchess"], "checkpoint")), \
                mock.patch.object(ctt.subprocess, "Popen", return_value=process) as popen, \
                mock.patch.object(ctt, "advance_openings") as advance:
            self.assertEqual(ctt.main(), 1)
        self.assertEqual(popen.call_args.kwargs["errors"], "replace")
        advance.assert_not_called()

    def test_clop_adapter_rejects_a_recovered_time_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            openings = pathlib.Path(directory, "openings.epd")
            openings.write_text("opening\n")
            process = types.SimpleNamespace(
                returncode=0,
                stdout="Finished game 1 (candidate vs baseline): 0-1 "
                       "{White loses on time (1ms overrun)}\n"
                       "Score of candidate vs baseline: 0 - 1 - 0\n")
            argv = [
                "clop_fastchess.py", "--fastchess", "fastchess", "--engine", "engine",
                "--openings", str(openings), "--opening-count", "1", "local", "0",
            ]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(clop.subprocess, "run", return_value=process), \
                    self.assertRaisesRegex(RuntimeError, "loses on time"):
                clop.main()

    def test_spsa_adapter_rejects_a_recovered_time_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            openings = directory / "openings.epd"
            openings.write_text("opening\n")
            logs = directory / "logs"
            logs.mkdir()
            process = types.SimpleNamespace(
                returncode=0,
                stdout=b"Finished game 1 (plus vs minus): 0-1 "
                       b"{White loses on time (1ms overrun)}\n"
                       b"Score of plus vs minus: 0 - 1 - 0\n")
            args = argparse.Namespace(
                fastchess="fastchess", engine="engine", engine_args="", tc="3+0.1",
                openings=str(openings), slots=1, logs=str(logs))
            with mock.patch.object(spsa.subprocess, "run", return_value=process), \
                    self.assertRaisesRegex(RuntimeError, "loses on time"):
                spsa.play(args, 0, 1, 1, {}, {})

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
                    "--baseline", "baseline", "--openings", "openings.epd"]
            with mock.patch.object(sys, "argv", argv):
                ctt_config.main()
            self.assertEqual(json.loads(config.read_text())["evaluate_points"], str(start))
            self.assertEqual(start.read_text(), "3,5\n")

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
                "    extra_points = load_points_to_evaluate(\n"
                "        space=opt.space,\n"
                "        csv_file=evaluate_points,\n"
                "    while True:\n        used_extra_point = False\n")
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
            self.assertLess(source.index("np.savez_compressed"), source.index("break"))
            self.assertIn('None if len(X) else settings.get("evaluate_points"', source)

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

            def play(args, number, opening, pairs, plus, minus):
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
