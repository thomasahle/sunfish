#!/usr/bin/env python3

import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import training_protocol


def seal(record):
    payload = {key: value for key, value in record.items() if key != "canonical_sha256"}
    record["canonical_sha256"] = training_protocol.canonical_digest(payload)
    return record


class TrainingProtocolTest(unittest.TestCase):
    def fixture(self, directory, opening_count=6, space_default=1, start_default=1, duplicate=False):
        directory = pathlib.Path(directory)
        files = {
            "engine": directory / "engine",
            "tables": directory / "tables",
            "fastchess": directory / "fastchess",
            "training_book": directory / "openings.epd",
            "opponent_panel": directory / "panel.json",
        }
        files["engine"].write_bytes(b"engine")
        files["tables"].write_bytes(b"tables")
        files["fastchess"].write_bytes(b"fastchess")
        positions = [f"position-{number}" for number in range(opening_count)]
        if duplicate:
            positions[-1] = positions[0]
        files["training_book"].write_text("\n".join(positions) + "\n")
        panel = [{"name": "master", "weight": 1}]
        files["opponent_panel"].write_text(json.dumps(panel, indent=2) + "\n")
        space = {
            "parameters": [{
                "name": "X", "type": "integer", "min": 0, "max": 2,
                "default": space_default,
            }],
        }
        space_path = directory / "space.json"
        space_path.write_text(json.dumps(space, indent=2) + "\n")
        artifacts = {
            name: {"sha256": training_protocol.file_digest(path)}
            for name, path in files.items()
        }
        artifacts["opponent_panel"]["canonical_sha256"] = (
            training_protocol.canonical_digest(panel))
        training = {
            "time_control": "3+0.1",
            "paired_observations": 6,
            "games": 12,
            "checkpoints": [0, 4, 8, 12],
            "openings": {
                "artifact": "training_book", "order": "sequential",
                "first_line": 1, "cycle": False,
            },
            "panel": {
                "artifact": "opponent_panel", "seed": 2026,
                "schedule": "weighted-shuffle-v1",
            },
        }
        methods = {
            "logistic_gp": seal({
                "paired_observations_per_update": 1,
                "implementation": {"version": "test-gp"},
                "settings": {
                    "pairs": 1, "total_batches": 6, "duel_fraction": 0.0,
                    "opening_order": "sequential", "cycle_openings": False,
                    "panel_seed": 2026,
                },
            }),
            "chess_tuning_tools": seal({
                "paired_observations_per_update": 1,
                "implementation": {"version": "test-ctt"},
                "settings": {"iterations": 6, "rounds": 1},
            }),
            "rbfopt": seal({
                "paired_observations_per_update": 1,
                "implementation": {"version": "test-rbfopt"},
                "settings": {"games": 12, "noisy_pairs": 1},
            }),
            "spsa": seal({
                "paired_observations_per_update": 2,
                "implementation": {"version": "test-spsa"},
                "common_random_numbers": {
                    "same_opening": True, "same_panel_member": True,
                },
                "settings": {"iterations": 3, "pairs_per_step": 2},
            }),
            "clop": seal({
                "paired_observations_per_update": 1,
                "implementation": {"version": "test-clop"},
                "settings": {"max_games": 12, "replications": 2},
            }),
        }
        protocol = seal({
            "version": 1,
            "id": "five-tuner-test",
            "artifacts": artifacts,
            "starts": {
                "05": {
                    "options": {"X": start_default},
                    "options_canonical_sha256": training_protocol.canonical_digest(
                        {"X": start_default}),
                    "space": {
                        "sha256": training_protocol.file_digest(space_path),
                        "canonical_sha256": training_protocol.canonical_digest(space),
                    },
                },
            },
            "training": training,
            "methods": methods,
        })
        protocol_path = directory / "protocol.json"
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
        return protocol_path, space_path, files, protocol

    def binding(self, protocol_path, space_path, files, protocol, **changes):
        values = {
            "path": protocol_path,
            "method_id": "logistic_gp",
            "start_id": "05",
            "artifacts": files,
            "space": space_path,
            "training": protocol["training"],
            "settings": protocol["methods"]["logistic_gp"]["settings"],
        }
        values.update(changes)
        return training_protocol.bind(**values)

    def test_binds_exact_portable_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            identity = self.binding(*fixture)
            self.assertEqual(identity["protocol"]["id"], "five-tuner-test")
            self.assertEqual(identity["method"]["id"], "logistic_gp")
            self.assertEqual(identity["start"]["id"], "05")
            self.assertNotIn("path", identity["artifacts"]["engine"])
            self.assertEqual(identity["training"]["checkpoints"], [0, 4, 8, 12])

    def test_freeze_refreshes_all_derived_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol_path, _, _, protocol = self.fixture(directory)
            protocol["canonical_sha256"] = "stale"
            protocol["methods"]["logistic_gp"]["canonical_sha256"] = "stale"
            protocol["starts"]["05"]["options_canonical_sha256"] = "stale"
            protocol_path.write_text(json.dumps(protocol))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                training_protocol.main([str(protocol_path)])
            self.assertEqual(json.loads(output.getvalue()), training_protocol.freeze(protocol))
            frozen = training_protocol.freeze(protocol)
            protocol_path.write_text(json.dumps(frozen))
            self.assertEqual(training_protocol.load(protocol_path), frozen)

    def test_rejects_manifest_and_method_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol_path, _, _, protocol = self.fixture(directory)
            protocol["training"]["games"] = 10
            protocol_path.write_text(json.dumps(protocol))
            with self.assertRaisesRegex(training_protocol.ProtocolError, "protocol canonical"):
                training_protocol.load(protocol_path)
            protocol["training"]["games"] = 12
            protocol["methods"]["logistic_gp"]["settings"]["pairs"] = 2
            seal(protocol)
            protocol_path.write_text(json.dumps(protocol))
            with self.assertRaisesRegex(training_protocol.ProtocolError, "method logistic_gp"):
                training_protocol.load(protocol_path)

    def test_rejects_runner_setting_and_training_disagreement(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            protocol = fixture[-1]
            settings = dict(protocol["methods"]["logistic_gp"]["settings"], pairs=2)
            with self.assertRaisesRegex(training_protocol.ProtocolError, "runner settings"):
                self.binding(*fixture, settings=settings)
            training = json.loads(json.dumps(protocol["training"]))
            training["time_control"] = "1+0.01"
            with self.assertRaisesRegex(training_protocol.ProtocolError, "training settings"):
                self.binding(*fixture, training=training)

    def test_rejects_a_sealed_unfair_gp_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol_path, _, _, protocol = self.fixture(directory)
            protocol["methods"]["logistic_gp"]["settings"]["duel_fraction"] = 0.2
            protocol = training_protocol.freeze(protocol)
            protocol_path.write_text(json.dumps(protocol))
            with self.assertRaisesRegex(training_protocol.ProtocolError, "common training"):
                training_protocol.load(protocol_path)

    def test_rejects_unknown_method_and_start(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            with self.assertRaisesRegex(training_protocol.ProtocolError, "unregistered tuner"):
                self.binding(*fixture, method_id="unknown")
            with self.assertRaisesRegex(training_protocol.ProtocolError, "unregistered recovery"):
                self.binding(*fixture, start_id="99")

    def test_checks_byte_and_canonical_artifact_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol_path, space_path, files, protocol = self.fixture(directory)
            files["opponent_panel"].write_text('[{"weight": 1, "name": "master"}]\n')
            with self.assertRaisesRegex(training_protocol.ProtocolError, "byte identity"):
                self.binding(protocol_path, space_path, files, protocol)
            files["opponent_panel"].write_text('[{"name": "peer", "weight": 1}]\n')
            protocol["artifacts"]["opponent_panel"]["sha256"] = (
                training_protocol.file_digest(files["opponent_panel"]))
            seal(protocol)
            protocol_path.write_text(json.dumps(protocol))
            with self.assertRaisesRegex(training_protocol.ProtocolError, "canonical identity"):
                self.binding(protocol_path, space_path, files, protocol)

    def test_requires_start_defaults_and_unique_opening_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            mismatch = self.fixture(directory, space_default=2, start_default=1)
            with self.assertRaisesRegex(training_protocol.ProtocolError, "space defaults"):
                self.binding(*mismatch)
        with tempfile.TemporaryDirectory() as directory:
            short = self.fixture(directory, opening_count=5)
            with self.assertRaisesRegex(training_protocol.ProtocolError, "unique opening line 6"):
                self.binding(*short)
        with tempfile.TemporaryDirectory() as directory:
            duplicate = self.fixture(directory, duplicate=True)
            with self.assertRaisesRegex(training_protocol.ProtocolError, "duplicate positions"):
                self.binding(*duplicate)

    def test_requires_sequential_noncycling_complete_pair_checkpoints(self):
        for field, value, message in (
                ("order", "random", "sequential"),
                ("cycle", True, "must not cycle")):
            with tempfile.TemporaryDirectory() as directory:
                protocol_path, _, _, protocol = self.fixture(directory)
                protocol["training"]["openings"][field] = value
                seal(protocol)
                protocol_path.write_text(json.dumps(protocol))
                with self.assertRaisesRegex(training_protocol.ProtocolError, message):
                    training_protocol.load(protocol_path)
        with tempfile.TemporaryDirectory() as directory:
            protocol_path, _, _, protocol = self.fixture(directory)
            protocol["training"]["checkpoints"] = [0, 3, 12]
            seal(protocol)
            protocol_path.write_text(json.dumps(protocol))
            with self.assertRaisesRegex(training_protocol.ProtocolError, "complete pairs"):
                training_protocol.load(protocol_path)

    def test_spsa_common_random_numbers_are_explicit_and_budgeted(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol_path, _, _, protocol = self.fixture(directory)
            spsa = protocol["methods"]["spsa"]
            spsa["common_random_numbers"]["same_opening"] = False
            seal(spsa)
            seal(protocol)
            protocol_path.write_text(json.dumps(protocol))
            with self.assertRaisesRegex(training_protocol.ProtocolError, "common-random"):
                training_protocol.load(protocol_path)
        with tempfile.TemporaryDirectory() as directory:
            protocol_path, _, _, protocol = self.fixture(directory)
            spsa = protocol["methods"]["spsa"]
            spsa["paired_observations_per_update"] = 4
            seal(spsa)
            seal(protocol)
            protocol_path.write_text(json.dumps(protocol))
            with self.assertRaisesRegex(training_protocol.ProtocolError, "does not divide"):
                training_protocol.load(protocol_path)
        with tempfile.TemporaryDirectory() as directory:
            protocol_path, _, _, protocol = self.fixture(directory)
            protocol["training"]["checkpoints"] = [0, 2, 12]
            seal(protocol)
            protocol_path.write_text(json.dumps(protocol))
            with self.assertRaisesRegex(training_protocol.ProtocolError, "split.*SPSA|split.*spsa"):
                training_protocol.load(protocol_path)


if __name__ == "__main__":
    unittest.main()
