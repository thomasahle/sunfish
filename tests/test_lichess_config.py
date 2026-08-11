"""Static checks on the lichess bundle: config.yml, greeting, patch, pins.

Fast and network-free (no BOT_CI gate).  These exist because of production
incidents that a static check would have caught:

* 2026-08-11: the configured greeting was 156-158 characters; lichess caps
  chat messages at 140.  The message was silently dropped -- and, worse, the
  rejected POST cancelled the bot's first move in every game (see
  lichess-bot.patch).  Greetings must fit for the longest possible opponent
  name (lichess usernames are at most 20 characters).
* The deployed lichess-bot tree is pin + patch; the pin recorded in setup.sh
  and the one the integration test runs against must be the same commit.
"""

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLE = REPO_ROOT / "tools" / "lichess"

MAX_CHAT_LEN = 140  # lichess's chat message cap
MAX_USERNAME_LEN = 20  # lichess's username cap
BOT_NAME = "sunfish-engine"


@pytest.fixture(scope="module")
def config():
    return yaml.safe_load((BUNDLE / "config.yml").read_text())


def test_config_parses_and_has_placeholders(config):
    assert config["token"] == "YOUR_TOKEN_HERE", \
        "config.yml must ship the token placeholder, never a real token"
    assert config["url"].startswith("https://lichess.org")


def test_engine_entry_exists(config):
    engine = config["engine"]
    assert engine["protocol"] == "uci"
    assert (REPO_ROOT / engine["name"]).exists(), \
        f"engine {engine['name']} not in this repository"
    assert int(engine["uci_options"]["TABLE_SIZE"]) > 0


def test_concurrency_is_one(config):
    # Decision (Thomas, 2026-08-11): one game at full strength, always.
    # Overflow games are aborted by the patched bridge, not slow-served.
    assert config["challenge"]["concurrency"] == 1


def test_greetings_fit_lichess_chat_cap(config):
    longest = {"opponent": "x" * MAX_USERNAME_LEN,
               "me": max(BOT_NAME, "x" * MAX_USERNAME_LEN, key=len)}
    for field, text in config.get("greeting", {}).items():
        rendered = text.format(**longest)
        assert len(rendered) <= MAX_CHAT_LEN, \
            (f"greeting.{field} is {len(rendered)} chars with a "
             f"{MAX_USERNAME_LEN}-char name; lichess silently drops "
             f"messages over {MAX_CHAT_LEN}: {rendered!r}")


def test_patch_ships_and_targets_the_bridge():
    patch = (BUNDLE / "lichess-bot.patch").read_text()
    assert patch.startswith("diff --git"), "lichess-bot.patch is not a git diff"
    for target in ("lib/lichess.py", "lib/lichess_bot.py"):
        assert f"a/{target}" in patch, f"patch no longer covers {target}"
    # The three production fixes must all be present.
    assert "EVENT_STREAM_SILENCE_LIMIT" in patch, "stream-death fix missing"
    assert "games_in_progress" in patch, "concurrency overflow fix missing"
    assert "move_due" in patch, "interrupted-move retry fix missing"


def test_setup_pin_matches_integration_test():
    setup = (BUNDLE / "setup.sh").read_text()
    pin = re.search(r"LICHESS_BOT_COMMIT=([0-9a-f]{40})", setup)
    assert pin, "setup.sh no longer pins lichess-bot"
    test_src = (REPO_ROOT / "tests" / "test_bot_integration.py").read_text()
    tested = re.search(r'"([0-9a-f]{40})"\)', test_src)
    assert tested and pin.group(1) == tested.group(1), \
        "setup.sh deploys a different lichess-bot commit than the one tested"
    assert "lichess-bot.patch" in setup, \
        "setup.sh does not apply the production patch"
