"""Small paired-opening result model shared by game-result tuners."""

import math
import re


WDL = re.compile(r"Score of.*?:\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)")
MATCH = re.compile(r"Score of (.*?) vs (.*?):")
FINISHED = re.compile(r"Finished game (\d+)")
GAME = re.compile(
    r"Finished game (\d+) \((.*?) vs (.*?)\):\s*(1-0|0-1|1/2-1/2)")
FAILURE = re.compile(
    r"(?:\b(?:White|Black) (?:disconnects|makes an illegal move|loses on time)\b|"
    r"\b(?:White|Black)'s connection stalls\b|\bstalled / disconnected\b|"
    r"\bEngine\b[^\n]*\b(?:disconnects|stalls|didn't respond|did not respond|"
    r"is not responsive|is non[- ]?responsive)\b|(?:^|[;:]\s*)Engine crashed\b|"
    r"^\s*Warning;\s*Illegal move\b|\btime forfeit\b|"
    r"^\s*(?:Timeouts|Crashed):\s*[1-9]\d*\s*$|"
    r"\b(?:disconnect(?:ed|s)?|stall(?:ed|s)?|crash(?:ed|es)?|forfeit(?:ed|s)?)\b"
    r"(?![- ]resistant|\s*[:=]\s*0\b))",
    re.IGNORECASE | re.MULTILINE)
PRIOR = (0.14 * 2.5, 0.19 * 2.5, 0.34 * 2.5, 0.19 * 2.5, 0.14 * 2.5)
PAIR_LOSS = (0, 0.25, 0.5, 0.75, 1)


class EngineFailure(RuntimeError):
    pass


def failure(output):
    """Return the first engine-failure marker reported by fastchess."""
    match = FAILURE.search(output)
    return match.group(0) if match else None


def reject_failures(output):
    """Refuse recovered fastchess output that reports an engine failure."""
    if marker := failure(output):
        raise EngineFailure(f"engine failure reported: {marker}")


def game_results(output, partial=False, subject=None):
    """Restore per-game W/L/D increments in game-number order."""
    reject_failures(output)
    snapshots = [tuple(map(int, match)) for match in WDL.findall(output)]
    matches = MATCH.findall(output)
    games = GAME.findall(output)
    if games and (matches or subject):
        subject = subject or matches[-1][0]
        results = {}
        for number, white, black, result in games:
            if result == "1/2-1/2":
                value = (0, 0, 1)
            else:
                winner = white if result == "1-0" else black
                value = (1, 0, 0) if winner == subject else (0, 1, 0)
            results[int(number)] = value
        order = sorted(results)
        if order != list(range(1, len(results) + 1)):
            raise ValueError("match output does not contain complete paired results")
        if partial:
            order = order[:len(order) // 2 * 2]
        elif len(order) % 2:
            raise ValueError("match output does not contain complete paired results")
        ordered = [results[number] for number in order]
        wdl = tuple(sum(result[i] for result in ordered) for i in range(3))
        if snapshots and sum(snapshots[-1]) == len(ordered) and wdl != snapshots[-1]:
            raise ValueError("per-game results disagree with the final score")
        return ordered, wdl
    order = [int(game) for game in FINISHED.findall(output)]
    if partial and not order and not snapshots:
        return [], (0, 0, 0)
    if len(snapshots) != len(order) or len(order) % 2:
        raise ValueError("match output does not contain complete paired results")
    previous = (0, 0, 0)
    results = []
    for snapshot in snapshots:
        results.append(tuple(value - old for value, old in zip(snapshot, previous)))
        previous = snapshot
    return [result for _, result in sorted(zip(order, results))], snapshots[-1]


def summarize(results):
    """Return pentanomial counts, W/D/L, and normalized pair scores."""
    counts, scores = [0] * 5, []
    wdl = tuple(sum(result[i] for result in results) for i in range(3))
    for first, second in zip(results[::2], results[1::2]):
        wins, losses, draws = (first[i] + second[i] for i in range(3))
        scores.append((wins + draws / 2) / 2)
        if wins == 2:
            counts[0] += 1
        elif wins == 1 and losses == 0:
            counts[1] += 1
        elif wins == 1 or draws == 2:
            counts[2] += 1
        elif losses == 1:
            counts[3] += 1
        else:
            counts[4] += 1
    return counts, wdl, scores


def pair_scores(output):
    """Return the candidate's normalized score for each color-swapped pair."""
    results, _ = game_results(output)
    return summarize(results)[2]


def parse(output):
    """Return WW/WD/WL-or-DD/LD/LL counts and final W/D/L."""
    results, _ = game_results(output)
    return summarize(results)[:2]


def posterior(counts, confidence=1.96):
    """Return loss-score mean, standard deviation, and clipped normal interval."""
    alpha = [count + prior for count, prior in zip(counts, PRIOR)]
    total = sum(alpha)
    mean = sum(a * score for a, score in zip(alpha, PAIR_LOSS)) / total
    second = sum(a * score ** 2 for a, score in zip(alpha, PAIR_LOSS)) / total
    deviation = math.sqrt(max(0, (second - mean ** 2) / (total + 1)))
    return mean, deviation, (max(0, mean - confidence * deviation),
                             min(1, mean + confidence * deviation))
