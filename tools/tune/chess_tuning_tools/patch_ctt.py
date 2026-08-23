"""Apply the reviewed commit's compatibility and finite-run patches."""

import pathlib

import tune


path = pathlib.Path(tune.__file__).with_name("local.py")
source = path.read_text()
old = "score = float(prob_to_elo(dist.mean().dot(scores), k=score_scale))"
new = "score = prob_to_elo(dist.mean().dot(scores), k=score_scale).item()"
if source.count(old) != 1:
    raise RuntimeError("the pinned CTT source no longer matches the reviewed patch")
path.write_text(source.replace(old, new))

path = pathlib.Path(tune.__file__).with_name("cli.py")
source = path.read_text()
old = "import logging\n"
new = "import logging\nimport os\nimport subprocess\n"
if source.count(old) != 1:
    raise RuntimeError("the pinned CTT import block no longer matches the reviewed patch")
source = source.replace(old, new)
old = "        for output_line in run_match(**match_settings):\n"
new = ('        os.environ["CTT_ITERATION"] = str(iteration)\n'
       '        for output_line in run_match(**match_settings):\n')
if source.count(old) != 1:
    raise RuntimeError("the pinned CTT match loop no longer matches the reviewed patch")
path.write_text(source.replace(old, new))

path = pathlib.Path(tune.__file__).with_name("local.py")
source = path.read_text()
old = "            if opt.space == old_opt.space:\n"
new = "            if opt.space == old_opt.space and list(old_opt.Xi) == X:\n"
if source.count(old) != 1:
    raise RuntimeError("the pinned CTT resume branch no longer matches the reviewed patch")
path.write_text(source.replace(old, new))

path = pathlib.Path(tune.__file__).with_name("local.py")
source = path.read_text()
old = "        return best_point, estimated_elo, float(best_std * 100)"
new = "        return best_point, estimated_elo, (best_std * 100).item()"
if source.count(old) != 1:
    raise RuntimeError("the pinned CTT result reporter no longer matches the reviewed patch")
path.write_text(source.replace(old, new))

path = pathlib.Path(tune.__file__).with_name("cli.py")
source = path.read_text()
old = "    while True:\n"
new = ('    max_iterations = settings.get("max_iterations", float("inf"))\n'
       '    while iteration <= max_iterations:\n')
if source.count(old) != 1:
    raise RuntimeError("the pinned CTT main loop no longer matches the reviewed patch")
source = source.replace(old, new)
old = "        used_extra_point = False\n"
new = '''        if iteration == max_iterations:
            with AtomicWriter(data_path, mode="wb", overwrite=True).open() as f:
                np.savez_compressed(
                    f, np.array(X), np.array(y), np.array(noise),
                    np.array(optima), np.array(performance), np.array(iteration),
                )
            if os.environ.get("CTT_OPENING_STATE"):
                subprocess.run(
                    ["cutechess-cli", "--commit-iteration", str(iteration)], check=True)
            break

        used_extra_point = False
'''
if source.count(old) != 1:
    raise RuntimeError("the pinned CTT experiment loop no longer matches the reviewed patch")
path.write_text(source.replace(old, new))

path = pathlib.Path(tune.__file__).with_name("utils.py")
source = path.read_text()
old = '''    raw_bounds = getattr(res.space, "bounds", None)
    minimize_bounds = None

    if raw_bounds is not None:
        minimize_bounds = []
        for bound in raw_bounds:
            lower, upper = bound
            lower_cast = None if lower is None else float(np.float64(lower))
            upper_cast = None if upper is None else float(np.float64(upper))
            minimize_bounds.append((lower_cast, upper_cast))
        minimize_bounds = tuple(minimize_bounds)
'''
new = '''    # The surrogate consumes transformed coordinates, not the raw option bounds.
    minimize_bounds = [(0.0, 1.0)] * res.space.transformed_n_dims
'''
if source.count(old) != 1:
    raise RuntimeError("the pinned CTT optimum finder no longer matches the reviewed patch")
path.write_text(source.replace(old, new))

path = pathlib.Path(tune.__file__).with_name("cli.py")
source = path.read_text()
old = '''    extra_points = load_points_to_evaluate(
        space=opt.space,
        csv_file=evaluate_points,
'''
new = '''    extra_points = load_points_to_evaluate(
        space=opt.space,
        csv_file=None if len(X) else settings.get("evaluate_points", evaluate_points),
'''
if source.count(old) != 1:
    raise RuntimeError("the pinned CTT initial-point loader no longer matches the reviewed patch")
path.write_text(source.replace(old, new))
