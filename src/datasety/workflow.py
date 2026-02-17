"""Run multi-step datasety workflows from YAML or JSON files."""

import json
import sys
from pathlib import Path


def _find_workflow_file(path=None):
    """Find a workflow file, auto-detecting if no path given.

    Searches for datasety.yaml, datasety.yml, datasety.json in the
    current directory.
    """
    if path:
        p = Path(path)
        if not p.exists():
            print(f"Error: Workflow file not found: {p}")
            sys.exit(1)
        return p

    for name in ["datasety.yaml", "datasety.yml", "datasety.json"]:
        p = Path(name)
        if p.exists():
            return p

    print("Error: No workflow file found. Create datasety.yaml or use --file/-f.")
    sys.exit(1)


def _load_workflow(path):
    """Load a workflow from YAML or JSON file."""
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    if suffix == ".json":
        return json.loads(text)

    # YAML
    try:
        import yaml
    except ImportError:
        print("Error: pyyaml is required for YAML workflow files.")
        print("Run: pip install pyyaml")
        sys.exit(1)

    return yaml.safe_load(text)


def _args_to_argv(command, args_dict):
    """Convert a command name + args dict to an argparse-compatible argv list.

    Example:
        _args_to_argv("resize", {"input": "./raw", "resolution": "768x1024"})
        => ["resize", "--input", "./raw", "--resolution", "768x1024"]
    """
    argv = [command]
    for key, value in args_dict.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        elif isinstance(value, list):
            for item in value:
                argv.extend([flag, str(item)])
        else:
            argv.extend([flag, str(value)])
    return argv


# ── Validation functions per command ──


def _validate_resize(args):
    """Validate resize command args."""
    errors = []
    if hasattr(args, "input") and args.input:
        if not Path(args.input).exists():
            errors.append(f"Input directory does not exist: {args.input}")
    if hasattr(args, "resolution") and args.resolution:
        try:
            w, h = map(int, args.resolution.lower().split("x"))
        except ValueError:
            errors.append(f"Invalid resolution format: {args.resolution} (use WIDTHxHEIGHT)")
    return errors


def _validate_caption(args):
    """Validate caption command args."""
    import os

    errors = []
    if hasattr(args, "input") and args.input:
        if not Path(args.input).exists():
            errors.append(f"Input directory does not exist: {args.input}")
    if getattr(args, "llm_api", False):
        if not os.environ.get("OPENAI_API_KEY"):
            errors.append("OPENAI_API_KEY environment variable is required for --llm-api")
    return errors


def _validate_align(args):
    """Validate align command args."""
    errors = []
    if hasattr(args, "target") and args.target:
        if not Path(args.target).exists():
            errors.append(f"Target directory does not exist: {args.target}")
    if hasattr(args, "control") and args.control:
        if not Path(args.control).exists():
            errors.append(f"Control directory does not exist: {args.control}")
    return errors


def _validate_shuffle(args):
    """Validate shuffle command args."""
    errors = []
    if hasattr(args, "input") and args.input:
        if not Path(args.input).exists():
            errors.append(f"Input directory does not exist: {args.input}")
    if not getattr(args, "group", None):
        errors.append("At least one --group is required")
    return errors


def _validate_synthetic(args):
    """Validate synthetic command args."""
    errors = []
    if hasattr(args, "input") and args.input:
        if not Path(args.input).exists():
            errors.append(f"Input directory does not exist: {args.input}")
    return errors


def _validate_mask(args):
    """Validate mask command args."""
    errors = []
    if hasattr(args, "input") and args.input:
        if not Path(args.input).exists():
            errors.append(f"Input directory does not exist: {args.input}")
    if hasattr(args, "keywords") and args.keywords:
        keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
        if not keywords:
            errors.append("No valid keywords provided")
    return errors


def _validate_degrade(args):
    """Validate degrade command args."""
    errors = []
    if hasattr(args, "input") and args.input:
        if not Path(args.input).exists():
            errors.append(f"Input directory does not exist: {args.input}")
    return errors


def _validate_character(args):
    """Validate character command args."""
    errors = []
    refs = getattr(args, "reference", None)
    if refs:
        for r in refs:
            if not Path(r).exists():
                errors.append(f"Reference image does not exist: {r}")
    llm_selected = (
        getattr(args, "llm_api", False)
        or getattr(args, "llm_ollama", "")
        or getattr(args, "llm_gguf", "")
        or getattr(args, "llm_model", "")
        or getattr(args, "prompts_file", "")
    )
    if not llm_selected:
        errors.append("An LLM backend or --prompts-file is required")
    return errors


_VALIDATORS = {
    "resize": _validate_resize,
    "caption": _validate_caption,
    "align": _validate_align,
    "shuffle": _validate_shuffle,
    "synthetic": _validate_synthetic,
    "mask": _validate_mask,
    "degrade": _validate_degrade,
    "character": _validate_character,
}


def cmd_workflow(args):
    """Execute the workflow command."""
    from datasety.cli import build_parser

    workflow_path = _find_workflow_file(args.file)
    print(f"Workflow file: {workflow_path}")

    data = _load_workflow(workflow_path)

    steps = data.get("steps", [])
    if not steps:
        print("Error: Workflow file contains no steps.")
        sys.exit(1)

    print(f"Found {len(steps)} step(s)")
    print("-" * 50)

    dry_run = args.dry_run

    # Build parser to validate args for each step
    parser = build_parser()

    all_ok = True

    for i, step in enumerate(steps, 1):
        command = step.get("command", "")
        step_args = step.get("args", {})

        if not command:
            print(f"Step {i}: [ERROR] Missing 'command' field")
            all_ok = False
            continue

        argv = _args_to_argv(command, step_args)
        step_desc = f"Step {i}: {command}"

        # Parse through argparse to validate
        try:
            parsed = parser.parse_args(argv)
        except SystemExit:
            print(f"{step_desc}: [ERROR] Invalid arguments: {' '.join(argv[1:])}")
            all_ok = False
            continue

        # Run validator if available
        validator = _VALIDATORS.get(command)
        if validator:
            errors = validator(parsed)
            if errors:
                print(f"{step_desc}: [FAIL]")
                for err in errors:
                    print(f"  - {err}")
                all_ok = False
                continue

        if dry_run:
            print(f"{step_desc}: [OK] {' '.join(argv[1:])}")
            continue

        # Execute the step
        print(f"{step_desc}: Running...")
        try:
            parsed.func(parsed)
            print(f"{step_desc}: [DONE]")
        except SystemExit as e:
            if e.code != 0:
                print(f"{step_desc}: [ERROR] Exited with code {e.code}")
                all_ok = False
                break
        except Exception as e:
            print(f"{step_desc}: [ERROR] {e}")
            all_ok = False
            break

    print("-" * 50)
    if dry_run:
        if all_ok:
            print("Dry run: All steps validated successfully.")
        else:
            print("Dry run: Some steps have errors.")
            sys.exit(1)
    else:
        if all_ok:
            print("Workflow completed successfully.")
        else:
            print("Workflow failed.")
            sys.exit(1)


def register_parser(subparsers):
    """Register the workflow subcommand."""
    wf_parser = subparsers.add_parser(
        "workflow", help="Run multi-step datasety workflows from YAML or JSON files"
    )
    wf_parser.add_argument(
        "--file",
        "-f",
        default=None,
        help="Path to workflow file (default: auto-detect datasety.yaml/yml/json)",
    )
    wf_parser.add_argument(
        "--dry-run", action="store_true", help="Validate workflow steps without executing them"
    )
    wf_parser.set_defaults(func=cmd_workflow)
