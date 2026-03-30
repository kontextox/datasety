#!/usr/bin/env python3
"""
datasety - resize, align, caption, shuffle, synthetic, mask, filter, degrade,
character, sweep, workflow, inspect, server.

Usage:
    datasety resize --input ./in --output ./out --resolution 768x1024 --crop-position top
    datasety align --target ./target --control ./control --dry-run
    datasety caption --input ./in --output ./out --trigger-word "[trigger]" --florence-2-base
    datasety shuffle --input ./in --output ./out --group "Hello.|Hey!" --group "World.|Earth!"
    datasety synthetic --input ./in --output ./out --prompt "add a winter hat"
    datasety mask --input ./in --output ./masks --keywords "face,hair" --model clipseg
    datasety filter --input ./in --output ./rejected --query "leg,male face" --action move
    datasety degrade --input ./in --output ./out --type jpeg --intensity 0.5
    datasety character --reference face.jpg --output ./dataset --llm-ollama qwen3.5:4b
    datasety workflow --file datasety.yaml --dry-run
    datasety inspect --input ./dataset --duplicates
    datasety server --input ./dataset --port 8080
"""

import argparse

from datasety import (
    align,
    caption,
    degrade,
    filter,
    inspect,
    mask,
    resize,
    server,
    shuffle,
    synthetic,
)

# Conditionally import new commands
try:
    from datasety import character
except ImportError:
    character = None

try:
    from datasety import workflow
except ImportError:
    workflow = None

try:
    from datasety import sweep
except ImportError:
    sweep = None

try:
    from datasety import train
except ImportError:
    train = None

try:
    from datasety import audio
except ImportError:
    audio = None

try:
    from datasety import upload
except ImportError:
    upload = None


def build_parser():
    """Build and return the main argument parser with all subcommands."""
    from datasety import __version__

    parser = argparse.ArgumentParser(
        prog="datasety",
        description="CLI tool for dataset preparation: image resizing, captioning, and more.",
    )
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    resize.register_parser(subparsers)
    caption.register_parser(subparsers)
    align.register_parser(subparsers)
    shuffle.register_parser(subparsers)
    synthetic.register_parser(subparsers)
    mask.register_parser(subparsers)
    filter.register_parser(subparsers)
    inspect.register_parser(subparsers)
    degrade.register_parser(subparsers)
    server.register_parser(subparsers)

    if character is not None:
        character.register_parser(subparsers)

    if workflow is not None:
        workflow.register_parser(subparsers)

    if sweep is not None:
        sweep.register_parser(subparsers)

    if train is not None:
        train.register_parser(subparsers)

    if audio is not None:
        audio.register_parser(subparsers)

    if upload is not None:
        upload.register_parser(subparsers)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
