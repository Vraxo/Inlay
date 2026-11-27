from pathlib import Path

from inlay_qwen3vl.cli import build_parser


def test_defaults_to_the_instruct_model_alias():
    args = build_parser().parse_args([str(Path("screen.png"))])

    assert args.model == "qwen3vl-8b"
