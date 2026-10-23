from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from direct_invension.route34_cli import run_route_evaluate_main


def main() -> None:
    run_route_evaluate_main(
        argv=sys.argv[1:],
        route_name="route4",
        route_title="Route IV eval",
    )


if __name__ == "__main__":
    main()
