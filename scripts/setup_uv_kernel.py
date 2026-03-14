#!/usr/bin/env python3

"""
This script creates a kernel with ephemeral environments for notebooks, every
kernel reload refresh the virtual environment, the packages are deduplicated and
retrieved and from uv cache

use on jupyter notebook cell to install a package:
!uv pip install <package>
"""
import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

UV = shutil.which("uv")
assert UV is not None, "uv not found in PATH"
UV_DIR = str(Path(UV).parent)


def main():
    parser = argparse.ArgumentParser(
        description="Setup Jupyter kernels for uv"
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        default=["3.11"],
        help="Python versions to configure (default: 3.11)",
    )

    args = parser.parse_args()

    # kernel_base = Path.home() / ".local" / "share" / "jupyter" / "kernels"
    kernel_base = Path(sys.prefix) / "share" / "jupyter" / "kernels"

    for version in args.versions:
        kernel_file = kernel_base / f"uv-{version}" / "kernel.json"

        kernel_file.parent.mkdir(parents=True, exist_ok=True)

        kernel_config = {
            "env": {
                "PATH": "${PATH}:" + UV_DIR,
                # Critical: Unset PYTHONPATH to prevent importing from host system
                "PYTHONPATH": "",
            },
            "argv": [
                UV,
                "run",
                "--python",
                version,
                "--with",
                "ipykernel",
                "--with",
                "ipywidgets",
                "--with",
                "black",
                "--with",
                "isort",
                "--no-project",
                "--isolated",
                # "--refresh",
                "python",
                "-m",
                "ipykernel_launcher",
                "-f",
                "{connection_file}",
            ],
            "display_name": f"uv-{version}",
            "language": "python",
            "metadata": {"debugger": True},
        }

        if kernel_file.exists():
            logger.info(f"Kernel already exists: uv-{version}")
            continue

        kernel_file.write_text(
            json.dumps(kernel_config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        logger.info(
            f"Kernel configured for Python {version} at: {kernel_file}"
        )

    logger.info("Refresh your list of kernels to see them.")
    logger.info(
        "Note: Kernels use isolated environments and will install ipykernel on first use."
    )


if __name__ == "__main__":
    main()
