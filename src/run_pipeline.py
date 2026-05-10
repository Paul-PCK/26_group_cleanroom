import argparse
import os
import subprocess
import sys
from pathlib import Path

from config import PROJECT_ROOT


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Run the cleanroom pipeline from raw images to outputs.")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--skip-animation", action="store_true")
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


def run_step(name, script_name, extra_args=None):
    command = [sys.executable, str(SCRIPT_DIR / script_name)]
    if extra_args:
        command.extend(extra_args)
    print(f"\n== {name} ==")
    print(" ".join(command))
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "tmp" / "matplotlib"))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, env=env)


def main():
    args = parse_args()
    max_image_args = []
    if args.max_images is not None:
        max_image_args = ["--max-images", str(args.max_images)]

    run_step("1. Preprocessing", "preprocessing.py", max_image_args)
    run_step("2. YOLO apply", "yolo_apply.py", ["--conf", str(args.conf), *max_image_args])
    run_step("3. Projection", "projection.py")
    run_step("4. Final table", "final_table.py")
    run_step("5. Object integration", "object_integration.py")
    run_step("6. Timeline CSV", "timeline.py")
    if not args.skip_animation:
        run_step("7. Animation", "animate_timeline.py")


if __name__ == "__main__":
    main()
