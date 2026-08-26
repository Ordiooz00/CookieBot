import argparse
import datetime as dt
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import msvcrt
import numpy as np
import pyautogui


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Image-matching auto clicker for games running in LDPlayer."
    )
    parser.add_argument(
        "--template",
        help="Path to one template image file.",
    )
    parser.add_argument(
        "--templates",
        nargs="+",
        help=(
            "Multiple template images in priority order. "
            "First match above threshold will be clicked."
        ),
    )
    parser.add_argument(
        "--numbered",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Load numbered templates in priority order, e.g. --numbered 1 8",
    )
    parser.add_argument(
        "--template-dir",
        default=".",
        help="Directory for numbered templates. Default: current folder",
    )
    parser.add_argument(
        "--template-ext",
        default=".png",
        help="File extension for numbered templates. Default: .png",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.88,
        help="Matching threshold in range [0.0, 1.0]. Default: 0.88",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.7,
        help="Delay (seconds) between scans. Default: 0.7",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=1.2,
        help="Minimum delay (seconds) between clicks. Default: 1.2",
    )
    parser.add_argument(
        "--loop-delay",
        type=float,
        default=2.0,
        help="Delay (seconds) after completing a full sequence loop. Default: 2.0",
    )
    parser.add_argument(
        "--region",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        help="Screen region to scan only. Example: --region 100 100 800 600",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Use grayscale matching for better speed.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one scan and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not click. Print detected coordinates only.",
    )
    return parser.parse_args()


def now() -> str:
    return dt.datetime.now().strftime("%H:%M:%S")


def handle_keyboard_input(templates: List[Tuple[str, np.ndarray]]) -> Tuple[bool, bool]:
    """Check keyboard input for controls: 'q' to quit, 'r' to reset sequence to template 1."""
    stop_requested = False
    reset_requested = False
    while msvcrt.kbhit():
        key = msvcrt.getwch().lower()
        if key == "q":
            stop_requested = True
        elif key == "r":
            reset_requested = True
            first_name = templates[0][0]
            print(f"[{now()}] Restart requested (r pressed): sequence reset to 1 ({first_name})")
    return stop_requested, reset_requested


def load_template(template_path: Path, grayscale: bool) -> np.ndarray:
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR

    # Use byte decoding directly so Windows Unicode paths do not trigger imread warnings.
    file_bytes = np.fromfile(str(template_path), dtype=np.uint8)
    template = cv2.imdecode(file_bytes, mode)

    if template is None:
        raise ValueError(f"Failed to read template image: {template_path}")
    return template


def resolve_template_paths(args: argparse.Namespace) -> List[Path]:
    paths: List[Path] = []
    if args.template:
        paths.append(Path(args.template).expanduser().resolve())
    if args.templates:
        paths.extend(Path(raw).expanduser().resolve() for raw in args.templates)

    if args.numbered:
        start, end = args.numbered
        if start > end:
            raise ValueError("--numbered requires START <= END")

        template_dir = Path(args.template_dir).expanduser().resolve()
        ext = args.template_ext if args.template_ext.startswith(".") else f".{args.template_ext}"

        for index in range(start, end + 1):
            paths.append((template_dir / f"{index}{ext}").resolve())

    if not paths:
        raise ValueError("Provide --template, --templates, or --numbered")
    return paths


def load_templates(paths: List[Path], grayscale: bool) -> List[Tuple[str, np.ndarray]]:
    loaded: List[Tuple[str, np.ndarray]] = []
    for path in paths:
        loaded.append((path.name, load_template(path, grayscale)))
    return loaded


def screenshot_to_cv(region, grayscale: bool) -> np.ndarray:
    shot = pyautogui.screenshot(region=region)
    frame = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)
    if grayscale:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def find_template(frame: np.ndarray, template: np.ndarray):
    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return max_val, max_loc


def compute_click_point(top_left, template_shape, region):
    tpl_h, tpl_w = template_shape[:2]
    x = top_left[0] + tpl_w // 2
    y = top_left[1] + tpl_h // 2
    if region:
        x += region[0]
        y += region[1]
    return x, y


def main() -> None:
    args = parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in range [0.0, 1.0]")

    template_paths = resolve_template_paths(args)
    templates = load_templates(template_paths, args.grayscale)

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.05

    print("Image bot started")
    print(f"- Loaded templates: {', '.join(name for name, _ in templates)}")
    print("- Move mouse to top-left corner to trigger PyAutoGUI failsafe")
    print("- Press q in this terminal window to stop")
    print("- Press r in this terminal window to restart sequence from template 1")
    print("- Starting in 3 seconds...")
    time.sleep(3)

    last_click_ts = 0.0
    use_sequence_mode = args.numbered is not None
    sequence_index = 0

    while True:
        stop_requested, reset_requested = handle_keyboard_input(templates)
        if stop_requested:
            print(f"[{now()}] Stop requested (q pressed).")
            break
        if reset_requested:
            sequence_index = 0

        frame = screenshot_to_cv(args.region, args.grayscale)

        best_name = ""
        best_score = -1.0
        matched = False

        if use_sequence_mode:
            templates_to_check = [templates[sequence_index]]
        else:
            templates_to_check = templates

        for tpl_name, tpl_img in templates_to_check:
            score, top_left = find_template(frame, tpl_img)
            if score > best_score:
                best_score = score
                best_name = tpl_name

            if score >= args.threshold:
                click_x, click_y = compute_click_point(top_left, tpl_img.shape, args.region)
                print(
                    f"[{now()}] Match {tpl_name} score={score:.3f} at ({click_x}, {click_y})"
                )

                seconds_since_last_click = time.time() - last_click_ts
                if seconds_since_last_click < args.cooldown:
                    wait_left = args.cooldown - seconds_since_last_click
                    print(f"[{now()}] Cooldown active ({wait_left:.2f}s left)")
                elif not args.dry_run:
                    pyautogui.click(click_x, click_y)
                    last_click_ts = time.time()
                    print(f"[{now()}] Clicked {tpl_name}")
                    if use_sequence_mode:
                        sequence_index = (sequence_index + 1) % len(templates)
                        if sequence_index == 0:
                            print(f"[{now()}] Loop completed! Waiting {args.loop_delay:.1f}s before restarting...")
                            time.sleep(args.loop_delay)
                        next_name = templates[sequence_index][0]
                        print(f"[{now()}] Next template: {next_name}")

                matched = True
                if args.once:
                    return
                break

        if not matched:
            if use_sequence_mode:
                current_name = templates[sequence_index][0]
                print(f"[{now()}] Waiting for {current_name} (best={best_name}:{best_score:.3f})")
            else:
                print(f"[{now()}] No match (best={best_name}:{best_score:.3f})")
            if args.once:
                break

        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except pyautogui.FailSafeException:
        print("Failsafe triggered. Exiting.")
    except KeyboardInterrupt:
        print("Interrupted by user.")
