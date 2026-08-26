import argparse
import datetime as dt
import random
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
        nargs="+",
        type=float,
        default=[0.5, 0.9],
        help="Delay (seconds) between scans. Single value or MIN MAX range. Default: 0.5 0.9",
    )
    parser.add_argument(
        "--cooldown",
        nargs="+",
        type=float,
        default=[1.0, 1.6],
        help="Minimum delay (seconds) between clicks. Single value or MIN MAX range. Default: 1.0 1.6",
    )
    parser.add_argument(
        "--loop-delay",
        nargs="+",
        type=float,
        default=[2.0, 6.0],
        help="Delay (seconds) after completing a full loop. Single value or MIN MAX range. Default: 2.0 6.0",
    )
    parser.add_argument(
        "--fallback-timeout",
        nargs="+",
        type=float,
        default=[1.8, 2.5],
        help="Time (seconds) to wait before falling back. Single value or MIN MAX range. Default: 1.8 2.5",
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


def get_random_value(values) -> float:
    """Return a random float between min and max if a list is provided, otherwise return float value."""
    if isinstance(values, (int, float)):
        return float(values)
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    return random.uniform(min(values), max(values))


def compute_click_point(top_left, template_shape, region):
    tpl_h, tpl_w = template_shape[:2]
    # Add slight random coordinate jitter (±2px) to prevent clicking the exact same pixel
    jitter_x = random.randint(-2, 2)
    jitter_y = random.randint(-2, 2)
    x = top_left[0] + tpl_w // 2 + jitter_x
    y = top_left[1] + tpl_h // 2 + jitter_y
    if region:
        x += region[0]
        y += region[1]
    return x, y


def do_click(x: int, y: int) -> None:
    """Move cursor to (x, y) and click immediately without delay."""
    try:
        import ctypes
        ctypes.windll.user32.SetCursorPos(int(x), int(y))
        ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTDOWN
        ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTUP
    except Exception:
        pyautogui.click(x, y)


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
    is_fallback = False
    search_start_ts = time.time()
    current_cooldown = get_random_value(args.cooldown)
    current_fallback_timeout = get_random_value(args.fallback_timeout)

    while True:
        stop_requested, reset_requested = handle_keyboard_input(templates)
        if stop_requested:
            print(f"[{now()}] Stop requested (q pressed).")
            break
        if reset_requested:
            sequence_index = 0
            is_fallback = False
            search_start_ts = time.time()
            current_fallback_timeout = get_random_value(args.fallback_timeout)

        frame = screenshot_to_cv(args.region, args.grayscale)

        best_name = ""
        best_score = -1.0
        matched = False

        if use_sequence_mode:
            check_index = (sequence_index - 1) % len(templates) if is_fallback else sequence_index
            templates_to_check = [templates[check_index]]
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
                if seconds_since_last_click < current_cooldown:
                    wait_left = current_cooldown - seconds_since_last_click
                    print(f"[{now()}] Cooldown active ({wait_left:.2f}s left)")
                elif not args.dry_run:
                    do_click(click_x, click_y)
                    last_click_ts = time.time()
                    current_cooldown = get_random_value(args.cooldown)
                    print(f"[{now()}] Clicked {tpl_name}")
                    if use_sequence_mode:
                        if is_fallback:
                            is_fallback = False
                            next_name = templates[sequence_index][0]
                            print(f"[{now()}] Fallback click succeeded! Returning to target template: {next_name}")
                        else:
                            sequence_index = (sequence_index + 1) % len(templates)
                            if sequence_index == 0:
                                delay = get_random_value(args.loop_delay)
                                print(f"[{now()}] Loop completed! Waiting {delay:.2f}s before restarting...")
                                time.sleep(delay)
                            next_name = templates[sequence_index][0]
                            print(f"[{now()}] Next template: {next_name}")
                        search_start_ts = time.time()
                        current_fallback_timeout = get_random_value(args.fallback_timeout)

                matched = True
                if args.once:
                    return
                break

        if not matched:
            if use_sequence_mode:
                current_name = templates[check_index][0]
                status_str = f"Fallback: searching {current_name}" if is_fallback else f"Waiting for {current_name}"
                print(f"[{now()}] {status_str} (best={best_name}:{best_score:.3f})")

                if time.time() - search_start_ts >= current_fallback_timeout:
                    if not is_fallback:
                        is_fallback = True
                        search_start_ts = time.time()
                        current_fallback_timeout = get_random_value(args.fallback_timeout)
                        prev_index = (sequence_index - 1) % len(templates)
                        prev_name = templates[prev_index][0]
                        print(
                            f"[{now()}] Timeout waiting for {templates[sequence_index][0]} "
                            f"({current_fallback_timeout:.1f}s). Falling back to previous: {prev_name}"
                        )
                    else:
                        is_fallback = False
                        search_start_ts = time.time()
                        current_fallback_timeout = get_random_value(args.fallback_timeout)
                        target_name = templates[sequence_index][0]
                        print(
                            f"[{now()}] Fallback timeout ({current_fallback_timeout:.1f}s). "
                            f"Returning to main target: {target_name}"
                        )
            else:
                print(f"[{now()}] No match (best={best_name}:{best_score:.3f})")
            if args.once:
                break

        time.sleep(get_random_value(args.interval))


if __name__ == "__main__":
    try:
        main()
    except pyautogui.FailSafeException:
        print("Failsafe triggered. Exiting.")
    except KeyboardInterrupt:
        print("Interrupted by user.")
