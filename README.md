# LDPlayer Image Matching Bot (Python)

This sample bot finds target images on the screen and clicks the center of the best matching button.
You can provide multiple templates in priority order for replay loops.

## 1) Install

```powershell
pip install -r requirements.txt
```

## 2) Prepare template image

1. Open your game in LDPlayer.
2. Take a screenshot of the button/icon you want to click.
3. Crop the screenshot so it contains only that button/icon.
4. Save as one or more files, for example:
	- replay.png
	- start.png
	- confirm.png

Tips:
- Use a sharp crop with minimal background.
- Keep LDPlayer window size and resolution fixed for more stable matching.

## 3) Run bot

Single template:

```powershell
python bot_image_match.py --template replay.png --threshold 0.9 --interval 0.7
```

Multi-template replay loop (recommended for CookieRun Classic):

```powershell
python bot_image_match.py --templates replay.png confirm.png start.png --threshold 0.9 --interval 0.6 --cooldown 1.2 --grayscale
```

Numbered templates (1 to 8):

```powershell
python bot_image_match.py --numbered 1 8 --template-dir . --template-ext .png --threshold 0.9 --interval 0.6 --cooldown 1.2 --grayscale
```

Useful options:

- `--region X Y W H` scan only a specific area for better speed and fewer false matches
- `--grayscale` faster matching in many cases
- `--templates A B C` check multiple images in order and click first matched item
- `--numbered START END` auto-load files like START..END (for example 1.png..8.png)
- `--template-dir` directory used by `--numbered`
- `--template-ext` extension used by `--numbered` (default `.png`)
- `--cooldown` minimum delay between clicks to prevent rapid double-clicks
- `--once` run one scan and exit
- `--dry-run` detect only, no clicking

Example with region:

```powershell
python bot_image_match.py --templates replay.png confirm.png start.png --threshold 0.9 --region 200 120 1000 700
```

Priority note:
- Put the most important button first in `--templates`.
- Example order for replay farming: replay -> confirm -> start.

## Stop and safety

- Press `q` in the terminal to stop.
- Move mouse to top-left corner to trigger PyAutoGUI failsafe.

## Notes

- Check game terms of service before automation.
- Start with `--dry-run` to confirm the detection is correct.
