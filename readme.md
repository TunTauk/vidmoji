# Vidmoji

## About

A real time hand gesture detection and emoji translation using `mediapipe`, `pillow` and `opencv`. Currently, the following emojis are supported for both left and right hand.

- closed fist ✊
- I love you 🤟
- open palm ✋
- pointing up ☝️
- victory ✌️

## Installation

I recommend you to create a new environment before installation. Current `mediapipe` only support python version 3.10.

```sh
conda create -n vidmoji python=3.10
```

```sh
conda activate vidmoji
```

```sh
conda install opencv pillow
pip install mediapipe
```

## Configuration

Currently, I don't support configuration over command line options. You can edit in the source code directly in `vidmoji.py`.

```python
# Config
DETECTION_TIME = 0.3  # seconds to wait before showing emoji
NUM_HANDS = 2  # number of hands to detect
EMOJI_DURATION = 2.0  # seconds to show
EMOJI_SIZE = 50 # size of emoji
EMOJI_PATH = "./emojis/" # path to emojis folder
```

## Usage

To run the application, just run the `vidmoji.py` and press `q` to quit.

```sh
python ./vidmoji.py`
```

## Possible application

In online casual video meeting, users can raise hand, send emojis to use gestures instead of pressing buttons.

## Next step

For the next step, I would like to implement emotion detection and emotions to emoji. This one is in progress. You can check in other branch.

## Credits

Credits for [Icon Pack](https://www.flaticon.com/packs/hand-gestures-17748111)
