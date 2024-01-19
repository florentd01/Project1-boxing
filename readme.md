# Boxing pose estimation project




## Splitter

To split videos to 2 use `splitter.py`, usage example: `python ./splitter.py ./Boxing_data`, where the only argument is the path to dir where all experiments are saved (each in its own directory).

This script will make `side` and `front` dirs and save left piece of the video to `side`, right - to `front`.

It also will make new videos with reduced sizes by 1.5, 2, 4 and save them in the same dirs but with prefixes `1.5_`/`2_`/`4_` respectively.