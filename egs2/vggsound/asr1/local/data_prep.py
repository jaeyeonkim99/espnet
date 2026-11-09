#!/usr/bin/env bash

# Copyright 2023  Siddhant Arora
#           2023  Carnegie Mellon University
# Apache 2.0

import json
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split

if len(sys.argv) != 2:
    print("Usage: python data_prep.py [vggsound_root]")
    sys.exit(1)
vggsound_root = sys.argv[1]

# Read CSV files from official GitHub repository
train_csv_url = "https://raw.githubusercontent.com/hche11/VGGSound/master/data/train.csv"
test_csv_url = "https://raw.githubusercontent.com/hche11/VGGSound/master/data/test.csv"

# CSV files don't have headers, so we specify column names
train_data = pd.read_csv(train_csv_url, header=None, names=["filename", "target"])
test_data = pd.read_csv(test_csv_url, header=None, names=["filename", "target"])

# Filter the entries that are not present in the audio folder
# This is youtube-based dataset, so the availability of files may vary
# Should I directly fix the file used in valid set rather than fixing random seed and splitting from train set?


train_data = train_data[
    train_data["filename"].apply(lambda x: Path(vggsound_root, x).is_file())
]
test_data = test_data[
    test_data["filename"].apply(lambda x: Path(vggsound_root, x).is_file())
]

print(f"\nAfter filtering:")
print(f"Valid train files: {len(train_data)}")
print(f"Valid test files: {len(test_data)}")

split_df = {}
split_df["test"] = test_data
split_df["train"], split_df["valid"] = train_test_split(
    train_data, test_size=0.2, random_state=1
)
dir_dict = split_df
for x in dir_dict:
    with open(os.path.join("data", x, "wav.scp"), "w") as wav_scp_f, open(
        os.path.join("data", x, "utt2spk"), "w"
    ) as utt2spk_f, open(os.path.join("data", x, "text"), "w") as text_f:
    
        filename = dir_dict[x]["filename"].values.tolist()
        label = dir_dict[x]["target"].values.tolist()
        for line_count in range(len(filename)):
            cls = "audio_class:" + str(label[line_count])
            # Support both .wav and .mp4 files
            utt_id = filename[line_count].replace(".wav", "").replace(".mp4", "")
            spk = utt_id
            data_dir = Path(vggsound_root, filename[line_count])
            
            # If MP4 file, use ffmpeg pipe; otherwise use direct path
            # Using 16kHz sampling rate (standard for audio event classification)
            # Change to 22050 or 44100 if higher quality is needed
            if str(data_dir).endswith(".mp4"):
                wav_scp_f.write(
                    f"{utt_id} ffmpeg -i {data_dir} -ac 1 -ar 16000 -f wav -vn "
                    f"-hide_banner -loglevel error - |\n"
                )
            else:
                wav_scp_f.write(utt_id + " " + str(data_dir) + "\n")
            text_f.write(utt_id + " " + cls + "\n")
            utt2spk_f.write(utt_id + " " + spk + "\n")
