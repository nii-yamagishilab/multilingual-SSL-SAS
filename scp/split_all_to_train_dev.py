import sys, os
from random import shuffle

# Read the full file paths into all_list
all_list = []
with open('librispeech_100/librispeech_100_wav16k_norm_full.lst') as f:
    for line in f:
        # Store the full path (without stripping directory and extension)
        all_list.append(line.strip())

# Shuffle the list
indices = list(range(len(all_list)))
shuffle(indices)
all_list = [all_list[idx] for idx in indices]

# Calculate split indices for training and development sets
num_train = int(len(all_list) * 0.95)
train_names, dev_names = all_list[:num_train], all_list[num_train:]


# Write the full paths to the output files
with open('librispeech_100/librispeech_100_wav16k_norm_train.lst', 'w') as f:
    for item in train_names:
        f.write("%s\n" % item)

with open('librispeech_100/librispeech_100_wav16k_norm_dev.lst', 'w') as f:
    for item in dev_names:
        f.write("%s\n" % item)
