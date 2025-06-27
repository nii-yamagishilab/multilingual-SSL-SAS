# Mitigating Language Mismatch in SSL-Based Speaker Anonymization
This repository contains training codes of the Interspeech 2025 paper
[Mitigating Language Mismatch in SSL-Based Speaker Anonymization]()
by Zhe Zhang, Wen-Chin Huang, Xin Wang, Xiaoxiao Miao, and Junichi Yamagishi

**Audio demos:** https://nii-yamagishilab.github.io/multilingual-SSL-SAS-samples/

---

## Dependencies

### Clone & Setup

```bash
git clone https://github.com/nii-yamagishilab/SSL-SAS.git
cd SSL-SAS
bash scripts/install.sh
```

Ensure you have `sox` and GNU `parallel` installed. If not, run:

```bash
source env.sh
conda install -c conda-forge sox
conda install -c conda-forge parallel
```

### Installation of FairSeq with mHuBERT support

```bash
# Clone the utter-project fork of fairseq
git clone https://github.com/utter-project/fairseq.git
cd fairseq

# Pin to the desired commit
git checkout 81b8b7cd1005470be0e3474ccbda0a091a268b54

# Install in editable mode without pulling other dependencies
pip install -e . --no-deps

# Return to project root
cd ..
```

### Installation of FAISS

```bash
# Install GPU-enabled FAISS without extra dependencies
pip install faiss-gpu --no-deps
```

### Download Pre-trained Models

To download and set up the pre-trained speaker vector and mHuBERT models, run:

```bash
bash scripts/download_pretrained_models.sh
```

This script performs the following steps:

1. Downloads an archive (`pretrained_models_anon_xv.tar.gz`) from [a Zenodo repo containing the anonymization models](https://zenodo.org/records/6529898).
2. Extracts the archive into `pretrained_models_anon_xv/`.
3. Within that directory, creates `pretrained_models_anon_xv/mhubert/` and downloads:

   * `checkpoint_best.pt`: The best checkpoint of the mHuBERT-147 model.
   * `mhubert147_faiss.index`: The FAISS index for the mHuBERT clusters.

Models will be placed under `pretrained_models_anon_xv/mhubert/`.

---

## Data Preparation

We offer an example of preparing the training data using [LibriSpeech train-clean-100](https://www.openslr.org/12):
1. Download the original dataset with `bash data/dataset_download_example.sh`.
2. Convert audio format and normalize the loudness with `bash data/downsample_and_norm_audio.sh`.
3. Generate an audio list using `bash data/generate_full_audio_list.sh`.
4. Repeat above steps to create the file list for dev/valid/test datasets.
5. Alternatively, modify and use `split_all_to_train_dev.py` to create a train/dev split.

---

## Training

### 1. Fine-tune Soft mHuBERT

Edit hyperparameters at the top of `soft_mhubert/train_soft_mhubert.py`, then:

```bash
python soft_mhubert/train_soft_mhubert.py
```

### 2. Train HiFi-GAN Vocoder

Prepare a config JSON (see `config/hifigan_*_mhubert_*.json`), then:

```bash
python adapted_from_facebookresearch/train.py \
  --checkpoint_path [CHECKPOINT_DIR] \
  --config [CONFIG_FILE]
```

---

## Inference

Given a test file list (one path per line), run:

```bash
python adapted_from_facebookresearch/inference.py \
  --input_test_file [LIST_FILE] \
  --checkpoint_file [CHECKPOINT_PATH] \
  --output_dir [OUTPUT_DIR]
```
This will resynthesis the input audio with original speaker vector, using the trained HiFi-GAN model and the corresponding mHuBERT model.

We do not release our Japanese speaker pool due to copyright constraints.
For anonymization usage, we refer the readers to the following resources:
- https://github.com/nii-yamagishilab/SSL-SAS?tab=readme-ov-file#english-anonymization
- https://github.com/nii-yamagishilab/SSL-SAS?tab=readme-ov-file#instructions-for-aonymization-for-your-own-dataset

---

### Troubleshooting

If you encounter

```
File "~/venv/lib/python3.8/site-packages/speechbrain/utils/profiling.py", line 11, in <module>
    from torch.autograd.profiler_util import (  # pytorch v1.10.1
ModuleNotFoundError: No module named 'torch.autograd.profiler_util'
```
or/and
```
File "~/venv/lib/python3.8/site-packages/speechbrain/utils/profiling.py", line 527, in <module>
    a: EventList, b: EventList, filter_by: str = "count",
NameError: name 'EventList' is not defined
```

Open `*/speechbrain/utils/profiling.py` and comment out or remove the offending function(s) as needed.

Ref: https://github.com/nii-yamagishilab/SSL-SAS?tab=readme-ov-file#some-potential-questions-you-may-have-and-how-to-solve-them

---

## Acknowledgments
This study is partially supported by JST AIP Acceleration Research (JPMJCR24U3) and by MEXT KAKENHI Grants (24K21324). This study was carried out using the TSUBAME4.0 supercomputer at Institute of Science Tokyo.

---

## License

The `adapted_from_facebookreaserch` subfolder has [Attribution-NonCommercial 4.0 International License](https://github.com/nii-yamagishilab/SSL-SAS/blob/main/adapted_from_facebookresearch/LICENSE). The `adapted_from_speechbrain` subfolder has [Apache License](https://github.com/nii-yamagishilab/SSL-SAS/blob/main/adapted_from_speechbrain/LICENSE). They were created by the [facebookreasearch](https://github.com/facebookresearch/speech-resynthesis/blob/main) and [speechbrain](https://github.com/speechbrain/speechbrain) orgnization, respectively. The `scripts` subfolder has the [MIT license](https://github.com/nii-yamagishilab/SSL-SAS/blob/main/scripts/LICENSE).

Because this source code was adapted from the facebookresearch and speechbrain, the whole project follows  
the [Attribution-NonCommercial 4.0 International License](https://github.com/nii-yamagishilab/SSL-SAS/blob/main/adapted_from_facebookresearch/LICENSE).

Copyright (c) 2025, Yamagishi Laboratory, National Institute of Informatics.
All rights reserved.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
