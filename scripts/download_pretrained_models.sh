#!/usr/bin/env bash
set -euo pipefail

#── Colors & Paths ──────────────────────────────────────────────────────────────
RED='\033[0;31m'    # red for warnings/errors
NC='\033[0m'        # no color
TARGET_DIR="pretrained_models_anon_xv"
MHUBERT_DIR="${TARGET_DIR}/mhubert"
ARCHIVE="${TARGET_DIR}.tar.gz"
ZENODO_URL="https://zenodo.org/record/6529898/files/${ARCHIVE}"

# Raw URLs on HuggingFace for Utter-project mHuBERT-147
HF_BASE="https://huggingface.co/utter-project/mHuBERT-147/resolve/main"
CKPT_URL="${HF_BASE}/checkpoint_best.pt"
INDEX_URL="${HF_BASE}/mhubert147_faiss.index"
#───────────────────────────────────────────────────────────────────────────────

# If the top‐level models folder doesn't exist, download & unpack it
if [[ ! -d "$TARGET_DIR" ]]; then
  # Clean up any stale archive
  if [[ -f "$ARCHIVE" ]]; then
    echo -e "${RED}Removing old archive ${ARCHIVE}${NC}"
    rm "$ARCHIVE"
  fi

  echo -e "${RED}Downloading pretrained anonymization models…${NC}"
  wget -O "$ARCHIVE" "$ZENODO_URL"

  echo -e "${RED}Extracting ${ARCHIVE}${NC}"
  tar -xzvf "$ARCHIVE"
else
  echo -e "${RED}${TARGET_DIR} already exists—skipping archive download.${NC}"
fi

# Make sure our mHuBERT folder exists
mkdir -p "$MHUBERT_DIR"

# In that folder, grab the new checkpoint + FAISS index
pushd "$MHUBERT_DIR" >/dev/null
  echo -e "${RED}Downloading mHuBERT checkpoint…${NC}"
  wget -nc "$CKPT_URL"

  echo -e "${RED}Downloading mHuBERT FAISS index…${NC}"
  wget -nc "$INDEX_URL"
popd >/dev/null

echo -e "${RED}Done. Models now in ${MHUBERT_DIR}${NC}"