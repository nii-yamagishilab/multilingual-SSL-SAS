#!/usr/bin/env bash
set -euo pipefail

#───────────────────────────────────────────────────────────────────────────────
# Configuration
#───────────────────────────────────────────────────────────────────────────────
URL="https://www.openslr.org/resources/12/train-clean-100.tar.gz"
ARCHIVE="$(basename "$URL")"        # train-clean-100.tar.gz
DEST_DIR="data/LibriSpeech"         # target directory for extraction
STRIP_COMPONENTS=1                  # remove top-level "LibriSpeech/" folder
#───────────────────────────────────────────────────────────────────────────────

# Create the destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Download the archive if it's not already present
if [[ ! -f "$ARCHIVE" ]]; then
    echo "Downloading ${ARCHIVE} from ${URL}..."
    wget -c "$URL" -O "$ARCHIVE"
else
    echo "Archive ${ARCHIVE} already exists — skipping download."
fi

# Extract into DEST_DIR, stripping the leading directory component
echo "Extracting ${ARCHIVE} into ${DEST_DIR}/ (strip ${STRIP_COMPONENTS} components)..."
tar -xzvf "$ARCHIVE" -C "$DEST_DIR" --strip-components="$STRIP_COMPONENTS"

echo "Done. You can find train-clean-100 in ${DEST_DIR}/train-clean-100/"