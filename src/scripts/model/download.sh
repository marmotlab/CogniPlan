#!/usr/bin/env bash
set -euo pipefail

OWNER="marmotlab"
REPO="CogniPlan"
TAG="ros_model_v1"
ASSET="ros_model_assets.tar.gz"
SHA256="ba0988c38e9968ba46f4980f4cf5bbf613dd8a041c76bd4ba5674b93ec96ba21"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_DIR="${SCRIPT_DIR}/.cache"

URL="https://github.com/${OWNER}/${REPO}/releases/download/${TAG}/${ASSET}"
CACHE_FILE="${CACHE_DIR}/${ASSET}"

mkdir -p "${CACHE_DIR}"

have() { command -v "$1" >/dev/null 2>&1; }

download() {
  echo "→ Downloading ${ASSET}"
  if have aria2c; then
    aria2c -x 16 -s 16 -k 1M -c -o "${CACHE_FILE}" "${URL}"
  elif have wget; then
    wget -c -O "${CACHE_FILE}" "${URL}"
  elif have curl; then
    curl -L --retry 5 --retry-delay 2 -C - -o "${CACHE_FILE}" "${URL}"
  else
    echo "Please install aria2c / wget / curl" >&2
    exit 1
  fi
}

sha_check() {
  if [[ -z "${SHA256}" || "${SHA256}" == "SKIP" ]]; then
    echo "⚠️  SHA256 check skipped"
    return
  fi

  if have sha256sum; then
    echo "${SHA256}  ${CACHE_FILE}" | sha256sum -c -
  elif have shasum; then
    echo "${SHA256}  ${CACHE_FILE}" | shasum -a 256 -c -
  else
    echo "⚠️  No sha256 tool found, skipping check"
  fi
}

already_ok() {
  [[ -f "${SCRIPT_DIR}/checkpoint.pth" \
  && -f "${SCRIPT_DIR}/generator.pt" \
  && -f "${SCRIPT_DIR}/config.yaml" ]]
}

if already_ok; then
  echo "✔ Model assets already exist, skip download."
  exit 0
fi

download
sha_check

echo "→ Extracting to ${SCRIPT_DIR}"
tar -xzf "${CACHE_FILE}" -C "${SCRIPT_DIR}"

# sanity check
test -f "${SCRIPT_DIR}/checkpoint.pth"
test -f "${SCRIPT_DIR}/generator.pt"
test -f "${SCRIPT_DIR}/config.yaml"

echo "✅ Model assets ready:"
ls -lh "${SCRIPT_DIR}/checkpoint.pth" "${SCRIPT_DIR}/generator.pt" "${SCRIPT_DIR}/config.yaml"
