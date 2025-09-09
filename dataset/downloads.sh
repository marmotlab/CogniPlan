#!/usr/bin/env bash
set -euo pipefail

OWNER="marmotlab"
REPO="CogniPlan"
TAG="paper_model_exploration"

# name|type|asset_filename|target_dir|sha256
ASSETS_ALL=(
  "checkpoints_cogniplan_nav_pred7|tgz|checkpoints_cogniplan_nav_pred7.tar.gz|checkpoints/cogniplan_nav_pred7|ad34986d27d66ff12368a70fb719d8659ad49dc0846ec35635d14c955cc9c156"
  "checkpoints_wgan_inpainting|tgz|checkpoints_wgan_inpainting.tar.gz|checkpoints/wgan_inpainting|64f3d64cd2b524e756fe0c80fae6fe7b253ec3e26c4ad0abe3b1495ff5112e69"
  "maps_train_nav|tgz|maps_train_nav.tar.gz|dataset/maps_train_nav|d3e6ad7527d38c907ad02fb0c5daccfbd67419516a8d21d424cd2068eaf07aa7"
  "maps_eval_nav|tgz|maps_eval_nav.tar.gz|dataset/maps_eval_nav|ef194667e4a74b83a8060cd3a76c5c058f345fe19fc2e0dcbc523280d37af2fd"
)

have() { command -v "$1" >/dev/null 2>&1; }

download() {
  local url="$1" out="$2"
  echo "→ Downloading: $url"
  if have aria2c; then
    aria2c -x 16 -s 16 -k 1M -c -o "$out" "$url"
  elif have wget; then
    wget -c -O "$out" "$url"
  elif have curl; then
    curl -L --retry 5 --retry-delay 2 -C - -o "$out" "$url"
  else
    echo "Please install aria2c / wget / curl" >&2; return 1
  fi
}

sha_check() {
  local file="$1" expect="$2"
  if [[ "$expect" == "SKIP" || -z "$expect" ]]; then
    echo "⚠️  No SHA256 configured, skipping check: $file"
    return 0
  fi
  if have sha256sum; then
    echo "${expect}  ${file}" | sha256sum -c -
  elif have shasum; then
    echo "${expect}  ${file}" | shasum -a 256 -c -
  else
    echo "⚠️  No sha256 tool found, skipping check: $file"
    return 0
  fi
}

extract_tgz_into() {
  local tgz="$1" dest="$2"
  mkdir -p "$dest"
  local tmpdir
  tmpdir="$(mktemp -d)"
  tar -xzf "$tgz" -C "$tmpdir"
  local top
  top="$(find "$tmpdir" -mindepth 1 -maxdepth 1 -type d | head -n1 || true)"
  if [[ -z "$top" ]]; then
    echo "Unexpected archive layout or empty archive: $tgz"
    rm -rf "$tmpdir"
    return 1
  fi
  rsync -a "$top"/ "$dest"/
  rm -rf "$tmpdir"
}

download_one() {
  local entry="$1"
  IFS='|' read -r name typ asset target sha <<<"$entry"
  local url="https://github.com/${OWNER}/${REPO}/releases/download/${TAG}/${asset}"
  local cache="dataset/.cache/${asset}"
  mkdir -p "dataset/.cache"

  if [[ "$typ" != "tgz" ]]; then
    echo "Unsupported type: $typ (expected 'tgz')"; return 1
  fi

  if [[ -d "$target" && -n "$(ls -A "$target" 2>/dev/null || true)" ]]; then
    echo "Already exists (non-empty), skip: $target"
    return 0
  fi

  download "$url" "$cache"
  sha_check "$cache" "$sha"
  extract_tgz_into "$cache" "$target"
  echo "✔ Restored to: $target"
}

for e in "${ASSETS_ALL[@]}"; do
  download_one "$e"
done

echo "✅ Done (all assets restored)"