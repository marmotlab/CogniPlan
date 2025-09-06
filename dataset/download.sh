#!/usr/bin/env bash
set -euo pipefail

OWNER="marmotlab"
REPO="CogniPlan"
TAG="paper_model_exploration"

# name|type|asset_filename|target_dir|sha256
ASSETS_REQUIRED=(
  # checkpoints (two packed folders)
  "checkpoints_cogniplan_exp_pred7|tgz|checkpoints_cogniplan_exp_pred7.tar.gz|checkpoints/cogniplan_exp_pred7|2d6023a414b02adf16351d5b4e8b2ac3d08d8e8a59b2b2df9a3ed3d703429596"
  "checkpoints_wgan_inpainting|tgz|checkpoints_wgan_inpainting.tar.gz|checkpoints/wgan_inpainting|64f3d64cd2b524e756fe0c80fae6fe7b253ec3e26c4ad0abe3b1495ff5112e69"
  # datasets (required)
  "maps_train|tgz|maps_train.tar.gz|dataset/maps_train|28b7227320cfd7794bfe29d9be75a022e39f33ba98283aaac2f877fc5c1b49c5"
  "maps_eval|tgz|maps_eval.tar.gz|dataset/maps_eval|b5f5bfb8fe6ec3aedd857b518c162eb63d4ecc12917af643cb95f3cde580f7e5"
)

ASSETS_OPTIONAL=(
  "maps_train_inpaint|tgz|maps_train_inpaint.tar.gz|dataset/maps_train_inpaint|cd4d05dbcce60c1412cca2af4de6d910754f8b8e8704e3d34f998e2f44045589"
  "maps_eval_inpaint|tgz|maps_eval_inpaint.tar.gz|dataset/maps_eval_inpaint|db611f68152a9a8a05d2a119ae259635da906605ae2608ff952e9413c3be998c"
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
  # merge-safe
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

  if [[ "$target" == checkpoints/* ]]; then
    # If the checkpoint folder already exists and is non-empty, skip
    if [[ -d "$target" && -n "$(ls -A "$target" 2>/dev/null || true)" ]]; then
      echo "Already exists (non-empty), skip: $target"
      return 0
    fi
  else
    # Dataset folders
    if [[ -d "$target" && -n "$(ls -A "$target" 2>/dev/null || true)" ]]; then
      echo "Already exists (non-empty), skip: $target"
      return 0
    fi
  fi

  download "$url" "$cache"
  sha_check "$cache" "$sha"
  extract_tgz_into "$cache" "$target"
  echo "✔ Restored to: $target"
}

download_group() {
  local -n arr="$1"
  for e in "${arr[@]}"; do
    download_one "$e"
  done
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [required|optional|all]
  required  -> checkpoints + maps_train + maps_eval (maps for planner training)
  optional  -> maps_train_inpaint + maps_eval_inpaint (maps for inpainting model and planner training)
  all       -> everything
Default: required
EOF
}

choice="${1:-required}"
case "$choice" in
  required) download_group ASSETS_REQUIRED ;;
  optional) download_group ASSETS_OPTIONAL ;;
  all)      download_group ASSETS_REQUIRED; download_group ASSETS_OPTIONAL ;;
  -h|--help) usage; exit 0 ;;
  *) echo "Unknown option: $choice"; usage; exit 1 ;;
esac

echo "✅ Done ($choice)"
