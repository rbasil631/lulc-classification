#!/usr/bin/env bash
#
# Remove the old flat layout from the repository.
#
# The restructured tree was pushed via the GitHub API, which can add and update
# files but cannot delete them. So the old paths (final.py, dataset/, ...) are
# still sitting alongside the new src/ layout. This script removes them.
#
# Run from the repository root, on a clean working tree.
#
#   bash tools/cleanup_old_repo.sh          # remove old paths, keep history
#   bash tools/cleanup_old_repo.sh --purge  # also strip dataset/ from history
#
# --purge rewrites history and requires a force-push. Read the warning below.

set -euo pipefail

PURGE=false
[[ "${1:-}" == "--purge" ]] && PURGE=true

if [[ ! -d .git ]]; then
  echo "ERROR: run this from the repository root." >&2
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: working tree is dirty. Commit or stash first." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 1. Back up the reference masks.
#
# dataset/images/ is regenerable (and broken anyway - single-band uint8).
# dataset/reference_masks/ is hand-annotated and CANNOT be regenerated.
# ---------------------------------------------------------------------------
BACKUP="../lulc-masks-backup-$(date +%Y%m%d-%H%M%S)"
if [[ -d dataset/reference_masks ]]; then
  echo "==> Backing up reference masks to $BACKUP"
  mkdir -p "$BACKUP"
  cp -r dataset/reference_masks "$BACKUP/"
  echo "    $(find "$BACKUP" -name '*.tif' | wc -l) mask files backed up"
else
  echo "==> No dataset/reference_masks found; skipping backup"
fi

# ---------------------------------------------------------------------------
# 2. Remove superseded paths from the working tree and index.
# ---------------------------------------------------------------------------
echo "==> Removing old layout"
for path in final.py finalist.py inference.py patch.py \
            corrected_xai.py __pycache__ dataset \
            lulc_results lulc_results12 lulc_results14 lulc_results16 \
            lulc_results21 inference_results_test2 presentable \
            final.ipynb lulc_torchgeo_pipeline.ipynb; do
  if git ls-files --error-unmatch "$path" >/dev/null 2>&1 || [[ -e "$path" ]]; then
    git rm -r --cached --quiet --ignore-unmatch "$path" || true
    rm -rf "$path"
    echo "    removed $path"
  fi
done

git commit -qm "Remove superseded flat layout; superseded by src/lulc package" || {
  echo "    nothing to commit"
}

echo "==> Done. Push with: git push origin main"

# ---------------------------------------------------------------------------
# 3. Optional: strip dataset/ from history.
# ---------------------------------------------------------------------------
if [[ "$PURGE" == true ]]; then
  cat <<'WARN'

============================================================================
WARNING - history rewrite

Deleting dataset/ in step 2 removes it from the current commit only. Every
blob is still in the pack file, so clones stay ~250 MB.

Purging rewrites every commit SHA. That means:
  - you must force-push
  - anyone else who cloned this repo must re-clone
  - open PRs against the old history will break
  - the GitHub-side copy of the dataset is gone for good

Your masks are backed up above. Make sure that backup is somewhere safe
before continuing.
============================================================================

WARN
  read -rp "Type PURGE to continue: " confirm
  [[ "$confirm" == "PURGE" ]] || { echo "Aborted."; exit 1; }

  if ! command -v git-filter-repo >/dev/null 2>&1; then
    echo "ERROR: git-filter-repo not found. Install it first:" >&2
    echo "  pip install git-filter-repo" >&2
    exit 1
  fi

  git filter-repo --path dataset/ --invert-paths --force

  echo
  echo "==> History rewritten. The 'origin' remote was dropped by filter-repo."
  echo "    Re-add it and force-push:"
  echo
  echo "      git remote add origin https://github.com/rbasil631/lulc-classification.git"
  echo "      git push --force origin main"
  echo
  echo "    Then verify size:  git count-objects -vH"
fi
