#!/bin/bash
# Run this ON YOUR MAC (not on the cluster).
# Pushes code + data to the STORAGE node (storge, 10.26.1.74) — NOT login01,
# NOT a GPU node. Files land on shared NFS (/share) and are then visible to
# every GPU node automatically.
#
# Usage:
#   bash transfer_from_mac.sh <your_cluster_username>
set -euo pipefail

CLUSTER_USER="${1:?usage: bash transfer_from_mac.sh <cluster_username>}"
STORGE="10.26.1.74"                                   # storage node (file I/O)
LOCAL_ROOT="/Users/xiaomaiyue/Downloads/slr/tse1"     # your Mac project root
DEST="~/tse1"                                         # -> /share/home/<user>/tse1

echo ">>> creating ~/tse1 on storge"
ssh "${CLUSTER_USER}@${STORGE}" "mkdir -p ~/tse1"

echo ">>> [1/2] code tree (excluding .git / ckpts / caches)"
rsync -avP \
  --exclude='.git' \
  --exclude='real-tse-ckpt' \
  --exclude='*.egg-info' \
  --exclude='__pycache__' \
  --exclude='mini_librimix_tse' \
  --exclude='exp' \
  "${LOCAL_ROOT}/wesep-real-tse" \
  "${CLUSTER_USER}@${STORGE}:${DEST}/"

echo ">>> [2/2] data (~3.1G: Libri2Mix wav16k/min + text_cues; raw LibriSpeech excluded)"
rsync -avP \
  --exclude='LibriSpeech' \
  --exclude='download' \
  --exclude='Libri2Mix/wav8k' \
  --exclude='Libri2Mix/wav16k/min/train_tiny' \
  "${LOCAL_ROOT}/libri_data" \
  "${CLUSTER_USER}@${STORGE}:${DEST}/"

echo ">>> done. Next: ssh ${CLUSTER_USER}@10.26.1.75 (login01) and run setup_cluster.sh"
