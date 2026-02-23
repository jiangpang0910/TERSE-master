#!/bin/bash
# Transfer experiments_logs from server to local
#
# STEP 1 — On the server (ssh pc), run:
#   cd ~/jp/terse-master/TERSE-master
#   tar -czvf experiments_logs.tar.gz experiments_logs
#   exit
#
# STEP 2 — From your local Mac, run this script:
#   ./sync_experiments.sh

set -e
REMOTE="pc"
REMOTE_PATH="$REMOTE:~/jp/terse-master/TERSE-master/experiments_logs.tar.gz"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Pulling experiments_logs.tar.gz from $REMOTE..."
scp "$REMOTE_PATH" "$LOCAL_DIR/"

echo "Extracting..."
tar -xzvf "$LOCAL_DIR/experiments_logs.tar.gz" -C "$LOCAL_DIR/"

echo "Done. Logs are in $LOCAL_DIR/experiments_logs/"
echo "You can remove the archive: rm $LOCAL_DIR/experiments_logs.tar.gz"
