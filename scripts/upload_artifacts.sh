#!/usr/bin/env bash
set -euo pipefail

BUCKET="${BUCKET:-badroohiman-m5-delayed-labels-artifacts}"
PREFIX="${PREFIX:-m5-delayed-labels}"

echo "Uploading to s3://$BUCKET/$PREFIX/"

if [ -d "data/processed" ]; then
  aws s3 sync "data/processed/" "s3://$BUCKET/$PREFIX/data/processed/" --only-show-errors
fi

if [ -d "artifacts/models" ]; then
  aws s3 sync "artifacts/models/" "s3://$BUCKET/$PREFIX/models/" --only-show-errors
elif [ -d "models" ]; then
  aws s3 sync "models/" "s3://$BUCKET/$PREFIX/models/" --only-show-errors
fi

if [ -d "reports" ]; then
  aws s3 sync "reports/" "s3://$BUCKET/$PREFIX/reports/" --only-show-errors
fi

if [ -d "logs" ]; then
  aws s3 sync "logs/" "s3://$BUCKET/$PREFIX/logs/" --only-show-errors
fi

echo "Done."
