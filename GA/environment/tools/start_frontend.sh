#!/bin/sh

GIT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)

if [ -z "$GIT_ROOT" ]; then
  echo "Not inside a Git repository."
  exit 1
fi

echo "Git root is: $GIT_ROOT"

TARGET_DIR="$GIT_ROOT/GA/environment/frontend_server"

if [ ! -d "$TARGET_DIR" ]; then
  echo "Target directory does not exist: $TARGET_DIR"
  exit 1
fi

cd "$TARGET_DIR"
echo "Starting web server..."

exec python3 manage.py runserver