#!/usr/bin/env bash
# Headless Google Drive mount for a CLI-created Colab session.
#
# `colab drivemount` prints a Google consent URL, then waits for Enter on stdin
# after the user has granted access in their browser (there is no code to paste).
# An agent's shell cannot press Enter on a waiting process, so this keeps the
# command alive on a fifo and lets a second call send the Enter.
#
# The kernel-side drive.mount() gives up about 120 s after `start`, so run
# `start` only when the user is ready to click right away.
#
#   colab-drivemount.sh start <session> [path]   prints the consent URL
#   colab-drivemount.sh done                     sends Enter, waits for the result
#   colab-drivemount.sh status                   tail of the log
set -euo pipefail

COLAB=${COLAB:-$HOME/colab-venv/bin/colab}
DIR=${COLAB_DRIVE_DIR:-${TMPDIR:-/tmp}/colab-drivemount}
FIFO=$DIR/stdin.fifo
LOG=$DIR/drivemount.log
PIDFILE=$DIR/pid

clean() { sed 's/\x1b\[[0-9;]*m//g' "$LOG" 2>/dev/null; }

case ${1:-} in
start)
  session=${2:?usage: colab-drivemount.sh start <session> [path]}
  path=${3:-/content/drive}
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    kill "$(cat "$PIDFILE")" 2>/dev/null || true
  fi
  mkdir -p "$DIR"
  rm -f "$FIFO" "$LOG"
  mkfifo "$FIFO"
  nohup bash -c 'exec 3<>"$1"; shift; exec "$@" <&3' _ "$FIFO" "$COLAB" drivemount -s "$session" "$path" >"$LOG" 2>&1 &
  echo $! >"$PIDFILE"
  for _ in $(seq 1 90); do
    if clean | grep -qE 'https://accounts.google.com/'; then
      clean | grep -oE 'https://accounts.google.com/[^[:space:]]+' | head -1
      exit 0
    fi
    kill -0 "$(cat "$PIDFILE")" 2>/dev/null || break
    sleep 1
  done
  echo "drivemount ended or stalled before printing a URL:" >&2
  clean | tail -n 20 >&2
  exit 1
  ;;
done)
  [ -p "$FIFO" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null || {
    echo "no drivemount is waiting; run 'start' again" >&2
    exit 1
  }
  printf '\n' >"$FIFO"
  for _ in $(seq 1 120); do
    kill -0 "$(cat "$PIDFILE")" 2>/dev/null || break
    if clean | grep -qiE 'mounted at|mount failed|Error'; then break; fi
    sleep 1
  done
  if clean | grep -qi 'mount failed'; then
    echo "mount failed (usually the 120 s window passed before the user granted access); run 'start' again" >&2
    exit 1
  fi
  clean | grep -v -E 'new version|update|upgrade|silence|https://accounts' | tail -n 5
  ;;
status)
  clean | tail -n 20
  ;;
*)
  sed -n '2,15p' "$0"
  exit 2
  ;;
esac
