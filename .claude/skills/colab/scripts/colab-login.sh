#!/usr/bin/env bash
# Headless sign-in for google-colab-cli inside a Claude Code cloud container.
#
# `colab new` asks for an authorization code on stdin the first time. An agent's
# shell has no stdin to type into, so this keeps `colab new` running in the
# background, reading from a fifo, and lets a second call feed it the code.
# The code is bound to that exact process (PKCE), so both steps must use the
# same running `colab new`.
#
#   colab-login.sh start <session> [colab new options...]
#       Starts `colab new -s <session> ...`. Prints the consent URL when sign-in
#       is needed, or "READY" when a saved token already worked.
#   colab-login.sh code <authorization-code>
#       Feeds the code the user pasted back and waits for the session.
#   colab-login.sh status
#       Prints the tail of the log.
set -euo pipefail

COLAB=${COLAB:-$HOME/colab-venv/bin/colab}
DIR=${COLAB_LOGIN_DIR:-${TMPDIR:-/tmp}/colab-login}
FIFO=$DIR/stdin.fifo
LOG=$DIR/new.log
PIDFILE=$DIR/pid

wait_for() { # pattern seconds
  local i
  for ((i = 0; i < $2; i++)); do
    grep -qE "$1" "$LOG" 2>/dev/null && return 0
    kill -0 "$(cat "$PIDFILE")" 2>/dev/null || return 1
    sleep 1
  done
  return 1
}

case ${1:-} in
start)
  session=${2:?usage: colab-login.sh start <session> [colab new options...]}
  shift 2
  [ -x "$COLAB" ] || { echo "colab CLI not found at $COLAB (see SKILL.md, Install)" >&2; exit 1; }
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "a colab new is already waiting (pid $(cat "$PIDFILE")); use 'code' or kill it first" >&2
    exit 1
  fi
  mkdir -p "$DIR"
  rm -f "$FIFO" "$LOG"
  mkfifo "$FIFO"
  # Holding the fifo open read-write keeps stdin from hitting EOF before the code arrives.
  nohup bash -c 'exec 3<>"$1"; shift; exec "$@" <&3' _ "$FIFO" "$COLAB" new -s "$session" "$@" >"$LOG" 2>&1 &
  echo $! >"$PIDFILE"
  if wait_for 'Session READY|https://accounts.google.com/' 60; then
    if grep -q 'Session READY' "$LOG"; then
      echo READY
    else
      grep -oE 'https://accounts.google.com/[^[:space:]]+' "$LOG" | head -1
    fi
  else
    echo "colab new ended or stalled before asking for sign-in:" >&2
    tail -n 20 "$LOG" >&2
    exit 1
  fi
  ;;
code)
  code=${2:?usage: colab-login.sh code <authorization-code>}
  [ -p "$FIFO" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null || {
    echo "no colab new is waiting; run 'start' again (codes only work for the process that made the URL)" >&2
    exit 1
  }
  printf '%s\n' "$code" >"$FIFO"
  if wait_for 'Session READY' 180; then
    echo READY
  else
    echo "session did not come up:" >&2
    tail -n 20 "$LOG" >&2
    exit 1
  fi
  ;;
status)
  tail -n 20 "$LOG"
  ;;
*)
  sed -n '2,17p' "$0"
  exit 2
  ;;
esac
