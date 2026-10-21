---
name: colab
description: Use Google Colab runtimes (CPU, GPU, TPU) from a Claude Code cloud container through the official google-colab-cli - install it, sign in without a terminal the user can type into, run Python on the VM, move files, and shut the VM down so it stops billing. Use this whenever the user asks to connect to / 连 Colab, wants a GPU or more memory than the container has, wants to train or run an experiment "on Colab", mentions colab new / colab exec / compute units, or asks to move data between the container, Colab and Google Drive - even if they only say "run it on a GPU".
---

# Colab from a Claude Code cloud container

The `colab` CLI (google-colab-cli) rents a Colab VM and runs a Jupyter kernel on it; you drive it with one-shot commands. Its own guide is in `references/colab-cli-skill.md` (command list, `colab run`, notebooks, recovery) — read it when you need a command not covered here. This file covers what that guide gets wrong or leaves out for this environment: installing in the container, signing in when nobody can type at your terminal, and not leaving VMs running.

## 1. Install

```bash
python3 -m venv ~/colab-venv && ~/colab-venv/bin/pip install google-colab-cli==0.7.4
```

- The PyPI package named `colab` is something else; install `google-colab-cli`. It needs Python 3.12+ and Linux or macOS.
- PyPI is reachable from the container without extra network setup. The venv keeps it out of the project's dependencies.
- Call it as `~/colab-venv/bin/colab` (the examples below write `colab`). Global flags go before the subcommand.

## 2. Sign in (once per container)

The container has no Google credentials, so the first `colab new` prints a consent URL and then waits for an authorization code on stdin. Your shell cannot type into a waiting process, and the code only works for the process that printed the URL, so use the bundled script, which keeps that process alive on a fifo:

```bash
.claude/skills/colab/scripts/colab-login.sh start colab            # CPU; add --gpu T4 etc. as for colab new
# prints READY (a saved token worked) or the consent URL
.claude/skills/colab/scripts/colab-login.sh code '<code the user pastes>'
```

When you get a URL, give it to the user and tell them:

- which Google account to pick: the one that owns their Colab and Drive files. People often have several and pick the wrong one;
- that the page then shows a command like `gcloud auth application-default login --no-launch-browser` followed by a code starting with `4/`. Only the code matters; they paste it back to you;
- that the code expires within minutes. If it fails, run `start` again for a fresh URL.

What the sign-in grants: Colab, Google Cloud (`cloud-platform`) and Drive access limited to files the app creates (`drive.file`). The CLI saves a long-lived refresh token in `~/.config/colab-cli/token.json`. Run `chmod 600` on it after sign-in, since it is created world-readable. Never print or read that file or `~/.config/colab-cli/colab.log` (which may also hold credentials). Tell the user the token stays in this container until the container is recycled, and that they can revoke it any time at https://myaccount.google.com/permissions; revoking it also signs out any other session that uses the same grant.

## 3. Check what is running and what it costs

```bash
colab sessions     # every runtime on the account; [?] = one this container did not create
colab usage        # compute-unit balance and the current hourly rate
```

Every runtime bills, including `[?]` ones. Those usually belong to another Claude session or to the user's browser, so ask before you stop one you did not create.

The user can also stop runtimes from the Colab web UI, yours included. When a command reports `Session '<name>' appears to be lost (404/401)`, run `colab sessions`. If the runtime is gone, ask before you start a new one rather than recreating it silently; they may have stopped it on purpose.

## 4. Run code

`colab exec` takes Python on stdin or from a local file. It does not accept code as an argument (`colab exec -s colab "print(1)"` fails with "unexpected extra arguments").

```bash
printf '%s\n' "import os" "print(os.cpu_count())" | colab exec -s colab
colab exec -s colab -f job.py --timeout 3600
```

- Don't count on `--timeout` (default 30 s) to cut a cell short. In 0.7.4 a 15 s cell run with `--timeout 3`, and a 40 s cell run with the default, both ran to the end and printed their output. Long cells simply hold the call open, so put your own Bash tool timeout above the cell's expected run time.
- The kernel keeps state between `exec` calls (imports, variables), and the working directory is `/content`. The VM has `git` and `node` (`/tools/node/bin/node`).
- A CPU runtime has no `nvidia-smi`. Check for a GPU with `shutil.which('nvidia-smi')` instead of calling it, or the cell raises.
- For a job longer than you want to hold a call open, start it from Python with `subprocess.Popen(..., stdout=open('/content/job.log', 'w'))`, return immediately, and poll the log with short `exec` calls.
- For a single script on a fresh VM that tears itself down afterwards, `colab run [--gpu T4] script.py args` is simpler (see the reference).

## 5. Move files

- One file: `colab upload <local> /content/<path>` and `colab download /content/<path> <local>`.
- A repository: clone it on the VM (a public GitHub repo needs no credentials), e.g. `subprocess.run(['git', 'clone', '--depth', '1', '<url>', '/content/<dir>'], check=True)`. That is faster than uploading files one at a time.
- Google Drive: `colab drivemount` prints a Google consent URL and then waits for Enter (there is no code; the user only clicks Allow). Use the bundled script, which keeps it alive on a fifo:

  ```bash
  .claude/skills/colab/scripts/colab-drivemount.sh start <session>   # prints the consent URL
  .claude/skills/colab/scripts/colab-drivemount.sh done              # after the user says they granted access
  ```

  - The kernel-side `drive.mount()` times out about **120 s after `start`**. A grant that came later failed with `ValueError: mount failed` (drive-timeout). So run `start` only when the user is ready, hand over the URL at once, and tell them to click Allow within two minutes and reply. If it fails, run `start` again for a fresh URL.
  - The URL pre-fills the account (`login_hint`); tell the user to keep that account.
  - Check the result with `os.path.isdir('/content/drive/MyDrive')`.
  - Don't run `drive.mount()` through `colab exec`: the auth request is not surfaced, and the cell blocks the kernel until its timeout.
  - Don't send the user to the `colab url` link to mount Drive there. In practice it opened a **new** runtime instead of attaching to the CLI's VM, so Drive was mounted on the wrong machine, and that extra runtime kept billing.
  - For a few small files, the Google Drive connector (if this session has it) is an alternative; it only takes content written out inline.

### Gotchas seen in practice

- `colab url -s <name>` can open a separate runtime in the browser. It then shows up in `colab sessions` as `[?]`, and `colab stop` cannot stop it ("Session not found"). Only the user can close it, from the Colab web UI: Runtime → Manage sessions → Terminate (运行时 → 管理会话 → 终止).
- Python scripts that start child processes with `spawn` (vLLM's engine, multiprocessing) must not be launched as `python script.py` when they lack an `if __name__ == "__main__":` guard: each child re-executes the whole script and fails with "An attempt has been made to start a new process before the current process has finished its bootstrapping phase". Launch with `python -c "p='/content/x.py'; exec(compile(open(p).read(), p, 'exec'))"`, and avoid initializing CUDA in the parent before the engine starts.
- `colab download` reads files without going through the kernel, so it still works while a long cell is blocking the kernel. Use it to read logs in that situation.
- Each new runtime is a fresh machine: nothing installed on one VM carries over to the next, and Drive has to be re-granted every time. Instead of caching packages, install with uv: `python -m pip install -q uv && python -m uv pip install --python "$(which python)" --target /content/pkgs ...`. A cold install of vLLM + transformers + TRL (8 GB) took 106 s with uv against about 6 min with pip, so a cache is not worth the Drive round trip. Put `/content/pkgs` first on `sys.path` and in `PYTHONPATH` so spawned children see it.
- A background job started with plain `subprocess.Popen` from the kernel can die with the kernel (for example after `colab restart-kernel`). Start long jobs with `setsid` / `start_new_session=True` and log to a file you read with `colab download`.
- Mounting Drive is per runtime and can't be made permanent: each new VM needs a fresh grant within the 120 s window.

## 6. Finish

Stop every runtime you created when the work is done, or when you hand back to the user and nothing is running: `colab stop -s <name>`. An idle runtime keeps billing until Colab reclaims it. Then report what you stopped, anything still running (yours or `[?]`), and the balance from `colab usage`.
