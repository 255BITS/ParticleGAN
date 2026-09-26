#!/usr/bin/env bash
# One independent GAN experiment attempt per invocation. Requires GNU timeout.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo="$script_dir/ParticleGAN-selected-h-stability-base"
base=HEAD
runs_dir="$script_dir/gan-attempts"
minutes=90
candidates=16
workers=3
gpu=1
focus='Make selected K3P continuous without a training horizon; preserve 22/22, hold and timely recovery.'
prompt_file=''
dry_run=false
codex_bin=${CODEX_BIN:-codex}
claude_bin=${CLAUDE_BIN:-claude}
grok_bin=${GROK_BIN:-grok}
opencode_bin=${OPENCODE_BIN:-opencode}
engine=${GAN_ENGINE:-codex}
model=''
budget=''
gan_python=${GAN_PYTHON:-/tmp/pr38-default-env/bin/python}

usage() {
    cat <<'EOF'
Usage: ./try-gan.sh [--engine codex|claude|grok|opencode] [--minutes N] [--candidates N]
                    [--workers N] [--model NAME] [--budget-usd AMOUNT]
                    [--focus TEXT] [--prompt-file FILE] [--repo DIR] [--base REF]
                    [--runs-dir DIR] [--gpu INDEX_OR_UUID] [--dry-run]

Launch one fresh GPU attempt from the committed search brief.
Each invocation creates a new branch/worktree from the selected committed base.
Default: ParticleGAN-selected-h-stability-base HEAD, 90 minutes, ./gan-attempts/.
Focused search: at most 16 proposals and 3 parallel benchmark workers.
--gpu selects one physical GPU (default 1); CUDA is required, with no CPU fallback.
Uses the selected commit's h_stability/SEARCH.md when present.
--prompt-file replaces the broad search brief with a focused experiment brief.
--dry-run prints the exact command and prompt without creating files or calling the agent.
--engine codex runs Codex gpt-6-astra/max; --engine claude runs Claude opus[1m]/max.
--engine grok runs Grok grok-4.7 with its default reasoning effort.
--engine opencode runs nano-gpt/meta/muse-spark-1.3-contributor through NanoGPT.
Engines can run concurrently: worktree creation is serialized with a
lock, each attempt owns its branch, run directory and GPU, and no state is shared.
--model overrides the engine default; --budget-usd caps spend (Claude only).
CODEX_BIN/CLAUDE_BIN/GROK_BIN override the CLI; GAN_PYTHON selects the benchmark Python.
OPENCODE_BIN overrides OpenCode. Use its nano-gpt login or NANO_GPT_API_KEY
(NANOGPT_API_KEY is accepted as an alias). Saved login takes precedence.
OpenCode --model takes provider/model (meta/... is shorthand for nano-gpt/meta/...).
Uses the existing agent login, no approval prompts, and full filesystem access.
EOF
}

die() { printf 'Error: %s\n' "$*" >&2; exit 2; }
while (($#)); do
    case "$1" in
        --repo|--base|--runs-dir|--minutes|--candidates|--workers|--focus|--prompt-file|--gpu|--engine|--model|--budget-usd)
            (($# >= 2)) || die "Missing value for $1"
            case "$1" in
                --repo) repo=$2;;
                --base) base=$2;;
                --runs-dir) runs_dir=$2;;
                --minutes) minutes=$2;;
                --candidates) candidates=$2;;
                --workers) workers=$2;;
                --gpu) gpu=$2;;
                --focus) focus=$2;;
                --prompt-file) prompt_file=$2;;
                --engine) engine=$2;;
                --model) model=$2;;
                --budget-usd) budget=$2;;
            esac
            shift 2;;
        --dry-run) dry_run=true; shift;;
        -h|--help) usage; exit 0;;
        *) die "Unknown argument: $1";;
    esac
done
# Permit help/dry-runs while paused, but never launch an attempt past STOP.
if ! "$dry_run" && [[ -f "$script_dir/gan-attempts/STOP" ]]; then
    printf 'GAN attempts are stopped by user request; see %s/gan-attempts/STOP\n' "$script_dir" >&2
    exit 75
fi
[[ "$minutes" =~ ^[1-9][0-9]{0,4}$ ]] || die '--minutes must be a positive integer (at most 99999)'
[[ "$candidates" =~ ^[1-9][0-9]{0,4}$ ]] || die '--candidates must be a positive integer (at most 99999)'
[[ "$workers" =~ ^[1-9][0-9]{0,2}$ ]] || die '--workers must be a positive integer (at most 999)'
[[ "$gpu" =~ ^([0-9]+|GPU-[a-fA-F0-9-]+)$ ]] || die '--gpu must name one physical index or GPU UUID'
effort=max
case "$engine" in
    codex) agent_bin=$codex_bin; model=${model:-gpt-6-astra};;
    claude) agent_bin=$claude_bin; model=${model:-'opus[1m]'};;
    grok) agent_bin=$grok_bin; model=${model:-grok-4.7}; effort=default;;
    opencode)
        agent_bin=$opencode_bin
        model=${model:-nano-gpt/meta/muse-spark-1.3-contributor}
        [[ "$model" != meta/* ]] || model="nano-gpt/$model"
        effort=default;;
    *) die '--engine must be codex, claude, grok or opencode';;
esac
[[ -z "$budget" ]] || [[ "$engine" == claude ]] || die '--budget-usd applies to --engine claude only'
[[ -z "$budget" || "$budget" =~ ^[0-9]+([.][0-9]+)?$ ]] || die '--budget-usd must be a dollar amount'
for executable in git timeout "$agent_bin"; do
    command -v "$executable" >/dev/null || die "Executable not found: $executable"
done
agent_bin=$(realpath -- "$(command -v "$agent_bin")")
renderer="$script_dir/claude-stream.py"
if [[ "$engine" != codex ]]; then
    command -v python3 >/dev/null || die 'Executable not found: python3 (renders the agent stream)'
    [[ -r "$renderer" ]] || die "Expected the stream renderer at $renderer"
fi
opencode_config=''
if [[ "$engine" == opencode ]]; then
    [[ -r "$script_dir/opencode-config.json" ]] || die "Expected $script_dir/opencode-config.json"
    opencode_config=$(cat -- "$script_dir/opencode-config.json")
fi
repo=$(git -C "$repo" rev-parse --show-toplevel) || die 'Expected a Git checkout at --repo'
source_sha=$(git -C "$repo" rev-parse --verify --end-of-options "${base}^{commit}")
board=reports/toy100/continuous-practical-leaderboard.md
base_brief=reports/toy100/h_stability/SEARCH.md
has_base_brief=false
if git -C "$repo" cat-file -e "$source_sha:$base_brief" 2>/dev/null; then
    has_base_brief=true
fi
if [[ -n "$prompt_file" ]]; then
    prompt_file=$(realpath -- "$prompt_file")
    [[ -s "$prompt_file" && -r "$prompt_file" ]] || die 'Expected a readable, nonempty --prompt-file'
elif ! "$has_base_brief"; then
    git -C "$repo" cat-file -e "$source_sha:$board" || die "The selected base has no $board"
fi
runs_dir=$(realpath -m -- "$runs_dir")
attempt_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
run_dir="$runs_dir/$attempt_id"
checkout="$run_dir/repo"
branch="codex/gan-attempt-$attempt_id"
if [[ ! -x "$gan_python" ]]; then
    gan_python=$(command -v python3) || die 'Set GAN_PYTHON to a Python with benchmark dependencies'
fi
# Preserve a venv's interpreter path: resolving its symlink can bypass the venv.
gan_python=$(realpath -s -- "$gan_python")
deadline=$(date -u -d "+$minutes minutes" '+%Y-%m-%d %H:%M:%S UTC')

prompt=$(cat <<EOF
Make a concrete GPU attempt from reports/toy100/current-research-base.json,
including its declared config, mechanism and probe. Read AGENTS.md and $board.
Preserve the selected schedules and auxiliary host terms as starting defaults.
Follow the declared failure gate order, then all 22 with frozen budgets, seeds
and scoring. Keep model training
and optimizer updates on CUDA. No seed sweeps or target-fitting replacement.
Implement and run bounded tests; save actual results, not theory.
EOF
)

if [[ -n "$prompt_file" ]]; then
    prompt=$(cat -- "$prompt_file")
elif "$has_base_brief"; then
    prompt=$(git -C "$repo" show "$source_sha:$base_brief")
fi
prompt+="$(cat <<EOF


Runtime instructions for this fresh, independent attempt:
- Read AGENTS.md. Work only in $checkout; do not modify other checkouts.
- Assigned focus: $focus
- Time budget: $minutes minutes, hard stop approximately $deadline.
- Candidate budget: at most $candidates distinct proposals; at most $workers parallel GPU
  benchmark workers, one CPU thread each. These are workers, not extra agents.
- Benchmark Python: $gan_python; physical GPU $gpu is visible as cuda:0.
  CUDA_VISIBLE_DEVICES=$gpu, CUBLAS_WORKSPACE_CONFIG=:4096:8, one CPU thread.
  Use FP32, deterministic algorithms and TF32 off as in the committed CUDA controls.
  CPU initialization is allowed; model training, gradients and optimizer state must be CUDA.
  Explicitly set CUDA_VISIBLE_DEVICES=$gpu and the above thread/determinism variables
  on benchmark subprocesses; do not assume the agent tool server inherited them.
- No nested agents or extra agent sessions (Codex, Claude, Grok or OpenCode), detached jobs,
  GitHub pushes or comments. You are the only agent on this attempt.
- Write concise progress and $run_dir/result.md with exact code/artifact paths,
  measured results, test totals, replay commands and remaining failures.
- Append each executed gate to $run_dir/tests.jsonl using candidate, gate,
  status (PASS/FAIL/ERROR/SKIPPED), seconds, metrics and artifact fields. Label
  warm-screen success as PASS on a clearly named warm_probe gate; it does not
  qualify cold acquisition or own-state stability. Record regression tests as
  candidate=regression. Do not count setup metadata as executed candidates.
- Before EACH experiment batch, read $run_dir/supervisor.md if it exists.
  This is the user's local supervisor steering the task. Follow its latest
  priorities; if it says STOP, save results and stop promptly.
- Fixed seeds, no seed sweeps. No metric weakening or frozen host/budget edits.
  Start from the selected research formulation, keeping its declared decay and auxiliary
  AE/token host losses. A focused formulation brief may authorize declared changes
  to adversarial losses, critic regularizers or optimizers. Do not introduce a
  non-GAN target fitter or metric feedback.
  The focused brief governs declared changes to schedules/noise and supersedes
  historical porting or constant-rate-only constraints.
- Start a real candidate test within five minutes. Candidate counts are caps, not quotas.
  Gate on measured failures before broad tests. Keep commands and logs easy to inspect.
EOF
)"

if [[ "$engine" == claude ]]; then
    prompt+="$(cat <<EOF

Claude runtime notes for this attempt:
- Bash calls default to a ten-minute timeout. For longer training, pass an explicit
  timeout up to 7200000 ms, or start it with run_in_background and poll its output.
- Subagents, workflows, web access and scheduling tools are disabled on purpose;
  parallel benchmark workers are plain background shell jobs you start and reap.
- Your final message is saved as $run_dir/final.md, but $run_dir/result.md is the
  deliverable: write it as you go so a hard stop still leaves the evidence.
EOF
)"
fi

# Plain exec starts a new thread. Ephemeral prevents history persistence; memory
# injection is separately disabled. Repository AGENTS.md still applies.
if [[ "$engine" == codex ]]; then
    command=("$agent_bin" --no-daemon exec --ignore-user-config --ephemeral
        --model "$model" -c 'model_reasoning_effort="max"'
        -c 'memories.use_memories=false' --disable memories
        --disable external_agent_memory_import --disable multi_agent
        -c 'approval_policy="never"' --sandbox danger-full-access
        --cd "$checkout" --color never --output-last-message "$run_dir/final.md" -)
elif [[ "$engine" == grok ]]; then
    command=("$agent_bin" --model "$model" --cwd "$checkout"
        --no-subagents --disable-web-search --no-plan
        --permission-mode bypassPermissions --output-format streaming-messages-json
        --disallowed-tools 'spawn_subagent,workflow,scheduler_create,scheduler_delete,scheduler_list,monitor,search_tool,use_tool,ask_user_question,send_feedback,image_gen,image_edit,image_to_video,reference_to_video'
        --prompt-file "$run_dir/prompt.md")
elif [[ "$engine" == opencode ]]; then
    # Fresh noninteractive session; stdin carries the prompt. The runtime config
    # disables nested agents and interactive tools while retaining shell/edit access.
    command=("$agent_bin" run --pure --model "$model" --agent build
        --format json --dir "$checkout")
else
    # The Codex flags one for one: fresh print-mode session, no stored history,
    # no user settings/MCP/skills, no nested agents, no approvals, full access.
    # The prompt arrives on stdin and the worktree is the working directory;
    # claude-stream.py renders the event stream and writes final.md.
    command=("$agent_bin" --print --model "$model" --effort max
        --output-format stream-json --verbose --no-session-persistence
        --setting-sources '' --strict-mcp-config --disable-slash-commands
        --settings "$run_dir/claude-settings.json"
        --permission-mode bypassPermissions --permission-prompts none
        --disallowed-tools 'Agent,Task,Workflow,TaskCreate,TaskStop,WebSearch,WebFetch,SendMessage,RemoteTrigger,PushNotification,ScheduleWakeup,CronCreate,CronDelete,CronList,DesignSync,EnterWorktree,ExitWorktree,Artifact')
    [[ -z "$budget" ]] || command+=(--max-budget-usd "$budget")
fi
log="$run_dir/$engine.log"

printf 'Engine: %s (%s)\nBase: %s\nBranch: %s\nWorktree: %s\nBudget: %s minutes; at most %s candidates; %s workers\n' \
    "$engine" "$model" "$source_sha" "$branch" "$checkout" "$minutes" "$candidates" "$workers"
if "$dry_run"; then
    if [[ "$engine" == opencode ]]; then
        printf 'OpenCode config (OPENCODE_CONFIG_CONTENT): %s\n' "$opencode_config"
        printf 'OpenCode config home: %s/opencode-config; project config and external skills disabled\n' "$run_dir"
    fi
    printf 'Command: '
    printf '%q ' timeout --signal=TERM --kill-after=30s "${minutes}m" "${command[@]}"
    printf '\n\n%s\n' "$prompt"
    exit 0
fi

mkdir -p -- "$run_dir"
printf '%s\n' "$prompt" > "$run_dir/prompt.md"
printf '%s\n' 'Continue the assigned focused attempt. Preserve failures and read this file before each new batch.' > "$run_dir/supervisor.md"
printf 'engine=%s\nbase=%s\nbranch=%s\nmodel=%s\neffort=%s\nminutes=%s\ncandidates=%s\nworkers=%s\nfocus=%s\ngpu=%s\nlog=%s\n' \
    "$engine" "$source_sha" "$branch" "$model" "$effort" "$minutes" "$candidates" "$workers" "$focus" "$gpu" "$log" \
    > "$run_dir/run.txt"
if [[ "$engine" == claude ]]; then
    printf '%s\n' '{"env": {"BASH_DEFAULT_TIMEOUT_MS": "600000", "BASH_MAX_TIMEOUT_MS": "7200000"}}' \
        > "$run_dir/claude-settings.json"
fi
if [[ "$engine" == opencode ]]; then
    printf '%s\n' "$opencode_config" > "$run_dir/opencode-config.json"
    mkdir -p -- "$run_dir/opencode-config"
fi
printf '%q ' "${command[@]}" > "$run_dir/command.txt"
printf '\n' >> "$run_dir/command.txt"
lock=''
command -v flock >/dev/null &&
    lock="$(git -C "$repo" rev-parse --path-format=absolute --git-common-dir)/try-gan-worktree.lock"
if [[ -n "$lock" ]]; then
    flock -w 300 -- "$lock" git -C "$repo" worktree add -b "$branch" "$checkout" "$source_sha"
else
    git -C "$repo" worktree add -b "$branch" "$checkout" "$source_sha"
fi
printf 'running\n' > "$run_dir/status.txt"

finish() {
    local code=$?
    trap - EXIT
    # Keep both commits and uncommitted edits; untracked files remain in checkout.
    git -C "$checkout" diff --binary "$source_sha" > "$run_dir/changes.patch" || true
    git -C "$checkout" status --short > "$run_dir/git-status.txt" || true
    printf '%s\n' "$code" > "$run_dir/exit-code.txt"
    case "$code" in
        0) printf 'completed (inspect result.md; exit 0 does not mean a win)\n';;
        124|137) printf 'timed out (partial artifacts retained)\n';;
        *) printf 'failed or interrupted (exit %s; artifacts retained)\n' "$code";;
    esac > "$run_dir/status.txt"
    printf '\nAttempt artifacts: %s\n' "$run_dir"
    exit "$code"
}
trap finish EXIT

# Avoid importing modules or session identity from the caller's active checkout.
unset PYTHONPATH CODEX_THREAD_ID LD_PRELOAD CLAUDECODE CLAUDE_CODE_ENTRYPOINT \
    CLAUDE_CODE_SESSION_ID CLAUDE_CODE_CHILD_SESSION CLAUDE_CODE_SESSION_ATTENDED \
    CLAUDE_CODE_MESSAGING_SOCKET CLAUDE_CODE_MESSAGING_TOKEN CLAUDE_CODE_EXECPATH \
    CLAUDE_EFFORT CLAUDE_PID CLAUDE_CONFIG_DIR CLAUDE_PROJECT_DIR
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2
export ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2 CUDA_VISIBLE_DEVICES="$gpu"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ "$engine" == opencode ]]; then
    unset OPENCODE_CONFIG OPENCODE_CONFIG_DIR OPENCODE_PERMISSION
    export OPENCODE_CONFIG_CONTENT="$opencode_config"
    # Use OpenCode's native auth resolution: saved login before env fallback.
    # Never serialize a credential into config, command or log files.
    export NANO_GPT_API_KEY="${NANO_GPT_API_KEY:-${NANOGPT_API_KEY:-}}"
    export XDG_CONFIG_HOME="$run_dir/opencode-config"
    export OPENCODE_DISABLE_PROJECT_CONFIG=true OPENCODE_DISABLE_CLAUDE_CODE=true
    export OPENCODE_DISABLE_EXTERNAL_SKILLS=true
fi
"$gan_python" -c 'import torch; assert torch.cuda.is_available(), "CUDA required; no CPU fallback"; print("GPU:", torch.cuda.get_device_name(0), "torch:", torch.__version__)'
printf 'Log: tail -f %q\n' "$log"
cd -- "$checkout"
if [[ "$engine" == codex ]]; then
    timeout --signal=TERM --kill-after=30s "${minutes}m" "${command[@]}" \
        < "$run_dir/prompt.md" 2>&1 | tee "$log"
else
    timeout --signal=TERM --kill-after=30s "${minutes}m" "${command[@]}" \
        < "$run_dir/prompt.md" 2>&1 | python3 "$renderer" --engine "$engine" --run-dir "$run_dir" | tee "$log"
fi
