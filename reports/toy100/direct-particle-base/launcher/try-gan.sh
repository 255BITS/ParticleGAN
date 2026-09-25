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
focus='Improve the selected direct-particle-response GAN; gate on unequal-width covariance first.'
prompt_file=''
dry_run=false
codex_bin=${CODEX_BIN:-codex}
gan_python=${GAN_PYTHON:-/tmp/pr38-default-env/bin/python}

usage() {
    cat <<'EOF'
Usage: ./try-gan.sh [--minutes N] [--candidates N] [--workers N]
                    [--focus TEXT] [--prompt-file FILE] [--repo DIR] [--base REF]
                    [--runs-dir DIR] [--gpu INDEX_OR_UUID] [--dry-run]

Launch one fresh Codex gpt-6-astra/max GPU attempt from the committed search brief.
Each invocation creates a new branch/worktree from the selected committed base.
Default: ParticleGAN-selected-h-stability-base HEAD, 90 minutes, ./gan-attempts/.
Focused search: at most 16 proposals and 3 parallel benchmark workers.
--gpu selects one physical GPU (default 1); CUDA is required, with no CPU fallback.
Uses the selected commit's h_stability/SEARCH.md when present.
--prompt-file replaces the broad search brief with a focused experiment brief.
--dry-run prints the exact command and prompt without creating files or calling Codex.
CODEX_BIN overrides the CLI executable; GAN_PYTHON selects the benchmark Python.
Uses the existing Codex login, no approval prompts, and full filesystem access.
EOF
}

die() { printf 'Error: %s\n' "$*" >&2; exit 2; }
while (($#)); do
    case "$1" in
        --repo|--base|--runs-dir|--minutes|--candidates|--workers|--focus|--prompt-file|--gpu)
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
            esac
            shift 2;;
        --dry-run) dry_run=true; shift;;
        -h|--help) usage; exit 0;;
        *) die "Unknown argument: $1";;
    esac
done
[[ "$minutes" =~ ^[1-9][0-9]{0,4}$ ]] || die '--minutes must be a positive integer (at most 99999)'
[[ "$candidates" =~ ^[1-9][0-9]{0,4}$ ]] || die '--candidates must be a positive integer (at most 99999)'
[[ "$workers" =~ ^[1-9][0-9]{0,2}$ ]] || die '--workers must be a positive integer (at most 999)'
[[ "$gpu" =~ ^([0-9]+|GPU-[a-fA-F0-9-]+)$ ]] || die '--gpu must name one physical index or GPU UUID'
for executable in git timeout "$codex_bin"; do
    command -v "$executable" >/dev/null || die "Executable not found: $executable"
done
codex_bin=$(command -v "$codex_bin")
codex_bin=$(realpath -- "$codex_bin")
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
- No nested agents, Codex sessions, detached jobs, GitHub pushes or comments.
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
  This porting task supersedes the old constant-rate research constraints.
- Start a real candidate test within five minutes. Candidate counts are caps, not quotas.
  Gate on measured failures before broad tests. Keep commands and logs easy to inspect.
EOF
)"

# Plain exec starts a new thread. Ephemeral prevents history persistence; memory
# injection is separately disabled. Repository AGENTS.md still applies.
command=("$codex_bin" --no-daemon exec --ignore-user-config --ephemeral
    --model gpt-6-astra -c 'model_reasoning_effort="max"'
    -c 'memories.use_memories=false' --disable memories
    --disable external_agent_memory_import --disable multi_agent
    -c 'approval_policy="never"' --sandbox danger-full-access
    --cd "$checkout" --color never --output-last-message "$run_dir/final.md" -)

printf 'Base: %s\nBranch: %s\nWorktree: %s\nBudget: %s minutes; at most %s candidates; %s workers\n' \
    "$source_sha" "$branch" "$checkout" "$minutes" "$candidates" "$workers"
if "$dry_run"; then
    printf 'Command: '
    printf '%q ' timeout --signal=TERM --kill-after=30s "${minutes}m" "${command[@]}"
    printf '\n\n%s\n' "$prompt"
    exit 0
fi

mkdir -p -- "$run_dir"
printf '%s\n' "$prompt" > "$run_dir/prompt.md"
printf '%s\n' 'Continue the assigned focused attempt. Preserve failures and read this file before each new batch.' > "$run_dir/supervisor.md"
printf 'base=%s\nbranch=%s\nmodel=gpt-6-astra\neffort=max\nminutes=%s\ncandidates=%s\nworkers=%s\nfocus=%s\ngpu=%s\n' \
    "$source_sha" "$branch" "$minutes" "$candidates" "$workers" "$focus" "$gpu" > "$run_dir/run.txt"
printf '%q ' "${command[@]}" > "$run_dir/command.txt"
printf '\n' >> "$run_dir/command.txt"
git -C "$repo" worktree add -b "$branch" "$checkout" "$source_sha"
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
unset PYTHONPATH CODEX_THREAD_ID LD_PRELOAD
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2
export ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2 CUDA_VISIBLE_DEVICES="$gpu"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$gan_python" -c 'import torch; assert torch.cuda.is_available(), "CUDA required; no CPU fallback"; print("GPU:", torch.cuda.get_device_name(0), "torch:", torch.__version__)'
printf 'Log: tail -f %q\n' "$run_dir/codex.log"
timeout --signal=TERM --kill-after=30s "${minutes}m" "${command[@]}" \
    < "$run_dir/prompt.md" 2>&1 | tee "$run_dir/codex.log"
