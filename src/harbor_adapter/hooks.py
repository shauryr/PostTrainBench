"""Trial lifecycle hooks for PostTrainBench.

Provides an AGENT_START hook that:
1. Pre-creates the .timer_start sentinel file with the current timestamp so
   timer.sh starts counting from the exact moment the agent begins, not from
   its first invocation.
2. Sets up a fuse-overlayfs over the HuggingFace cache Modal volume so the
   agent can read the shared 500GB cache but writes are isolated to a local
   overlay layer.

Usage:
    from harbor.trial.trial import Trial
    from hooks import register_agent_start_hook

    trial = Trial(config)
    register_agent_start_hook(trial, num_hours=10)
    result = await trial.run()
"""

from harbor.environments.base import BaseEnvironment
from harbor.trial.hooks import TrialEvent, TrialHookEvent

# Paths inside the container
HF_CACHE_VOLUME_MOUNT = "/hf-cache-volume"
HF_HOME_PATH = "/hf-home"
WORKSPACE = "/home/agent/workspace"


_OVERLAY_SETUP_SCRIPT = """\
if [ -d "{volume_mount}" ] && [ "$(ls -A {volume_mount} 2>/dev/null)" ]; then
    mkdir -p /tmp/hf-overlay/upper /tmp/hf-overlay/work
    if command -v fuse-overlayfs &>/dev/null; then
        fuse-overlayfs \
            -o "lowerdir={volume_mount},upperdir=/tmp/hf-overlay/upper,workdir=/tmp/hf-overlay/work" \
            "{hf_home}" 2>&1
        if [ $? -eq 0 ]; then
            echo "HF cache overlay mounted at {hf_home}"
        else
            echo "WARNING: fuse-overlayfs failed, falling back to symlinks" >&2
            cp -as "{volume_mount}/." "{hf_home}/" 2>/dev/null || true
        fi
    else
        echo "WARNING: fuse-overlayfs not found, falling back to symlinks" >&2
        cp -as "{volume_mount}/." "{hf_home}/" 2>/dev/null || true
    fi
else
    echo "HF cache volume not mounted or empty at {volume_mount}, using empty cache at {hf_home}"
fi
"""


def register_agent_start_hook(
    trial,  # harbor.trial.trial.Trial
    *,
    num_hours: int,
    hf_cache_volume_mount: str = HF_CACHE_VOLUME_MOUNT,
    hf_home_path: str = HF_HOME_PATH,
) -> None:
    """Register an AGENT_START hook that initializes timer and HF cache overlay.

    Must be called after Trial.__init__ (which creates the environment) but
    before Trial.run().

    Args:
        trial: The Trial object to register the hook on.
        num_hours: Number of hours for the timer.
        hf_cache_volume_mount: Where the Modal volume is mounted (read-only base layer).
        hf_home_path: Where the overlay merged dir will appear (agent's HF_HOME).
    """
    environment: BaseEnvironment = trial._environment

    async def on_agent_start(_event: TrialHookEvent) -> None:
        # 1. Set up HuggingFace cache overlay (if volume is mounted).
        #    Do this first so the overlay setup time doesn't count against
        #    the agent's timer.
        overlay_cmd = _OVERLAY_SETUP_SCRIPT.format(
            volume_mount=hf_cache_volume_mount,
            hf_home=hf_home_path,
        )
        result = await environment.exec(overlay_cmd)
        if result.stdout:
            print(f"[PostTrainBench overlay] {result.stdout.strip()}")
        if result.stderr:
            print(f"[PostTrainBench overlay] {result.stderr.strip()}")

        # 2. Pre-create the .timer_start sentinel so timer.sh counts from
        #    right now, after setup is done, rather than from the first
        #    timer.sh invocation.
        result = await environment.exec(
            f"date +%s > {WORKSPACE}/.timer_start"
        )
        if result.return_code != 0:
            raise RuntimeError(
                f"Failed to create .timer_start sentinel: {result.stderr}"
            )

    trial.add_hook(TrialEvent.AGENT_START, on_agent_start)
