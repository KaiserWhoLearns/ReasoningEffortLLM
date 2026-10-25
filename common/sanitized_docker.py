"""A mini-swe-agent Docker environment that runs a setup command before the agent starts.

SWE-Bench Pro-Corrected uses it to strip the container's git history
(`swebench_pro/configs/strip_history.yaml`) before the agent's first step. mini 2.4's own hook,
`run.env_startup_command`, calls `env.execute(<str>)` where `DockerEnvironment.execute` expects an
action dict, so it cannot be used. This subclass:

* fails the instance if the setup command exits non-zero, so an instance never runs on an
  unprepared container;
* gives the setup command its own timeout (sanitizing a large repository takes longer than the
  agent's 60 s command timeout);
* records the command's output in the trajectory (`info.setup_output`).

With no `setup_command` it behaves exactly like `DockerEnvironment`.
"""
from __future__ import annotations

from minisweagent.environments.docker import DockerEnvironment, DockerEnvironmentConfig


class SetupCommandDockerEnvironmentConfig(DockerEnvironmentConfig):
    setup_command: str = ""
    """Shell command to run in the container after it starts, before the agent's first step.
    Empty = do nothing. A non-zero exit aborts the instance."""
    setup_timeout: int = 900
    """Timeout (s) for `setup_command`. Separate from `timeout`, which caps the agent's own steps
    and is far too short for e.g. re-adding every file of a large repo."""


class SetupCommandDockerEnvironment(DockerEnvironment):
    """DockerEnvironment + one command run at container start, whose failure aborts the instance."""

    def __init__(self, *, config_class: type = SetupCommandDockerEnvironmentConfig, **kwargs):
        self.setup_output: str = ""
        super().__init__(config_class=config_class, **kwargs)
        if self.config.setup_command.strip():
            self._run_setup()

    def _run_setup(self) -> None:
        result = self.execute({"command": self.config.setup_command},
                             timeout=self.config.setup_timeout)
        self.setup_output = result.get("output", "")
        if result["returncode"] != 0:
            # Raised inside process_instance's try block: the instance is recorded with this
            # exception as its exit status and an empty patch.
            raise RuntimeError(
                f"setup_command failed (returncode={result['returncode']}) in container "
                f"{self.container_id}; refusing to run the agent on an unprepared environment.\n"
                f"--- setup output ---\n{self.setup_output}")
        self.logger.info(f"setup_command ok: {self.setup_output.strip()[-500:]}")

    def serialize(self) -> dict:
        """Environment record saved into the trajectory, plus what the setup command printed."""
        data = super().serialize()
        data["info"]["setup_output"] = self.setup_output
        return data


class GitStrippedDockerEnvironment(SetupCommandDockerEnvironment):
    """A container whose git history is stripped at startup.

    The stripping is the shell script in the benchmark config, which is also saved in every run's
    `config.merged.yaml`.
    """
