import subprocess

from va.actions import ActionError

last_active_player: str | None = None


def _run(verb: str) -> None:
    try:
        command = ["playerctl"]
        if last_active_player:
            command += ["--player", last_active_player]
        command.append(verb)
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        raise ActionError(e)


def play_music() -> None:
    _run("play")


def next_track() -> None:
    _run("next")


def previous_track() -> None:
    _run("previous")


def pause_music() -> None:
    global last_active_player
    try:
        active = subprocess.run(
            ["playerctl", "--no-messages", "metadata", "--format", "{{playerName}}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if active:
            last_active_player = active
        subprocess.run(["playerctl", "--all-players", "pause"], check=True)
    except subprocess.CalledProcessError as e:
        raise ActionError(e)
