import subprocess

from va.actions import ActionError

last_active_player: str | None = None


def _run(*args: str) -> None:
    try:
        command = ["playerctl"]
        if last_active_player:
            command.extend(["--player", last_active_player])
        command.extend(args)
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        raise ActionError(e)


def play_music() -> None:
    _run("play")


def next_track() -> None:
    _run("next")


def previous_track() -> None:
    _run("previous")


def volume_down() -> None:
    _run("volume", "0.1-")


def volume_up() -> None:
    _run("volume", "0.1+")


def volume_much_down() -> None:
    _run("volume", "0.3-")


def volume_much_up() -> None:
    _run("volume", "0.3+")


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
