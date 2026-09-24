import subprocess

from va.actions import ActionError

last_active_player: str | None = None


def play_music() -> None:
    try:
        if last_active_player:
            command = ["playerctl", "--player", last_active_player, "play"]
        else:
            command = ["playerctl", "play"]
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        raise ActionError(e)


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
