import asyncio
import subprocess

import yt_dlp
import ytmusicapi

from va.actions import ActionError, AssistantResponse

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


async def play_music(song: str | None = None) -> AssistantResponse | None:
    if song:
        return await _play_from_yt_in_mpv(song)
    else:
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


def _current_volume() -> float | None:
    try:
        command = ["playerctl", "--no-messages"]
        if last_active_player:
            command.extend(["--player", last_active_player])
        command.append("volume")
        volume = subprocess.run(
            command, check=True, capture_output=True, text=True
        ).stdout.strip()
        return float(volume)
    except (subprocess.CalledProcessError, ValueError):
        return None


async def _play_from_yt_in_mpv(song: str) -> AssistantResponse:
    ytmusic = ytmusicapi.YTMusic()
    tracks = await asyncio.to_thread(ytmusic.search, song, filter="songs")
    track = tracks[0] if tracks else None
    if track is None:
        return AssistantResponse(f'Не удалось найти трек "{song}"')

    artists = f"{', '.join(a['name'] for a in track['artists'])}"
    title = track["title"]
    video_id = track["videoId"]

    link = f"https://music.youtube.com/watch?v={video_id}"
    opts = {
        "extract_audio": True,
        "format": "bestaudio",
    }
    with yt_dlp.YoutubeDL(opts) as ytdl:
        assert isinstance(ytdl, yt_dlp.YoutubeDL)
        response = await asyncio.to_thread(ytdl.extract_info, link, download=False)
        stream_url = response["url"]

    pause_music()

    command = ["mpv", "--no-video", "--force-window=no"]
    volume = _current_volume()
    if volume is not None:
        command.append(f"--volume={int(volume * 100):d}")
    command.append(stream_url)
    await asyncio.create_subprocess_exec(*command, start_new_session=True)
    return AssistantResponse(f'Играет "{artists} - {title}"', f'Играет "{artists} – {title}"')
