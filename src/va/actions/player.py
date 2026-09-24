import asyncio
import subprocess

import yt_dlp
import ytmusicapi
import ytmusicapi.exceptions

from va.actions import ActionError, AssistantResponse

last_active_player: str | None = None
_mpv: asyncio.subprocess.Process | None = None


def _stop_mpv() -> None:
    global _mpv
    if _mpv is not None and _mpv.returncode is None:
        try:
            _mpv.kill()
        except (ProcessLookupError, ChildProcessError):
            pass
    _mpv = None


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


def now_playing() -> AssistantResponse:
    command = ["playerctl", "--no-messages"]
    if last_active_player:
        command.extend(["--player", last_active_player])
    try:
        status = subprocess.run(
            [*command, "status"], check=True, capture_output=True, text=True
        ).stdout.strip()
        meta = subprocess.run(
            [*command, "metadata", "--format", "{{artist}} — {{title}}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return AssistantResponse("Сейчас ничего не играет")

    if status == "Paused":
        return AssistantResponse(f"Сейчас на паузе: {meta}")
    return AssistantResponse(f"Сейчас играет {meta}")


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
    try:
        playlist = await asyncio.to_thread(ytmusic.search, song, filter="songs")
        track = playlist[0] if playlist else None
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
    except (
        OSError,
        KeyError,
        IndexError,
        ytmusicapi.exceptions.YTMusicError,
        yt_dlp.DownloadError,
    ):
        return AssistantResponse(f'Не удалось разобрать трек "{song}"')

    try:
        pause_music()
    except ActionError:
        pass
    volume = _current_volume()

    queue = []
    try:
        watch_playlist = await asyncio.to_thread(ytmusic.get_watch_playlist, video_id)
        playlist = watch_playlist["tracks"]
        assert isinstance(playlist, list)
        queue = [
            f"https://music.youtube.com/watch?v={t['videoId']}"
            for t in playlist
            if t.get("videoId") and t["videoId"] != video_id
        ]
    except (KeyError, AssertionError, OSError, ytmusicapi.exceptions.YTMusicError):
        queue = []

    global _mpv, last_active_player
    _stop_mpv()
    command = ["mpv", "--no-video", "--force-window=no"]
    if volume is not None:
        command.append(f"--volume={int(volume * 100):d}")
    command.extend([stream_url, *queue])
    try:
        _mpv = await asyncio.create_subprocess_exec(*command, start_new_session=True)
    except OSError as e:
        raise ActionError(e)
    last_active_player = "mpv"

    return AssistantResponse(
        f'Играет "{artists} - {title}"', f'Играет "{artists} – {title}"'
    )
