import typing

import llama_cpp

SYSTEM_PROMPT = (
    "Ты помогаешь извлечь название трека или исполнителя из реплики "
    "пользователя для поиска в YouTube Music. Отвечай строго одной строкой: "
    "название или НЕТ. Название должно быть дословной фразой из самой реплики "
    "пользователя."
)

EXAMPLES = """
Фразы и ответы:
"включи песню группы спиритбокс" -> спиритбокс
"хочу послушать платинум" -> платинум
"включи музыку Gojira Born in Winter" -> Gojira Born in Winter
"поставь музыку" -> НЕТ
"""


def extract_music_artist_and_or_title(llm: llama_cpp.Llama, s: str) -> str | None:
    user = f'{EXAMPLES}\n\nФраза: "{s}"\nОтвет:'
    messages: list[llama_cpp.ChatCompletionRequestMessage] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    out = llm.create_chat_completion(
        messages=messages,
        temperature=0,
        max_tokens=32,
    )
    out = typing.cast(llama_cpp.CreateChatCompletionResponse, out)
    text = out["choices"][0]["message"]["content"]
    assert text is not None
    text = text.strip().splitlines()[0]
    if not text or text.upper() == "НЕТ":
        return None
    text = " ".join(text.split())
    print(text)
    return text or None
