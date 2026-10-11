import re
import typing

import llama_cpp

from va import models

_CJK_REGEX = re.compile(r"[\u4E00-\u9FFF]+")
_CJK_CHAR = re.compile(r"[\u4E00-\u9FFF]")
_CYRILLIC_CHAR = re.compile(r"[\u0400-\u04FF]")


def create_chat_completion(
    messages: list[
        llama_cpp.ChatCompletionRequestSystemMessage
        | llama_cpp.ChatCompletionRequestUserMessage
        | llama_cpp.ChatCompletionRequestAssistantMessage
        | llama_cpp.ChatCompletionRequestToolMessage
        | llama_cpp.ChatCompletionRequestFunctionMessage
    ],
):
    out = models.QWEN_2_5_1_5B.create_chat_completion(messages=messages)
    out = typing.cast(llama_cpp.CreateChatCompletionResponse, out)
    content = out["choices"][0]["message"]["content"]
    assert content is not None
    content = content.strip().splitlines()[0]
    if _is_random_cjk(content):
        content = _replace_cjk_tokens(content)
    return content


def _is_random_cjk(content: str) -> bool:
    cyrillic = len(_CYRILLIC_CHAR.findall(content))
    cjk = len(_CJK_CHAR.findall(content))
    return 0 < cjk < cyrillic


def _replace_cjk_tokens(content: str) -> str:
    encoding = models.QWEN_2_5_1_5B_TOKENIZER(
        content, return_offsets_mapping=True, add_special_tokens=False
    )  # ty: ignore[call-non-callable]

    res = ""
    i = 0
    for start, end in encoding["offset_mapping"]:
        match = _CJK_REGEX.search(content[start:end])
        if match is None:
            continue
        cjk_start = start + match.start()
        cjk_end = start + match.end()
        res += content[i:cjk_start]
        masked = f"{res}{models.FILL_MASK_RU_MODERN_BERT.tokenizer.mask_token}{content[cjk_end:]}"  # ty: ignore[unresolved-attribute]
        token = models.FILL_MASK_RU_MODERN_BERT(masked)[0]["token_str"]
        print("MATCHED", content[cjk_start:cjk_end], token)
        res += token
        i = cjk_end
    res += content[i:]
    return res
