import json
import logging
import os
from pathlib import Path
from typing import Any

from llama_cpp import Llama
from llama_cpp import llama_cpp
from openinference.semconv.trace import SpanAttributes
from smolagents.models import ChatMessage, MessageRole, Model

from bookmarks.models.gpu import get_gpu_config
from bookmarks.tracing import get_tracer


def _coerce_message_content(content: str | list[dict[str, Any]] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


class LlamaCppChatModel(Model):
    def __init__(
        self,
        model_path: Path | str,
        *,
        n_ctx: int = 4096,
        n_batch: int = 512,
        temperature: float = 0.0,
        verbose: bool | None = None,
        **kwargs: Any,
    ):
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.temperature = temperature
        self.verbose = verbose if verbose is not None else os.getenv("BOOKMARKS_LLAMA_VERBOSE", "").lower() in (
            "1",
            "true",
            "yes",
        )

        gpu_config = get_gpu_config()
        if gpu_config.is_accelerated:
            logging.info("gpu acceleration enabled: %s backend", gpu_config.backend)

        try:
            system_info = llama_cpp.llama_print_system_info()
            info_text = system_info.decode("utf-8", errors="ignore").lower()
            if gpu_config.backend == "cuda" and "cuda" not in info_text:
                logging.warning("llama-cpp-python is not built with cuda; rebuild it to enable gpu layers")
        except Exception:
            pass

        self._llama = Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_batch=self.n_batch,
            n_threads=max(os.cpu_count() or 1, 1),
            n_gpu_layers=gpu_config.n_gpu_layers,
            verbose=self.verbose,
        )

        super().__init__(model_id=str(self.model_path), **kwargs)

    def generate(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,  # noqa: ARG002 - not supported by llama.cpp
        tools_to_call_from=None,  # noqa: ANN001 - ignored (no tool calling)
        **kwargs: Any,
    ) -> ChatMessage:
        request = []
        for message in messages:
            request.append(
                {
                    "role": message.role.value,
                    "content": _coerce_message_content(message.content),
                }
            )

        temperature = float(kwargs.get("temperature", self.temperature))
        max_tokens = kwargs.get("max_tokens")

        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("llama_cpp.chat_completion") as span:
            span.set_attribute(SpanAttributes.LLM_PROVIDER, "llama_cpp")
            span.set_attribute(SpanAttributes.LLM_MODEL_NAME, str(self.model_path))
            span.set_attribute(
                SpanAttributes.LLM_INVOCATION_PARAMETERS,
                json.dumps(
                    {"temperature": temperature, "max_tokens": max_tokens, "stop": stop_sequences},
                    ensure_ascii=False,
                ),
            )
            span.set_attribute(SpanAttributes.INPUT_VALUE, json.dumps(request, ensure_ascii=False))

            try:
                result = self._llama.create_chat_completion(
                    messages=request,
                    temperature=temperature,
                    stop=stop_sequences,
                    max_tokens=max_tokens,
                )
            except TypeError:
                # older llama-cpp-python builds may not accept certain kwargs.
                result = self._llama.create_chat_completion(
                    messages=request,
                    temperature=temperature,
                    stop=stop_sequences,
                )

            content = (
                result.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(content))

        return ChatMessage(role=MessageRole.ASSISTANT, content=str(content), raw=result)
