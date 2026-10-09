"""OpenAI-compatible HTTP API.

Endpoints:
    POST /v1/completions       — text completion
    POST /v1/chat/completions  — chat completion
    GET  /health               — health check

Supports streaming via SSE for both endpoints.
Concurrent requests are batched by the scheduler's background step loop.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Optional

from serve.runtime_warnings import import_torch_safely

torch = import_torch_safely()
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field

from fastapi.responses import JSONResponse

from serve.api.webui import build_chat_page
from serve.engine.sampling import SamplingParams
from serve.engine.serving import ServingEngine
from serve.logging import configure_logging, get_logger

LOGGER = get_logger(__name__)


# -- request/response models -----------------------------------------------

class CompletionRequest(BaseModel):
    model: str = ""
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = Field(default=-1, alias="top_k")
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int | None = None
    logprobs: int | None = None
    stream: bool = False
    stop: list[str] | None = None
    timeout: float | None = None  # Request timeout in seconds.

    model_config = {"populate_by_name": True}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = Field(default=-1, alias="top_k")
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int | None = None
    logprobs: int | None = None
    stream: bool = False
    stop: list[str] | None = None
    timeout: float | None = None

    model_config = {"populate_by_name": True}


# -- server ----------------------------------------------------------------

class ServingApp:
    """Wraps the ServingEngine in an HTTP server."""

    def __init__(
        self,
        engine: ServingEngine,
        *,
        enable_webui: bool = False,
    ):
        self.engine = engine
        self.model_name = engine.model_path.split("/")[-1]
        self.app = FastAPI(title="b12x serve")
        self.app.add_api_route("/v1/completions", self._completions, methods=["POST"])
        self.app.add_api_route("/v1/chat/completions", self._chat_completions, methods=["POST"])
        self.app.add_api_route("/v1/models", self._models, methods=["GET"])
        self.app.add_api_route("/health", self._health, methods=["GET"])
        if enable_webui:
            self.app.add_api_route("/", self._webui_root, methods=["GET"], response_class=RedirectResponse)
            self.app.add_api_route("/ui", self._webui, methods=["GET"], response_class=HTMLResponse)
        self.app.add_exception_handler(Exception, self._handle_error)

    async def _handle_error(self, request, exc):
        msg = str(exc)
        status = 503 if "queue full" in msg else 500
        return JSONResponse(
            status_code=status,
            content={"error": {"message": msg, "type": type(exc).__name__}},
        )

    async def _health(self):
        import serve
        loop_health = self.engine.server_loop_health()
        return {
            "status": loop_health["status"],
            "version": serve.__version__,
            "model": self.model_name,
            "scheduler": self.engine.scheduler.stats,
            "server_loop": loop_health,
            "runtime_policy": self.engine.runtime_policy(),
        }

    async def _models(self):
        return {
            "object": "list",
            "data": [{
                "id": self.model_name,
                "object": "model",
                "owned_by": "b12x",
            }],
        }

    async def _webui_root(self):
        return RedirectResponse(url="/ui", status_code=307)

    async def _webui(self):
        return HTMLResponse(build_chat_page(model_name=self.model_name))

    # -- /v1/completions ---------------------------------------------------

    async def _completions(self, req: CompletionRequest):
        params = self._to_sampling_params(req)

        if req.stream:
            return StreamingResponse(
                self._stream_completion(req.prompt, params, req.timeout),
                media_type="text/event-stream",
            )

        input_ids = self.engine.tokenizer(req.prompt, return_tensors="pt").input_ids[0].tolist()
        request = self.engine.submit(input_ids, params, timeout_s=req.timeout)
        await asyncio.to_thread(request._done_event.wait)
        self._raise_for_failed_request(request)

        text = self.engine.tokenizer.decode(request.output_ids, skip_special_tokens=True)
        result = self.engine._to_result(request)
        return self._completion_response(text, result, "text_completion")

    # -- /v1/chat/completions ----------------------------------------------

    async def _chat_completions(self, req: ChatCompletionRequest):
        params = self._to_sampling_params(req)
        messages = [{"role": m.role, "content": m.content} for m in req.messages]

        if req.stream:
            return StreamingResponse(
                self._stream_chat(messages, params, req.timeout),
                media_type="text/event-stream",
            )

        formatted = self.engine.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = self.engine.tokenizer(formatted, return_tensors="pt").input_ids[0].tolist()
        request = self.engine.submit(input_ids, params, timeout_s=req.timeout)
        await asyncio.to_thread(request._done_event.wait)
        self._raise_for_failed_request(request)

        text = self.engine.tokenizer.decode(request.output_ids, skip_special_tokens=True)
        result = self.engine._to_result(request)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": result.finish_reason or "length",
            }],
            "usage": {
                "prompt_tokens": len(result.prompt_ids),
                "completion_tokens": len(result.generated_ids),
                "total_tokens": len(result.prompt_ids) + len(result.generated_ids),
            },
        }

    # -- streaming ---------------------------------------------------------

    async def _stream_tokens(self, request, completion_id, obj_type, make_choice):
        """Yield SSE chunks as tokens are generated.

        Uses request._token_event to wake up on each new token instead of polling.
        Cancels the request if the client disconnects.
        """
        try:
            prev_len = 0
            while not request.is_finished:
                await asyncio.to_thread(request._token_event.wait)
                request._token_event.clear()
                for tok_id in request.output_ids[prev_len:]:
                    text = self.engine.tokenizer.decode([tok_id], skip_special_tokens=True)
                    chunk = {
                        "id": completion_id,
                        "object": obj_type,
                        "created": int(time.time()),
                        "model": self.model_name,
                        "choices": [make_choice(text, None)],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                prev_len = len(request.output_ids)

            # Final chunk with finish_reason.
            chunk = {
                "id": completion_id,
                "object": obj_type,
                "created": int(time.time()),
                "model": self.model_name,
                "choices": [make_choice("", request.finished_reason or "stop")],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        except (asyncio.CancelledError, GeneratorExit):
            request.cancel()
            raise

    async def _stream_completion(self, prompt, params, timeout=None):
        completion_id = f"cmpl-{uuid.uuid4().hex[:12]}"
        input_ids = self.engine.tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
        request = self.engine.submit(input_ids, params, timeout_s=timeout)

        def _choice(text, reason):
            return {"text": text, "index": 0, "finish_reason": reason}

        async for chunk in self._stream_tokens(request, completion_id, "text_completion", _choice):
            yield chunk

    async def _stream_chat(self, messages, params, timeout=None):
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        formatted = self.engine.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = self.engine.tokenizer(formatted, return_tensors="pt").input_ids[0].tolist()
        request = self.engine.submit(input_ids, params, timeout_s=timeout)

        def _choice(text, reason):
            c = {"index": 0, "finish_reason": reason}
            c["delta"] = {"content": text} if text else {}
            return c

        async for chunk in self._stream_tokens(request, completion_id, "chat.completion.chunk", _choice):
            yield chunk

    # -- helpers -----------------------------------------------------------

    def _to_sampling_params(self, req) -> SamplingParams:
        unsupported = []
        if getattr(req, "seed", None) is not None:
            unsupported.append("seed")
        if getattr(req, "logprobs", None) is not None:
            unsupported.append("logprobs")
        if unsupported:
            raise HTTPException(
                status_code=400,
                detail=f"unsupported request fields: {', '.join(unsupported)}",
            )
        stop_sequences = None
        if req.stop:
            stop_sequences = []
            for s in req.stop:
                ids = self.engine.tokenizer.encode(s, add_special_tokens=False)
                if ids:
                    stop_sequences.append(ids)
        return SamplingParams(
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            min_p=req.min_p,
            repetition_penalty=req.repetition_penalty,
            presence_penalty=req.presence_penalty,
            frequency_penalty=req.frequency_penalty,
            max_new_tokens=req.max_tokens,
            stop_sequences=stop_sequences,
        )

    def _raise_for_failed_request(self, request) -> None:
        reason = request.finished_reason
        if reason in {None, "length", "stop"}:
            return
        if reason == "context_too_long":
            raise HTTPException(status_code=400, detail="prompt exceeds the configured KV-cache budget")
        if reason == "timeout":
            raise HTTPException(status_code=504, detail="request timed out before completion")
        if reason == "oom":
            raise HTTPException(status_code=503, detail="server ran out of cache capacity for this request")
        if reason == "engine_error":
            raise HTTPException(status_code=503, detail="generation failed because the engine became unhealthy")
        if reason == "cancelled":
            raise HTTPException(status_code=499, detail="request was cancelled")
        raise HTTPException(status_code=500, detail=f"generation failed with reason={reason}")

    def _completion_response(self, text, result, obj_type):
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": obj_type,
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [{
                "text": text,
                "index": 0,
                "finish_reason": result.finish_reason or "length",
            }],
            "usage": {
                "prompt_tokens": len(result.prompt_ids),
                "completion_tokens": len(result.generated_ids),
                "total_tokens": len(result.prompt_ids) + len(result.generated_ids),
            },
        }

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        self.engine.start_server_loop()
        uvicorn.run(self.app, host=host, port=port)


if __name__ == "__main__":
    configure_logging("info", rank=0)
    LOGGER.info("Use 'python -m serve.cli MODEL --serve' for the API server.")
    LOGGER.info("It supports TP, prefill graphs, and all other options.")
