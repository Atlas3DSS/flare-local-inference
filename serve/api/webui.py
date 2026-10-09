"""Small optional web UI for chat over the existing HTTP API."""

from __future__ import annotations

import html
import json


def build_chat_page(*, model_name: str) -> str:
    safe_model_name = html.escape(model_name)
    initial_config = json.dumps({"model": model_name}).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_model_name} · b12x chat</title>
  <style>
    :root {{
      --bg: #f3efe4;
      --paper: rgba(255, 252, 245, 0.92);
      --ink: #1e1d1a;
      --muted: #676457;
      --line: rgba(43, 40, 34, 0.14);
      --accent: #0f766e;
      --accent-strong: #115e59;
      --user: #ece8db;
      --assistant: #fffdf8;
      --shadow: 0 22px 60px rgba(30, 29, 26, 0.12);
    }}

    * {{ box-sizing: border-box; }}

    body {{
      margin: 0;
      min-height: 100vh;
      font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.14), transparent 30rem),
        radial-gradient(circle at bottom right, rgba(184, 115, 51, 0.14), transparent 26rem),
        linear-gradient(180deg, #f8f4ea 0%, var(--bg) 100%);
    }}

    .shell {{
      width: min(1040px, calc(100vw - 2rem));
      margin: 1.25rem auto;
      border: 1px solid var(--line);
      border-radius: 28px;
      overflow: hidden;
      background: var(--paper);
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
    }}

    .masthead {{
      display: flex;
      justify-content: space-between;
      gap: 1rem;
      padding: 1rem 1.25rem;
      border-bottom: 1px solid var(--line);
      background:
        linear-gradient(135deg, rgba(15, 118, 110, 0.08), rgba(255, 255, 255, 0)),
        rgba(255, 255, 255, 0.54);
    }}

    .masthead h1 {{
      margin: 0;
      font-size: 1.1rem;
      font-weight: 600;
      letter-spacing: 0.02em;
    }}

    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      align-items: center;
      color: var(--muted);
      font-size: 0.88rem;
    }}

    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      padding: 0.3rem 0.65rem;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.7);
    }}

    .workspace {{
      display: grid;
      grid-template-rows: 1fr auto;
      min-height: calc(100vh - 4rem);
      max-height: calc(100vh - 4rem);
    }}

    #messages {{
      overflow-y: auto;
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 0.95rem;
      scroll-behavior: smooth;
    }}

    .welcome {{
      padding: 1.4rem;
      border: 1px dashed var(--line);
      border-radius: 22px;
      color: var(--muted);
      background: rgba(255, 255, 255, 0.65);
    }}

    .bubble {{
      max-width: min(52rem, 92%);
      padding: 0.95rem 1rem;
      border-radius: 20px;
      line-height: 1.58;
      white-space: pre-wrap;
      word-break: break-word;
      border: 1px solid var(--line);
      animation: rise 160ms ease-out;
    }}

    .bubble.user {{
      align-self: flex-end;
      background: var(--user);
      border-bottom-right-radius: 6px;
    }}

    .bubble.assistant {{
      align-self: flex-start;
      background: var(--assistant);
      border-bottom-left-radius: 6px;
    }}

    .bubble.error {{
      align-self: center;
      background: #fff0ec;
      border-color: rgba(153, 27, 27, 0.18);
      color: #8b2f1a;
    }}

    @keyframes rise {{
      from {{ opacity: 0; transform: translateY(8px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}

    .composer {{
      padding: 1rem 1.1rem 1.1rem;
      border-top: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.82);
    }}

    .controls {{
      display: flex;
      gap: 0.75rem;
      align-items: end;
      flex-wrap: wrap;
    }}

    .input-wrap {{
      flex: 1 1 36rem;
      display: grid;
      gap: 0.55rem;
    }}

    textarea {{
      width: 100%;
      min-height: 6.5rem;
      resize: vertical;
      padding: 0.9rem 1rem;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.95);
      color: var(--ink);
      font: inherit;
      line-height: 1.45;
    }}

    textarea:focus {{
      outline: 2px solid rgba(15, 118, 110, 0.16);
      border-color: rgba(15, 118, 110, 0.35);
    }}

    .side-controls {{
      display: grid;
      gap: 0.55rem;
      min-width: 11rem;
    }}

    .row {{
      display: flex;
      gap: 0.6rem;
    }}

    label {{
      display: grid;
      gap: 0.35rem;
      color: var(--muted);
      font-size: 0.82rem;
    }}

    input[type="number"] {{
      width: 100%;
      padding: 0.6rem 0.7rem;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.95);
      font: inherit;
    }}

    button {{
      border: 0;
      border-radius: 999px;
      padding: 0.82rem 1rem;
      font: inherit;
      cursor: pointer;
      transition: transform 120ms ease, opacity 120ms ease, background 120ms ease;
    }}

    button:hover {{ transform: translateY(-1px); }}
    button:disabled {{ cursor: not-allowed; opacity: 0.6; transform: none; }}

    .primary {{
      background: var(--accent);
      color: white;
    }}

    .primary:hover {{ background: var(--accent-strong); }}

    .ghost {{
      background: rgba(255, 255, 255, 0.8);
      color: var(--ink);
      border: 1px solid var(--line);
    }}

    .footer-note {{
      margin-top: 0.55rem;
      color: var(--muted);
      font-size: 0.78rem;
    }}

    @media (max-width: 820px) {{
      .shell {{
        width: 100vw;
        margin: 0;
        border-radius: 0;
        border-left: 0;
        border-right: 0;
      }}

      .workspace {{
        min-height: 100vh;
        max-height: 100vh;
      }}

      .masthead {{
        flex-direction: column;
        align-items: start;
      }}

      .side-controls {{
        width: 100%;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="masthead">
      <div>
        <h1>b12x chat</h1>
        <div class="meta">
          <span class="pill">Model: <strong>{safe_model_name}</strong></span>
          <span class="pill" id="status-pill">Ready</span>
        </div>
      </div>
      <div class="meta">
        <button class="ghost" id="clear-chat" type="button">Clear Chat</button>
      </div>
    </div>
    <div class="workspace">
      <div id="messages">
        <div class="welcome">
          This is a small built-in chat UI for the existing `/v1/chat/completions` API.
          It keeps conversation state in the page and streams tokens from the server.
        </div>
      </div>
      <form class="composer" id="chat-form">
        <div class="controls">
          <div class="input-wrap">
            <label for="prompt">Message</label>
            <textarea id="prompt" name="prompt" placeholder="Ask something..." required></textarea>
          </div>
          <div class="side-controls">
            <div class="row">
              <label>
                Temperature
                <input id="temperature" type="number" min="0" max="2" step="0.1" value="0.7">
              </label>
              <label>
                Max Tokens
                <input id="max-tokens" type="number" min="1" max="4096" step="1" value="256">
              </label>
            </div>
            <div class="row">
              <button class="primary" id="send" type="submit">Send</button>
              <button class="ghost" id="stop" type="button" disabled>Stop</button>
            </div>
          </div>
        </div>
        <div class="footer-note">No build step, no npm. This page talks directly to the local API.</div>
      </form>
    </div>
  </div>

  <script>
    const config = {initial_config};
    const messagesEl = document.getElementById("messages");
    const promptEl = document.getElementById("prompt");
    const formEl = document.getElementById("chat-form");
    const statusEl = document.getElementById("status-pill");
    const stopEl = document.getElementById("stop");
    const sendEl = document.getElementById("send");
    const clearEl = document.getElementById("clear-chat");
    const temperatureEl = document.getElementById("temperature");
    const maxTokensEl = document.getElementById("max-tokens");

    const conversation = [];
    let activeController = null;

    function setStatus(text) {{
      statusEl.textContent = text;
    }}

    function scrollToBottom() {{
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }}

    function addBubble(role, text, extraClass = "") {{
      const bubble = document.createElement("div");
      bubble.className = ["bubble", role, extraClass].filter(Boolean).join(" ");
      bubble.textContent = text;
      messagesEl.appendChild(bubble);
      scrollToBottom();
      return bubble;
    }}

    function resetConversation() {{
      conversation.length = 0;
      messagesEl.innerHTML = "";
      const welcome = document.createElement("div");
      welcome.className = "welcome";
      welcome.textContent =
        "This is a small built-in chat UI for the existing /v1/chat/completions API. " +
        "It keeps conversation state in the page and streams tokens from the server.";
      messagesEl.appendChild(welcome);
      setStatus("Ready");
    }}

    function setBusy(isBusy) {{
      promptEl.disabled = isBusy;
      sendEl.disabled = isBusy;
      stopEl.disabled = !isBusy;
      temperatureEl.disabled = isBusy;
      maxTokensEl.disabled = isBusy;
    }}

    async function streamChat(messages, assistantBubble) {{
      activeController = new AbortController();
      const response = await fetch("/v1/chat/completions", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify({{
          model: config.model,
          messages,
          stream: true,
          temperature: Number(temperatureEl.value),
          max_tokens: Number(maxTokensEl.value),
        }}),
        signal: activeController.signal,
      }});

      if (!response.ok || !response.body) {{
        let errorText = `HTTP ${{response.status}}`;
        try {{
          const payload = await response.json();
          errorText = payload.detail || payload.error?.message || errorText;
        }} catch (_err) {{}}
        throw new Error(errorText);
      }}

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let fullText = "";

      while (true) {{
        const {{ value, done }} = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), {{ stream: !done }});

        while (true) {{
          const splitAt = buffer.indexOf("\\n\\n");
          if (splitAt === -1) {{
            break;
          }}
          const frame = buffer.slice(0, splitAt);
          buffer = buffer.slice(splitAt + 2);
          if (!frame.startsWith("data: ")) {{
            continue;
          }}
          const data = frame.slice(6).trim();
          if (data === "[DONE]") {{
            return fullText;
          }}
          const payload = JSON.parse(data);
          const delta = payload.choices?.[0]?.delta?.content || "";
          if (delta) {{
            fullText += delta;
            assistantBubble.textContent = fullText;
            scrollToBottom();
          }}
        }}

        if (done) {{
          return fullText;
        }}
      }}
    }}

    formEl.addEventListener("submit", async (event) => {{
      event.preventDefault();
      const prompt = promptEl.value.trim();
      if (!prompt) {{
        return;
      }}

      if (messagesEl.querySelector(".welcome")) {{
        messagesEl.innerHTML = "";
      }}

      conversation.push({{ role: "user", content: prompt }});
      addBubble("user", prompt);
      const assistantBubble = addBubble("assistant", "");
      promptEl.value = "";
      setBusy(true);
      setStatus("Streaming");

      try {{
        const text = await streamChat(conversation, assistantBubble);
        conversation.push({{ role: "assistant", content: text }});
        setStatus("Ready");
      }} catch (error) {{
        assistantBubble.remove();
        addBubble("error", error.name === "AbortError" ? "Generation stopped." : String(error.message), "error");
        setStatus(error.name === "AbortError" ? "Stopped" : "Error");
      }} finally {{
        activeController = null;
        setBusy(false);
        promptEl.focus();
      }}
    }});

    stopEl.addEventListener("click", () => {{
      if (activeController) {{
        activeController.abort();
      }}
    }});

    clearEl.addEventListener("click", () => {{
      if (activeController) {{
        activeController.abort();
      }}
      resetConversation();
      promptEl.focus();
    }});

    promptEl.focus();
  </script>
</body>
</html>
"""
