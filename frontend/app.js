const state = {
  sessionId: null,
  busy: false,
};

const els = {
  sessionId: document.querySelector("#sessionId"),
  messages: document.querySelector("#messages"),
  form: document.querySelector("#chatForm"),
  input: document.querySelector("#messageInput"),
  send: document.querySelector("#sendButton"),
  newSession: document.querySelector("#newSessionButton"),
  deleteMemory: document.querySelector("#deleteMemoryButton"),
  stm: document.querySelector("#stmList"),
  ltm: document.querySelector("#ltmList"),
  written: document.querySelector("#writtenList"),
};

function setBusy(busy) {
  state.busy = busy;
  els.send.disabled = busy;
  els.newSession.disabled = busy;
  els.deleteMemory.disabled = busy;
  els.input.disabled = busy;
}

function setSession(sessionId) {
  state.sessionId = sessionId;
  els.sessionId.textContent = sessionId;
}

function appendMessage(role, label, text) {
  const node = document.createElement("article");
  node.className = `message ${role}`;
  node.innerHTML = `<span class="message-label"></span><div></div>`;
  node.querySelector(".message-label").textContent = label;
  node.querySelector("div").textContent = text;
  els.messages.append(node);
  els.messages.scrollTop = els.messages.scrollHeight;
}

function renderList(element, items, emptyText) {
  element.innerHTML = "";
  element.classList.toggle("empty", items.length === 0);
  const values = items.length ? items : [emptyText];
  for (const item of values) {
    const li = document.createElement("li");
    li.textContent = item;
    element.append(li);
  }
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || `Request failed: ${response.status}`);
  }
  return payload;
}

async function createSession() {
  const payload = await api("/api/sessions", { method: "POST" });
  setSession(payload.session_id);
  els.messages.innerHTML = "";
  renderList(els.stm, [], "No short-term memory yet.");
  renderList(els.ltm, [], "No long-term memory retrieved yet.");
  renderList(els.written, [], "No long-term memory written yet.");
}

async function sendMessage(message) {
  setBusy(true);
  appendMessage("user", "👤 You", message);
  try {
    const payload = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ session_id: state.sessionId, message }),
    });
    setSession(payload.session_id);
    appendMessage("ai", "🤖 AI", payload.assistant_message);
    renderList(els.stm, payload.short_term_memory, "No short-term memory yet.");
    renderList(els.ltm, payload.long_term_memory, "No long-term memory retrieved yet.");
    renderList(
      els.written,
      payload.extracted_long_term_memory,
      "No long-term memory written this turn."
    );
  } catch (error) {
    appendMessage("system", "Error", error.message);
  } finally {
    setBusy(false);
    els.input.focus();
  }
}

async function deleteSessionMemory() {
  if (!state.sessionId) {
    return;
  }
  setBusy(true);
  try {
    await api(`/api/sessions/${encodeURIComponent(state.sessionId)}/memory`, { method: "DELETE" });
    renderList(els.stm, [], "No short-term memory yet.");
  } catch (error) {
    appendMessage("system", "Error", error.message);
  } finally {
    setBusy(false);
  }
}

els.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = els.input.value.trim();
  if (!message || state.busy) {
    return;
  }
  els.input.value = "";
  sendMessage(message);
});

els.newSession.addEventListener("click", () => {
  if (!state.busy) {
    createSession().catch((error) => appendMessage("system", "Error", error.message));
  }
});

els.deleteMemory.addEventListener("click", () => {
  if (!state.busy) {
    deleteSessionMemory();
  }
});

createSession().catch((error) => appendMessage("system", "Error", error.message));
