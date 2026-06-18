const state = {
  threadId: null,
  threads: [],
  busy: false,
};

const els = {
  threadList: document.querySelector("#threadList"),
  newThread: document.querySelector("#newThreadButton"),
  threadTitle: document.querySelector("#threadTitle"),
  messages: document.querySelector("#messages"),
  form: document.querySelector("#chatForm"),
  input: document.querySelector("#messageInput"),
  send: document.querySelector("#sendButton"),
  clearMemory: document.querySelector("#clearMemoryButton"),
  stm: document.querySelector("#stmList"),
  ltm: document.querySelector("#ltmList"),
  written: document.querySelector("#writtenList"),
};

const EMPTY = {
  stm: "No short-term memory yet.",
  ltm: "No long-term memory retrieved yet.",
  written: "No long-term memory written this turn.",
};

function setBusy(busy) {
  state.busy = busy;
  els.send.disabled = busy;
  els.newThread.disabled = busy;
  els.clearMemory.disabled = busy;
  els.input.disabled = busy;
}

function relativeTime(epochSeconds) {
  if (!epochSeconds) {
    return "";
  }
  const diff = Date.now() / 1000 - epochSeconds;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
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

function showTyping() {
  const node = document.createElement("article");
  node.className = "message ai typing";
  node.innerHTML =
    `<span class="message-label">🤖 Adviser</span>` +
    `<div class="typing-dots"><span></span><span></span><span></span></div>`;
  els.messages.append(node);
  els.messages.scrollTop = els.messages.scrollHeight;
  return node;
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

function clearMemoryPanels() {
  renderList(els.stm, [], EMPTY.stm);
  renderList(els.ltm, [], EMPTY.ltm);
  renderList(els.written, [], EMPTY.written);
}

function renderThreadList() {
  els.threadList.innerHTML = "";
  els.threadList.classList.toggle("empty", state.threads.length === 0);
  if (state.threads.length === 0) {
    const li = document.createElement("li");
    li.className = "thread-empty";
    li.textContent = "No conversations yet.";
    els.threadList.append(li);
    return;
  }
  for (const thread of state.threads) {
    const li = document.createElement("li");
    li.className = "thread-item";
    if (thread.thread_id === state.threadId) {
      li.classList.add("active");
    }
    li.innerHTML = `<span class="thread-title"></span><span class="thread-meta"></span>`;
    li.querySelector(".thread-title").textContent = thread.title || "New conversation";
    li.querySelector(".thread-meta").textContent = relativeTime(thread.last_active);
    li.addEventListener("click", () => {
      if (!state.busy && thread.thread_id !== state.threadId) {
        selectThread(thread.thread_id);
      }
    });
    els.threadList.append(li);
  }
}

function setActiveThread(threadId, title) {
  state.threadId = threadId;
  if (title !== undefined) {
    els.threadTitle.textContent = title || "New conversation";
  }
  renderThreadList();
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

async function loadThreads() {
  const payload = await api("/api/threads");
  state.threads = payload.threads || [];
  renderThreadList();
  return state.threads;
}

async function selectThread(threadId) {
  setBusy(true);
  try {
    const thread = state.threads.find((t) => t.thread_id === threadId);
    setActiveThread(threadId, thread ? thread.title : undefined);
    els.messages.innerHTML = "";
    clearMemoryPanels();

    const [history, memory] = await Promise.all([
      api(`/api/threads/${encodeURIComponent(threadId)}/messages`),
      api(`/api/threads/${encodeURIComponent(threadId)}/memory`),
    ]);
    for (const message of history.messages || []) {
      const isUser = message.role === "user";
      appendMessage(isUser ? "user" : "ai", isUser ? "👤 You" : "🤖 Adviser", message.text);
    }
    renderList(els.stm, memory.short_term_memory || [], EMPTY.stm);
  } catch (error) {
    appendMessage("system", "Error", error.message);
  } finally {
    setBusy(false);
    els.input.focus();
  }
}

async function createThread() {
  setBusy(true);
  try {
    const thread = await api("/api/threads", { method: "POST" });
    state.threads.unshift(thread);
    setActiveThread(thread.thread_id, thread.title);
    els.messages.innerHTML = "";
    clearMemoryPanels();
  } catch (error) {
    appendMessage("system", "Error", error.message);
  } finally {
    setBusy(false);
    els.input.focus();
  }
}

async function sendMessage(message) {
  setBusy(true);
  appendMessage("user", "👤 You", message);
  const typing = showTyping();
  try {
    const payload = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ thread_id: state.threadId, message }),
    });
    typing.remove();
    setActiveThread(payload.thread_id, payload.title);
    appendMessage("ai", "🤖 Adviser", payload.assistant_message);
    renderList(els.stm, payload.short_term_memory, EMPTY.stm);
    renderList(els.ltm, payload.long_term_memory, EMPTY.ltm);
    renderList(els.written, payload.extracted_long_term_memory, EMPTY.written);
    // Refresh the sidebar so the auto-generated title and ordering update.
    await loadThreads();
  } catch (error) {
    typing.remove();
    appendMessage("system", "Error", error.message);
  } finally {
    setBusy(false);
    els.input.focus();
  }
}

async function clearWorkingMemory() {
  if (!state.threadId) {
    return;
  }
  setBusy(true);
  try {
    await api(`/api/threads/${encodeURIComponent(state.threadId)}/memory`, { method: "DELETE" });
    renderList(els.stm, [], EMPTY.stm);
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

els.newThread.addEventListener("click", () => {
  if (!state.busy) {
    createThread();
  }
});

els.clearMemory.addEventListener("click", () => {
  if (!state.busy) {
    clearWorkingMemory();
  }
});

async function init() {
  try {
    const threads = await loadThreads();
    if (threads.length > 0) {
      await selectThread(threads[0].thread_id);
    } else {
      await createThread();
    }
  } catch (error) {
    appendMessage("system", "Error", error.message);
  }
}

init();
