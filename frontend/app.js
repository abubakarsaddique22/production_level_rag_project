/* Nexora Knowledge Assistant - frontend.
 * Plain JS, no build step. Served by FastAPI at /ui (same origin as the API).
 * Endpoints used: /v1/auth/login, /v1/chat, /v1/agent/chat,
 *                 /v1/sessions, /v1/sessions/{id}, /v1/feedback
 * All text from the server is inserted with textContent (never innerHTML),
 * so a malicious document or answer cannot inject HTML.
 */
(() => {
  "use strict";

  const API = ""; // same origin. If you serve this file elsewhere, set e.g. "http://localhost:8000"

  const $ = (sel) => document.querySelector(sel);
  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  };

  const state = {
    token: sessionStorage.getItem("token"),
    sessionId: sessionStorage.getItem("sessionId"),
    mode: sessionStorage.getItem("mode") || "rag", // "rag" | "agent"
    busy: false,
    sessions: [],
  };

  /* ---------------- helpers ---------------- */

  function toast(msg) {
    const t = $("#toast");
    t.textContent = msg;
    t.hidden = false;
    clearTimeout(toast._timer);
    toast._timer = setTimeout(() => (t.hidden = true), 4000);
  }

  function parseJwt(token) {
    try {
      const payload = token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/");
      return JSON.parse(atob(payload));
    } catch {
      return {};
    }
  }

  async function api(path, { method = "GET", body, auth = true } = {}) {
    const headers = { "Content-Type": "application/json" };
    if (auth && state.token) headers.Authorization = `Bearer ${state.token}`;

    let res;
    try {
      res = await fetch(API + path, {
        method,
        headers,
        body: body ? JSON.stringify(body) : undefined,
      });
    } catch {
      throw new Error("Cannot reach the server. Is the API running?");
    }

    let data = null;
    try { data = await res.json(); } catch { /* empty body */ }

    if (res.status === 401 && auth) {
      logout("Your session expired. Please sign in again.");
      throw new Error("unauthorized");
    }
    if (res.status === 429) throw new Error("Too many requests. Please wait a minute and try again.");
    if (res.status === 504) throw new Error("The agent took too long. Try again or switch to RAG mode.");
    if (!res.ok) {
      const detail = data && typeof data.detail === "string" ? data.detail : null;
      throw new Error(detail || `Request failed (${res.status})`);
    }
    return data;
  }

  /* ---------------- views ---------------- */

  function showLogin(message) {
    $("#app-view").hidden = true;
    $("#login-view").hidden = false;
    const err = $("#login-error");
    err.hidden = !message;
    err.textContent = message || "";
  }

  function showApp() {
    $("#login-view").hidden = true;
    $("#app-view").hidden = false;
    const claims = parseJwt(state.token);
    $("#user-badge").textContent = claims.role ? `role: ${claims.role}` : "signed in";
    setMode(state.mode);
    $("#question").focus();
  }

  function logout(message) {
    state.token = null;
    state.sessionId = null;
    state.sessions = [];
    sessionStorage.removeItem("token");
    sessionStorage.removeItem("sessionId");
    resetChat();
    renderSessions();
    closeSidebar();
    showLogin(message);
  }

  function setMode(mode) {
    state.mode = mode;
    sessionStorage.setItem("mode", mode);
    $("#mode-rag").classList.toggle("active", mode === "rag");
    $("#mode-agent").classList.toggle("active", mode === "agent");
  }

  const openSidebar = () => $("#sidebar").classList.add("open");
  const closeSidebar = () => $("#sidebar").classList.remove("open");

  /* ---------------- chat history sidebar ---------------- */

  function renderSessions() {
    const nav = $("#session-list");
    nav.replaceChildren();
    if (!state.sessions.length) {
      nav.appendChild(el("p", "muted small side-empty", "No chats yet."));
      return;
    }
    for (const s of state.sessions) {
      const btn = el("button", "session-item", s.title);
      btn.type = "button";
      btn.dataset.id = s.session_id;
      btn.title = `${s.title}\n${new Date(s.updated_at).toLocaleString()}`;
      if (s.session_id === state.sessionId) btn.classList.add("active");
      btn.addEventListener("click", () => openSession(s.session_id));
      nav.appendChild(btn);
    }
  }

  function markActiveSession() {
    for (const b of document.querySelectorAll(".session-item")) {
      b.classList.toggle("active", b.dataset.id === state.sessionId);
    }
  }

  async function loadSessions() {
    try {
      state.sessions = await api("/v1/sessions");
      renderSessions();
    } catch (e) {
      if (e.message !== "unauthorized") toast("Could not load chat history.");
    }
  }

  async function openSession(id) {
    closeSidebar();
    if (state.busy || id === state.sessionId) return;
    state.sessionId = id;
    sessionStorage.setItem("sessionId", id);
    resetChat();
    markActiveSession();
    await restoreSession();
    markActiveSession();
  }

  function newChat() {
    closeSidebar();
    if (state.busy) return;
    state.sessionId = null;
    sessionStorage.removeItem("sessionId");
    resetChat();
    markActiveSession();
    $("#question").focus();
  }

  /* ---------------- message rendering ---------------- */

  const messagesEl = () => $("#messages");

  function scrollDown() {
    const m = messagesEl();
    m.scrollTop = m.scrollHeight;
  }

  // Tiny, safe formatter: **bold**, [n] citation chips, "- " bullets, line breaks.
  function renderText(container, text, sourcesEl) {
    const lines = String(text).split(/\r?\n/);
    for (const raw of lines) {
      if (!raw.trim()) continue;
      const bullet = /^\s*([-*]|\d+\.)\s+/.test(raw);
      const line = el("div", bullet ? "line bullet" : "line");
      const content = bullet ? raw.replace(/^\s*([-*]|\d+\.)\s+/, "") : raw;
      for (const part of content.split(/(\*\*[^*]+\*\*|\[\d+\])/g)) {
        if (!part) continue;
        if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
          line.appendChild(el("strong", "", part.slice(2, -2)));
        } else if (/^\[\d+\]$/.test(part)) {
          const id = part.slice(1, -1);
          const chip = el("button", "cite", part);
          chip.type = "button";
          chip.title = "Show source";
          chip.addEventListener("click", () => {
            const target = sourcesEl && sourcesEl.querySelector(`[data-src="${id}"]`);
            if (!target) return;
            target.open = true;
            target.classList.add("flash");
            target.scrollIntoView({ behavior: "smooth", block: "nearest" });
            setTimeout(() => target.classList.remove("flash"), 1200);
          });
          line.appendChild(chip);
        } else {
          line.appendChild(document.createTextNode(part));
        }
      }
      container.appendChild(line);
    }
  }

  function renderSources(sources) {
    const wrap = el("div", "sources");
    for (const s of sources) {
      const d = el("details", "source");
      d.dataset.src = String(s.id);
      const sum = el("summary");
      sum.appendChild(el("span", "src-id", `[${s.id}]`));
      const title = s.title || s.doc_id || "Document";
      sum.appendChild(document.createTextNode(s.page ? `${title}, page ${s.page}` : title));
      d.appendChild(sum);
      if (s.snippet) d.appendChild(el("div", "snippet", s.snippet));
      wrap.appendChild(d);
    }
    return wrap;
  }

  function addUserMessage(text) {
    $("#empty").hidden = true;
    const msg = el("div", "msg user");
    msg.appendChild(el("div", "bubble", text));
    messagesEl().appendChild(msg);
    scrollDown();
  }

  function addAssistantMessage({ answer, sources = [], trace_id, latency_ms, route, grounded, tool_calls }) {
    $("#empty").hidden = true;
    const msg = el("div", "msg assistant");
    const bubble = el("div", "bubble");

    const sourcesEl = sources.length ? renderSources(sources) : null;
    renderText(bubble, answer, sourcesEl);
    msg.appendChild(bubble);
    if (sourcesEl) msg.appendChild(sourcesEl);

    const meta = el("div", "meta");
    if (latency_ms != null) meta.appendChild(el("span", "", `${(latency_ms / 1000).toFixed(1)} s`));
    if (route) {
      let info = `agent: ${route}`;
      if (grounded != null) info += grounded ? ", grounded" : ", not grounded";
      if (tool_calls) info += `, ${tool_calls} tool call${tool_calls > 1 ? "s" : ""}`;
      meta.appendChild(el("span", "", info));
    }
    if (trace_id) meta.appendChild(feedbackButtons(trace_id));
    if (meta.childNodes.length) msg.appendChild(meta);

    messagesEl().appendChild(msg);
    scrollDown();
  }

  function feedbackButtons(traceId) {
    const box = el("span");
    const up = el("button", "fb", "\u{1F44D}");
    const down = el("button", "fb", "\u{1F44E}");
    up.type = down.type = "button";
    up.title = "Helpful";
    down.title = "Not helpful";

    async function send(rating) {
      let comment = null;
      if (rating === -1) {
        comment = window.prompt("What was wrong with this answer? (optional)") || null;
      }
      try {
        await api("/v1/feedback", { method: "POST", body: { trace_id: traceId, rating, comment } });
        up.classList.toggle("active", rating === 1);
        down.classList.toggle("active", rating === -1);
        toast("Thanks for the feedback.");
      } catch (e) {
        if (e.message !== "unauthorized") toast(e.message);
      }
    }
    up.addEventListener("click", () => send(1));
    down.addEventListener("click", () => send(-1));
    box.append(up, " ", down);
    return box;
  }

  function showTyping() {
    const msg = el("div", "msg assistant");
    msg.id = "typing";
    const bubble = el("div", "bubble typing");
    bubble.append(el("span"), el("span"), el("span"));
    msg.appendChild(bubble);
    messagesEl().appendChild(msg);
    scrollDown();
  }
  const hideTyping = () => { const t = $("#typing"); if (t) t.remove(); };

  function resetChat() {
    const m = messagesEl();
    [...m.querySelectorAll(".msg")].forEach((n) => n.remove());
    $("#empty").hidden = false;
  }

  /* ---------------- chat flow ---------------- */

  async function ask(question) {
    if (state.busy || !question.trim()) return;
    state.busy = true;
    $("#send").disabled = true;

    addUserMessage(question);
    showTyping();

    try {
      const path = state.mode === "agent" ? "/v1/agent/chat" : "/v1/chat";
      const data = await api(path, {
        method: "POST",
        body: { question, session_id: state.sessionId },
      });
      state.sessionId = data.session_id;
      sessionStorage.setItem("sessionId", data.session_id);
      hideTyping();
      addAssistantMessage(data);
      loadSessions(); // new chat appears in the sidebar, updated chat moves to the top
    } catch (e) {
      hideTyping();
      if (e.message !== "unauthorized") {
        addAssistantMessage({ answer: e.message });
      }
    } finally {
      state.busy = false;
      $("#send").disabled = false;
      $("#question").focus();
    }
  }

  async function restoreSession() {
    if (!state.sessionId) return;
    try {
      const data = await api(`/v1/sessions/${encodeURIComponent(state.sessionId)}`);
      for (const m of data.messages) {
        if (m.role === "user") addUserMessage(m.content);
        else addAssistantMessage({ answer: m.content, sources: m.sources || [], trace_id: m.trace_id });
      }
    } catch (e) {
      if (e.message !== "unauthorized") {
        state.sessionId = null;
        sessionStorage.removeItem("sessionId");
      }
    }
  }

  /* ---------------- wiring ---------------- */

  function autoGrow(t) {
    t.style.height = "auto";
    t.style.height = Math.min(t.scrollHeight, 160) + "px";
  }

  function init() {
    // login
    $("#login-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const btn = $("#login-btn");
      btn.disabled = true;
      try {
        const data = await api("/v1/auth/login", {
          method: "POST",
          auth: false,
          body: { email: $("#email").value.trim(), password: $("#password").value },
        });
        state.token = data.access_token;
        sessionStorage.setItem("token", state.token);
        // a new login always starts on a fresh chat (history is in the sidebar)
        state.sessionId = null;
        sessionStorage.removeItem("sessionId");
        $("#password").value = "";
        showApp();
        resetChat();
        await loadSessions();
      } catch (err) {
        showLogin(err.message);
      } finally {
        btn.disabled = false;
      }
    });

    $("#demo-accounts").addEventListener("click", (e) => {
      const email = e.target.dataset && e.target.dataset.email;
      if (email) { $("#email").value = email; $("#password").focus(); }
    });

    // chat
    const q = $("#question");
    $("#chat-form").addEventListener("submit", (e) => {
      e.preventDefault();
      const text = q.value;
      q.value = "";
      autoGrow(q);
      ask(text);
    });
    q.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        $("#chat-form").requestSubmit();
      }
    });
    q.addEventListener("input", () => autoGrow(q));

    $("#examples").addEventListener("click", (e) => {
      if (e.target.classList.contains("chip")) ask(e.target.textContent);
    });

    // sidebar and toolbar
    $("#mode-rag").addEventListener("click", () => setMode("rag"));
    $("#mode-agent").addEventListener("click", () => setMode("agent"));
    $("#new-chat").addEventListener("click", newChat);
    $("#logout").addEventListener("click", () => logout());
    $("#menu-btn").addEventListener("click", () => $("#sidebar").classList.toggle("open"));

    // start
    if (state.token) {
      showApp();
      loadSessions();
      restoreSession();
    } else {
      showLogin();
    }
  }

  init();
})();
