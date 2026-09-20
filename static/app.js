(() => {
  const $ = (sel) => document.querySelector(sel);
  const els = {
    banner: $("#banner"), dropzone: $("#dropzone"), fileInput: $("#file-input"), dropError: $("#drop-error"),
    loading: $("#loading"), doc: $("#doc"), fileName: $("#file-name"), summary: $("#summary"),
    chunks: $("#chunks"), replace: $("#replace"), messages: $("#messages"), empty: $("#empty"),
    suggest: $("#suggest"), form: $("#composer"), question: $("#question"), send: $("#send"), role: $("#role"),
  };

  const state = { sessionId: null, busy: false, history: [] };
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const isDesktop = () => window.matchMedia("(min-width: 960px)").matches;

  const BASE_QUESTIONS = [
    "What technical skills does the candidate have?",
    "Which programming languages does the candidate know?",
    "Summarize the candidate's professional profile.",
    "What are the candidate's key qualifications and experiences?",
  ];
  const FIT_QUESTION = "Is the candidate a good fit for this role?";

  // ---------- Small helpers ----------
  const fmt = (n) => Number(n).toLocaleString();

  function renderMarkdown(src) {
    const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    const inline = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/`([^`]+)`/g, "<code>$1</code>");
    let html = "", list = null, m;
    const close = () => { if (list) { html += `</${list}>`; list = null; } };
    for (const raw of src.split(/\r?\n/)) {
      const line = raw.trim();
      if (!line) { close(); continue; }
      if ((m = line.match(/^[-*•]\s+(.*)/))) {
        if (list !== "ul") { close(); html += "<ul>"; list = "ul"; }
        html += `<li>${inline(m[1])}</li>`;
      } else if ((m = line.match(/^\d+[.)]\s+(.*)/))) {
        if (list !== "ol") { close(); html += "<ol>"; list = "ol"; }
        html += `<li>${inline(m[1])}</li>`;
      } else if ((m = line.match(/^#{1,4}\s+(.*)/))) {
        close(); html += `<h4>${inline(m[1])}</h4>`;
      } else {
        close(); html += `<p>${inline(line)}</p>`;
      }
    }
    close();
    return html;
  }

  function errorText(data, fallback) {
    return data && typeof data.detail === "string" ? data.detail : fallback;
  }

  // ---------- Suggested questions ----------
  function renderSuggestions() {
    const list = [...BASE_QUESTIONS];
    if (els.role.value.trim()) list.push(FIT_QUESTION);
    els.suggest.innerHTML = "";
    for (const q of list) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "chip";
      b.textContent = q;
      b.disabled = !state.sessionId || state.busy;
      b.addEventListener("click", () => ask(q));
      els.suggest.appendChild(b);
    }
  }

  function syncControls() {
    const ready = !!state.sessionId && !state.busy;
    els.question.disabled = !state.sessionId;
    els.send.disabled = !ready;
    els.suggest.querySelectorAll(".chip").forEach((c) => (c.disabled = !ready));
  }

  // ---------- Upload ----------
  function showDropError(msg) {
    els.dropError.textContent = msg;
    els.dropError.hidden = !msg;
  }

  async function upload(file) {
    showDropError("");
    if (!file) return;
    if (!/\.pdf$/i.test(file.name) && file.type !== "application/pdf") {
      showDropError("Please choose a PDF file.");
      return;
    }
    if (file.size > 5 * 1024 * 1024) {
      showDropError("That file is larger than 5 MB. Choose a smaller PDF.");
      return;
    }

    els.dropzone.hidden = true;
    els.doc.hidden = true;
    els.loading.hidden = false;

    const body = new FormData();
    body.append("file", file);
    try {
      const res = await fetch("/api/upload", { method: "POST", body });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(errorText(data, "The upload failed. Try again."));
      loadResume(data);
    } catch (err) {
      els.loading.hidden = true;
      els.dropzone.hidden = false;
      showDropError(err.message || "The upload failed. Try again.");
    }
  }

  function loadResume(data) {
    if (state.sessionId) fetch(`/api/session/${state.sessionId}`, { method: "DELETE" }).catch(() => {});
    state.sessionId = data.session_id;
    state.history = [];

    els.fileName.textContent = data.filename;
    els.summary.textContent =
      `Read ${data.pages} ${data.pages === 1 ? "page" : "pages"} (${fmt(data.characters)} characters), ` +
      `split it into ${data.chunks.length} chunks, and stored each as a ${data.embedding_dim}-dimension vector in a FAISS index.`;

    els.chunks.innerHTML = "";
    data.chunks.forEach((text, i) => {
      const li = document.createElement("li");
      li.className = "chunk";
      li.id = `chunk-${i}`;
      li.innerHTML = `<div class="chunk-meta"><span>Chunk ${i + 1}</span><span class="chunk-used">Used in the latest answer</span></div>`;
      const span = document.createElement("span");
      span.className = "chunk-text";
      span.textContent = text;
      li.appendChild(span);
      els.chunks.appendChild(li);
    });

    els.messages.innerHTML = "";
    els.messages.appendChild(makeAssistantNote("Resume ready. Ask a question below or pick a suggestion."));
    els.loading.hidden = true;
    els.doc.hidden = false;
    renderSuggestions();
    syncControls();
    els.question.focus({ preventScroll: true });
  }

  function makeAssistantNote(text) {
    const d = document.createElement("div");
    d.className = "empty";
    d.textContent = text;
    return d;
  }

  function expireSession() {
    state.sessionId = null;
    state.history = [];
    els.doc.hidden = true;
    els.dropzone.hidden = false;
    showDropError("This session expired. Upload the resume again.");
    syncControls();
    renderSuggestions();
  }

  // ---------- Highlighting retrieved chunks ----------
  function highlight(ids, scroll) {
    els.chunks.querySelectorAll(".chunk.hit").forEach((el) => el.classList.remove("hit"));
    ids.forEach((i) => $(`#chunk-${i}`)?.classList.add("hit"));
    if (scroll && ids.length && isDesktop()) {
      $(`#chunk-${Math.min(...ids)}`)?.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "center" });
    }
  }

  // ---------- Chat ----------
  function scrollChat() { els.messages.scrollTop = els.messages.scrollHeight; }

  function addUser(text) {
    const d = document.createElement("div");
    d.className = "msg user";
    d.textContent = text;
    els.messages.appendChild(d);
    scrollChat();
  }

  function addTyping() {
    const d = document.createElement("div");
    d.className = "msg assistant";
    d.innerHTML = '<span class="typing" role="status" aria-label="Thinking"><i></i><i></i><i></i></span>';
    els.messages.appendChild(d);
    scrollChat();
    return d;
  }

  function addError(text) {
    const d = document.createElement("div");
    d.className = "msg error";
    d.setAttribute("role", "alert");
    d.textContent = text;
    els.messages.appendChild(d);
    scrollChat();
  }

  function addAnswer(data) {
    const ids = data.retrieved.map((r) => r.id);
    const d = document.createElement("div");
    d.className = "msg assistant";
    d.innerHTML = renderMarkdown(data.answer);
    if (ids.length) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "trace";
      b.textContent = `Show where this came from (chunks ${ids.map((i) => i + 1).join(", ")})`;
      b.addEventListener("click", () => highlight(ids, true));
      d.appendChild(b);
    }
    els.messages.appendChild(d);
    highlight(ids, true);
    scrollChat();
  }

  async function ask(question) {
    const q = (question || "").trim();
    if (!q || !state.sessionId || state.busy) return;
    state.busy = true;
    syncControls();
    addUser(q);
    els.question.value = "";
    autoGrow();
    const typing = addTyping();

    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: state.sessionId,
          question: q,
          job_role: els.role.value.trim(),
          history: state.history.slice(-6),
        }),
      });
      const data = await res.json().catch(() => ({}));
      typing.remove();
      if (res.status === 404) { expireSession(); return; }
      if (!res.ok) { addError(errorText(data, "Something went wrong. Try again.")); return; }
      addAnswer(data);
      state.history.push({ role: "user", content: q }, { role: "assistant", content: data.answer });
    } catch {
      typing.remove();
      addError("Could not reach the server. Check your connection and try again.");
    } finally {
      state.busy = false;
      syncControls();
    }
  }

  function autoGrow() {
    els.question.style.height = "auto";
    els.question.style.height = Math.min(els.question.scrollHeight, 140) + "px";
  }

  // ---------- Events ----------
  els.dropzone.addEventListener("click", () => els.fileInput.click());
  els.dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); els.fileInput.click(); }
  });
  els.fileInput.addEventListener("change", () => { upload(els.fileInput.files[0]); els.fileInput.value = ""; });
  ["dragenter", "dragover"].forEach((t) =>
    els.dropzone.addEventListener(t, (e) => { e.preventDefault(); els.dropzone.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((t) =>
    els.dropzone.addEventListener(t, (e) => { e.preventDefault(); els.dropzone.classList.remove("drag"); }));
  els.dropzone.addEventListener("drop", (e) => upload(e.dataTransfer.files[0]));

  els.replace.addEventListener("click", () => els.fileInput.click());

  els.form.addEventListener("submit", (e) => { e.preventDefault(); ask(els.question.value); });
  els.question.addEventListener("input", autoGrow);
  els.question.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(els.question.value); }
  });
  els.role.addEventListener("input", renderSuggestions);

  // ---------- Startup ----------
  renderSuggestions();
  syncControls();
  fetch("/api/health")
    .then((r) => r.json())
    .then((h) => {
      if (!h.api_key_configured) {
        els.banner.textContent = "The server has no GEMINI_API_KEY set, so uploads will fail. Add the key and restart the server.";
        els.banner.hidden = false;
      }
    })
    .catch(() => {});
})();
