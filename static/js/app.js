/**
 * Frontend Controller for Smart Campus Assistant (RAG + Ollama)
 * Course: CSC-320(L) – Artificial Intelligence | Section B | Group 1
 */

document.addEventListener("DOMContentLoaded", () => {
  // 1. Tab Navigation
  const navTabs = document.querySelectorAll(".nav-tab");
  const tabPanels = document.querySelectorAll(".tab-panel");

  function switchTab(targetId) {
    navTabs.forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.tab === targetId);
    });
    tabPanels.forEach((panel) => {
      panel.classList.toggle("active", panel.id === targetId);
    });
    if (targetId === "tab-kb") {
      loadKnowledgeBase();
    }
  }

  navTabs.forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });

  // 2. Chat & RAG Inspector Elements
  const askForm = document.getElementById("askForm");
  const questionInput = document.getElementById("questionInput");
  const askSubmitBtn = document.getElementById("askSubmitBtn");
  const askBtnText = document.getElementById("askBtnText");
  const chatMessages = document.getElementById("chatMessages");
  const clearChatBtn = document.getElementById("clearChatBtn");

  const llmModelSelect = document.getElementById("llmModelSelect");
  const embedModelSelect = document.getElementById("embedModelSelect");
  const topKSelect = document.getElementById("topKSelect");
  const thresholdInput = document.getElementById("thresholdInput");

  function escapeHtml(str) {
    if (str === null || str === undefined) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function appendUserMessage(text) {
    const div = document.createElement("div");
    div.className = "msg user";
    div.innerHTML = `
      <div class="msg-avatar">YOU</div>
      <div class="msg-bubble">${escapeHtml(text)}</div>
    `;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function appendAssistantMessage(data) {
    const div = document.createElement("div");
    div.className = "msg assistant";

    const isAbstained = Boolean(data.abstained);
    const bubbleClass = isAbstained ? "abstained" : "grounded";
    const statusBadge = isAbstained
      ? `<span class="badge badge-abstained">⚠ Abstained (Outside KB Coverage)</span>`
      : `<span class="badge badge-grounded">✓ Grounded in Verified KB</span>`;

    const sourcesHtml = (data.sources || [])
      .map(
        (s) =>
          `<span class="badge badge-source" title="${escapeHtml(s.source)}">${escapeHtml(s.id)}: ${escapeHtml(s.title)} (cos=${s.cosine_similarity})</span>`
      )
      .join(" ");

    const totalMs = data.timing_ms ? data.timing_ms.total_ms : 0;
    const formattedAnswer = escapeHtml(data.answer).replace(/\n/g, "<br />");

    div.innerHTML = `
      <div class="msg-avatar">AI</div>
      <div class="msg-bubble ${bubbleClass}">
        <div>${formattedAnswer}</div>
        <div class="msg-meta-footer">
          ${statusBadge}
          ${sourcesHtml}
          <span>• Max Cosine: <strong>${data.max_cosine_similarity}</strong></span>
          <span>• ${totalMs} ms (${escapeHtml(data.llm_model)})</span>
        </div>
      </div>
    `;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function updateInspector(data) {
    const nlp = data.nlp || {};
    const timing = data.timing_ms || {};
    const passages = data.retrieved_passages || [];

    document.getElementById("inspectorStatusBadge").textContent = data.abstained
      ? "Abstained (Unsupported Query)"
      : "Grounded Answer Generated";
    document.getElementById("inspectorStatusBadge").className = data.abstained
      ? "badge badge-abstained"
      : "badge badge-grounded";

    // Step 1: NLP
    document.getElementById("nlpTimeBadge").textContent = `${timing.nlp_ms || 0} ms`;
    document.getElementById("inspCleanedQuery").textContent = nlp.cleaned_question || "—";
    document.getElementById("inspExpandedQuery").textContent = nlp.expanded_question || "—";
    document.getElementById("inspKeywords").textContent =
      (nlp.keywords || []).join(", ") || "None";
    document.getElementById("inspCategory").textContent =
      nlp.predicted_category || "General";

    // Step 2: Retrieval
    document.getElementById("retTimeBadge").textContent = `${
      (timing.embedding_ms || 0) + (timing.retrieval_ms || 0)
    } ms (Embed: ${timing.embedding_ms || 0} ms)`;
    document.getElementById("inspEmbedModel").textContent = data.embed_model || "—";
    document.getElementById("inspMaxCosine").textContent = `${data.max_cosine_similarity} (Combined: ${data.max_combined_score})`;
    document.getElementById("inspRelevanceStatus").innerHTML = data.abstained
      ? `<span style="color: var(--accent-amber);">Below Threshold / Out-of-Scope (${escapeHtml(data.abstention_reason || "")})</span>`
      : `<span style="color: var(--accent-emerald);">Relevant Passages Retrieved</span>`;

    const listEl = document.getElementById("inspPassagesList");
    if (!passages.length) {
      listEl.innerHTML = `<div style="font-size:0.8rem; color:var(--text-muted);">No passages retrieved.</div>`;
    } else {
      listEl.innerHTML = passages
        .map((p, i) => {
          const pct = Math.max(5, Math.min(100, Math.round(p.cosine_similarity * 100)));
          return `
            <div class="retrieved-passage-card">
              <div class="passage-top-row">
                <span class="passage-title">#${i + 1} [${escapeHtml(p.id)}] ${escapeHtml(p.title)}</span>
                <span class="badge badge-source">Cosine: ${p.cosine_similarity}</span>
              </div>
              <div class="sim-bar-wrap">
                <div class="sim-bar-fill" style="width: ${pct}%;"></div>
              </div>
              <div class="passage-snippet">${escapeHtml(p.content)}</div>
              <div class="passage-source-line">📄 Source: ${escapeHtml(p.source)}</div>
            </div>
          `;
        })
        .join("");
    }

    // Step 3: LLM
    document.getElementById("llmTimeBadge").textContent = `${timing.llm_ms || 0} ms`;
    document.getElementById("inspLlmModel").textContent = data.llm_model || "—";
    document.getElementById("inspGenMode").textContent = data.abstained
      ? "Abstention Guard Triggered (No Hallucination)"
      : "Strict Context-Grounded RAG Synthesis";
    document.getElementById("inspTotalTime").textContent = `${timing.total_ms || 0} ms`;
  }

  async function submitQuestion(questionText) {
    const q = (questionText || "").trim();
    if (!q) return;

    switchTab("tab-assistant");
    appendUserMessage(q);
    questionInput.value = "";
    askSubmitBtn.disabled = true;
    askBtnText.textContent = "Querying Ollama...";

    try {
      const resp = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: q,
          llm_model: llmModelSelect.value,
          embed_model: embedModelSelect.value,
          top_k: parseInt(topKSelect.value, 10) || 3,
          similarity_threshold: parseFloat(thresholdInput.value) || 0.36,
        }),
      });
      const data = await resp.json();
      if (!resp.ok || data.error) {
        appendAssistantMessage({
          answer: data.error || "Error communicating with local Ollama service.",
          abstained: true,
          sources: [],
          max_cosine_similarity: 0,
          llm_model: llmModelSelect.value,
          timing_ms: { total_ms: 0 },
        });
      } else {
        appendAssistantMessage(data);
        updateInspector(data);
      }
    } catch (err) {
      appendAssistantMessage({
        answer: `Network error: ${err.message}. Please ensure Flask and Ollama are running.`,
        abstained: true,
        sources: [],
        max_cosine_similarity: 0,
        llm_model: llmModelSelect.value,
        timing_ms: { total_ms: 0 },
      });
    } finally {
      askSubmitBtn.disabled = false;
      askBtnText.textContent = "Ask Assistant";
    }
  }

  askForm.addEventListener("submit", (e) => {
    e.preventDefault();
    submitQuestion(questionInput.value);
  });

  // Sample Prompt Chips
  document.querySelectorAll(".sample-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      submitQuestion(chip.dataset.q);
    });
  });

  // Clear Chat
  clearChatBtn.addEventListener("click", () => {
    chatMessages.innerHTML = `
      <div class="msg assistant">
        <div class="msg-avatar">AI</div>
        <div class="msg-bubble grounded">
          Chat history cleared. Ask any question about campus facilities, admissions, fees, library rules, office timings, or exams!
        </div>
      </div>
    `;
  });

  // 3. Knowledge Base Tab Logic
  const kbSearchInput = document.getElementById("kbSearchInput");
  const kbCategoryFilter = document.getElementById("kbCategoryFilter");
  const kbCardsContainer = document.getElementById("kbCardsContainer");
  const addKbForm = document.getElementById("addKbForm");
  const addKbFeedback = document.getElementById("addKbFeedback");

  async function loadKnowledgeBase() {
    const search = kbSearchInput.value.trim();
    const category = kbCategoryFilter.value;
    const params = new URLSearchParams({ search, category });

    const resp = await fetch(`/api/knowledge-base?${params.toString()}`);
    const data = await resp.json();
    const entries = data.entries || [];

    if (data.stats) {
      document.getElementById("headerKbCount").textContent = data.stats.total_entries;
      document.getElementById("navKbBadge").textContent = data.stats.total_entries;
      document.getElementById("kbStatTotal").textContent = data.stats.total_entries;
    }

    kbCardsContainer.innerHTML = entries
      .map(
        (e) => `
        <div class="kb-item-card">
          <div>
            <div class="kb-item-header">
              <span class="kb-id-pill">${escapeHtml(e.id)}</span>
              <span class="kb-cat-pill">${escapeHtml(e.category)}</span>
            </div>
            <div class="kb-item-title" style="margin-top: 0.45rem;">${escapeHtml(e.title)}</div>
            <div class="kb-item-content" style="margin-top: 0.4rem;">${escapeHtml(e.content)}</div>
          </div>
          <div class="kb-item-footer">
            <span>📄 ${escapeHtml(e.source)}</span>
            <button type="button" class="btn-secondary kb-ask-btn" data-q="Tell me about ${escapeHtml(e.title)}">Ask</button>
          </div>
        </div>
      `
      )
      .join("");

    document.querySelectorAll(".kb-ask-btn").forEach((btn) => {
      btn.addEventListener("click", () => submitQuestion(btn.dataset.q));
    });
  }

  kbSearchInput.addEventListener("input", loadKnowledgeBase);
  kbCategoryFilter.addEventListener("change", loadKnowledgeBase);

  addKbForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    addKbFeedback.textContent = "Embedding and saving passage via Ollama...";
    const payload = {
      title: document.getElementById("newKbTitle").value,
      category: document.getElementById("newKbCategory").value,
      content: document.getElementById("newKbContent").value,
      source: document.getElementById("newKbSource").value,
      keywords: document.getElementById("newKbKeywords").value,
    };

    const resp = await fetch("/api/knowledge-base", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    if (resp.ok) {
      addKbFeedback.textContent = `✓ ${data.message}`;
      addKbForm.reset();
      loadKnowledgeBase();
    } else {
      addKbFeedback.textContent = `Error: ${data.error || "Could not add entry"}`;
    }
  });

  // 4. Evaluation Benchmark Tab Logic
  const runEvalBtn = document.getElementById("runEvalBtn");
  const runEvalBtnText = document.getElementById("runEvalBtnText");
  const evalTableBody = document.getElementById("evalTableBody");

  function bindTestInChatButtons() {
    document.querySelectorAll(".test-in-chat-btn").forEach((btn) => {
      btn.addEventListener("click", () => submitQuestion(btn.dataset.q));
    });
  }
  bindTestInChatButtons();

  runEvalBtn.addEventListener("click", async () => {
    runEvalBtn.disabled = true;
    runEvalBtnText.textContent = "⏳ Running 25 Questions on Local Ollama...";

    try {
      const resp = await fetch("/api/evaluation/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          top_k: parseInt(topKSelect.value, 10) || 3,
          similarity_threshold: parseFloat(thresholdInput.value) || 0.36,
        }),
      });
      const report = await resp.json();
      const s = report.summary || {};

      document.getElementById("evalHit3").textContent = `${s.retrieval_hit_at_3_rate}%`;
      document.getElementById("evalHit1").textContent = `${s.retrieval_hit_at_1_rate}%`;
      document.getElementById("evalAnswerAcc").textContent = `${s.in_scope_answer_accuracy}%`;
      document.getElementById("evalAbstainAcc").textContent = `${s.out_of_scope_abstention_accuracy}%`;
      document.getElementById("evalOverallAcc").textContent = `${s.overall_system_accuracy}%`;
      document.getElementById("evalAvgTime").textContent = `${s.avg_total_time_ms} ms`;
      document.getElementById("evalEmbedTime").textContent = s.avg_embedding_time_ms;
      document.getElementById("evalLlmTime").textContent = s.avg_llm_time_ms;

      evalTableBody.innerHTML = (report.results || [])
        .map((r) => {
          const expectedHtml = r.should_abstain
            ? `<span class="badge badge-abstained">Should Abstain</span>`
            : (r.expected_kb_ids || [])
                .map((id) => `<span class="badge badge-source">${escapeHtml(id)}</span>`)
                .join(" ");

          const retrievedHtml = (r.retrieved_kb_ids || [])
            .map((id) => `<span class="badge badge-source">${escapeHtml(id)}</span>`)
            .join(" ");

          let statusHtml = "";
          if (r.should_abstain) {
            statusHtml = r.abstained
              ? `<span class="badge badge-grounded">✓ Correctly Abstained</span>`
              : `<span class="badge badge-abstained">✗ False Answer</span>`;
          } else {
            statusHtml = r.answer_correct
              ? `<span class="badge badge-grounded">✓ Grounded & Accurate</span>`
              : `<span class="badge badge-abstained">⚠ Partial / Check</span>`;
          }

          return `
            <tr>
              <td><strong>${escapeHtml(r.id)}</strong></td>
              <td>
                <div style="font-weight: 600; color: var(--text-primary);">${escapeHtml(r.question)}</div>
                <div style="font-size: 0.74rem; color: var(--text-muted); margin-top: 0.2rem;">${escapeHtml(r.category)}</div>
                <div style="font-size: 0.76rem; color: var(--text-secondary); margin-top: 0.35rem; background: rgba(15,23,42,0.6); padding: 0.4rem 0.6rem; border-radius: 6px;">
                  ${escapeHtml(r.answer)}
                </div>
              </td>
              <td>${expectedHtml}</td>
              <td>${retrievedHtml}</td>
              <td><strong>${r.max_cosine_similarity}</strong></td>
              <td>${statusHtml}</td>
              <td>${r.timing_ms ? r.timing_ms.total_ms : 0} ms</td>
              <td>
                <button type="button" class="btn-secondary test-in-chat-btn" data-q="${escapeHtml(r.question)}">Test in Chat</button>
              </td>
            </tr>
          `;
        })
        .join("");

      bindTestInChatButtons();
    } finally {
      runEvalBtn.disabled = false;
      runEvalBtnText.textContent = "▶ Re-Run All 25 Test Questions on Ollama";
    }
  });
});
