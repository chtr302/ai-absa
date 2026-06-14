const CATEGORIES = [
  "PERFORMANCE",
  "INTELLIGENCE",
  "RESOURCES",
  "BEHAVIOR",
  "TECHNICAL",
  "SOFTWARE",
  "COMPARATIVE",
];

const SENTIMENTS = ["Positive", "Negative", "Neutral"];

const EXAMPLES = [
  {
    sentence: "I like it personally, Llama 3.1 8B is my go-to model for fine tuning. I may switch to Gemma 3 though based on what I'm reading.",
    results: {
      advanced: {
        model_type: "advanced",
        sentence: "I like it personally, Llama 3.1 8B is my go-to model for fine tuning. I may switch to Gemma 3 though based on what I'm reading.",
        quads: [
          {
            aspect: "Llama 3.1 8B",
            opinion: "go-to model for fine tuning",
            category: "RESOURCES",
            sentiment: "Positive",
          },
        ],
      },
    },
  },
  {
    sentence: "Claude 3.5 is great, but modernbert is faster.",
    results: {
      advanced: {
        model_type: "advanced",
        sentence: "Claude 3.5 is great, but modernbert is faster.",
        quads: [
          {
            aspect: "Claude 3.5",
            opinion: "great",
            category: "INTELLIGENCE",
            sentiment: "Positive",
          },
          {
            aspect: "Claude 3.5",
            opinion: "faster",
            category: "INTELLIGENCE",
            sentiment: "Positive",
          },
        ],
      },
    },
  },
  {
    sentence: "Sounds like we may have a decent coding model for the GPU poor. The old 30B A3B ran surprisingly well on CPU only.",
    results: {
      advanced: {
        model_type: "advanced",
        sentence: "Sounds like we may have a decent coding model for the GPU poor. The old 30B A3B ran surprisingly well on CPU only.",
        quads: [
          {
            aspect: "old",
            opinion: "ran surprisingly well on CPU only",
            category: "PERFORMANCE",
            sentiment: "Positive",
          },
          {
            aspect: "30B A3B",
            opinion: "ran surprisingly well on CPU only",
            category: "PERFORMANCE",
            sentiment: "Positive",
          },
        ],
      },
    },
  }
];

const EXAMPLE_PREDICTION = EXAMPLES[0];
let currentExampleIndex = 0;

const state = {
  currentPage: 1,
  pageSize: 20,
  search: "",
  category: "",
  sentiment: "",
  quadType: "all",
  benchmarkFocus: "category",
  benchmarkSort: "support",
  benchmarkRows: [],
  selectedBenchmarkKey: "",
  overview: null,
  modelComparison: null,
  categoryDistribution: [],
  sentimentDistribution: [],
  categoryMatrix: [],
  topAspects: [],
  factorFocus: { category: [], sentiment: [], aspect: [] },
};

const TAB_IDS = ["predict", "benchmark", "explorer", "dataset", "presentation"];
const LEGACY_TAB_MAP = {
  stats: "dataset",
  overview: "dataset",
  distribution: "dataset",
  matrix: "dataset",
};

async function apiGet(url) {
  const response = await fetch(url);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || `GET ${url} failed`);
  }
  return data;
}

async function apiPost(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || `POST ${url} failed`);
  }
  return data;
}

function setMessage(id, text, kind = "") {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = text || "";
  el.className = `message ${kind}`.trim();
}

function fmtNumber(value) {
  return Number(value || 0).toLocaleString();
}

function fmtMetric(value, digits = 3) {
  if (value === null || value === undefined) return "-";
  return Number(value).toFixed(digits);
}

function fmtPercent(count, total) {
  if (!total) return "0.0%";
  return `${((Number(count || 0) / total) * 100).toFixed(1)}%`;
}

function fmtSigned(value, suffix = "", digits = 3) {
  if (value === null || value === undefined) return "-";
  const number = Number(value);
  const sign = number > 0 ? "+" : "";
  return `${sign}${number.toFixed(digits)}${suffix}`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function loadOverview() {
  try {
    const data = await apiGet("/api/dataset/overview");
    state.overview = data;
    renderOverview(data);
    renderBenchmarkContext();
    setMessage("overviewMessage", "");
  } catch (error) {
    setMessage("overviewMessage", error.message, "error");
  }
}

async function loadCategoryDistribution() {
  try {
    const data = await apiGet("/api/dataset/category-distribution");
    state.categoryDistribution = data;
    renderBars("categoryBars", data, "category");
  } catch (error) {
    document.getElementById("categoryBars").innerHTML = `<p class="message error">${escapeHtml(error.message)}</p>`;
  }
}

async function loadSentimentDistribution() {
  try {
    const data = await apiGet("/api/dataset/sentiment-distribution");
    state.sentimentDistribution = data;
    renderBars("sentimentBars", data, "sentiment");
  } catch (error) {
    document.getElementById("sentimentBars").innerHTML = `<p class="message error">${escapeHtml(error.message)}</p>`;
  }
}

async function loadMatrix() {
  try {
    const data = await apiGet("/api/dataset/category-sentiment-matrix");
    state.categoryMatrix = data;
    renderMatrix(data);
    renderBenchmarkTable();
  } catch (error) {
    document.getElementById("matrixBody").innerHTML = `<tr><td colspan="5">${escapeHtml(error.message)}</td></tr>`;
    const benchmarkBody = document.getElementById("benchmarkMatrixBody");
    if (benchmarkBody) benchmarkBody.innerHTML = `<tr><td colspan="5">${escapeHtml(error.message)}</td></tr>`;
  }
}

async function loadTopAspects() {
  try {
    state.topAspects = await apiGet("/api/dataset/top-aspects?limit=12");
    renderTopAspects();
    renderBenchmarkTable();
  } catch (error) {
    const body = document.getElementById("topAspectBody");
    if (body) body.innerHTML = `<tr><td colspan="4">${escapeHtml(error.message)}</td></tr>`;
  }
}

async function loadFactorFocus() {
  try {
    state.factorFocus = await apiGet("/api/dataset/factor-focus?main_limit=36&related_limit=8");
    renderBenchmarkTable();
  } catch (error) {
    const benchmarkBody = document.getElementById("benchmarkMatrixBody");
    const benchmarkDetail = document.getElementById("benchmarkDetail");
    if (benchmarkBody) benchmarkBody.innerHTML = `<tr><td colspan="5">${escapeHtml(error.message)}</td></tr>`;
    if (benchmarkDetail) benchmarkDetail.innerHTML = `<p class="message error">${escapeHtml(error.message)}</p>`;
  }
}

async function loadSamples() {
  const params = new URLSearchParams({
    page: state.currentPage,
    page_size: state.pageSize,
    search: state.search,
    category: state.category,
    sentiment: state.sentiment,
    quad_type: state.quadType,
  });

  try {
    const data = await apiGet(`/api/dataset/samples?${params.toString()}`);
    renderSamples(data);
    setMessage("samplesMessage", `${fmtNumber(data.total)} matching samples`);
  } catch (error) {
    setMessage("samplesMessage", error.message, "error");
  }
}

async function loadModelComparison() {
  try {
    const data = await apiGet("/api/model-comparison");
    renderModelComparison(data);
  } catch (error) {
    document.getElementById("modelCards").innerHTML = `<div class="panel message error">${escapeHtml(error.message)}</div>`;
  }
}

function renderOverview(data) {
  const cards = [
    ["Total Sentences", data.total_sentences, "all"],
    ["Total Quads", data.total_quads, "with_quads"],
    ["With Quads", data.sentences_with_quads, "with_quads"],
    ["No Quad", data.sentences_without_quads, "no_quad"],
    ["Single Quad", data.single_quad_sentences, "single_quad"],
    ["Multi Quad", data.multi_quad_sentences, "multi_quad"],
    ["Avg Quad / Sentence", data.avg_quads_per_sentence, ""],
  ];

  document.getElementById("overviewCards").innerHTML = cards
    .map(([label, value, action]) => `
      <article class="overview-card" ${action ? `data-overview-action="${action}" role="button" tabindex="0"` : ""}>
        <div class="overview-label">${label}</div>
        <div class="overview-value">${label.startsWith("Avg") ? fmtMetric(value, 2) : typeof value === "number" ? fmtNumber(value) : escapeHtml(value)}</div>
      </article>
    `)
    .join("");
}

function renderBars(container, data, labelKey) {
  const el = document.getElementById(container);
  el.innerHTML = data
    .map((item) => {
      const label = item[labelKey];
      const percent = Math.max(0, Math.min(100, Number(item.percent || 0)));
      return `
        <div class="bar-row" data-filter-type="${escapeHtml(labelKey)}" data-filter-value="${escapeHtml(label)}">
          <div class="bar-top">
            <span class="bar-label">${escapeHtml(label)}</span>
            <span class="bar-meta">${fmtNumber(item.count)} (${percent.toFixed(2)}%)</span>
          </div>
          <div class="bar-track" aria-label="${escapeHtml(label)} ${percent.toFixed(2)}%">
            <div class="bar-fill" style="width: ${percent}%"></div>
          </div>
        </div>
      `;
    })
    .join("");
}

function renderMatrix(data) {
  document.getElementById("matrixBody").innerHTML = data
    .map((row) => `
      <tr data-matrix-category="${escapeHtml(row.category)}">
        <td><strong>${escapeHtml(row.category)}</strong></td>
        <td>${fmtNumber(row.Positive)}</td>
        <td>${fmtNumber(row.Negative)}</td>
        <td>${fmtNumber(row.Neutral)}</td>
        <td>${fmtNumber(row.total)}</td>
      </tr>
    `)
    .join("");
}

function renderQuad(quad) {
  return `
    <div class="quad-chip">
      <div class="quad-line"><strong>aspect</strong>: ${escapeHtml(quad.aspect || "-")}</div>
      <div class="quad-line"><strong>opinion</strong>: ${escapeHtml(quad.opinion || "-")}</div>
      <div class="prediction-meta">
        <span class="category-badge">${escapeHtml(quad.category || "UNKNOWN")}</span>
        <span class="sentiment-badge sentiment-${escapeHtml(quad.sentiment || "Neutral")}">${escapeHtml(quad.sentiment || "UNKNOWN")}</span>
      </div>
    </div>
  `;
}

function renderPredictionResult(label, result, statusText = "") {
  const quads = Array.isArray(result?.quads) ? result.quads : [];
  return `
    <article class="model-prediction">
      <div class="model-title-row">
        <h3>${escapeHtml(label)}</h3>
        <span class="badge">${fmtNumber(quads.length)} quads</span>
      </div>
      <div class="prediction-sentence">
        <strong>Sentence</strong>
        <div>${escapeHtml(result?.sentence || "")}</div>
        <div class="prediction-meta">
          ${statusText ? `<span class="badge">${escapeHtml(statusText)}</span>` : ""}
          ${result?.model_path ? `<span class="badge">${escapeHtml(result.model_path)}</span>` : ""}
        </div>
      </div>
      ${
        quads.length
          ? `<div class="quad-list">${quads.map(renderQuad).join("")}</div>`
          : `<p class="empty-state">No quads returned by this model.</p>`
      }
    </article>
  `;
}

function renderPrediction(result, statusText = "") {
  const output = document.getElementById("predictOutput");
  if (result?.results) {
    const blocks = [renderPredictionResult("Advanced Model", result.results.advanced, statusText)];
    if (result.results.baseline) {
      blocks.push(renderPredictionResult("Baseline Model", result.results.baseline, statusText));
    }
    output.innerHTML = blocks.join("");
    return;
  }

  output.innerHTML = renderPredictionResult("Advanced Model", result, statusText);
}

function renderSamples(data) {
  document.getElementById("samplesBody").innerHTML = data.items
    .map((item) => `
      <tr>
        <td>${fmtNumber(item.id)}</td>
        <td><div class="sentence-cell" title="${escapeHtml(item.sentence)}">${escapeHtml(item.sentence)}</div></td>
        <td>${fmtNumber(item.quad_count)}</td>
        <td>
          ${
            item.quads.length
              ? `<div class="quad-list">${item.quads.map(renderQuad).join("")}</div>`
              : `<span class="empty-state">No quads</span>`
          }
        </td>
      </tr>
    `)
    .join("");

  state.currentPage = data.page;
  document.getElementById("pageIndicator").textContent = `Page ${data.page} / ${data.total_pages}`;
  document.getElementById("prevPage").disabled = data.page <= 1;
  document.getElementById("nextPage").disabled = data.page >= data.total_pages;
}

function renderModelCard(label, result) {
  if (!result) {
    return `
      <article class="model-card">
        <div class="model-title-row">
          <h3>${label}</h3>
          <span class="status-pill status-slow">Missing</span>
        </div>
        <p class="empty-state">Model file not found in the packaged runtime path.</p>
      </article>
    `;
  }

  const metrics = [
    ["Model Name", result.model_name],
    ["Model Size", `${fmtMetric(result.model_size_mb, 2)} MB`],
    ["Runtime", result.runtime || "-"],
    ["CPU Profile", result.cpu_profile || "2 CPU cores"],
    ["Model Path", result.model_path || "-"],
  ];

  return `
    <article class="model-card">
      <div class="model-title-row">
        <h3>${label}</h3>
        <span class="status-pill status-good">Ready</span>
      </div>
      <div class="metric-grid">
        ${metrics.map(([name, value]) => `
          <div class="metric-item">
            <div class="metric-label">${escapeHtml(name)}</div>
            <div class="metric-value">${escapeHtml(value)}</div>
          </div>
        `).join("")}
      </div>
    </article>
  `;
}

function renderModelComparison(data) {
  state.modelComparison = data;
  document.getElementById("modelCards").innerHTML = [
    renderModelCard("Baseline Model", data.baseline),
    renderModelCard("Advanced Model", data.advanced),
  ].join("");

  renderBenchmarkContext();
}

function getSelectedBenchmarkRow() {
  return state.benchmarkRows.find((item) => item.key === state.selectedBenchmarkKey) || state.benchmarkRows[0] || null;
}

function getBestModel() {
  const data = state.modelComparison || {};
  return [data.baseline, data.advanced].filter(Boolean).length;
}

function renderBenchmarkContext() {
  const el = document.getElementById("benchmarkContext");
  if (!el) return;

  const selected = getSelectedBenchmarkRow();
  const readyModels = getBestModel();
  const overview = state.overview || {};
  const sortLabel = state.benchmarkSort === "positive"
    ? "positive"
    : state.benchmarkSort === "negative"
      ? "negative"
      : "support";

  const cards = [
    ["Packaged Models", `${fmtNumber(readyModels)} / 2 ready`, "Baseline and advanced ONNX files"],
    ["Dataset Pool", fmtNumber(overview.total_sentences), `${fmtNumber(overview.total_quads)} quads available`],
    ["Data Focus", getFocusName(), `Sorted by ${sortLabel}`],
    ["Selected", selected ? selected.item : "-", selected ? `${fmtNumber(selected.support)} supporting quads` : "Pick a distribution row"],
  ];

  el.innerHTML = cards
    .map(([label, value, note]) => `
      <article class="insight-card">
        <span>${escapeHtml(label)}</span>
        <strong>${escapeHtml(value)}</strong>
        <small>${escapeHtml(note)}</small>
      </article>
    `)
    .join("");
}

function sentimentTotals() {
  const totals = { Positive: 0, Negative: 0, Neutral: 0 };
  state.categoryMatrix.forEach((row) => {
    SENTIMENTS.forEach((sentiment) => {
      totals[sentiment] += Number(row[sentiment] || 0);
    });
  });
  return SENTIMENTS.map((sentiment) => ({
    key: sentiment,
    item: sentiment,
    support: totals[sentiment],
    Positive: sentiment === "Positive" ? totals[sentiment] : 0,
    Negative: sentiment === "Negative" ? totals[sentiment] : 0,
    Neutral: sentiment === "Neutral" ? totals[sentiment] : 0,
  }));
}

function aspectRows() {
  return state.topAspects.map((item) => ({
    key: item.aspect,
    item: item.aspect,
    support: item.count,
    Positive: item.sentiments?.Positive || 0,
    Negative: item.sentiments?.Negative || 0,
    Neutral: item.sentiments?.Neutral || 0,
    category: item.top_category,
  }));
}

function categoryRows() {
  return state.categoryMatrix.map((row) => ({
    key: row.category,
    item: row.category,
    support: row.total,
    Positive: row.Positive,
    Negative: row.Negative,
    Neutral: row.Neutral,
  }));
}

function getBenchmarkRows() {
  const focusRows = state.factorFocus?.[state.benchmarkFocus] || [];
  const rows = focusRows.length
    ? focusRows
    : state.benchmarkFocus === "sentiment"
      ? sentimentTotals()
      : state.benchmarkFocus === "aspect"
        ? aspectRows()
        : categoryRows();

  const sortKey = state.benchmarkSort === "positive"
    ? "Positive"
    : state.benchmarkSort === "negative"
      ? "Negative"
      : "support";

  return [...rows].sort((a, b) => Number(b[sortKey] || 0) - Number(a[sortKey] || 0));
}

function renderBenchmarkTable() {
  const body = document.getElementById("benchmarkMatrixBody");
  if (!body) return;

  const rows = getBenchmarkRows();
  state.benchmarkRows = rows;
  if ((!state.selectedBenchmarkKey || !rows.some((row) => row.key === state.selectedBenchmarkKey)) && rows.length) {
    state.selectedBenchmarkKey = rows[0].key;
  }

  body.innerHTML = rows
    .slice(0, 10)
    .map((row) => {
      const selected = row.key === state.selectedBenchmarkKey ? " is-selected" : "";
      return `
        <tr class="interactive-row${selected}" data-benchmark-key="${escapeHtml(row.key)}">
          <td><strong>${escapeHtml(row.item)}</strong>${row.category ? `<div class="bar-meta">${escapeHtml(row.category)}</div>` : ""}</td>
          <td>${fmtNumber(row.support)}</td>
          <td>${fmtNumber(row.Positive)}</td>
          <td>${fmtNumber(row.Negative)}</td>
          <td>${fmtNumber(row.Neutral)}</td>
        </tr>
      `;
    })
    .join("");

  renderBenchmarkDetail();
  renderBenchmarkContext();
}

function getFocusName(focus = state.benchmarkFocus) {
  const names = {
    category: "Category",
    sentiment: "Sentiment",
    aspect: "Aspect",
  };
  return names[focus] || "Focus";
}

function dominantSentiment(row) {
  return SENTIMENTS
    .map((sentiment) => ({ sentiment, count: Number(row[sentiment] || 0) }))
    .sort((a, b) => b.count - a.count)[0];
}

function getRelationPanels(row) {
  if (state.benchmarkFocus === "category") {
    return [
      ["Top sentiments", "sentiment", row.sentiments || []],
      ["Top aspects in this category", "aspect", row.aspects || []],
    ];
  }
  if (state.benchmarkFocus === "sentiment") {
    return [
      ["Top categories for this sentiment", "category", row.categories || []],
      ["Top aspects for this sentiment", "aspect", row.aspects || []],
    ];
  }
  return [
    ["Categories for this aspect", "category", row.categories || []],
    ["Sentiments for this aspect", "sentiment", row.sentiments || []],
  ];
}

function relationItemButton(item, type) {
  return `
    <button class="relation-item" type="button" data-focus-type="${escapeHtml(type)}" data-focus-key="${escapeHtml(item.key)}">
      <span class="relation-meta-text">
        <strong>${escapeHtml(item.item)}</strong>
        <small>${fmtNumber(item.count)} quads &middot; ${fmtMetric(item.percent, 1)}%</small>
      </span>
      <span class="relation-meter" aria-hidden="true">
        <span style="width: ${Math.max(2, Math.min(100, Number(item.percent || 0)))}%"></span>
      </span>
    </button>
  `;
}

function renderBenchmarkRelations(row) {
  const container = document.getElementById("benchmarkRelations");
  if (!container) return;

  container.innerHTML = getRelationPanels(row)
    .map(([title, type, items]) => `
      <section class="relation-panel">
        <div class="relation-heading">
          <span>${escapeHtml(title)}</span>
          <b>${fmtNumber(items.reduce((sum, item) => sum + Number(item.count || 0), 0))}</b>
        </div>
        <div class="relation-list">
          ${
            items.length
              ? items.map((item) => relationItemButton(item, type)).join("")
              : `<p class="empty-state">No relation data for this focus yet.</p>`
          }
        </div>
      </section>
    `)
    .join("");
}

function renderBenchmarkDetail() {
  const detail = document.getElementById("benchmarkDetail");
  if (!detail) return;
  const row = getSelectedBenchmarkRow();
  if (!row) {
    detail.innerHTML = `<p class="empty-state">No benchmark insight available yet.</p>`;
    const relations = document.getElementById("benchmarkRelations");
    if (relations) relations.innerHTML = "";
    return;
  }

  const support = Number(row.support || 0);
  const positiveShare = fmtPercent(row.Positive, support);
  const negativeShare = fmtPercent(row.Negative, support);
  const neutralShare = fmtPercent(row.Neutral, support);
  const topSentiment = dominantSentiment(row);
  const topRelations = getRelationPanels(row)
    .map(([title, , items]) => [title, items[0]?.item || "-"])
    .filter(([, value]) => value !== "-");
  const signal = topSentiment.count
    ? `${topSentiment.sentiment} is the dominant sentiment in this focus.`
    : "This focus does not have a dominant sentiment yet.";
  const explorerAction = state.benchmarkFocus === "category"
    ? `<button type="button" data-open-category="${escapeHtml(row.item)}">Open category samples</button>`
    : state.benchmarkFocus === "sentiment"
      ? `<button type="button" data-open-sentiment="${escapeHtml(row.item)}">Open sentiment samples</button>`
      : `<button type="button" data-open-search="${escapeHtml(row.item)}">Search aspect samples</button>`;
  const datasetAction = state.benchmarkFocus === "category"
    ? `<button class="secondary-button" type="button" data-show-dataset-category="${escapeHtml(row.item)}">Show in dataset matrix</button>`
    : `<button class="secondary-button" type="button" data-show-dataset>Show dataset overview</button>`;

  detail.innerHTML = `
    <div class="detail-card">
      <strong>${escapeHtml(getFocusName())}: ${escapeHtml(row.item)}</strong>
      <div class="detail-row"><span>Support</span><b>${fmtNumber(support)}</b></div>
      <div class="detail-row"><span>Positive share</span><b>${positiveShare}</b></div>
      <div class="detail-row"><span>Negative share</span><b>${negativeShare}</b></div>
      <div class="detail-row"><span>Neutral share</span><b>${neutralShare}</b></div>
    </div>
    <div class="detail-card">
      <strong>Insight</strong>
      <p>${signal}</p>
      ${
        topRelations.length
          ? `<p class="empty-state">${topRelations.map(([label, value]) => `${escapeHtml(label)}: ${escapeHtml(value)}`).join(" - ")}</p>`
          : `<p class="empty-state">Click related factors below to move the focus across the benchmark map.</p>`
      }
      <div class="detail-actions">
        ${explorerAction}
        ${datasetAction}
      </div>
    </div>
  `;
  renderBenchmarkRelations(row);
}

function renderTopAspects() {
  const body = document.getElementById("topAspectBody");
  if (!body) return;
  body.innerHTML = state.topAspects
    .map((item) => `
      <tr data-top-aspect="${escapeHtml(item.aspect)}">
        <td><strong>${escapeHtml(item.aspect)}</strong></td>
        <td>${fmtNumber(item.count)}</td>
        <td><span class="category-badge">${escapeHtml(item.top_category)}</span></td>
        <td><span class="sentiment-badge sentiment-${escapeHtml(item.top_sentiment)}">${escapeHtml(item.top_sentiment)}</span></td>
      </tr>
    `)
    .join("");
}

async function submitPrediction(event) {
  event.preventDefault();
  const button = document.getElementById("predictButton");
  const payload = {
    sentence: document.getElementById("predictSentenceInput").value.trim(),
    include_baseline: document.getElementById("includeBaselineInput").checked,
  };

  button.disabled = true;
  button.textContent = "Predicting...";
  setMessage("predictMessage", "Prediction running...");

  try {
    const result = await apiPost("/api/predict", payload);
    setMessage("predictMessage", "Prediction completed.", "success");
    renderPrediction(result, "model output");
  } catch (error) {
    setMessage("predictMessage", error.message, "error");
  } finally {
    button.disabled = false;
    button.textContent = "Predict";
  }
}

function loadExamplePrediction() {
  const example = EXAMPLES[currentExampleIndex];
  document.getElementById("predictSentenceInput").value = example.sentence;
  renderPrediction(example, "sample");
  setMessage("predictMessage", `Loaded Sample #${currentExampleIndex + 1} with ${example.results.advanced.quads.length} quads.`, "success");
  
  // Cycle to next index
  currentExampleIndex = (currentExampleIndex + 1) % EXAMPLES.length;
}

function debounce(fn, delay) {
  let timer = null;
  return (...args) => {
    window.clearTimeout(timer);
    timer = window.setTimeout(() => fn(...args), delay);
  };
}

function activateTab(tabId, updateHash = true) {
  const normalizedTab = LEGACY_TAB_MAP[tabId] || tabId;
  const nextTab = TAB_IDS.includes(normalizedTab) ? normalizedTab : "predict";

  document.querySelectorAll("[data-tab-target]").forEach((button) => {
    const isActive = button.dataset.tabTarget === nextTab;
    button.classList.toggle("is-active", isActive);
    button.setAttribute("aria-selected", String(isActive));
  });

  document.querySelectorAll("[data-tab-panel]").forEach((panel) => {
    const isActive = panel.dataset.tabPanel === nextTab;
    panel.classList.toggle("is-active", isActive);
    panel.hidden = !isActive;
  });

  if (updateHash) {
    history.replaceState(null, "", `#${nextTab}`);
  }
}

function setControlValue(id, value) {
  const el = document.getElementById(id);
  if (el) el.value = value;
}

function applyExplorerFilters({ search = "", category = "", sentiment = "", quadType = "all" } = {}) {
  state.search = search;
  state.category = category;
  state.sentiment = sentiment;
  state.quadType = quadType;
  state.currentPage = 1;

  setControlValue("searchInput", search);
  setControlValue("categorySelect", category);
  setControlValue("sentimentSelect", sentiment);
  setControlValue("quadTypeSelect", quadType);

  activateTab("explorer");
  loadSamples();
}

function selectBenchmarkLens(focus, key = "") {
  state.benchmarkFocus = focus;
  state.selectedBenchmarkKey = key;
  setControlValue("benchmarkFocusSelect", focus);
  renderBenchmarkTable();
}

function handleOverviewAction(action) {
  if (!action) return;
  const quadType = ["no_quad", "single_quad", "multi_quad"].includes(action) ? action : "all";
  applyExplorerFilters({ quadType });
}

function handleDistributionClick(row) {
  const type = row.dataset.filterType;
  const value = row.dataset.filterValue;
  if (!type || !value) return;

  if (type === "category") {
    selectBenchmarkLens("category", value);
    applyExplorerFilters({ category: value });
    return;
  }

  if (type === "sentiment") {
    selectBenchmarkLens("sentiment", value);
    applyExplorerFilters({ sentiment: value });
  }
}

function bindEvents() {
  document.querySelectorAll("[data-tab-target]").forEach((button) => {
    button.addEventListener("click", () => activateTab(button.dataset.tabTarget));
  });

  window.addEventListener("hashchange", () => {
    activateTab(window.location.hash.replace("#", ""), false);
  });

  document.getElementById("searchInput").addEventListener("input", debounce((event) => {
    state.search = event.target.value;
    state.currentPage = 1;
    loadSamples();
  }, 300));

  [
    ["categorySelect", "category"],
    ["sentimentSelect", "sentiment"],
    ["quadTypeSelect", "quadType"],
    ["pageSizeSelect", "pageSize"],
  ].forEach(([id, key]) => {
    document.getElementById(id).addEventListener("change", (event) => {
      state[key] = key === "pageSize" ? Number(event.target.value) : event.target.value;
      state.currentPage = 1;
      loadSamples();
    });
  });

  const benchmarkFocusSelect = document.getElementById("benchmarkFocusSelect");
  if (benchmarkFocusSelect) {
    benchmarkFocusSelect.addEventListener("change", (event) => {
      state.benchmarkFocus = event.target.value;
      state.selectedBenchmarkKey = "";
      renderBenchmarkTable();
    });
  }

  const benchmarkSortSelect = document.getElementById("benchmarkSortSelect");
  if (benchmarkSortSelect) {
    benchmarkSortSelect.addEventListener("change", (event) => {
      state.benchmarkSort = event.target.value;
      state.selectedBenchmarkKey = "";
      renderBenchmarkTable();
    });
  }

  const benchmarkMatrixBody = document.getElementById("benchmarkMatrixBody");
  if (benchmarkMatrixBody) {
    benchmarkMatrixBody.addEventListener("click", (event) => {
      const row = event.target.closest("[data-benchmark-key]");
      if (!row) return;
      state.selectedBenchmarkKey = row.dataset.benchmarkKey;
      renderBenchmarkTable();
    });
  }

  const benchmarkDetail = document.getElementById("benchmarkDetail");
  if (benchmarkDetail) {
    benchmarkDetail.addEventListener("click", (event) => {
      const button = event.target.closest("button");
      if (!button) return;

      if (button.dataset.openCategory) {
        applyExplorerFilters({ category: button.dataset.openCategory });
        return;
      }

      if (button.dataset.openSentiment) {
        applyExplorerFilters({ sentiment: button.dataset.openSentiment });
        return;
      }

      if (button.dataset.openSearch) {
        applyExplorerFilters({ search: button.dataset.openSearch });
        return;
      }

      if (button.dataset.showDatasetCategory) {
        activateTab("dataset");
        return;
      }

      if (button.hasAttribute("data-show-dataset")) {
        activateTab("dataset");
      }
    });
  }

  const benchmarkRelations = document.getElementById("benchmarkRelations");
  if (benchmarkRelations) {
    benchmarkRelations.addEventListener("click", (event) => {
      const button = event.target.closest("[data-focus-type]");
      if (!button) return;
      selectBenchmarkLens(button.dataset.focusType, button.dataset.focusKey);
    });
  }

  const topAspectBody = document.getElementById("topAspectBody");
  if (topAspectBody) {
    topAspectBody.addEventListener("click", (event) => {
      const row = event.target.closest("[data-top-aspect]");
      if (!row) return;
      selectBenchmarkLens("aspect", row.dataset.topAspect);
    });
  }

  document.getElementById("overviewCards").addEventListener("click", (event) => {
    const card = event.target.closest("[data-overview-action]");
    if (!card) return;
    handleOverviewAction(card.dataset.overviewAction);
  });

  document.getElementById("overviewCards").addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    const card = event.target.closest("[data-overview-action]");
    if (!card) return;
    event.preventDefault();
    handleOverviewAction(card.dataset.overviewAction);
  });

  ["categoryBars", "sentimentBars"].forEach((id) => {
    document.getElementById(id).addEventListener("click", (event) => {
      const row = event.target.closest("[data-filter-type]");
      if (!row) return;
      handleDistributionClick(row);
    });
  });

  document.getElementById("matrixBody").addEventListener("click", (event) => {
    const row = event.target.closest("[data-matrix-category]");
    if (!row) return;
    const category = row.dataset.matrixCategory;
    selectBenchmarkLens("category", category);
    applyExplorerFilters({ category });
  });

  document.getElementById("prevPage").addEventListener("click", () => {
    state.currentPage = Math.max(1, state.currentPage - 1);
    loadSamples();
  });

  document.getElementById("nextPage").addEventListener("click", () => {
    state.currentPage += 1;
    loadSamples();
  });

  document.getElementById("loadExampleButton").addEventListener("click", loadExamplePrediction);
  document.getElementById("predictForm").addEventListener("submit", submitPrediction);

  const getEmbedUrl = (url) => {
    url = url.trim();
    if (url.includes("canva.com/design/")) {
      let cleanUrl = url.split('?')[0];
      if (cleanUrl.endsWith('/edit') || cleanUrl.endsWith('/watch') || cleanUrl.endsWith('/view')) {
        cleanUrl = cleanUrl.substring(0, cleanUrl.lastIndexOf('/'));
      }
      if (!cleanUrl.endsWith('/view')) {
        cleanUrl = cleanUrl + '/view';
      }
      return cleanUrl + '?embed';
    }
    if (url.includes("docs.google.com/presentation/")) {
      if (url.includes("/pub")) {
        return url.replace("/pub", "/embed");
      }
      const match = url.match(/(https:\/\/docs\.google\.com\/presentation\/d\/[a-zA-Z0-9_-]+)/);
      if (match) {
        return match[1] + "/embed";
      }
    }
    return url;
  };


  const loadPresentationBtn = document.getElementById("loadPresentationButton");
  const urlInput = document.getElementById("presentationUrlInput");
  const wrapper = document.getElementById("presentationIframeWrapper");

  if (loadPresentationBtn && urlInput && wrapper) {
    loadPresentationBtn.addEventListener("click", () => {
      const rawUrl = urlInput.value.trim();
      if (!rawUrl) {
        return;
      }
      const embedUrl = getEmbedUrl(rawUrl);
      localStorage.setItem("presentation_url", rawUrl);
      wrapper.innerHTML = `<iframe src="${escapeHtml(embedUrl)}" allowfullscreen="true"></iframe>`;
    });

    // Auto-sync when typing (debounced)
    urlInput.addEventListener("input", debounce(() => {
      loadPresentationBtn.click();
    }, 600));

    // Load persisted URL on startup
    const savedUrl = localStorage.getItem("presentation_url");
    if (savedUrl) {
      urlInput.value = savedUrl;
      setTimeout(() => {
        loadPresentationBtn.click();
      }, 300);
    }
  }

  const loadPresetCanvaBtn = document.getElementById("loadPresetCanvaButton");
  if (loadPresetCanvaBtn && urlInput && loadPresentationBtn) {
    loadPresetCanvaBtn.addEventListener("click", () => {
      urlInput.value = "https://www.canva.com/design/DAGIrO1M2tI/w4sB7F_4J1pXv2j1_m4R2A/view?embed";
      loadPresentationBtn.click();
    });
  }
}


async function init() {
  activateTab(window.location.hash.replace("#", ""), false);
  bindEvents();
  await Promise.all([
    loadOverview(),
    loadCategoryDistribution(),
    loadSentimentDistribution(),
    loadMatrix(),
    loadTopAspects(),
    loadFactorFocus(),
    loadSamples(),
    loadModelComparison(),
  ]);
}

document.addEventListener("DOMContentLoaded", () => {
  init().catch((error) => {
    setMessage("overviewMessage", error.message, "error");
  });
});
