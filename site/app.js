/* When keyword matching fails — static site logic. All numbers come from data/site.json. */
"use strict";

const MODEL_LABELS = { logreg: "Logistic regression", lgbm_pointwise: "LightGBM pointwise", lgbm_lambdarank: "LightGBM LambdaRank" };
const METHOD_LABELS = {
  random: "Random (20 seeds)", bm25_title: "BM25 title", bm25_all: "BM25 all fields",
  encoder_cosine: "Encoder cosine (zero-shot)", ...MODEL_LABELS,
};
const METHOD_ORDER = ["random", "bm25_title", "bm25_all", "encoder_cosine", "logreg", "lgbm_pointwise", "lgbm_lambdarank"];
const LABEL_NAMES = { E: "Exact", S: "Substitute", C: "Complement", I: "Irrelevant" };

let S = null;          // site.json
let I = null;          // inspector.json
const charts = {};     // id -> Chart
const state = { cat: null, outcome: null, bin: null, qid: null, sel: null, testSet: "all" };

/* ---------------- helpers ---------------- */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const get = (obj, path) => path.split(".").reduce((o, k) => (o == null ? undefined : o[k]), obj);
const f4 = (v, d = 4) => (v == null || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const signed = (v, d = 4) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d);
const pct = (v, d = 1) => (100 * v).toFixed(d) + "%";
const ciText = (c, d = 4) => `${f4(c.mean, d)} [${f4(c.lo, d)}, ${f4(c.hi, d)}]`;
const ciSigned = (c, d = 4) => `${signed(c.mean, d)} [${signed(c.lo, d)}, ${signed(c.hi, d)}]`;
const esc = (s) => String(s).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const modelName = () => MODEL_LABELS[S.cv.chosen.family];

function el(tag, attrs = {}, html = "") {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  e.innerHTML = html;
  return e;
}

function table(id, head, rows, opts = {}) {
  const t = document.getElementById(id);
  if (!t) return;
  const numCols = opts.num || [];
  const th = head.map((h, i) => `<th class="${numCols.includes(i) ? "num" : ""}">${h}</th>`).join("");
  const tb = rows.map((r) => {
    const cls = r.hl ? ' class="hl"' : "";
    const cells = (r.cells || r).map((c, i) => `<td class="${numCols.includes(i) ? "num" : ""}">${c}</td>`).join("");
    return `<tr${cls}>${cells}</tr>`;
  }).join("");
  t.innerHTML = `<thead><tr>${th}</tr></thead><tbody>${tb}</tbody>`;
}

/* ---------------- theme ---------------- */
function currentTheme() {
  const set = document.documentElement.dataset.theme;
  if (set) return set;
  return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}
function initTheme() {
  try {
    const saved = localStorage.getItem("theme");
    if (saved) document.documentElement.dataset.theme = saved;
  } catch (e) { /* storage unavailable */ }
  $$(".theme-btn").forEach((b) => b.addEventListener("click", () => {
    const next = currentTheme() === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch (e) { /* ignore */ }
    refreshCharts();
    drawLinks();
  }));
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { refreshCharts(); drawLinks(); });
}

/* ---------------- routing ---------------- */
const pages = () => $$(".page");
function buildNav() {
  const nav = $("#nav");
  pages().forEach((p, i) => {
    nav.appendChild(el("a", { href: "#" + p.id, "data-id": p.id }, `<span class="num">${i + 1}</span><span>${p.dataset.title}</span>`));
  });
  nav.addEventListener("click", () => closeMenu());
}
function show() {
  const id = (location.hash || "#question").slice(1).split("?")[0];
  const list = pages();
  const idx = Math.max(0, list.findIndex((p) => p.id === id));
  list.forEach((p, i) => p.classList.toggle("active", i === idx));
  $$("#nav a").forEach((a) => a.classList.toggle("active", a.dataset.id === list[idx].id));
  const prev = list[idx - 1], next = list[idx + 1];
  $("#pager").innerHTML =
    (prev ? `<a href="#${prev.id}"><span class="dir">← Previous</span>${prev.dataset.title}</a>` : "<span></span>") +
    (next ? `<a href="#${next.id}" style="text-align:right"><span class="dir">Next →</span>${next.dataset.title}</a>` : "<span></span>");
  document.title = `${list[idx].dataset.title} · When Keyword Matching Fails`;
  window.scrollTo(0, 0);
  if (S) renderSectionCharts(list[idx].id);
  if (list[idx].id === "inspector" && I) renderInspector();
}
function openMenu() { $("#sidebar").classList.add("open"); $("#scrim").hidden = false; $("#menu-btn").setAttribute("aria-expanded", "true"); }
function closeMenu() { $("#sidebar").classList.remove("open"); $("#scrim").hidden = true; $("#menu-btn").setAttribute("aria-expanded", "false"); }

/* ---------------- bindings ---------------- */
function bindValues() {
  $$("[data-v]").forEach((e) => {
    const v = get(S, e.dataset.v);
    if (v == null) { e.textContent = "–"; return; }
    if ("pct" in e.dataset) e.textContent = pct(v);
    else if ("sign" in e.dataset) e.textContent = signed(v, e.dataset.f ? +e.dataset.f : 4);
    else if (e.dataset.f) e.textContent = Number(v).toLocaleString("en-US", { minimumFractionDigits: +e.dataset.f, maximumFractionDigits: +e.dataset.f });
    else e.textContent = f4(v);
  });
  $$("[data-ci-range]").forEach((e) => { const c = get(S, e.dataset.ciRange); if (c) e.textContent = `[${signed(c.lo)}, ${signed(c.hi)}]`; });
  $$("[data-text]").forEach((e) => { const v = get(S, e.dataset.text); e.textContent = v == null ? "–" : v; });
  $$("[data-json]").forEach((e) => { e.textContent = JSON.stringify(get(S, e.dataset.json)); });
  $$("[data-model-name]").forEach((e) => { e.textContent = modelName(); });
}

/* ---------------- static tables ---------------- */
function renderTables() {
  table("t-sizes", ["Split", "Queries", "Pairs", "Products", "Candidates per query (min / median / max)"],
    S.audit.sizes.map((s) => [s.split, s.queries.toLocaleString(), s.pairs.toLocaleString(), s.products.toLocaleString(),
      `${s.cand_min} / ${s.cand_median} / ${s.cand_max}`]), { num: [1, 2, 3] });
  const miss = S.audit.missing;
  table("t-missing", ["Field", "Train", "Test"], Object.keys(miss.train).map((f) => [f, pct(miss.train[f]), pct(miss.test[f])]), { num: [1, 2] });
  table("t-features", ["Feature", "Group", "Definition", "Why it might help"],
    S.features.map((f) => [`<code>${f.name}</code>`, f.group, esc(f.formula), esc(f.why)]));

  const bmCv = S.cv.baselines.bm25_title.ndcg.mean;
  const cvRows = METHOD_ORDER.filter((m) => S.cv.baselines[m] || S.cv.models[m]).map((m) => {
    const s = (S.cv.baselines[m] || S.cv.models[m].summary);
    const ms = (k) => `${f4(s[k].mean)} ± ${f4(s[k].std)}`;
    return { hl: m === S.cv.chosen.family, cells: [METHOD_LABELS[m], ms("ndcg"), ms("ndcg5"), ms("p1"), signed(s.ndcg.mean - bmCv)] };
  });
  table("t-cv", ["Method", "nDCG", "nDCG@5", "Exact-P@1", "Δ nDCG vs BM25 title"], cvRows, { num: [1, 2, 3, 4] });
  renderTestTables();

  const cats = S.test.categories;
  table("t-cats", ["Category", "Queries", "BM25 title", "Model", "Δ nDCG [95% CI]", "Win / loss"],
    Object.entries(cats).map(([c, v]) => [
      `${c}${v.analysis_only ? " <span class='muted small'>(analysis only)</span>" : ""}<div class="muted small">${esc(v.description)}</div>`,
      v.n.toLocaleString(), f4(v.bm25_title_ndcg.mean), f4(v.model_ndcg.mean),
      `<span class="${v.delta_vs_bm25_title.mean >= 0 ? "pos" : "neg"}">${ciSigned(v.delta_vs_bm25_title)}</span>`, `${v.win} / ${v.loss}`]),
    { num: [1, 2, 3, 4, 5] });

  table("t-runtime", ["Phase", "Seconds"], Object.entries(S.runtime || {}).map(([k, v]) => [k, Math.round(v.seconds).toLocaleString()]), { num: [1] });
  renderExamples();
  renderConclusion();
}

function renderTestTables() {
  const set = S.test[state.testSet];
  const rows = METHOD_ORDER.filter((m) => set.table[m]).map((m) => ({
    hl: m === S.cv.chosen.family,
    cells: [METHOD_LABELS[m], ciText(set.table[m].ndcg), ciText(set.table[m].ndcg5), ciText(set.table[m].p1)],
  }));
  table("t-test", ["Method", "nDCG", "nDCG@5", "Exact-P@1"], rows, { num: [1, 2, 3] });
  const diffRows = Object.entries(set.diff).map(([k, v]) => [
    k.replace("model", modelName()).replace("bm25_title", "BM25 title").replace("bm25_all", "BM25 all").replace("encoder_cosine", "encoder cosine").replace("random", "random"),
    ...["ndcg", "ndcg5", "p1"].map((m) => `<span class="${v[m].lo > 0 ? "pos" : v[m].hi < 0 ? "neg" : ""}">${ciSigned(v[m])}</span>`)]);
  table("t-diff", ["Paired difference", "Δ nDCG", "Δ nDCG@5", "Δ Exact-P@1"], diffRows, { num: [1, 2, 3] });
}

function renderExamples() {
  for (const kind of ["helps", "hurts"]) {
    const box = document.getElementById("ex-" + kind);
    box.innerHTML = "";
    for (const e of S.examples[kind]) {
      const li = (x) => `<li><span class="chip ${x.label}">${x.label}</span> ${esc(x.title)}</li>`;
      const d = el("div", { class: "ex" },
        `<div><span class="ex-q" data-qid="${e.query_id}">${esc(e.query)}</span>
           <span class="${e.delta >= 0 ? "pos" : "neg"}"> Δ ${signed(e.delta, 3)}</span>
           <span class="muted small">(BM25 ${f4(e.bm25_ndcg, 3)} → model ${f4(e.model_ndcg, 3)})</span></div>
         <div class="ex-top"><div><div class="small muted">BM25 top 3</div><ol>${e.bm25_top3.map(li).join("")}</ol></div>
         <div><div class="small muted">Model top 3</div><ol>${e.model_top3.map(li).join("")}</ol></div></div>
         <div class="diag">${esc(e.diagnosis)}</div>`);
      box.appendChild(d);
    }
  }
  $$(".ex-q").forEach((q) => q.addEventListener("click", () => openInspector({ qid: +q.dataset.qid })));
}

function renderConclusion() {
  const t = S.test.all, d = t.diff["model - bm25_title"].ndcg, rnd = t.table.random.ndcg.mean;
  const bm = t.table.bm25_title.ndcg.mean, mod = t.table.model.ndcg.mean, enc = t.table.encoder_cosine;
  const gap = bm - rnd;
  const sig = d.lo > 0 ? "better than" : d.hi < 0 ? "worse than" : "not distinguishable from";
  const cats = Object.entries(S.test.categories).filter(([, v]) => !v.analysis_only);
  const worst = cats.reduce((a, b) => (b[1].delta_vs_bm25_title.mean < a[1].delta_vs_bm25_title.mean ? b : a));
  const best = cats.reduce((a, b) => (b[1].delta_vs_bm25_title.mean > a[1].delta_vs_bm25_title.mean ? b : a));
  const abl = S.cv.ablation;
  const dEnc = t.diff["model - encoder_cosine"];
  const wtl = S.test.win_tie_loss;
  const extra = $("#conclusion-extra") ? $("#conclusion-extra").innerHTML : "";
  $("#conclusion-body").innerHTML = `
    <p class="lead">On the official test split, the ${esc(modelName())} reaches nDCG <b>${f4(mod)}</b>, ${sig}
      BM25 on titles (<b>${f4(bm)}</b>): a paired difference of <b>${ciSigned(d)}</b>.</p>
    <ul>
      <li><b>Size of the gain.</b> Random ordering already scores ${f4(rnd)}; BM25 adds ${signed(gap)} on top of random,
        and the model adds ${signed(mod - bm)} on top of BM25 (${Math.round(100 * (mod - bm) / gap)}% of the random → BM25 gap).
        The model wins on ${wtl.win.toLocaleString()} queries, ties on ${wtl.tie.toLocaleString()} and loses on ${wtl.loss.toLocaleString()}.</li>
      ${enc ? `<li><b>Against meaning-based matching.</b> The zero-shot encoder alone scores ${f4(enc.ndcg.mean)};
        the model − encoder difference is ${ciSigned(dEnc.ndcg)}. Much of what the model adds over BM25 is semantic similarity,
        combined with attribute and within-list signals.</li>` : ""}
      <li><b>Feature groups.</b> In cross-validation, LambdaRank with lexical features only scores ${f4(abl[0].summary.ndcg.mean)};
        with all ${abl[abl.length - 1].n_features} features ${f4(abl[abl.length - 1].summary.ndcg.mean)}.</li>
      <li><b>Where it helps and fails.</b> The largest gain is on <i>${best[0]}</i> queries (Δ ${ciSigned(best[1].delta_vs_bm25_title)});
        the smallest on <i>${worst[0]}</i> queries (Δ ${ciSigned(worst[1].delta_vs_bm25_title)}).
        See the Errors section for the full table and per-query examples.</li>
      <li><b>Robustness.</b> Removing the ${S.test.n_near_dup} near-duplicate test queries changes the model − BM25 difference to
        ${ciSigned(S.test.without_near_dup.diff["model - bm25_title"].ndcg)}.</li>
    </ul>${extra}`;
}

/* ---------------- charts ---------------- */
const ciPlugin = {
  id: "ci",
  afterDatasetsDraw(chart) {
    const { ctx } = chart;
    chart.data.datasets.forEach((ds, di) => {
      if (!ds.ci) return;
      const meta = chart.getDatasetMeta(di);
      const horizontal = chart.options.indexAxis === "y";
      const scale = horizontal ? chart.scales.x : chart.scales.y;
      ctx.save();
      ctx.strokeStyle = cssVar("--text");
      ctx.lineWidth = 1.5;
      meta.data.forEach((bar, i) => {
        const c = ds.ci[i];
        if (!c) return;
        const a = scale.getPixelForValue(c[0]), b = scale.getPixelForValue(c[1]);
        ctx.beginPath();
        if (horizontal) {
          ctx.moveTo(a, bar.y); ctx.lineTo(b, bar.y);
          ctx.moveTo(a, bar.y - 5); ctx.lineTo(a, bar.y + 5); ctx.moveTo(b, bar.y - 5); ctx.lineTo(b, bar.y + 5);
        } else {
          ctx.moveTo(bar.x, a); ctx.lineTo(bar.x, b);
          ctx.moveTo(bar.x - 6, a); ctx.lineTo(bar.x + 6, a); ctx.moveTo(bar.x - 6, b); ctx.lineTo(bar.x + 6, b);
        }
        ctx.stroke();
      });
      ctx.restore();
    });
  },
};

function baseOptions(extra = {}) {
  const text = cssVar("--muted"), grid = cssVar("--border");
  return Object.assign({
    responsive: true, maintainAspectRatio: false, animation: false,
    plugins: { legend: { labels: { color: text } }, tooltip: {} },
    scales: {
      x: { ticks: { color: text }, grid: { color: grid } },
      y: { ticks: { color: text }, grid: { color: grid } },
    },
  }, extra);
}

const CHARTS_BY_SECTION = {
  data: ["c-labels"], evaluation: ["c-ablation", "c-importance"], results: ["c-delta"], errors: ["c-cats"],
};

function makeChart(id) {
  if (typeof Chart === "undefined") return;
  const canvas = document.getElementById(id);
  if (!canvas) return;
  if (charts[id]) charts[id].destroy();
  const cfg = chartConfig(id);
  if (cfg) charts[id] = new Chart(canvas, cfg);
}

function chartConfig(id) {
  const pos = cssVar("--pos"), neg = cssVar("--neg"), accent = cssVar("--accent"), muted = cssVar("--muted");
  if (id === "c-labels") {
    const splits = ["train", "test"];
    const datasets = ["E", "S", "C", "I"].map((l) => ({
      label: `${l} — ${LABEL_NAMES[l]}`, data: splits.map((s) => S.audit.label_mix[s][l]), backgroundColor: cssVar("--" + l),
    }));
    const o = baseOptions({ indexAxis: "y" });
    o.scales.x.stacked = true; o.scales.y.stacked = true; o.scales.x.max = 1;
    o.scales.x.ticks.callback = (v) => pct(v, 0);
    o.plugins.tooltip.callbacks = { label: (c) => `${c.dataset.label}: ${pct(c.raw)}` };
    return { type: "bar", data: { labels: splits, datasets }, options: o };
  }
  if (id === "c-ablation") {
    const a = S.cv.ablation;
    const vals = a.map((s) => s.summary.ndcg.mean);
    const bm = S.cv.baselines.bm25_title.ndcg.mean;
    const lo = Math.min(bm, ...a.map((s) => s.ndcg_ci.lo));
    const o = baseOptions();
    o.scales.y.min = Math.floor((lo - 0.01) * 100) / 100;
    o.scales.y.title = { display: true, text: "CV nDCG", color: muted };
    o.plugins.legend.display = false;
    o.plugins.tooltip.callbacks = { label: (c) => c.dataIndex === 0 ? `BM25 title: ${f4(bm)}` : `nDCG ${f4(c.raw)} [${f4(a[c.dataIndex - 1].ndcg_ci.lo)}, ${f4(a[c.dataIndex - 1].ndcg_ci.hi)}]` };
    return {
      type: "bar",
      data: {
        labels: ["BM25 title (baseline)", ...a.map((s) => `${s.name} (${s.n_features})`)],
        datasets: [{ data: [bm, ...vals], backgroundColor: [muted, ...a.map(() => accent)], ci: [null, ...a.map((s) => [s.ndcg_ci.lo, s.ndcg_ci.hi])] }],
      },
      options: o, plugins: [ciPlugin],
    };
  }
  if (id === "c-importance") {
    const imp = S.cv.importance;
    const labels = imp.map((r) => (S.features.find((f) => f.name === r.feature) || {}).label || r.feature);
    const o = baseOptions({ indexAxis: "y" });
    o.plugins.legend.display = false;
    o.scales.x.title = { display: true, text: "mean |SHAP|", color: muted };
    o.plugins.tooltip.callbacks = { label: (c) => `${imp[c.dataIndex].feature} (${imp[c.dataIndex].group}): ${f4(c.raw)}` };
    return { type: "bar", data: { labels, datasets: [{ data: imp.map((r) => r.mean_abs_shap), backgroundColor: accent }] }, options: o };
  }
  if (id === "c-delta") {
    const h = S.test.delta_hist;
    const labels = h.counts.map((_, i) => `${signed(h.edges[i], 2)} … ${signed(h.edges[i + 1], 2)}`);
    const colors = h.counts.map((_, i) => (h.edges[i] >= 0 ? pos : h.edges[i + 1] <= 0 ? neg : muted));
    const o = baseOptions();
    o.plugins.legend.display = false;
    o.scales.x.title = { display: true, text: `ΔnDCG per query (${modelName()} − BM25 title); end bins include everything beyond`, color: muted };
    o.scales.y.title = { display: true, text: "queries", color: muted };
    o.onClick = (_, els) => { if (els.length) openInspector({ bin: els[0].index }); };
    o.onHover = (e, els) => { e.native.target.style.cursor = els.length ? "pointer" : "default"; };
    return { type: "bar", data: { labels, datasets: [{ data: h.counts, backgroundColor: colors, barPercentage: 1, categoryPercentage: 0.95 }] }, options: o };
  }
  if (id === "c-cats") {
    const entries = Object.entries(S.test.categories);
    const o = baseOptions({ indexAxis: "y" });
    o.plugins.legend.display = false;
    o.scales.x.title = { display: true, text: `ΔnDCG vs BM25 title (95% CI)`, color: muted };
    o.plugins.tooltip.callbacks = { label: (c) => { const v = entries[c.dataIndex][1]; return `Δ ${ciSigned(v.delta_vs_bm25_title)} · ${v.n} queries`; } };
    o.onClick = (_, els) => { if (els.length) openInspector({ cat: I ? I.categories.indexOf(entries[els[0].index][0]) : null }); };
    o.onHover = (e, els) => { e.native.target.style.cursor = els.length ? "pointer" : "default"; };
    return {
      type: "bar",
      data: {
        labels: entries.map(([c, v]) => `${c}${v.analysis_only ? " (analysis only)" : ""} · n=${v.n}`),
        datasets: [{
          data: entries.map(([, v]) => v.delta_vs_bm25_title.mean),
          backgroundColor: entries.map(([, v]) => (v.delta_vs_bm25_title.mean >= 0 ? pos : neg)),
          ci: entries.map(([, v]) => [v.delta_vs_bm25_title.lo, v.delta_vs_bm25_title.hi]),
        }],
      },
      options: o, plugins: [ciPlugin],
    };
  }
  return null;
}

function renderSectionCharts(section) { (CHARTS_BY_SECTION[section] || []).forEach((id) => { if (!charts[id]) makeChart(id); }); }
function refreshCharts() {
  Object.keys(charts).forEach((id) => { charts[id].destroy(); delete charts[id]; });
  const active = $(".page.active");
  if (active && S) renderSectionCharts(active.id);
}

/* ---------------- inspector ---------------- */
function binOf(d) {
  const e = S.test.delta_hist.edges;
  const x = Math.min(Math.max(d, e[0]), e[e.length - 1] - 1e-9);
  for (let i = 0; i < e.length - 1; i++) if (x >= e[i] && x < e[i + 1]) return i;
  return e.length - 2;
}

function filteredQueries() {
  return I.queries.filter((q) =>
    (state.cat == null || q.c.includes(state.cat)) &&
    (state.outcome == null || q.o === state.outcome) &&
    (state.bin == null || binOf(q.d) === state.bin));
}

function openInspector(opts) {
  Object.assign(state, { cat: null, outcome: null, bin: null, qid: null, sel: null }, opts);
  if (location.hash !== "#inspector") location.hash = "inspector";
  else renderInspector();
}

function buildChips() {
  const box = $("#chips");
  box.innerHTML = "";
  I.categories.forEach((c, i) => {
    const b = el("button", { class: "fchip" + (state.cat === i ? " on" : "") }, c.replaceAll("_", " "));
    b.addEventListener("click", () => { state.cat = state.cat === i ? null : i; state.qid = null; renderInspector(); });
    box.appendChild(b);
  });
  ["helps", "hurts", "tie"].forEach((o, j) => {
    const b = el("button", { class: "fchip" + (state.outcome === o ? " on" : "") + (j === 0 ? " sep" : "") }, o);
    b.addEventListener("click", () => { state.outcome = state.outcome === o ? null : o; state.qid = null; renderInspector(); });
    box.appendChild(b);
  });
  const af = $("#active-filter");
  if (state.bin != null) {
    const e = S.test.delta_hist.edges;
    af.innerHTML = `Δ bin ${signed(e[state.bin], 2)} … ${signed(e[state.bin + 1], 2)} <button class="fchip" id="clear-bin">clear ×</button>`;
    $("#clear-bin").addEventListener("click", () => { state.bin = null; renderInspector(); });
  } else af.textContent = "";
}

function renderInspector() {
  buildChips();
  const list = filteredQueries();
  $("#qlist-count").textContent = `${list.length.toLocaleString()} queries`;
  if (state.qid == null || !list.some((q) => q.id === state.qid)) state.qid = list.length ? list[0].id : null;
  const ul = $("#qlist");
  ul.innerHTML = "";
  list.slice(0, 400).forEach((q) => {
    const li = el("li", { "data-id": q.id, class: q.id === state.qid ? "active" : "" },
      `<span>${esc(q.q)}</span><span class="d ${q.d > 0 ? "pos" : q.d < 0 ? "neg" : "muted"}">${signed(q.d, 3)}</span>`);
    li.addEventListener("click", () => { state.qid = q.id; state.sel = null; renderInspector(); });
    ul.appendChild(li);
  });
  if (list.length > 400) ul.appendChild(el("li", { class: "muted" }, `… ${list.length - 400} more — use search or filters`));
  renderQuery();
}

function renderQuery() {
  const q = I.queries.find((x) => x.id === state.qid);
  if (!q) { $("#q-head").innerHTML = "<p class='muted'>No query matches these filters.</p>"; $("#col-bm25").innerHTML = ""; $("#col-model").innerHTML = ""; $("#links").innerHTML = ""; $("#contrib").hidden = true; return; }
  const cats = q.c.map((i) => `<span class="fchip">${I.categories[i].replaceAll("_", " ")}</span>`).join(" ");
  $("#q-head").innerHTML = `<div class="q">${esc(q.q)}</div>
    <div class="scores"><span>nDCG BM25 <b>${f4(q.nb, 3)}</b></span><span>nDCG ${esc(modelName())} <b>${f4(q.nm, 3)}</b></span>
    <span>Δ <b class="${q.d > 0 ? "pos" : q.d < 0 ? "neg" : ""}">${signed(q.d, 3)}</b></span><span>${q.p.length} candidates</span></div>
    <div class="chips" style="margin-top:.35rem">${cats}</div>`;
  const item = (p, i, rank) => `<li class="prod${state.sel === i ? " sel" : ""}" data-i="${i}">
      <span class="rk">${rank}</span><span><span class="chip ${p[2]}" title="${LABEL_NAMES[p[2]]}">${p[2]}</span></span>
      <span><span class="t">${esc(p[0])}</span><span class="b">${esc(p[1] || "no brand")}</span></span></li>`;
  const idx = q.p.map((_, i) => i);
  $("#col-bm25").innerHTML = [...idx].sort((a, b) => q.p[a][3] - q.p[b][3]).map((i) => item(q.p[i], i, q.p[i][3])).join("");
  $("#col-model").innerHTML = [...idx].sort((a, b) => q.p[a][4] - q.p[b][4]).map((i) => item(q.p[i], i, q.p[i][4])).join("");
  $$("#rankings .prod").forEach((li) => li.addEventListener("click", () => { state.sel = +li.dataset.i; renderQuery(); }));
  requestAnimationFrame(drawLinks);
  renderContrib(q);
}

function drawLinks() {
  const svg = $("#links");
  const q = I && I.queries.find((x) => x.id === state.qid);
  if (!svg || !q || !$("#inspector").classList.contains("active")) return;
  const sr = svg.getBoundingClientRect();
  const w = sr.width;
  const centre = (col, i) => {
    const li = $(`#${col} .prod[data-i="${i}"]`);
    if (!li) return null;
    const r = li.getBoundingClientRect();
    return r.top + r.height / 2 - sr.top;
  };
  let lines = "";
  q.p.forEach((p, i) => {
    const y1 = centre("col-bm25", i), y2 = centre("col-model", i);
    if (y1 == null || y2 == null) return;
    lines += `<line x1="0" y1="${y1}" x2="${w}" y2="${y2}" stroke="${cssVar("--" + p[2])}" class="${state.sel === i ? "sel" : ""}"/>`;
  });
  svg.innerHTML = lines;
}

function renderContrib(q) {
  const box = $("#contrib");
  if (state.sel == null) {
    box.hidden = false;
    box.innerHTML = `<div class="muted small">Click a product to see its top 5 feature contributions in the ${esc(modelName())}.</div>`;
    return;
  }
  const p = q.p[state.sel];
  const maxAbs = Math.max(...p[5].map((c) => Math.abs(c[1])), 1e-9);
  const bars = p[5].map(([fi, v]) => {
    const w = (50 * Math.abs(v)) / maxAbs;
    const style = v >= 0 ? `left:50%;width:${w}%` : `left:${50 - w}%;width:${w}%`;
    return `<div class="bar-row"><span>${esc(I.features[fi])}</span><span class="bar-track"><span class="bar ${v >= 0 ? "pos" : "neg"}" style="${style}"></span></span><span class="v ${v >= 0 ? "pos" : "neg"}">${signed(v, 2)}</span></div>`;
  }).join("");
  box.hidden = false;
  box.innerHTML = `<div class="title"><span class="chip ${p[2]}">${p[2]}</span> ${esc(p[0])}</div>
    <div class="small muted" style="margin-bottom:.5rem">BM25 rank ${p[3]} → model rank ${p[4]} · top 5 contributions to the model's score
    (${S.cv.chosen.family === "logreg" ? "log-odds units: coefficient × standardised value" : "LightGBM pred_contrib (SHAP) units"})</div>${bars}`;
}

function initSearch() {
  const input = $("#q-search"), box = $("#q-suggest");
  let sel = -1, items = [];
  const close = () => { box.hidden = true; sel = -1; };
  const choose = (q) => { input.value = q.q; close(); Object.assign(state, { cat: null, outcome: null, bin: null, qid: q.id, sel: null }); renderInspector(); };
  input.addEventListener("input", () => {
    const s = input.value.trim().toLowerCase();
    if (!s) { close(); return; }
    items = I.queries.filter((q) => q.q.toLowerCase().includes(s)).slice(0, 12);
    box.innerHTML = items.length ? items.map((q, i) => `<li data-i="${i}">${esc(q.q)}</li>`).join("") : `<li class="muted">No match in the sample</li>`;
    box.hidden = false; sel = -1;
    $$("li[data-i]", box).forEach((li) => li.addEventListener("mousedown", (e) => { e.preventDefault(); choose(items[+li.dataset.i]); }));
  });
  input.addEventListener("keydown", (e) => {
    const lis = $$("li[data-i]", box);
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      sel = (sel + (e.key === "ArrowDown" ? 1 : -1) + lis.length) % Math.max(lis.length, 1);
      lis.forEach((li, i) => li.classList.toggle("sel", i === sel));
    } else if (e.key === "Enter" && items.length) { choose(items[Math.max(sel, 0)]); }
    else if (e.key === "Escape") close();
  });
  input.addEventListener("blur", () => setTimeout(close, 100));
}

/* ---------------- init ---------------- */
async function init() {
  initTheme();
  buildNav();
  $("#menu-btn").addEventListener("click", () => ($("#sidebar").classList.contains("open") ? closeMenu() : openMenu()));
  $("#scrim").addEventListener("click", closeMenu);
  window.addEventListener("hashchange", show);
  window.addEventListener("resize", () => requestAnimationFrame(drawLinks));
  $$("#nd-toggle .seg-btn").forEach((b) => b.addEventListener("click", () => {
    state.testSet = b.dataset.set;
    $$("#nd-toggle .seg-btn").forEach((x) => x.classList.toggle("active", x === b));
    renderTestTables();
  }));
  show();
  try {
    const [s, i] = await Promise.all([fetch("data/site.json").then((r) => r.json()), fetch("data/inspector.json").then((r) => r.json())]);
    S = s; I = i;
  } catch (e) {
    $("#load-error").hidden = false;
    return;
  }
  bindValues();
  renderTables();
  initSearch();
  show();
}
document.addEventListener("DOMContentLoaded", init);
