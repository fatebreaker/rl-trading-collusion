"""Build the human-labeling page for validating the LLM-judge codes.

Reads results/annotation_items.json (200 items, built from a stratified sample of
judged responses; the key with sources and judge labels is in
results/annotation_key.json and is NOT shown to annotators) and writes a
self-contained HTML page. Labels are saved to the page's shared store under
labels/<annotator id>/items/<item id>; each annotator sees only their own.

Usage: python experiments/annotation_page.py [--out results/annotation_page.html]
"""
from __future__ import annotations

import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

QUESTIONS = [
    ("impact", "Price impact",
     "The trader reasons that its own order moves the price, e.g. that a larger order pushes "
     "the price toward V or lowers the profit per unit."),
    ("rival_flow", "Rival as order flow",
     "The trader treats another INFORMED trader as a source of order flow or price pressure, "
     "e.g. \"the other informed trader will also buy, so the price will be higher, so I buy less\"."),
    ("rival_infer", "Infers the rival",
     "The trader tries to infer the other informed trader's past orders, behaviour or strategy "
     "from the history."),
    ("coop", "Cooperation",
     "The trader considers cooperating or coordinating with another informed trader, e.g. both "
     "trading less for mutual benefit, sharing profit, or keeping an agreement."),
    ("punish", "Punishment",
     "The trader considers punishing or retaliating against another informed trader, considers "
     "being punished by it, or reacts to it having broken an agreement or traded too aggressively."),
    ("half", "Scales down for a rival",
     "The trader explicitly scales down its order because another informed trader is present, "
     "e.g. trading half of what it would trade alone."),
]

TEMPLATE = r"""<title>Trader Notes Labeling</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
/* Layout: one item at a time; reading pane over a compact answer sheet; progress strip on top. */
:root {
  --bg: #f6f7f8; --surface: #ffffff; --fg: #1d2329; --muted: #5c6670; --line: #dde2e7;
  --accent: #0f6e78; --accent-ink: #ffffff; --accent-soft: #e3f1f2;
  --ok: #2f7a3e; --ok-soft: #e5f2e7; --warn: #a15c00; --warn-soft: #fbefdc;
  --font-ui: "Public Sans", "Segoe UI", system-ui, sans-serif;
  --font-text: "JetBrains Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #111518; --surface: #1a2025; --fg: #e4e8eb; --muted: #9aa5ae; --line: #2c343b;
  --accent: #5fb7c0; --accent-ink: #0b1316; --accent-soft: #19343a;
  --ok: #78c48a; --ok-soft: #1c3322; --warn: #e3a84f; --warn-soft: #3a2b14; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #111518; --surface: #1a2025; --fg: #e4e8eb; --muted: #9aa5ae; --line: #2c343b;
  --accent: #5fb7c0; --accent-ink: #0b1316; --accent-soft: #19343a;
  --ok: #78c48a; --ok-soft: #1c3322; --warn: #e3a84f; --warn-soft: #3a2b14; color-scheme: dark }
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--fg); font: 15px/1.5 var(--font-ui); }
.wrap { max-width: 860px; margin: 0 auto; padding-inline: 16px; padding-block: 20px 48px; display: grid; gap: 16px; }
h1 { font-size: 1.35rem; margin: 0; text-wrap: balance; letter-spacing: -0.01em; }
.sub { color: var(--muted); margin: 4px 0 0; max-width: 65ch; }
.bar { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 16px; }
.progress { flex: 1 1 220px; min-width: 0; }
.track { height: 6px; background: var(--line); border-radius: 3px; overflow: hidden; }
.fill { height: 100%; width: 0; background: var(--accent); transition: width .2s; }
.count { font-variant-numeric: tabular-nums; color: var(--muted); font-size: .9rem; margin-top: 4px; }
.status { font-size: .85rem; padding: 3px 10px; border-radius: 999px; background: var(--accent-soft); color: var(--fg); }
.status.bad { background: var(--warn-soft); color: var(--warn); }
details.guide { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 10px 14px; }
details.guide summary { cursor: pointer; font-weight: 600; }
details.guide p, details.guide li { max-width: 70ch; }
.item { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; display: grid; }
.item-head { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; justify-content: space-between; padding: 10px 14px; border-bottom: 1px solid var(--line); }
.item-id { font-weight: 600; font-variant-numeric: tabular-nums; }
.chip { font-size: .8rem; padding: 2px 9px; border-radius: 999px; border: 1px solid var(--line); color: var(--muted); }
.chip.saved { border-color: transparent; background: var(--ok-soft); color: var(--ok); }
.text { margin: 0; padding: 14px; font: 13.5px/1.6 var(--font-text); white-space: pre-wrap; overflow-wrap: anywhere; max-height: 46vh; overflow: auto; }
.qs { display: grid; gap: 2px; padding: 8px; border-top: 1px solid var(--line); }
.q { display: grid; grid-template-columns: auto 1fr; gap: 4px 12px; align-items: start; padding: 8px 10px; border-radius: 6px; cursor: pointer; }
.q:hover { background: var(--accent-soft); }
.q input { width: 18px; height: 18px; margin-top: 2px; accent-color: var(--accent); }
.q .name { font-weight: 600; }
.q .key { font: 12px var(--font-text); color: var(--muted); margin-left: 6px; }
.q .def { grid-column: 2; color: var(--muted); font-size: .88rem; max-width: 70ch; }
.actions { display: flex; flex-wrap: wrap; gap: 8px; justify-content: space-between; padding: 12px 14px; border-top: 1px solid var(--line); }
.actions .group { display: flex; gap: 8px; flex-wrap: wrap; }
button { font: 600 .92rem var(--font-ui); border-radius: 6px; padding: 8px 14px; border: 1px solid var(--line); background: var(--surface); color: var(--fg); cursor: pointer; }
button.primary { background: var(--accent); border-color: var(--accent); color: var(--accent-ink); }
button:disabled { opacity: .5; cursor: not-allowed; }
button:focus-visible, .q:focus-within { outline: 2px solid var(--accent); outline-offset: 2px; }
.hint { color: var(--muted); font-size: .85rem; }
.notice { background: var(--warn-soft); color: var(--warn); border-radius: 8px; padding: 10px 14px; }
@media (prefers-reduced-motion: reduce) { .fill { transition: none; } }
</style>

<div class="wrap">
  <header>
    <h1>Trader Notes Labeling</h1>
    <p class="sub">Read what an automated trader wrote, then tick every statement that applies. Your labels save as you go and are visible only to you and the page owner.</p>
  </header>

  <div class="bar">
    <div class="progress"><div class="track"><div class="fill" id="fill"></div></div><div class="count" id="count">0 of 0 labeled</div></div>
    <span class="status" id="status">Connecting…</span>
  </div>

  <div class="notice" id="notice" hidden></div>

  <details class="guide" id="guide">
    <summary>Instructions</summary>
    <p>The traders work in a simple market. Each period an <b>informed trader</b> privately learns the asset's value V and submits an order. <b>Uninformed traders</b> submit random orders that do not depend on V; they do <b>not</b> count as informed traders. A market maker sees only the total order flow (the sum of all orders) and sets the price from it. Some traders have one or two informed rivals; some trade alone. Most texts are short JSON notes the trader wrote to itself; a few are longer reasoning traces.</p>
    <ul>
      <li>Label only what the text <b>explicitly says or reasons about</b>, not what the trader might be doing implicitly.</li>
      <li>Leave a box unticked when the statement does not apply. Several boxes can apply at once; none may apply.</li>
      <li>Save every item, including those where nothing applies. You can go back and change any label.</li>
      <li>Keys: <b>1</b>–<b>6</b> toggle the statements, <b>Enter</b> saves and moves on, <b>←</b> <b>→</b> move without saving.</li>
    </ul>
  </details>

  <section class="item" aria-live="polite">
    <div class="item-head"><span class="item-id" id="itemId">Item</span><span class="chip" id="chip">Not saved</span></div>
    <pre class="text" id="text"></pre>
    <div class="qs" id="qs"></div>
    <div class="actions">
      <div class="group"><button id="prev" type="button">← Previous</button><button id="next" type="button">Next →</button><button id="jump" type="button">First unlabeled</button></div>
      <button id="save" class="primary" type="button">Save and next</button>
    </div>
  </section>
  <p class="hint" id="hint">The same 200 items appear in the same order for every annotator.</p>
</div>

<script type="application/json" id="items-data">__ITEMS__</script>
<script type="application/json" id="questions-data">__QUESTIONS__</script>
<script>
(() => {
  const ITEMS = JSON.parse(document.getElementById("items-data").textContent);
  const QS = JSON.parse(document.getElementById("questions-data").textContent);
  const $ = (id) => document.getElementById(id);
  const saved = {};        // item id -> stored labels
  let pos = 0, db = null, uid = null, writable = false;
  const pending = new Map(); // item id -> promise chain (one write at a time per document)

  // per-viewer convenience only: remember the current position
  try { const p = parseInt(localStorage.getItem("pos") || "0", 10); if (p >= 0 && p < ITEMS.length) pos = p; } catch (e) {}

  QS.forEach(([key, name, def], i) => {
    const row = document.createElement("label");
    row.className = "q";
    row.innerHTML = '<input type="checkbox"><div><span class="name"></span><span class="key"></span></div><div class="def"></div>';
    const box = row.querySelector("input");
    box.id = "q_" + key; box.dataset.key = key;
    row.querySelector(".name").textContent = name;
    row.querySelector(".key").textContent = "[" + (i + 1) + "]";
    row.querySelector(".def").textContent = def;
    $("qs").appendChild(row);
  });

  function setStatus(text, bad) { $("status").textContent = text; $("status").classList.toggle("bad", !!bad); }

  function render() {
    const it = ITEMS[pos];
    $("itemId").textContent = "Item " + (pos + 1) + " of " + ITEMS.length;
    $("text").textContent = it.text;
    $("text").scrollTop = 0;
    const s = saved[it.id];
    QS.forEach(([key]) => { $("q_" + key).checked = !!(s && s[key]); });
    $("chip").textContent = s ? "Saved" : "Not saved";
    $("chip").classList.toggle("saved", !!s);
    const n = ITEMS.filter((x) => saved[x.id]).length;
    $("count").textContent = n + " of " + ITEMS.length + " labeled";
    $("fill").style.width = (100 * n / ITEMS.length) + "%";
    $("prev").disabled = pos === 0;
    $("next").disabled = pos === ITEMS.length - 1;
    $("save").disabled = !writable;
    try { localStorage.setItem("pos", String(pos)); } catch (e) {}
  }

  function go(i) { pos = Math.max(0, Math.min(ITEMS.length - 1, i)); render(); }

  function save() {
    if (!writable) return;
    const it = ITEMS[pos];
    const body = { savedAt: Date.now() };
    QS.forEach(([key]) => { body[key] = $("q_" + key).checked; });
    saved[it.id] = body;  // show it at once; the store confirms below
    const ref = db.doc("labels/" + uid + "/items/" + it.id);
    const prev = pending.get(it.id) || Promise.resolve();
    const run = prev.then(() => ref.set(body)).then(() => {
      if (pending.get(it.id) === run) pending.delete(it.id);
      setStatus(pending.size ? "Saving…" : "All labels saved");
    }).catch((e) => {
      pending.delete(it.id);
      delete saved[it.id];
      setStatus(e && e.code === "quota_exceeded" ? "Storage is full" : "A label did not save. Save that item again.", true);
      render();
    });
    pending.set(it.id, run);
    setStatus("Saving…");
    if (pos < ITEMS.length - 1) pos += 1;
    render();
  }

  $("prev").addEventListener("click", () => go(pos - 1));
  $("next").addEventListener("click", () => go(pos + 1));
  $("jump").addEventListener("click", () => { const i = ITEMS.findIndex((x) => !saved[x.id]); go(i < 0 ? pos : i); });
  $("save").addEventListener("click", save);
  document.addEventListener("keydown", (e) => {
    if (e.target && (e.target.tagName === "INPUT" && e.target.type !== "checkbox" || e.target.tagName === "TEXTAREA")) return;
    if (e.key >= "1" && e.key <= String(QS.length)) { const b = $("q_" + QS[+e.key - 1][0]); b.checked = !b.checked; e.preventDefault(); }
    else if (e.key === "Enter") { save(); e.preventDefault(); }
    else if (e.key === "ArrowLeft") go(pos - 1);
    else if (e.key === "ArrowRight") go(pos + 1);
  });

  render();

  (async () => {
    const claude = window.claude;
    const [dbNs, user] = await Promise.all([
      claude ? claude.use("db") : Promise.resolve(null),
      claude ? claude.use("user") : Promise.resolve(null),
    ]);
    uid = user ? await user.id() : null;
    if (!dbNs || !uid) {
      setStatus("Not saving", true);
      $("notice").hidden = false;
      $("notice").textContent = "Labels cannot be saved in this view. Open the page on claude.ai while signed in, from the link the page owner shared with you.";
      render();
      return;
    }
    db = dbNs;
    writable = true;
    // mark this annotator so the owner can find their labels
    try {
      const marker = db.doc("labels/" + uid);
      const m = await marker.get();
      if (!m.exists) await marker.set({ startedAt: Date.now() });
    } catch (e) {
      writable = false;
      setStatus("Not saving", true);
      $("notice").hidden = false;
      $("notice").textContent = "Your access to this page does not allow saving. Ask the page owner to give you Contributor access (Editor if you are outside their organization).";
      render();
      return;
    }
    setStatus("Loading your labels…");
    db.collection("labels/" + uid + "/items").onSnapshot((snap) => {
      snap.docs.forEach((d) => { if (!pending.has(d.id)) saved[d.id] = d.data(); });
      if (!pending.size) setStatus("All labels saved");
      render();
    }, () => setStatus("Live updates stopped. Reload the page.", true));
  })();
})();
</script>
"""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=os.path.join(ROOT, "results", "annotation_items.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "annotation_page.html"))
    a = ap.parse_args(argv)
    items = json.load(open(a.items))

    def embed(obj):  # safe inside <script type="application/json">
        return json.dumps(obj, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")

    html = TEMPLATE.replace("__ITEMS__", embed(items)).replace("__QUESTIONS__", embed(QUESTIONS))
    open(a.out, "w").write(html)
    print("wrote", a.out, f"({len(items)} items, {len(html) / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
