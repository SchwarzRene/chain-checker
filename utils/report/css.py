def shared_css() -> str:
    return """
:root {
  --bg: #f4f5f7;
  --card-bg: #ffffff;
  --text: #1c1f26;
  --muted: #6b7280;
  --border: #e3e5ea;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14161b;
    --card-bg: #1d2027;
    --text: #eceef2;
    --muted: #9aa1ad;
    --border: #2b2f38;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 2rem;
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
h1, h2 { margin: 0; }
.run { max-width: 1100px; margin: 0 auto 3rem auto; }
.run-title {
  font-size: 1.4rem;
  padding-bottom: 0.75rem;
  border-bottom: 2px solid var(--border);
  margin-bottom: 1.5rem;
}
.big-stats {
  display: flex;
  justify-content: center;
  align-items: baseline;
  gap: 3rem;
  padding: 1.5rem 0 2rem 0;
  flex-wrap: wrap;
}
.big-stat { text-align: center; }
.big-stat-value {
  font-size: 2.75rem;
  font-weight: 700;
  line-height: 1;
}
.big-stat-primary { font-size: 5rem; }
.big-stat-label {
  color: var(--muted);
  font-size: 1rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-top: 0.25rem;
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 1.25rem;
}
.card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1rem 1.25rem 1.25rem 1.25rem;
}
.card-header {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  margin-bottom: 0.9rem;
}
.card-title { font-weight: 600; font-size: 0.95rem; }
.card-badge {
  font-size: 0.8rem;
  color: var(--muted);
  font-weight: 600;
}
.empty { color: var(--muted); font-size: 0.85rem; }
.raw {
  font-family: ui-monospace, monospace;
  font-size: 0.75rem;
  white-space: pre-wrap;
  color: var(--muted);
}

/* run-info chips (chain or modifier tier/model) - shared by both report types. */
.run-info {
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem;
  margin-bottom: 0.5rem;
}
.run-info-chip {
  display: flex;
  align-items: baseline;
  gap: 0.4rem;
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 0.3rem 0.85rem;
  font-size: 0.8rem;
}
.run-info-key {
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-size: 0.68rem;
}
.run-info-value { font-weight: 600; }

/* prompt box */
.prompt-card { margin-top: 1.25rem; }
.prompt-text {
  font-family: ui-monospace, monospace;
  font-size: 0.8rem;
  white-space: pre-wrap;
  word-break: break-word;
  margin: 0;
  color: var(--text);
  max-height: 480px;
  overflow-y: auto;
}

/* collapsed prompt - shared by "Prompt evolution" and the comparison report. */
details.prompt-entry {
  border: 1px solid var(--border);
  border-radius: 12px;
  margin-bottom: 0.6rem;
  background: var(--card-bg);
}
details.prompt-entry summary {
  cursor: pointer;
  padding: 0.7rem 1rem;
  font-weight: 600;
  font-size: 0.85rem;
  list-style: none;
}
details.prompt-entry summary::-webkit-details-marker { display: none; }
details.prompt-entry summary::before { content: "▸ "; color: var(--muted); }
details.prompt-entry[open] summary::before { content: "▾ "; }
details.prompt-entry .prompt-text { padding: 0 1rem 1rem 1rem; max-height: 400px; }
"""


def epoch_report_css() -> str:
    return """
/* confusion rectangle */
.confusion {
  display: grid;
  grid-template-columns: auto 1fr 1fr;
  gap: 4px;
  align-items: stretch;
}
.cm-axis {
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.72rem;
  color: var(--muted);
  text-align: center;
  padding: 0.25rem;
}
.cm-cell {
  border: 1px solid;
  border-radius: 8px;
  padding: 0.6rem 0.4rem;
  text-align: center;
}
.cm-value { font-size: 1.6rem; font-weight: 700; }
.cm-label { font-size: 0.68rem; color: var(--muted); margin-top: 0.15rem; }

/* bar chart (categorical) */
.bars {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  max-height: 280px;
  overflow-y: auto;
  overflow-x: hidden;
  padding-right: 2px;
}
.bar-row {
  display: grid;
  grid-template-columns: 90px 1fr 40px;
  align-items: center;
  gap: 0.5rem;
  flex: 0 0 auto;
}
.bar-label {
  font-size: 0.8rem;
  color: var(--muted);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.bar-track { background: var(--border); border-radius: 4px; height: 10px; overflow: hidden; }
.bar-fill { background: var(--bar-color, #4f6df5); height: 100%; border-radius: 4px; }
.bar-value { font-size: 0.8rem; text-align: right; }

/* histogram (numeric / binned) */
.histogram {
  display: flex;
  align-items: flex-end;
  gap: 6px;
  height: 140px;
  overflow-x: auto;
  overflow-y: hidden;
  padding-bottom: 2px;
}
.hist-col {
  flex: 1 0 28px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-end;
  height: 100%;
}
.hist-count { font-size: 0.72rem; color: var(--muted); margin-bottom: 0.2rem; }
.hist-bar {
  width: 100%;
  min-height: 2px;
  background: var(--bar-color, #4f6df5);
  border-radius: 3px 3px 0 0;
}
.hist-label { font-size: 0.65rem; color: var(--muted); margin-top: 0.3rem; text-align: center; }
/* paired bar chart (true vs. predicted per entry) */
.paired-legend {
  display: flex;
  gap: 1.25rem;
  font-size: 0.75rem;
  color: var(--muted);
  margin-bottom: 0.75rem;
}
.legend-item { display: flex; align-items: center; gap: 0.4rem; }
.legend-dot { width: 10px; height: 10px; border-radius: 3px; display: inline-block; }
.paired-chart {
  display: flex;
  align-items: flex-end;
  gap: 14px;
  height: 140px;
  overflow-x: auto;
  padding-bottom: 2px;
}
.paired-group {
  flex: 0 0 auto;
  min-width: 36px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-end;
  height: 100%;
}
.paired-bars { display: flex; align-items: flex-end; gap: 4px; height: 100%; }
.paired-bar-col {
  width: 14px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-end;
  height: 100%;
}
.paired-value { font-size: 0.6rem; color: var(--muted); margin-bottom: 2px; white-space: nowrap; }
.paired-bar { width: 100%; min-height: 2px; border-radius: 3px 3px 0 0; }
.paired-label { font-size: 0.65rem; color: var(--muted); margin-top: 0.35rem; }

.comparison { max-width: 1100px; margin: 0 auto 2.5rem auto; }

/* mispredicted entries */
.mispred-section { margin-top: 1.5rem; }
.mispred-list { display: flex; flex-direction: column; gap: 1rem; }
.mispred-text {
  font-size: 0.85rem;
  color: var(--muted);
  font-style: italic;
  margin-bottom: 0.9rem;
  white-space: pre-wrap;
}
.mispred-rows { display: flex; flex-direction: column; gap: 0.6rem; }
.mispred-key { border: 1px solid var(--border); border-radius: 8px; padding: 0.6rem 0.75rem; }
.mispred-key-diff { border-color: #d6454566; background: rgba(214, 69, 69, 0.07); }
.mispred-key-name { font-size: 0.78rem; font-weight: 600; margin-bottom: 0.5rem; }
.mispred-key-sides { display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; }
.mispred-side-label {
  font-size: 0.65rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--muted);
}
.mispred-side-value { font-size: 0.88rem; font-weight: 600; margin: 0.15rem 0 0.25rem 0; }
.mispred-reason { font-size: 0.76rem; color: var(--muted); line-height: 1.4; }
"""


def summary_report_css() -> str:
    return """
.epoch-count {
  color: var(--muted);
  font-size: 0.95rem;
  text-align: center;
  margin-top: -1rem;
  margin-bottom: 1rem;
}

/* chain-info / modifier-info: two separate boxed groups so it reads as
   "this is what's being trained" vs. "this is what's rewriting it", not
   one flat row of same-looking tags where a repeated Tier/Model is
   ambiguous about which group it belongs to. */
.run-info-groups {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  margin-bottom: 0.5rem;
}
.run-info-group {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 0.55rem 0.9rem 0.7rem 0.9rem;
  flex: 1 1 260px;
}
.run-info-group .run-info { margin-bottom: 0; }
.run-info-group-label {
  color: var(--muted);
  font-size: 0.7rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 0.45rem;
}

.section { max-width: 1100px; margin: 0 auto 2.5rem auto; }
.section-title { font-size: 1.1rem; margin-bottom: 1rem; }

.card { position: relative; }
.chart-title { font-weight: 600; font-size: 0.95rem; margin-bottom: 0.75rem; }
.chart-svg { width: 100%; height: auto; overflow: visible; cursor: crosshair; }
.chart-line { fill: none; stroke-width: 2; }
.chart-area { stroke: none; fill-opacity: 0.55; }
.chart-point { stroke: var(--card-bg); stroke-width: 1.5; }
.chart-axis { stroke: var(--border); stroke-width: 1; }
.chart-axis-label { fill: var(--muted); font-size: 10px; }
.chart-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem 1.1rem;
  margin-top: 0.75rem;
  font-size: 0.78rem;
  color: var(--muted);
}
.legend-item { display: flex; align-items: center; gap: 0.4rem; }
.legend-dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; flex: 0 0 auto; }

/* hover crosshair (see hover_script.HOVER_SCRIPT) - a vertical guide line
   that snaps to the nearest epoch's x position, plus a small floating
   readout of every series' value at that point */
.chart-crosshair {
  stroke: var(--muted);
  stroke-width: 1;
  stroke-dasharray: 3 3;
  pointer-events: none;
  display: none;
}
.chart-hover-dot { pointer-events: none; display: none; stroke: var(--card-bg); stroke-width: 1.5; }
.chart-tooltip {
  position: absolute;
  z-index: 5;
  pointer-events: none;
  display: none;
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.45rem 0.65rem;
  font-size: 0.75rem;
  line-height: 1.45;
  white-space: nowrap;
  box-shadow: 0 4px 14px rgba(0,0,0,0.18);
}
.chart-tooltip .chart-tooltip-title { font-weight: 600; margin-bottom: 0.15rem; }
.chart-tooltip .chart-tooltip-row { color: var(--muted); }
.chart-tooltip .chart-tooltip-row .chart-tooltip-swatch {
  display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 0.35rem;
}
.chart-tooltip .chart-tooltip-row .chart-tooltip-value { color: var(--text); font-weight: 600; }
.chart-tooltip .chart-tooltip-row-strong {
  border-top: 1px solid var(--border);
  margin-top: 0.3rem;
  padding-top: 0.3rem;
  color: var(--text);
  font-weight: 600;
}
"""
