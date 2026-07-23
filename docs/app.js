// Shared helpers for index.html / run.html. No build step, no dependencies --
// plain ES module served as-is by GitHub Pages.

export async function fetchJSON(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.json();
}

export async function fetchJSONL(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  const text = await response.text();
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

export function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = value == null ? "" : String(value);
  return div.innerHTML;
}

export function formatPct(value) {
  return value == null ? "-" : `${(value * 100).toFixed(1)}%`;
}

export function formatDelta(single, agentic) {
  if (single == null || agentic == null) return "-";
  const delta = agentic - single;
  const sign = delta >= 0 ? "+" : "";
  const cls = delta >= 0 ? "delta-pos" : "delta-neg";
  return `<span class="${cls}">${sign}${(delta * 100).toFixed(1)}%</span>`;
}

export function badge(correct) {
  return correct
    ? '<span class="badge good">correct</span>'
    : '<span class="badge bad">wrong</span>';
}

export function qs(name) {
  return new URLSearchParams(window.location.search).get(name);
}
