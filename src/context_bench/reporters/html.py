"""Self-contained HTML reporter with styled tables and charts."""

from __future__ import annotations

import html
import json

from context_bench.results import EvalResult


def to_html(result: EvalResult) -> str:
    """Generate a self-contained HTML report from an EvalResult."""
    if not result.summary:
        return "<html><body><p>No results to report.</p></body></html>"

    systems = list(result.summary.keys())
    all_metrics: list[str] = []
    for sys_metrics in result.summary.values():
        for m in sys_metrics:
            if m not in all_metrics:
                all_metrics.append(m)

    # Build summary table rows
    table_rows = ""
    for system in systems:
        cells = f"<td><strong>{html.escape(system)}</strong></td>"
        for metric in all_metrics:
            v = result.summary[system].get(metric)
            if v is None:
                cells += "<td>—</td>"
            elif isinstance(v, float) and v != float("inf"):
                cells += f"<td>{v:.4f}</td>"
            else:
                cells += f"<td>{html.escape(str(v))}</td>"
        table_rows += f"<tr>{cells}</tr>\n"

    header_cells = "<th>System</th>" + "".join(
        f"<th>{html.escape(m)}</th>" for m in all_metrics
    )

    # Timing section
    timing_html = ""
    if result.timing:
        timing_items = "".join(
            f"<li><strong>{html.escape(s)}</strong>: {t:.2f}s</li>"
            for s, t in result.timing.items()
        )
        timing_html = f"<h2>Timing</h2><ul>{timing_items}</ul>"

    # Per-row data for the interactive table
    row_data = json.dumps([
        {
            "system": r.system,
            "example_id": str(r.example_id),
            "dataset": r.dataset,
            "latency": round(r.latency, 4),
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            **{k: round(v, 4) for k, v in r.scores.items()},
        }
        for r in result.rows
    ])

    num_examples = result.config.get("num_examples", "?")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>context-bench Results</title>
<style>
  :root {{
    --bg: #ffffff; --fg: #1a1a2e; --accent: #0f3460;
    --border: #e0e0e0; --hover: #f0f4ff; --good: #2e7d32; --bad: #c62828;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #1a1a2e; --fg: #e0e0e0; --accent: #4fc3f7;
      --border: #333; --hover: #16213e; --good: #66bb6a; --bad: #ef5350;
    }}
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg); color: var(--fg); padding: 2rem; line-height: 1.6;
  }}
  h1 {{ margin-bottom: 1.5rem; color: var(--accent); }}
  h2 {{ margin: 1.5rem 0 0.75rem; color: var(--accent); }}
  table {{
    border-collapse: collapse; width: 100%; margin: 1rem 0;
    font-size: 0.9rem;
  }}
  th, td {{ padding: 0.5rem 0.75rem; border: 1px solid var(--border); text-align: right; }}
  th {{ background: var(--accent); color: #fff; text-align: center; position: sticky; top: 0; }}
  td:first-child {{ text-align: left; font-weight: 600; }}
  tr:hover {{ background: var(--hover); }}
  ul {{ margin: 0.5rem 0 0.5rem 1.5rem; }}
  .meta {{ color: #888; font-size: 0.85rem; margin-top: 1rem; }}
  .bar-container {{ display: flex; align-items: center; gap: 0.5rem; }}
  .bar {{ height: 18px; background: var(--accent); border-radius: 3px; min-width: 2px; }}
  #filter {{ margin: 0.5rem 0; padding: 0.4rem 0.6rem; width: 300px; border: 1px solid var(--border); border-radius: 4px; }}
  #details {{ max-height: 500px; overflow: auto; }}
  #details table {{ font-size: 0.8rem; }}
</style>
</head>
<body>
<h1>context-bench Results</h1>

<h2>Summary</h2>
<table>
<thead><tr>{header_cells}</tr></thead>
<tbody>{table_rows}</tbody>
</table>

{timing_html}

<h2>Per-Example Details</h2>
<input type="text" id="filter" placeholder="Filter by system or dataset..." oninput="filterRows()">
<div id="details"></div>

<p class="meta">{num_examples} examples evaluated</p>

<script>
const rowData = {row_data};
const detailDiv = document.getElementById('details');
const filterInput = document.getElementById('filter');

function renderTable(data) {{
  if (!data.length) {{ detailDiv.innerHTML = '<p>No rows.</p>'; return; }}
  const cols = Object.keys(data[0]);
  let h = '<table><thead><tr>' + cols.map(c => '<th>' + c + '</th>').join('') + '</tr></thead><tbody>';
  for (const row of data) {{
    h += '<tr>' + cols.map(c => '<td>' + (row[c] ?? '') + '</td>').join('') + '</tr>';
  }}
  h += '</tbody></table>';
  detailDiv.innerHTML = h;
}}

function filterRows() {{
  const q = filterInput.value.toLowerCase();
  if (!q) {{ renderTable(rowData); return; }}
  renderTable(rowData.filter(r =>
    r.system.toLowerCase().includes(q) || r.dataset.toLowerCase().includes(q)
  ));
}}

renderTable(rowData);
</script>
</body>
</html>"""
