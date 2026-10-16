"""Build the Bellman / lambda page from its template and results/viz-export.json.

usage: python viz_export.py > results/viz-export.json && python viz/build.py
Published as a claude.ai artifact: https://claude.ai/artifact/15uw8AewMu7ZNdPRhobum1
"""
from pathlib import Path

here = Path(__file__).parent
data = (here.parent / "results" / "viz-export.json").read_text().strip()
page = (here / "bellman-lambda.template.html").read_text().replace("__DATA__", data)
(here / "bellman-lambda.html").write_text(page)
