"""Build four presentation previews from the live demo markup.

Keeping one HTML source means each preview uses the same cases, controls, audio,
MIDI rolls, paper results, provenance, and accessibility labels as index.html.
"""

from pathlib import Path


DOCS = Path(__file__).resolve().parents[1] / "docs"
SOURCE = (DOCS / "index.html").read_text()

STYLES = {
    "01": ("paper", "Research page", "A compact, paper-like layout with teal headings and pale MIDI plots."),
    "02": ("editorial", "Editorial", "Wide typography, generous whitespace, and a stacked listening comparison."),
    "03": ("gallery", "Gallery", "A warm specimen gallery with four methods across the page."),
    "04": ("lab", "Lab desk", "A bright analysis workspace with the track selector in a side rail."),
}

for number, (slug, title, description) in STYLES.items():
    html = SOURCE.replace(
        '<link rel="stylesheet" href="style.css?v=white-layout-4">',
        '<link rel="stylesheet" href="style.css?v=white-layout-4">\n  <link rel="stylesheet" href="style-previews.css">\n  <meta name="robots" content="noindex">',
        1,
    ).replace(
        "<body>", f'<body class="style-preview preview-{slug}">', 1
    ).replace(
        "<title>Beyond Piano · Diff-SFProxy demo</title>",
        f"<title>{number} · {title} | Beyond Piano demo</title>", 1,
    ).replace(
        'Compare four presentation styles. The current demo remains the published layout.',
        f'{number} / {title}: {description} <a class="return-current" href="index.html#listen">Return to current demo ↗</a>',
        1,
    )
    html = html.replace(
        f'<a href="style-{number}.html#listen">',
        f'<a class="is-current" aria-current="page" href="style-{number}.html#listen">',
        1,
    )
    (DOCS / f"style-{number}.html").write_text(html)
