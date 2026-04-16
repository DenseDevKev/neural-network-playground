import re

with open('apps/web/index.html', 'r') as f:
    html = f.read()

# Add Google Fonts
fonts_link = """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@400;500;600&display=swap" rel="stylesheet">
</head>
"""

html = html.replace('</head>', fonts_link)

with open('apps/web/index.html', 'w') as f:
    f.write(html)
