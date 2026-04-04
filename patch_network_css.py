with open('apps/web/src/styles/index.css', 'r') as f:
    css = f.read()

network_styles = """
/* Network Visualization Specifics */
.network-edge {
    transition: stroke-width 0.2s, stroke-opacity 0.2s;
    filter: drop-shadow(0 0 4px var(--color-primary-glow));
}

.network-node {
    filter: drop-shadow(0 0 8px var(--color-primary-glow));
}
"""

css += network_styles

with open('apps/web/src/styles/index.css', 'w') as f:
    f.write(css)
