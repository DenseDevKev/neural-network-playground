import re

with open('apps/web/src/styles/index.css', 'r') as f:
    css = f.read()

# Replace variables
replacements = {
    r'--bg-base:.*': '--bg-base: #0e0e0e;',
    r'--bg-surface:.*': '--bg-surface: #131313;',
    r'--bg-elevated:.*': '--bg-elevated: #1a1a1a;',
    r'--bg-hover:.*': '--bg-hover: #262626;',
    r'--bg-input:.*': '--bg-input: #1a1a1a;',

    r'--color-primary:.*': '--color-primary: #81ecff;',
    r'--color-primary-dim:.*': '--color-primary-dim: #00d4ec;',
    r'--color-primary-glow:.*': '--color-primary-glow: rgba(129, 236, 255, 0.25);',
    r'--color-accent:.*': '--color-accent: #bc87fe;',
    r'--color-accent-dim:.*': '--color-accent-dim: #ba85fb;',
    r'--color-accent-glow:.*': '--color-accent-glow: rgba(188, 135, 254, 0.25);',

    r'--border-subtle:.*': '--border-subtle: rgba(255, 255, 255, 0.05);',
    r'--border-default:.*': '--border-default: #262626;',

    r'--text-primary:.*': '--text-primary: #ffffff;',
    r'--text-secondary:.*': '--text-secondary: #adaaaa;',

    r'--font-sans:.*': '--font-sans: "Inter", system-ui, sans-serif;',
    r'--font-mono:.*': '--font-mono: "Space Grotesk", monospace;'
}

for pattern, replacement in replacements.items():
    css = re.sub(pattern, replacement, css)

with open('apps/web/src/styles/index.css', 'w') as f:
    f.write(css)

print("Replaced variables.")
