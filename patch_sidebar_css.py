with open('apps/web/src/styles/index.css', 'r') as f:
    css = f.read()

sidebar_styles = """
/* Sidebar toggling */
.sidebar {
    transition: width 0.3s ease;
    display: flex;
    flex-direction: column;
}

.sidebar--collapsed {
    width: 60px;
}

.sidebar__toggle-container {
    padding: 10px;
    display: flex;
    justify-content: flex-end;
}

.sidebar__toggle {
    background: transparent;
    border: none;
    color: var(--text-secondary);
    cursor: pointer;
    font-size: 1.2rem;
}
.sidebar__toggle:hover {
    color: var(--text-primary);
}
"""

css += sidebar_styles

with open('apps/web/src/styles/index.css', 'w') as f:
    f.write(css)
