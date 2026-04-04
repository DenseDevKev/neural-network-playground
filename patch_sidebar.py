import re

with open('apps/web/src/components/layout/Sidebar.tsx', 'r') as f:
    content = f.read()

# Add a toggle button for the whole sidebar
import_str = "import { memo, useState } from 'react';"
content = content.replace("import { memo } from 'react';", import_str)

component_str = """
export const Sidebar = memo(function Sidebar({ onReset }: SidebarProps) {
    const hiddenLayers = usePlaygroundStore((s) => s.network.hiddenLayers);
    const [isSidebarExpanded, setIsSidebarExpanded] = useState(true);

    return (
        <aside className={`sidebar ${isSidebarExpanded ? '' : 'sidebar--collapsed'}`} role="complementary" aria-label="Configuration">
            <div className="sidebar__toggle-container">
                <button
                    className="sidebar__toggle btn btn--icon"
                    onClick={() => setIsSidebarExpanded(!isSidebarExpanded)}
                    aria-label={isSidebarExpanded ? 'Collapse sidebar' : 'Expand sidebar'}
                >
                    {isSidebarExpanded ? '←' : '→'}
                </button>
            </div>
            <div className="sidebar__content" style={{ display: isSidebarExpanded ? 'block' : 'none' }}>
"""

end_str = """
            </div>
        </aside>
    );
});
"""

content = re.sub(r'export const Sidebar = memo\(function Sidebar\(\{ onReset \}: SidebarProps\) \{[^\}]*return \(\s*<aside[^>]*>', component_str, content)
content = re.sub(r'</aside>\s*\);\s*\}\);', end_str, content)

with open('apps/web/src/components/layout/Sidebar.tsx', 'w') as f:
    f.write(content)
