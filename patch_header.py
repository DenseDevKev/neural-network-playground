import re

with open('apps/web/src/components/layout/Header.tsx', 'r') as f:
    content = f.read()

# Replace Neural Network Playground with Neural Architect
content = content.replace('<span>Neural Network</span> Playground', '<span>Neural</span> Architect')
content = content.replace('<div className="header__logo">NN</div>', '')

with open('apps/web/src/components/layout/Header.tsx', 'w') as f:
    f.write(content)
