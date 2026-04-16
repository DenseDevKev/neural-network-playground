import re

with open('apps/web/src/components/visualization/NetworkGraph.tsx', 'r') as f:
    content = f.read()

# Change the node colors
content = re.sub(
    r'if \(value > 0\) return `rgba\(249, 115, 22, \$\{0\.3 \+ abs \* 0\.7\}\)`;\n\s*return `rgba\(59, 130, 246, \$\{0\.3 \+ abs \* 0\.7\}\)`;',
    'if (value > 0) return `rgba(129, 236, 255, ${0.4 + abs * 0.6})`;\n    return `rgba(188, 135, 254, ${0.4 + abs * 0.6})`;',
    content
)

# Change the edge colors
content = re.sub(
    r'if \(weight > 0\) return `rgba\(249, 115, 22, \$\{0\.15 \+ abs \* 0\.6\}\)`;\n\s*return `rgba\(59, 130, 246, \$\{0\.15 \+ abs \* 0\.6\}\)`;',
    'if (weight > 0) return `rgba(129, 236, 255, ${0.2 + abs * 0.6})`;\n    return `rgba(188, 135, 254, ${0.2 + abs * 0.6})`;',
    content
)

with open('apps/web/src/components/visualization/NetworkGraph.tsx', 'w') as f:
    f.write(content)
