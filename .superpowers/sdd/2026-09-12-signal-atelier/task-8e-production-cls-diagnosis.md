# Production CLS diagnosis

Read-only probe against the existing normal-production preview at http://127.0.0.1:4173 reproduced one shift at each width: 1440: 0.000026131519793810012; 768: 0.0000406274112654321. Both sources were Text node `.` with parent `<span class="atelier-artifact-step">Grid at step 1100.</span>`. The full-evaluation caption was not the source. Parent rectangles stayed fixed (1440: x1091.84375 y794.53125 w316.140625 h22.390625; 768: x396 y890.390625 w348 h22.390625).

Cause: the separate static period sibling moves horizontally as the dynamic step gains a digit. Proposed fix: render the identical Grid caption using one interpolated text node, preserving the block dimensions and all provenance.

Reproduce: `node .superpowers/sdd/2026-09-12-signal-atelier/task-8e-production-cls-probe.mjs`. Raw receipt: `/tmp/task8e-production-cls-details.json`; log: `/tmp/task8e-production-cls-probe.log`. This diagnostic did not modify app sources, rebuild, or restart servers.
