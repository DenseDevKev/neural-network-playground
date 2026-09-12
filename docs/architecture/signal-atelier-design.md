# Signal Atelier design contract

The complete September 12, 2026 NN·FORGE vision is an interactive scientific workspace. The primary composition is data → network → prediction. Generated reference images set composition and hierarchy; all displayed values come from accepted engine artifacts.

Runtime `apps/web/src/styles/atelier.css` is the canonical token source. Theme preferences are System, Light and Dark, initialized before paint and synchronized with device changes only in System mode. A storage failure keeps the current session choice usable.

| Role | Light | Dark |
|---|---|---|
| Canvas | #F7F6F2 | #17191B |
| Control surface | #EFEEE9 | #202326 |
| Main text | #202225 | #EEEDE8 |
| Supporting text | #666970 | #BCC0C2 |
| Rules | #D7D7D2 | #43484D |
| Accent | #D64C28 | #EF653F |
| Primary action | #C74424 / white | #EF653F / #17191B |

Use installed Inter for text and Space Grotesk for numeric evidence. Body 16px, labels 14px, page headings 32px; spacing 4/8/12/16/24/32/48. Surfaces stay flat, controls lightly rounded, neuron activation maps square. Class and signed-weight colors retain their meanings between themes and have labels or line styles.

Desktop from 1200px uses horizontal experiment composition. From 760–1199px, supporting regions stack around a dedicated graph. Below 760px, Data, Network and Prediction use focused region tabs; graph/table overflow stays local. Controls have 44px targets, safe-area padding, visible keyboard focus, and respect reduced motion and forced colors.

One modal owns focus at a time. Escape dismisses the innermost dismissible surface. Essential errors persist beside their actions; transient status is reserved for acknowledgements. Setup changes have a shared Apply changes / Cancel footer and a guarded exit. Numeric drafts may be incomplete; canonical candidate validation determines whether they can apply.

Every visual acceptance capture must use deterministic recipes and real engine output. Reference accuracy values must never become fixture UI data. Review light/dark captures at 1440/1280/1024/768/390/360px, breakpoint boundaries, landscape and 200% zoom before creating screenshot baselines.

Responsive rules use the `atelier` root container so browser/document zoom also
changes composition. `useAtelierViewport` exposes the usable width/height for
compact mounting and virtual-keyboard-aware forms. The training transport stays in normal flow above the content at every width;
the workspace tab row remains reachable. Setup uses a bounded scrolling form and
a separate Apply/Cancel footer, keeping actions available without covering fields. Large graphs keep local pan,
zoom, and a bounded structural summary instead of shrinking neuron targets.

The approved source gallery contains 20 states. Their complete feature mapping
and release checklist are maintained in `docs/qa/signal-atelier-acceptance.md`.
