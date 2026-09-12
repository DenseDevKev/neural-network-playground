# Task 6 report

Implemented the Signal Atelier lesson library and persistent guided panel in the owned files. All ten canonical IDs and revision-pinned recipe references remain unchanged. Library detail selection is separate from active lesson state and never applies a recipe. Details show real dataset, hidden-layer architecture, activation, distinct dataset/model seeds, duration, summary, and step outline. Starting/replacing and restarting display the existing explicit reset consequences. Preparation retains deduplication, stale-request protection, failure reporting/retry, and one onReset call after the exact recipe prepares.

The optional onNavigate callback wraps the full start, step, resume, Show me, All lessons, finish/exit and restart action. Parent must pass requestGuardedNavigation and keep this controller mounted. Deferred transitions leave the current active state/index and target unchanged. Active lessons remain intact when other detail rows are browsed.

Targets use openSetup, navigate and setResultsTab rather than legacy view state. Features route to Setup Network; hyperparameters route to Setup Training; data to Setup Dataset; observations to Results Boundary/Learning (inspection mapping supported). Setup instructions explicitly require shared Apply changes and distinguish the two seeds. Observable setup completion now checks actual destination/workspaceTab/setupTab. Training completion snapshots the pre-reset model before reset can erase it and excludes that generation even if delayed evidence advances its revision.

Scoped semantic CSS implements a split library/detail layout and desktop active panel, with compact collapsible mobile controls, native select, keyboard focus and 44px targets. Lazy loading remains through EducationContent.ts. No new dependencies, scientific schema or engine changes.

Verification: 43 focused registry/panel tests pass, including 70 catalog-source to canonical-lesson starts, exact preparation and stale request assertions, keyboard controls, error/retry availability, selection-only browsing, dirty guard deferral, and pre-reset generation evidence rejection. Web production and test TypeScript checks pass. Owned ESLint passes. No full-suite rerun. Reference images 10 and 11 inspected; browser matrix and aggregate bundle gates remain parent-owned.

Integration concerns: parent must position .atelier-lesson-host beside the active workspace and pass onNavigate. Active panel alone cannot enforce dirty draft behavior without that callback. This report does not claim visual browser validation or aggregate bundle acceptance.

## Fix round 1 — pending preparation lifecycle

Read task-6-review.md and addressed the P2 restart/exit race using its permitted consistent-disable approach. While lesson preparation is pending, Exit, Finish, Previous, Continue, Show me, All lessons and Resume are disabled. Deferred step/exit/navigation guard callbacks also check the synchronous in-flight reference, so an already queued callback cannot alter lesson progress during preparation. A status message explains why navigation is temporarily unavailable. The existing preparation ownership and single reset after success remain unchanged.

Added delayed active-restart coverage proving Exit and transitions cannot change state before completion, progress resets exactly once after successful preparation, and Exit then clears active state. Added delayed replacement coverage proving Resume cannot override the selected replacement. Focused panel/registry tests now pass 45/45; owned ESLint passes. No broad suite or additional scope changes.
