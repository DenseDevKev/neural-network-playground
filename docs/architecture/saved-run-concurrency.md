# Saved-run write ownership

All current tabs coordinate the V2 localStorage envelope with an exclusive Web Lock named after its storage key. A tab retains its own action queue, acquires the origin lock, rereads and validates storage, applies its operation, verifies serialization, and writes once before releasing the lock. Hydration uses the same lock so an asynchronous read cannot publish a stale view after a newer coordinated write.

Operations follow lock-acquisition order. Saves add an exact validated worker artifact; rename/remove use record IDs; clear removes current valid records. There is no silent capacity eviction. Retries add the original pending artifact to the latest envelope and never contact the worker or recapture the model. A failed write retains the artifact and leaves persisted bytes unchanged.

Rejected deletion follows selected raw bytes instead of a stale array index. Deleting incompatible or legacy storage requires the exact originally selected file. Mutations that would promote rejected duplicates into accepted runs are blocked until those rejected entries are explicitly removed. Neither the storage schema nor the scientific identities change.

Browsers without Web Locks can read/export records, but writes fail with a visible error and a retained artifact. Supported deployment is HTTPS (or localhost for development). Web Locks are cooperative: reload already-open older app versions after upgrading. A byte check immediately before writing additionally catches uncoordinated changes during asynchronous validation; it does not claim to make arbitrary external localStorage writers transactional.

Regression coverage: `experimentMemoryStore.test.ts`, the save and storage-sync hook suites, and `saved-run-concurrency.spec.ts` with two real tabs in Chromium and WebKit. The browser tests separately cover stale hydration and a second writer waiting while the first validates its envelope under the lock.

[Web Locks specification](https://w3c.github.io/web-locks/).
