# Uni5DB DataFusion maintenance branch

This is a full Apache DataFusion Git checkout, not a set of unpacked Cargo
packages. Unchanged code and history belong to Apache DataFusion. Local commits
record Uni5DB's imported modifications.

## Baseline and remotes

- Upstream: https://github.com/apache/datafusion.git
- Fork: git@github.com:chenyangzhi88/datafusion.git
- Official release tag: `53.1.0`
- Verified upstream baseline: `eae7bf4fa1c037c0a065d1f36d0669f5bb97a9cf`
- Unmodified baseline branch: `upstream-base`
- Maintenance branch: `uni5db-53.1`
- `main` remains an upstream-only branch. Product changes belong on the
  maintenance branch or focused contribution branches, never on `main`.

All 258 Rust source files in the four original official 53.1.0 package archives
were byte-identical to the corresponding files at this upstream tag. All 274
source files in the four previous local snapshots, including added tests, were
preserved byte-for-byte during migration. Cargo-published normalized manifests,
package metadata and outdated snapshot provenance files were not imported as
source modifications; this checkout retains upstream workspace manifests.

## Imported changes

| Commit | Responsibility |
| --- | --- |
| `cbb4d4e87fde` | feat(core): preserve session state when replacing the query planner |
| `b3201af00ae0` | feat(expr): expose callable identity and aggregate output contracts |
| `5a93d423e3e6` | fix(physical-expr): mark null-safe comparisons as non-nullable |
| `5b3074fda1f2` | fix(async): preserve row cardinality through asynchronous evaluation |
| `be4991e771cc` | feat(unnest): admit and retain output allocations under the memory budget |
| `2c2de1367f37` | feat(execution): preserve grouping masks and observe sort and aggregate work |

The expression commit also preserves the existing move of private upstream tests
from inline modules into separate test files. Such moves are local organization
changes, not newly authored upstream test logic. Use Git's move-aware diff and
blame to inspect their origins.

Only these upstream crate source trees differ from the baseline:

- `datafusion/core`: 1 changed or added files.
- `datafusion/expr`: 8 changed or added files.
- `datafusion/physical-expr`: 9 changed or added files.
- `datafusion/physical-plan`: 22 changed or added files.

The product-owned GROUPING analyzer remains in the parent project's
`crates/compute/src/datafusion/grouping.rs`. The upstream optimizer source is
unmodified, even when built from this coherent workspace checkout.

## Inspect provenance

Run inside this repository:

```sh
git log --reverse --oneline eae7bf4fa1c037c0a065d1f36d0669f5bb97a9cf..HEAD
git diff --color-moved eae7bf4fa1c037c0a065d1f36d0669f5bb97a9cf...HEAD -- datafusion
git blame -C -C -- datafusion/physical-expr/src/expressions/binary.rs
```

## Integrate upstream updates

Fresh submodule clones pin a detached HEAD and do not inherit this machine's
extra remotes or local branch aliases. If `upstream` is absent, first add
`https://github.com/apache/datafusion.git` as that remote. The inspection commands
above use the immutable baseline SHA and work without a local tag or branch.

Fetch from `upstream`, select a specific upstream release or commit, and merge
it on an integration branch based on `uni5db-53.1`. Resolve semantic conflicts
and validate the parent product before updating its pinned submodule commit.
Advance `upstream-base` to the newly integrated upstream commit only after
acceptance. Do not rebase or force-push a shared maintenance branch.

## Contribute a focused change

Start a separate contribution branch from current `upstream/main`, cherry-pick
the relevant functional commit(s), and adapt to upstream APIs, test conventions,
lint and performance requirements. Submit only that change, not the entire
maintenance branch. A cherry-pick may need conflict resolution; an independent
local commit does not prove that current upstream can accept it unchanged.

Pushing maintenance/contribution branches and opening PRs require the parent's
explicit instruction. No remote branches are changed by the local migration.
