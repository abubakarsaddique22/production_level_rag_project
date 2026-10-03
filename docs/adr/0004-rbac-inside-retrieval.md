# ADR 0004: RBAC enforced inside retrieval and inside the cache key

- **Status:** Accepted
- **Date:** 2026-09

## Context

Four departments (HR, Engineering, Finance, Product) and five roles. A role must never receive content from a department it cannot read.
Filtering in the prompt ("do not reveal Finance data") is not a control: the model has already seen the text.

## Decision

- Roles map to departments in one place (`api/deps.py`, `ROLE_DEPARTMENTS`):

  | Role | Departments |
  |---|---|
  | employee, hr | HR, Product |
  | engineer | Engineering, HR, Product |
  | finance | HR, Finance, Product |
  | admin | all four |

- The department list is passed to retrieval and applied **inside** the Qdrant query (`MatchAny`) and inside the BM25 search.
- `HybridRetriever` has a safety net: any chunk whose department is not allowed is dropped and logged, even if a filter upstream failed.
- The **answer cache key includes the departments**, so a cached answer built from Finance documents can never be served to an employee.
- Agent tools call the same `RagService` with the same departments, so the agent path is not a bypass.

## Consequences

- (+) The model never sees chunks the user is not allowed to read.
- (+) Verified by `tests/integration/test_rbac.py` (28 tests) and the RBAC cases in the red-team suite.
- (-) `test_rbac.py` needs a filled Qdrant collection and the real models, so it runs locally and is **not** in CI.
- (-) Roles are hard-coded in Python. Changing them needs a deploy. A real company would load them from the database or an identity provider.
