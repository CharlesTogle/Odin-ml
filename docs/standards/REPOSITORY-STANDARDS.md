# Repository Standards

Enforceable engineering standards for Odin-ML, the Python/FastAPI machine-learning service.

---

## Scope

These standards apply to code in `app/`, `tests/`, and `training/`. This repository is a Python service; the frontend/TypeScript standards of the `odin` application repository do not apply here.

---

## Domain and Data Scope

- Odin is a single-user-account application. Model the system as `1 user = 1 account`; do not introduce tenant, organization, or workspace architecture.
- All user-owned data must be scoped to the authenticated `user_id` boundary, on both read and write paths.
- Validate foreign keys and related record references against the user's ownership boundary before use.
- Do not fetch, update, or delete user-owned records without explicit user scoping.

---

## Input Validation and Sanitization

- Treat all request bodies, query params, and route params as untrusted input.
- Validate, clean, and sanitize data at the API boundary before persistence or downstream processing.
- Prefer explicit Pydantic schemas and narrow accepted values over permissive input handling.
- Whitelist constrained values such as sort orders, filter keys, and enum-backed options.
- Normalize values so equivalent inputs do not create inconsistent stored data.

---

## Dependency Discipline

- Do not add a new dependency when an existing project dependency solves the problem adequately.
- Add Python and ML dependencies only to this repository.
- Respect the pinned versions in `requirements.txt` and `requirements-dev.txt`.
- Install dependencies in a virtual environment only.

---

## Structural Discipline

- Organize by responsibility, not by arbitrary preference.
- Separate transport (`app/api/`), business logic (`app/services/`), model wrappers (`app/models/`), and schemas (`app/schemas/`).
- Avoid dumping unrelated files into the repository root.
- Create new top-level directories only when they represent a durable architectural boundary.
- Use clear, stable naming for files and folders.

---

## Function and Module Design

- Keep functions focused on a single responsibility.
- Decompose large functions into smaller, composable units.
- Do not create god classes or god services. Split classes that accumulate unrelated responsibilities.
- Move logic out of route handlers into services. Route handlers should be thin transport adapters.

---

## DRY

- Do not repeat logic that can be shared safely.
- Extract repeated logic into a reusable function, helper, or module at the correct boundary.
- Prefer a single maintained source of truth over copied parallel implementations.

---

## Logging

- Never log secrets, access tokens, passwords, or raw credentials.
- Do not log personally sensitive user data unless there is a clear operational need and the value is minimized.
- Prefer structured logs with safe identifiers and diagnostic context (e.g., `user_id`, request IDs, counts, status) over dumping raw objects.
- Never return raw exception messages to clients. Log full exception objects server-side.

---

## API Response and Error Handling

- Use `403` for permission denials on authenticated users; reserve `401` for authentication failures.
- Never comment out authorization checks. Verify write endpoints explicitly.
- Enforce ownership checks on read paths as well as write paths.
- Set explicit timeouts on all outbound HTTP calls.

---

## Model Serving

- Each model module must return a structured contract and must never throw an unhandled exception on failure; return the structured contract with fallback values.
- Model artifacts are versioned and stored in object storage with metadata (training-data hash, performance metrics, dependency versions).
- Each service exposes `/health`, `/ready`, and `/metrics` endpoints.
- Keep model loading and inference code separate from API transport code.

---

## Testing

- Test real behavior, not assumed behavior. Read the source before writing assertions.
- Mock data must match the real runtime shape returned by the service or data layer.
- Keep test method signatures aligned with the production methods they exercise.
- Place tests in `tests/`, near the code they validate.
- Run the suite with `pytest`.

---

## Documentation Discipline

- When structure or dependency rules change, update `AGENTS.md`.
- When coding standards change, update `REPOSITORY-STANDARDS.md`.
- When setup instructions change, update `README.md`.
- Do not let repository documentation drift behind the actual implementation.
- Follow the shared formatting rules in `documentation-format.md`.

---

## Remediation

- When a security or correctness issue is found in one place, sweep the codebase for the same pattern before considering the fix complete.
