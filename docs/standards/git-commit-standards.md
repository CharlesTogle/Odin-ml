# Git Commit Message Standards

Shared commit message format for all Odin repositories.

## Format

```text
<type>(<scope>): <brief message>

[optional body]

[optional footer]
```

- Use lowercase `type` and `scope` tokens.
- Always include a scope. Do not use unscoped subjects such as `fix: ...`.
- Keep the brief message specific, imperative, and focused on the change.

## Commit Types

| Type | Use For |
| :--- | :--- |
| `build` | Build system, packaging, or compile pipeline changes |
| `chore` | Maintenance tasks that do not change functionality |
| `ci` | Continuous integration, checks, or release automation |
| `docs` | Documentation-only changes |
| `feat` | New features or capabilities |
| `fix` | Bug fixes or behavior corrections |
| `perf` | Performance improvements |
| `refactor` | Structural improvements without changing behavior |
| `revert` | Reverts of prior commits |
| `style` | Formatting-only changes with no behavior impact |
| `test` | Test-only changes |

## Commit Scopes (Odin-ML)

| Scope | Use For |
| :--- | :--- |
| `api` | API routing, transport, request handling, or API contracts |
| `ml` | Model service, training, evaluation, forecasting, or ML contracts |
| `data` | Data sources, datasets, or data pipeline changes |
| `training` | Training pipeline scripts or configuration |
| `docs` | Documentation-only updates |
| `config` | Repo configuration, tooling, or runtime pins |
| `deps` | Dependency additions, removals, or upgrades |
| `tests` | Test fixtures, helpers, or coverage |
| `standards` | Shared engineering standards and agent guidance |

Examples:

```text
feat(api): add pfp classification endpoint
fix(ml): scope anomaly baseline by user id
docs(standards): document commit message format
```

## Subject Rules

- 50-72 characters when possible.
- Imperative mood.
- Lowercase unless it begins with a proper noun, acronym, or code identifier.
- No trailing period.

## Body Rules

- Blank line between subject and body.
- Wrap at 72 characters.
- Explain what changed and why.
- Prefer context over repeating diff details.

Suggested body order:

1. Current situation
2. Reason for change
3. Action taken
4. Impact or notes

## Footer Rules

- Use `Fixes:`, `Closes:`, `Refs:`, `See also:`, or `BREAKING CHANGE:`.
- Keep concise.

## Quality Rule

A valid commit subject completes this sentence:

```text
If applied, this commit will <your subject line here>
```
