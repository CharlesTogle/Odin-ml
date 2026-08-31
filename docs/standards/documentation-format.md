# Documentation Format

Shared formatting conventions for all Odin thesis repositories (Odin-Paper, Odin-Literature, Odin-ML).

---

## Metadata Block

Every versioned document includes a JSON metadata block immediately after the first heading:

```json
{
  "document-type": "<type>",
  "version": "<semver>",
  "date": "YYYY.MM.DD",
  "authors": ["Last, First M."]
}
```

| Field | Rule |
| :--- | :--- |
| `document-type` | Lowercase-kebab: `system-spec`, `prd`, `mdd`, `benchmark`, `workflow`, etc. |
| `version` | Semantic versioning (`Major.Minor.Patch`). Major = adviser-backed revisions. Minor = researcher revisions within adviser assumptions. Patch = trivial or general revisions. |
| `date` | `YYYY.MM.DD` format. |
| `authors` | Full last-name-first format. Group-4 roster when applicable. |

Non-versioned documents (indices, standards, notes) may omit the metadata block.

---

## Heading Hierarchy

- `#` — Document title (one per file).
- `##` — Major sections.
- `###` — Subsections.
- Never skip levels (`#` directly to `###` is not allowed).
- Use ATX-style headings (`#`, `##`, `###`), not underline style.

---

## Horizontal Rules

Use `---` to separate major sections within a document. Do not use `***` or `___`.

---

## Tables

- Use pipe-delimited markdown tables.
- Align colons for readability: `| :--- |` for left, `| :---: |` for center.
- Include a header row and a separator row in every table.

---

## Code Blocks

- Use fenced code blocks with a language tag: ` ```python `, ` ```json `, ` ```text `, etc.
- Use ` ```text ` for ASCII diagrams, terminal output, or when no language applies.
- Do not indent code blocks inside list items (use standard nesting).

---

## Cross-References

Use relative markdown links:

```markdown
See [System Specification](../requirements-engineering/system-spec.md) for details.
```

Rules:
- Always verify the target path exists before committing.
- Use paths relative to the file containing the reference.
- Prefer linking to section anchors when referencing specific sections: `[§5.3](../spec.md#53-label-space)`.

---

## README.md Structure

Every repository root `README.md` follows this order:

```markdown
# Repo-Name

One-line purpose statement.

## What's Here

| Directory / File | Purpose |
| :--- | :--- |
| ... | ... |

## Setup

Prerequisites and install steps.

## Usage

How to run or use the repository.

## Navigation

Link to `INDEX.md` for full navigation.
```

Keep the README concise. INDEX.md is the navigation hub; the README is the entry point.

---

## AGENTS.md Structure

Every repository root `AGENTS.md` follows this order:

```markdown
# Repo-Name — Agent Guide

**Thesis**: <full thesis title>
**Group 4, III-DCSAD, University of Makati**

---

## Repository Role

One paragraph describing what lives here and what does not.

## Coding Standards

| Standard | Location |
| :--- | :--- |
| ... | ... |

## Top-Level Directory Layout

```text
<ASCII tree>
```

## Navigation

| Document | Purpose |
| :--- | :--- |
| ... | ... |

## Important Gotchas

- Bullet list of critical warnings and known issues.
```

Keep AGENTS.md focused on orientation and navigation. Detailed standards belong in `docs/standards/`.

---

## INDEX.md Structure

Every repository root `INDEX.md` (and `docs/INDEX.md` where applicable) follows this order:

```markdown
# Repo-Name — Repository Index

- **Project:** <thesis title>
- **Institution:** University of Makati | Group 4, III-DCSAD
- **Last indexed:** YYYY-MM-DD

---

## How to Use This Index

| Need | Go to |
| :--- | :--- |
| ... | ... |

## Repository Map

| Path | Purpose |
| :--- | :--- |
| ... | ... |

---

## <Section per directory>

| File | Purpose |
| :--- | :--- |
| ... | ... |
```

The `Last indexed` date must be updated whenever the index is modified.

---

## Placeholder / TODO Format

When a document is not yet written, use this format:

```markdown
# Document Title

> **Status:** TODO — not yet written.

## Expected Content

- Bullet list of what this document will contain.
- Reference to the authoritative source if content exists elsewhere.

## Owner

- Who is responsible for writing this.
```

Do not leave files as empty stubs or bare HTML comments. Every placeholder must state what belongs in it.

---

## Deprecated / Superseded Documents

Mark superseded files with `(OLD)` in the filename and add a notice at the top of the file body:

```markdown
> **Deprecated.** This document is superseded by `../path/to/current.md`.
> Retained for historical reference only.
```

Do not delete superseded files outright; they may contain context needed for audit trails.

---

## Language and Tone

- Formal academic English.
- No emojis in committed documentation.
- No internal commentary, chat notes, or humor in committed documents.
- Use present tense for describing the system. Use past tense for describing completed work.
- Prefer active voice.
