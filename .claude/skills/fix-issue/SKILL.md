---
name: fix-issue
description: Fix a GitHub issue end-to-end
disable-model-invocation: true
---

# Fix GitHub Issue

Analyze and fix the GitHub issue: $ARGUMENTS

## Steps

1. Use `gh issue view $ARGUMENTS` to read the issue details
2. Understand the problem — reproduce if possible
3. Search the codebase for relevant files using Grep/Glob
4. Implement the fix with minimal changes
5. Write or update tests to cover the fix
6. Run `uv run pytest` to verify tests pass
7. Run `uv run ruff check .` and `uv run mypy src/` for lint/type checks
8. Create a commit with message: `fix: <description> (closes #$ARGUMENTS)`
9. Push and create a PR with `gh pr create`
