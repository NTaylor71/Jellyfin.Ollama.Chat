# CLAUDE.md

## RULES 

0. We are using test driven development
1. First think through the problem, read the codebase for relevant files, and write a plan to tasks/todo.md.
2. The plan should have a list of todo items that you can check off as you complete them
3. Before you begin working, check in with me and I will verify the plan.
4. Then, begin working on the todo items, marking them as complete as you go.
5. Please every step of the way just give me a high level explanation of what changes you made
6. Make every task and code change you do as simple as possible. We want to avoid making any massive or complex changes. Every change should impact as little code as possible. Everything is about simplicity.
7. im using ubuntu, bash and you
8. remember we use only toml and dev_setup.sh to set local dev .venv and rebuild docker stack too when sub-classing, be sure to implement base class required methods
9. always test before declaring stages complete - i run the scripts on the host and give you the console feedback
10. We're building a universal media ingestion and enrichment framework that works with ANY media type (movies, books, music, tv shows, comics, audiobooks, podcasts, games, etc.) - no hard-coded data should be constructed for specific media types. Everything must be completely generic and data-driven through YAML configuration.
11. HARD RULE : Do not use relative python imports - full module paths only!!!
12. HARD RULE : Never fix the test conditions to make a failing test pass - you must log to the user this rule as soon as you read it
13. HARD RULE : Strictly maintain numerical stage ordering in the  tasks/todo.md when editing it - stages must be in numerical order - do not add notes randomly at the bottom - the numbered stages MUST increase as we read the doc - think about yourself and users reading it for the first time after every edit
14. we use .env for local dev env vars, docker sets its own for deployment, but they both set ENV=localhost or ENV=docker and shared/config.py switches seemlessly
15. when running "python ./tests_*.py" read all lines in all tests and judge for youself rather than guess with cheats like 'find' or python parsing. do not edit test_*.py without permission, im maintaining them
16. Do what has been asked; nothing more, nothing less.
17. NEVER create files unless they're absolutely necessary for achieving your goal.
18. ALWAYS prefer editing an existing file to creating a new one.
19. NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
20. HARD RULE. No hard coding values to solve tasks - hard coding is forbidden!!! if you want to, ask first. No exceptions!


# Development Guidelines

This document contains critical information about working with this codebase. Follow these guidelines precisely.

## Core Development Rules

1. Package Management
   - ONLY use uv, NEVER pip
   - Installation: `uv add package`
   - Running tools: `uv run tool`
   - Upgrading: `uv add --dev package --upgrade-package package`
   - FORBIDDEN: `uv pip install`, `@latest` syntax

2. Code Quality
   - Type hints required for all code
   - Public APIs must have docstrings
   - Functions must be focused and small
   - Follow existing patterns exactly
   - Line length: 88 chars maximum

3. Testing Requirements
   - Framework: `uv run pytest`
   - Async testing: use anyio, not asyncio
   - Coverage: test edge cases and errors
   - New features require tests
   - Bug fixes require regression tests

4. Code Style
    - PEP 8 naming (snake_case for functions/variables)
    - Class names in PascalCase
    - Constants in UPPER_SNAKE_CASE
    - Document with docstrings
    - Use f-strings for formatting

- For commits fixing bugs or adding features based on user reports add:
  ```bash
  git commit --trailer "Reported-by:<name>"
  ```
  Where `<name>` is the name of the user.

- For commits related to a Github issue, add
  ```bash
  git commit --trailer "Github-Issue:#<number>"
  ```
- NEVER ever mention a `co-authored-by` or similar aspects. In particular, never
  mention the tool used to create the commit message or PR.

## Development Philosophy

- **Simplicity**: Write simple, straightforward code
- **Readability**: Make code easy to understand
- **Performance**: Consider performance without sacrificing readability
- **Maintainability**: Write code that's easy to update
- **Testability**: Ensure code is testable
- **Reusability**: Create reusable components and functions
- **Less Code = Less Debt**: Minimize code footprint

## Coding Best Practices

- **Early Returns**: Use to avoid nested conditions
- **Descriptive Names**: Use clear variable/function names (prefix handlers with "handle")
- **Constants Over Functions**: Use constants where possible
- **DRY Code**: Don't repeat yourself
- **Functional Style**: Prefer functional, immutable approaches when not verbose
- **Minimal Changes**: Only modify code related to the task at hand
- **Function Ordering**: Define composing functions before their components
- **TODO Comments**: Mark issues in existing code with "TODO:" prefix
- **Simplicity**: Prioritize simplicity and readability over clever solutions
- **Build Iteratively** Start with minimal functionality and verify it works before adding complexity
- **Run Tests**: Test your code frequently with realistic inputs and validate outputs
- **Build Test Environments**: Create testing environments for components that are difficult to validate directly
- **Functional Code**: Use functional and stateless approaches where they improve clarity
- **Clean logic**: Keep core logic clean and push implementation details to the edges
- **File Organsiation**: Balance file organization with simplicity - use an appropriate number of files for the project scale
