---
name: code-quality
description: Generate production-grade code accordingly to industry best practices and standards. Use this skill when the user asks to implementing features, refactoring or writing new code.
---

## Step 1: Analyze the problem

- Identify:
  - Inputs / outputs
  - Constraints
  - Expected scale
- Decide complexity level:
  - Simple: minimal structure
  - Medium: light structure
  - Complex: apply patterns

## Step 2: Choose style

- Prefer **functional/procedural** when:
  - Data transformation
  - Stateless logic
  - Simple flows

- Use **OOP** when:
  - Managing state
  - Complex domain modeling
  - Multiple behaviors

## Step 3: Apply coding principles

### Always

- Use clear and semantic naming
- Keep functions small
- Avoid deep nesting
- Prefer early returns and guardian clauses over `elif` and `else`

### Functional style

- Prefer pure functions
- Avoid shared mutable state
- Isolate side effects

### OOP (only if needed)

- Apply SOLID principles ONLY when complexity requires
- Avoid deep inheritance
- Prefer composition over inheritance

## Step 4: Control complexity

Before adding abstractions, ask:

- Is this reused?
- Will it change?
- Does it reduce complexity?

If NO, keep it simple

## Step 5: Structure the code

- Separate:
  - Business logic
  - I/O (API, DB, file)
- Keep flat structure first
- Add layers only when needed

## Step 6: Error handling

- Fail fast
- Use clear error messages
- Do not silently ignore errors

## Step 7: Output

Produce:

- Complete working code
- Readable and maintainable structure

## Anti-patterns to avoid

- Overengineering
- Premature abstraction
- Excessive design patterns
- Framework-like complexity for simple problems
