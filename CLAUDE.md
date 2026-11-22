The ``Project Instructions Email.pdf`` is the ultimate source of truth for the end goal of this project. It is clear from these instructions that the design of this transonic turbine will need to use modular and iterative code. Therefore, the exact hardware values/parameters are not too relevant especially for unit testing of module code.

## How to Write Code:
Use type annotation for function arguments and returns.
It maybe helpful to use extra variable declarations or more lines if a single variable assignment or function call is excessively long; this improves code readability.

**No comments** for explaining code logic. If you need comments to explain a workaround, then your logic is flawed to begin with. **Well-written code should read like natural language.**

The only acceptable comments are for: indicating dimensions/units of variables, categorization/dividing long code segments (assigning titles to said segments/categories are acceptable), and elaborating on what a variable is if the assigned name is not sufficiently descriptive. This applies to docstrings as well; they should be 1 or 2 sentences/lines at most.

## Appending to ``CLAUDE.md`` (this file):
This applies only to LLMs.
Do not use notes in ``CLAUDE.md`` to justify code logic/workarounds.
Nor should you use this to explain the code, as it would then turn this file into a liability once the content of this file is no longer in congruence with the actual source code.
Any amendments made by Claude or any LLM should be under the LLM Notes section.

## LLM Notes: