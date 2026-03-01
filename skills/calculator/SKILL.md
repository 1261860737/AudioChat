---
name: calculator
description: 专门用于精确数学计算的工具。当用户询问任何数学问题时，必须优先使用此工具，而不是自己计算。
---
# Calculator

Use this tool to safely calculate basic math expressions.

## Usage
args: {"expression": "2 + 2 * 5"}

## Supported operations
- Numbers: integers and floats
- Binary ops: `+`, `-`, `*`, `/`, `//`, `%`, `**`
- Unary ops: `+`, `-`

## Output
Returns a JSON object: `{"value": <number>}`.
