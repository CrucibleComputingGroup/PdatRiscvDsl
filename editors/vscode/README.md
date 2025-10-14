# PDAT DSL for Visual Studio Code

Syntax highlighting for PDAT DSL files (`.dsl` extension).

## Features

- Syntax highlighting for all DSL keywords
- Comment support
- Register name highlighting
- Number literals (hex, binary, decimal)
- Instruction and extension name highlighting

## Installation

### From VSIX (recommended)

1. Package the extension:
   ```bash
   cd editors/vscode
   vsce package
   ```

2. Install in VS Code:
   - Open VS Code
   - Go to Extensions (Ctrl+Shift+X)
   - Click "..." → "Install from VSIX"
   - Select `pdat-dsl-0.1.0.vsix`

### Manual Installation

Copy this directory to your VS Code extensions folder:

```bash
# Linux/Mac
cp -r editors/vscode ~/.vscode/extensions/pdat-dsl-0.1.0

# Windows
cp -r editors/vscode %USERPROFILE%\.vscode\extensions\pdat-dsl-0.1.0
```

Then reload VS Code.

## Usage

Any file with `.dsl` extension will automatically get syntax highlighting.

## Example

```dsl
# Require RV32I base instruction set
require RV32I

# Limit to 16 registers
require_registers x0-x15

# Outlaw multiplication instructions
instruction MUL
instruction DIV { rd = x0 }

# Low-level pattern
pattern 0x02000033 mask 0xFE00707F
```

## License

CC-BY-NC-SA-4.0 - Copyright 2025 Nathan Bleier
