# Editor Support for PDAT DSL

Syntax highlighting for `.dsl` files in various editors.

## Visual Studio Code

Location: `vscode/`

### Installation

**Option 1: Install from directory**
```bash
cd editors/vscode
code --install-extension .
```

**Option 2: Manual copy**
```bash
cp -r editors/vscode ~/.vscode/extensions/pdat-dsl-0.1.0
```

Then reload VS Code.

## Neovim / Helix / Emacs (Tree-sitter)

Location: `../tree-sitter-pdat-dsl/`

### Neovim Setup

Add to your Tree-sitter config:
```lua
local parser_config = require("nvim-treesitter.parsers").get_parser_configs()
parser_config.pdat_dsl = {
  install_info = {
    url = "~/path/to/PdatDsl/tree-sitter-pdat-dsl",
    files = {"src/parser.c"},
  },
  filetype = "dsl",
}
```

### Build the parser

```bash
cd ../tree-sitter-pdat-dsl
npm install
npx tree-sitter generate
npx tree-sitter test
```

## Vim (without Tree-sitter)

Create `~/.vim/syntax/pdat-dsl.vim`:

```vim
" PDAT DSL syntax highlighting
if exists("b:current_syntax")
  finish
endif

syn keyword pdatKeyword require require_registers instruction pattern mask
syn match pdatComment "#.*$"
syn match pdatRegister "\<x[0-9]\{1,2\}\>"
syn match pdatNumber "\<0x[0-9a-fA-F_]\+\>"
syn match pdatNumber "\<0b[01_]\+\>"
syn match pdatNumber "\<[0-9][0-9_]*\>"
syn match pdatInstruction "\<[A-Z][A-Z0-9_]*\>" contained
syn region pdatInstrRule start="instruction" end="$" contains=pdatKeyword,pdatInstruction

hi def link pdatKeyword Keyword
hi def link pdatComment Comment
hi def link pdatRegister Identifier
hi def link pdatNumber Number
hi def link pdatInstruction Function

let b:current_syntax = "pdat-dsl"
```

Add to `~/.vim/ftdetect/pdat-dsl.vim`:
```vim
au BufRead,BufNewFile *.dsl set filetype=pdat-dsl
```

## Contributing

Contributions to editor support are welcome! Please test thoroughly before submitting.

## License

CC-BY-NC-SA-4.0 - Copyright 2025 Nathan Bleier
