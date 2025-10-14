# tree-sitter-pdat-dsl

Tree-sitter grammar for PDAT DSL (RISC-V ISA subset specification language).

## Installation

### Neovim

```lua
-- Add to your Tree-sitter config
require'nvim-treesitter.configs'.setup {
  ensure_installed = { "pdat_dsl" },
  -- or install manually:
  -- :TSInstall pdat_dsl
}
```

### Helix

Add to `~/.config/helix/languages.toml`:

```toml
[[language]]
name = "pdat-dsl"
scope = "source.pdat-dsl"
file-types = ["dsl"]
roots = []
comment-token = "#"
grammar = "pdat_dsl"

[[grammar]]
name = "pdat_dsl"
source = { git = "https://github.com/yourusername/tree-sitter-pdat-dsl", rev = "main" }
```

### Emacs

```elisp
(use-package tree-sitter-langs
  :config
  (add-to-list 'tree-sitter-major-mode-language-alist '(pdat-dsl-mode . pdat-dsl)))
```

## Development

```bash
# Generate parser
npm install
npx tree-sitter generate

# Test grammar
npx tree-sitter test

# Parse a file
npx tree-sitter parse examples/test.dsl
```

## License

CC-BY-NC-SA-4.0 - Copyright 2025 Nathan Bleier
