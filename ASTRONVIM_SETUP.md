# PDAT DSL Syntax Highlighting in AstroNvim

Your AstroNvim config has been updated to support PDAT DSL syntax highlighting.

## What Was Changed

1. **`~/.config/nvim/lua/plugins/treesitter.lua`**
   - Added custom parser configuration for `pdat_dsl`
   - Points to local tree-sitter grammar at `~/PdatProject/PdatDsl/tree-sitter-pdat-dsl`

2. **`~/.config/nvim/lua/polish.lua`**
   - Added filetype detection: `.dsl` files → `pdat-dsl` filetype

## Installation Steps (COMPLETED)

The parser has been **manually installed** to avoid ABI version conflicts.

### Installation Location

- **Parser**: `~/.local/share/nvim/site/parser/pdat_dsl.so` (ABI v14)
- **Queries**: `~/.local/share/nvim/site/queries/pdat_dsl/highlights.scm`

### Manual Installation Command (if needed)

If you need to reinstall or update:

```bash
cd /home/nbleier/PdatProject/PdatDsl

# Compile parser
gcc -o ~/.local/share/nvim/site/parser/pdat_dsl.so \
    -shared tree-sitter-pdat-dsl/src/parser.c \
    -I./tree-sitter-pdat-dsl/src -Os -fPIC

# Copy queries
mkdir -p ~/.local/share/nvim/site/queries/pdat_dsl
cp tree-sitter-pdat-dsl/queries/highlights.scm \
   ~/.local/share/nvim/site/queries/pdat_dsl/
```

### Verify Installation

Restart Neovim and open any `.dsl` file:

```bash
nvim examples/example_16reg.dsl
```

You should see:
- Keywords (`require`, `instruction`) highlighted
- Extension names (RV32I) highlighted
- Register names (x0, x15) highlighted
- Comments in different color
- Numbers with appropriate highlighting

Check that Tree-sitter is active:
```vim
:lua =vim.treesitter.highlighter.active[vim.api.nvim_get_current_buf()]
```

Should return a table (not nil) when viewing a `.dsl` file.

## Troubleshooting

### Parser not found

If `:TSInstall pdat_dsl` fails:

```vim
:messages
```

Check for errors. Make sure the path in `treesitter.lua` is correct:
```lua
url = vim.fn.expand("~/PdatProject/PdatDsl/tree-sitter-pdat-dsl")
```

### Filetype not detected

Check filetype:
```vim
:set filetype?
```

Should show `filetype=pdat-dsl` when you open a `.dsl` file.

### No syntax highlighting

1. Ensure parser is installed: `:TSInstallInfo`
2. Check that highlights.scm exists: `tree-sitter-pdat-dsl/queries/highlights.scm`
3. Restart Neovim

## Manual Parser Build (if needed)

If automatic installation doesn't work:

```bash
cd ~/PdatProject/PdatDsl/tree-sitter-pdat-dsl
npx tree-sitter generate
gcc -o ~/.local/share/nvim/tree-sitter-parsers/pdat_dsl.so \
    -shared src/parser.c -I./src -Os
```

## Alternative: Use vim.treesitter.language.add

If you prefer to compile yourself, add to `polish.lua`:

```lua
vim.treesitter.language.add('pdat_dsl', {
  path = vim.fn.expand("~/PdatProject/PdatDsl/tree-sitter-pdat-dsl/pdat_dsl.so")
})
```

## Next Steps

Once syntax highlighting is working:
- Colors are customizable via your AstroNvim theme
- Tree-sitter provides code folding automatically
- Query-based features like text objects will work

Enjoy syntax-highlighted DSL files!
