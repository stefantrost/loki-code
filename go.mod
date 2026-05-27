module loki-code

go 1.26.3

require (
	github.com/charmbracelet/bubbles v1.0.0                                        // allow: scrollable viewport + text-input components
	github.com/charmbracelet/bubbletea v1.3.10                                     // allow: TUI event loop (Elm architecture)
	github.com/charmbracelet/glamour v1.0.0                                        // allow: markdown rendering post-stream
	github.com/charmbracelet/lipgloss v1.1.1-0.20250404203927-76690c660834        // allow: terminal layout and styling
)

require (
	github.com/alecthomas/chroma/v2 v2.20.0 // indirect; allow: syntax highlighting via glamour
	github.com/atotto/clipboard v0.1.4 // indirect; allow: clipboard support via bubbles/textinput
	github.com/aymanbagabas/go-osc52/v2 v2.0.1 // indirect; allow: OSC 52 clipboard escape sequences via lipgloss
	github.com/aymerick/douceur v0.2.0 // indirect; allow: CSS sanitiser via glamour
	github.com/charmbracelet/colorprofile v0.4.1 // indirect; allow: terminal colour-profile detection via bubbletea
	github.com/charmbracelet/x/ansi v0.11.6 // indirect; allow: ANSI sequence helpers via bubbletea
	github.com/charmbracelet/x/cellbuf v0.0.15 // indirect; allow: cell-buffer rendering via bubbletea
	github.com/charmbracelet/x/exp/slice v0.0.0-20250327172914-2fdc97757edf // indirect; allow: slice utilities via charmbracelet ecosystem
	github.com/charmbracelet/x/term v0.2.2 // indirect; allow: terminal size detection via bubbletea
	github.com/clipperhouse/displaywidth v0.9.0 // indirect; allow: Unicode display-width via glamour
	github.com/clipperhouse/stringish v0.1.1 // indirect; allow: string utilities via displaywidth
	github.com/clipperhouse/uax29/v2 v2.5.0 // indirect; allow: Unicode word-break rules via displaywidth
	github.com/dlclark/regexp2 v1.11.5 // indirect; allow: full-featured regexp via chroma syntax highlighting
	github.com/erikgeiser/coninput v0.0.0-20211004153227-1c3628e74d0f // indirect; allow: Windows console input via bubbletea
	github.com/gorilla/css v1.0.1 // indirect; allow: CSS parser via glamour/bluemonday
	github.com/lucasb-eyer/go-colorful v1.3.0 // indirect; allow: colour conversion via lipgloss
	github.com/mattn/go-isatty v0.0.20 // indirect; allow: TTY detection via bubbletea/lipgloss
	github.com/mattn/go-localereader v0.0.1 // indirect; allow: locale-aware stdin reader via bubbletea
	github.com/mattn/go-runewidth v0.0.19 // indirect; allow: rune display-width via bubbles
	github.com/microcosm-cc/bluemonday v1.0.27 // indirect; allow: HTML sanitiser via glamour
	github.com/muesli/ansi v0.0.0-20230316100256-276c6243b2f6 // indirect; allow: ANSI string helpers via muesli/reflow
	github.com/muesli/cancelreader v0.2.2 // indirect; allow: cancellable io.Reader via bubbletea
	github.com/muesli/reflow v0.3.0 // indirect; allow: terminal text reflow via glamour
	github.com/muesli/termenv v0.16.0 // indirect; allow: terminal environment detection via lipgloss
	github.com/rivo/uniseg v0.4.7 // indirect; allow: Unicode segmentation via go-runewidth
	github.com/xo/terminfo v0.0.0-20220910002029-abceb7e1c41e // indirect; allow: terminfo database via muesli/termenv
	github.com/yuin/goldmark v1.7.13 // indirect; allow: Markdown parser via glamour
	github.com/yuin/goldmark-emoji v1.0.6 // indirect; allow: emoji extension via glamour/goldmark
	golang.org/x/net v0.55.0 // indirect; allow: HTML handling via bluemonday (upgraded from v0.38.0 to fix GO-2026-4440/4441/4918/5025-5030)
	golang.org/x/sys v0.45.0 // indirect; allow: OS-level primitives via bubbletea/term (upgraded from v0.38.0 to fix GO-2026-5024)
	golang.org/x/term v0.43.0 // indirect; allow: terminal raw-mode via bubbletea
	golang.org/x/text v0.37.0 // indirect; allow: Unicode text normalization via goldmark
)
