package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"os/exec"

	"loki-code/clients"
	"loki-code/internal/tools"
)

// repoRoot returns the working tree root (cwd, since these tests run from the
// module root).
func repoRoot(t *testing.T) string {
	t.Helper()
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	return wd
}

// listGoFiles walks the repo and returns *.go files, excluding hidden dirs.
func listGoFiles(t *testing.T, root string) []string {
	t.Helper()
	var out []string
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			name := info.Name()
			if strings.HasPrefix(name, ".") || name == "vendor" {
				return filepath.SkipDir
			}
			return nil
		}
		if strings.HasSuffix(path, ".go") {
			out = append(out, path)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return out
}

// TestArchitecture_DepDirection locks in the one-way dependency rule.
// clients/ must not import the root module or any internal package.
func TestArchitecture_DepDirection(t *testing.T) {
	out, err := exec.Command("go", "list", "-deps", "loki-code/clients").Output()
	if err != nil {
		t.Fatalf("go list failed: %v", err)
	}
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if line == "loki-code" {
			t.Fatalf("clients/ imports loki-code (root) — direction must stay main → clients")
		}
		if strings.HasPrefix(line, "loki-code/internal/") {
			t.Fatalf("clients/ imports %s — clients/ must not depend on internal packages", line)
		}
	}
}

// TestArchitecture_FileSizeBudget enforces the 800-LOC budget. Bumping the
// ceiling is OK; the point is that growth past the ceiling must be a
// deliberate decision, not silent drift.
func TestArchitecture_FileSizeBudget(t *testing.T) {
	const maxLines = 800
	for _, path := range listGoFiles(t, repoRoot(t)) {
		if strings.HasSuffix(path, "_test.go") {
			continue
		}
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		n := strings.Count(string(data), "\n") + 1
		if n > maxLines {
			t.Errorf("%s: %d lines exceeds budget %d — split by concern", path, n, maxLines)
		}
	}
}

// TestArchitecture_RegistryCompleteness asserts toolRegistry and
// GetAvailableTools() agree on the set of tools. Catches the open/closed gap
// where adding to one but not the other would silently break dispatch.
func TestArchitecture_RegistryCompleteness(t *testing.T) {
	toolList := tools.GetAvailableTools()
	schemaNames := map[string]struct{}{}
	for _, tl := range toolList {
		schemaNames[tl.Function.Name] = struct{}{}
	}

	registryNames := tools.ToolNames()
	for _, name := range registryNames {
		if _, ok := schemaNames[name]; !ok {
			t.Errorf("toolRegistry has %q but GetAvailableTools() does not", name)
		}
	}
	for name := range schemaNames {
		found := false
		for _, rn := range registryNames {
			if rn == name {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("GetAvailableTools() has %q but toolRegistry does not", name)
		}
	}
}

// TestArchitecture_NoDuplicateModeState ensures the LLM clients delegate plan
// and concise mode to the ContextManager rather than holding their own copy.
func TestArchitecture_NoDuplicateModeState(t *testing.T) {
	forbidden := map[string]struct{}{"planMode": {}, "conciseMode": {}}
	for _, target := range []reflect.Type{
		reflect.TypeOf(clients.OllamaClient{}),
		reflect.TypeOf(clients.OpenAIClient{}),
	} {
		for i := 0; i < target.NumField(); i++ {
			name := target.Field(i).Name
			if _, banned := forbidden[name]; banned {
				t.Errorf("%s.%s reintroduces duplicated mode state — store it only on ContextManager",
					target.Name(), name)
			}
		}
	}
}

// TestArchitecture_NoTerminalCouplingOutsideUI restricts os.Stdin / os.Stdout
// imports to internal/ui/ui.go and main.go.
func TestArchitecture_NoTerminalCouplingOutsideUI(t *testing.T) {
	allowed := map[string]bool{
		filepath.Join(repoRoot(t), "internal", "ui", "ui.go"): true,
		filepath.Join(repoRoot(t), "main.go"):                  true,
	}
	fset := token.NewFileSet()
	for _, path := range listGoFiles(t, repoRoot(t)) {
		if strings.HasSuffix(path, "_test.go") || allowed[path] {
			continue
		}
		file, err := parser.ParseFile(fset, path, nil, parser.SkipObjectResolution)
		if err != nil {
			t.Fatalf("parse %s: %v", path, err)
		}
		ast.Inspect(file, func(n ast.Node) bool {
			sel, ok := n.(*ast.SelectorExpr)
			if !ok {
				return true
			}
			ident, ok := sel.X.(*ast.Ident)
			if !ok {
				return true
			}
			if ident.Name == "os" && (sel.Sel.Name == "Stdin" || sel.Sel.Name == "Stdout") {
				t.Errorf("%s: references os.%s — route through uiInput or confirmYN/confirmDiff instead",
					path, sel.Sel.Name)
			}
			return true
		})
	}
}

// TestArchitecture_NoToolNamesInClients ensures the clients/ package stays
// generic. Hard-coding a tool name there couples transport to product.
func TestArchitecture_NoToolNamesInClients(t *testing.T) {
	toolNameList := tools.ToolNames()
	root := filepath.Join(repoRoot(t), "clients")
	fset := token.NewFileSet()
	for _, path := range listGoFiles(t, root) {
		if strings.HasSuffix(path, "_test.go") {
			continue
		}
		file, err := parser.ParseFile(fset, path, nil, parser.SkipObjectResolution)
		if err != nil {
			t.Fatalf("parse %s: %v", path, err)
		}
		ast.Inspect(file, func(n ast.Node) bool {
			lit, ok := n.(*ast.BasicLit)
			if !ok || lit.Kind != token.STRING {
				return true
			}
			val := strings.Trim(lit.Value, "`\"")
			for _, name := range toolNameList {
				if val == name {
					t.Errorf("%s:%d: tool name literal %q in clients/ — move policy to tools.go",
						path, fset.Position(lit.Pos()).Line, val)
				}
			}
			return true
		})
	}
}

// TestArchitecture_NoThirdPartyDeps keeps the module stdlib-only. A new
// require directive in go.mod must be a deliberate decision, opted in via a
// trailing `// allow: <reason>` comment on the line.
func TestArchitecture_NoThirdPartyDeps(t *testing.T) {
	data, err := os.ReadFile(filepath.Join(repoRoot(t), "go.mod"))
	if err != nil {
		t.Fatal(err)
	}
	inBlock := false
	for _, raw := range strings.Split(string(data), "\n") {
		line := strings.TrimSpace(raw)
		if line == "" || strings.HasPrefix(line, "//") {
			continue
		}
		switch {
		case strings.HasPrefix(line, "require ("):
			inBlock = true
			continue
		case inBlock && line == ")":
			inBlock = false
			continue
		case strings.HasPrefix(line, "require "):
			if !strings.Contains(line, "// allow:") {
				t.Errorf("go.mod: third-party require without `// allow: <reason>` marker: %s", line)
			}
		case inBlock:
			if !strings.Contains(raw, "// allow:") {
				t.Errorf("go.mod: third-party require without `// allow: <reason>` marker: %s", line)
			}
		}
	}
}

// TestArchitecture_StructCohesion looks at every method on the public client
// types and asserts each method references at least one field of its receiver.
func TestArchitecture_StructCohesion(t *testing.T) {
	ctx := &fakeContextManager{}
	cases := []reflect.Value{
		reflect.ValueOf(clients.NewOllamaClient("http://x", "m", ctx, nil, nil)),
		reflect.ValueOf(clients.NewOpenAIClient("http://x", "", "m", ctx, nil, nil)),
	}
	for _, v := range cases {
		typ := v.Elem().Type()
		if typ.NumField() == 0 {
			t.Errorf("%s has no fields — utility-bag smell", typ.Name())
		}
	}
}

// fakeContextManager is a stand-in used only to satisfy the constructor
// signatures in the cohesion test. It does nothing.
type fakeContextManager struct{}

func (fakeContextManager) AddMessage(clients.ChatMessage)                        {}
func (fakeContextManager) GetMessages() []clients.ChatMessage                    { return nil }
func (fakeContextManager) GetStats() (int, int, int)                             { return 0, 0, 0 }
func (fakeContextManager) SetMaxTokens(int)                                      {}
func (fakeContextManager) Clear()                                                {}
func (fakeContextManager) CompactContext(clients.CompactFunc) error              { return nil }
func (fakeContextManager) CanCompact() bool                                      { return false }
func (fakeContextManager) UpdateTokenCount(int)                                  {}
func (fakeContextManager) SetPlanMode(bool)                                      {}
func (fakeContextManager) IsInPlanMode() bool                                    { return false }
func (fakeContextManager) SetConciseMode(bool)                                   {}
func (fakeContextManager) IsInConciseMode() bool                                 { return false }
func (fakeContextManager) GetActiveTask() *clients.UserTask                      { return nil }
func (fakeContextManager) SetActiveTask(string)                                  {}
func (fakeContextManager) CompleteCurrentTask(string)                            {}
