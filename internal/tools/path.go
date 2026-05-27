package tools

import (
	"fmt"
	"path/filepath"
	"strings"
)

// validatePath constrains a path to the current working tree. It resolves
// symlinks where possible (falling back to the deepest existing ancestor for
// files that don't exist yet, so create_file targets still get checked).
// The final rel-path against the cwd must not start with "..".
func validatePath(path string) error {
	cwd, err := filepath.Abs(".")
	if err != nil {
		return fmt.Errorf("failed to get current directory: %v", err)
	}

	abs, err := filepath.Abs(path)
	if err != nil {
		return fmt.Errorf("failed to resolve path: %v", err)
	}

	if resolved, err := filepath.EvalSymlinks(abs); err == nil {
		abs = resolved
	} else {
		dir := filepath.Dir(abs)
		for dir != filepath.Dir(dir) {
			if resolved, err := filepath.EvalSymlinks(dir); err == nil {
				abs = filepath.Join(resolved, abs[len(dir):])
				break
			}
			dir = filepath.Dir(dir)
		}
	}

	rel, err := filepath.Rel(cwd, abs)
	if err != nil {
		return fmt.Errorf("path outside working tree: %s", path)
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return fmt.Errorf("access outside current directory not allowed: %s", path)
	}

	return nil
}
