package tools

import (
	"os"
	"os/exec"
	"strings"
)

// ProjectInfo summarizes what kind of project is rooted at the cwd, along
// with the static-analysis tools available for it. Used to wire the
// analyze_code tool's schema and dispatch.
type ProjectInfo struct {
	Language    string
	HasConfig   bool
	ConfigFiles []string
	Analyzers   []AnalyzerInfo
}

// AnalyzerInfo describes one runnable analyzer (e.g. golangci-lint, ruff).
type AnalyzerInfo struct {
	Name       string
	Command    string
	Args       []string
	Available  bool
	ConfigFile string
}

func commandExists(command string) bool {
	_, err := exec.LookPath(command)
	return err == nil
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func findConfigFile(candidates []string) string {
	for _, file := range candidates {
		if fileExists(file) {
			return file
		}
	}
	return ""
}

func hasPythonFiles() bool {
	pythonIndicators := []string{
		"requirements.txt", "pyproject.toml", "setup.py",
		"setup.cfg", "Pipfile", "poetry.lock",
	}
	for _, indicator := range pythonIndicators {
		if fileExists(indicator) {
			return true
		}
	}
	entries, err := os.ReadDir(".")
	if err != nil {
		return false
	}
	for _, entry := range entries {
		if !entry.IsDir() && strings.HasSuffix(entry.Name(), ".py") {
			return true
		}
	}
	return false
}

func detectProject() ProjectInfo {
	if fileExists("go.mod") {
		return detectGoProject()
	}
	if fileExists("package.json") {
		return detectNodeProject()
	}
	if hasPythonFiles() {
		return detectPythonProject()
	}
	return ProjectInfo{Language: "unknown"}
}

func detectGoProject() ProjectInfo {
	analyzers := []AnalyzerInfo{}

	if commandExists("golangci-lint") {
		analyzers = append(analyzers, AnalyzerInfo{
			Name:       "golangci-lint",
			Command:    "golangci-lint",
			Args:       []string{"run"},
			Available:  true,
			ConfigFile: findConfigFile([]string{".golangci.yml", ".golangci.yaml"}),
		})
	}

	if commandExists("go") {
		analyzers = append(analyzers, AnalyzerInfo{
			Name:      "go vet",
			Command:   "go",
			Args:      []string{"vet", "./..."},
			Available: true,
		})
	}

	return ProjectInfo{
		Language:  "go",
		HasConfig: fileExists("go.mod"),
		Analyzers: analyzers,
	}
}

func detectPythonProject() ProjectInfo {
	analyzers := []AnalyzerInfo{}

	if commandExists("ruff") {
		analyzers = append(analyzers, AnalyzerInfo{
			Name:       "ruff",
			Command:    "ruff",
			Args:       []string{"check"},
			Available:  true,
			ConfigFile: findConfigFile([]string{"ruff.toml", "pyproject.toml"}),
		})
	}
	if commandExists("pylint") {
		analyzers = append(analyzers, AnalyzerInfo{
			Name:       "pylint",
			Command:    "pylint",
			Args:       []string{},
			Available:  true,
			ConfigFile: findConfigFile([]string{".pylintrc", "pylint.ini"}),
		})
	}
	if commandExists("flake8") {
		analyzers = append(analyzers, AnalyzerInfo{
			Name:       "flake8",
			Command:    "flake8",
			Args:       []string{},
			Available:  true,
			ConfigFile: findConfigFile([]string{".flake8", "setup.cfg"}),
		})
	}
	if commandExists("python3") {
		analyzers = append(analyzers, AnalyzerInfo{
			Name:      "python3",
			Command:   "python3",
			Args:      []string{"-m", "py_compile"},
			Available: true,
		})
	} else if commandExists("python") {
		analyzers = append(analyzers, AnalyzerInfo{
			Name:      "python",
			Command:   "python",
			Args:      []string{"-m", "py_compile"},
			Available: true,
		})
	}

	hasConfig := fileExists("pyproject.toml") || fileExists("requirements.txt") || fileExists("setup.py")

	return ProjectInfo{
		Language:  "python",
		HasConfig: hasConfig,
		Analyzers: analyzers,
	}
}

func detectNodeProject() ProjectInfo {
	analyzers := []AnalyzerInfo{}
	isTypeScript := fileExists("tsconfig.json")

	if isTypeScript && commandExists("tsc") {
		analyzers = append(analyzers, AnalyzerInfo{
			Name:       "tsc",
			Command:    "tsc",
			Args:       []string{"--noEmit"},
			Available:  true,
			ConfigFile: "tsconfig.json",
		})
	}

	eslintConfig := findConfigFile([]string{
		".eslintrc.js", ".eslintrc.json", ".eslintrc.yml",
		".eslintrc.yaml", "eslint.config.js", ".eslintrc",
	})
	if eslintConfig != "" && commandExists("eslint") {
		analyzers = append(analyzers, AnalyzerInfo{
			Name:       "eslint",
			Command:    "eslint",
			Args:       []string{},
			Available:  true,
			ConfigFile: eslintConfig,
		})
	}

	if commandExists("jshint") {
		analyzers = append(analyzers, AnalyzerInfo{
			Name:      "jshint",
			Command:   "jshint",
			Args:      []string{},
			Available: true,
		})
	}

	if commandExists("node") {
		analyzers = append(analyzers, AnalyzerInfo{
			Name:      "node",
			Command:   "node",
			Args:      []string{"--check"},
			Available: true,
		})
	}

	language := "javascript"
	if isTypeScript {
		language = "typescript"
	}

	return ProjectInfo{
		Language:  language,
		HasConfig: fileExists("package.json"),
		Analyzers: analyzers,
	}
}

func getAnalyzerNames(analyzers []AnalyzerInfo) []string {
	var names []string
	for _, analyzer := range analyzers {
		if analyzer.Available {
			names = append(names, analyzer.Name)
		}
	}
	return names
}
