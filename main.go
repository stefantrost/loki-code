package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"

	"loki-code/clients"
	"loki-code/internal/agent"
	"loki-code/internal/config"
	"loki-code/internal/session"
	"loki-code/internal/tools"
	"loki-code/internal/view"
)

var version = "dev"

const (
	logDir  = "logs"
	logFile = "loki-code.log"
)

func newLogger(w io.Writer) *slog.Logger {
	return slog.New(slog.NewTextHandler(w, &slog.HandlerOptions{Level: slog.LevelDebug}))
}

// setupLogger opens logs/loki-code.log (creating the directory if needed) and
// configures slog to write there. The log file is always at DEBUG level so
// every run is fully auditable; debug mode additionally tees output to stderr.
// Returns the open log file so the caller can defer Close (nil on error).
func setupLogger(debug bool) *os.File {
	if err := os.MkdirAll(logDir, 0750); err != nil {
		slog.SetDefault(newLogger(os.Stderr))
		return nil
	}

	lf, err := os.OpenFile(filepath.Join(logDir, logFile),
		os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		slog.SetDefault(newLogger(os.Stderr))
		return nil
	}

	var w io.Writer = lf
	if debug {
		w = io.MultiWriter(lf, os.Stderr)
	}
	slog.SetDefault(newLogger(w))
	return lf
}

// enableStderrTee rewires the default logger to tee to stderr in addition to
// the existing log file. Called after config load when cfg.Debug is true but
// --debug was not passed on the CLI (env-based debug activation).
func enableStderrTee(lf *os.File) {
	if lf == nil {
		return
	}
	slog.SetDefault(newLogger(io.MultiWriter(lf, os.Stderr)))
}

func fatal(msg string, err error) {
	slog.Error(msg, "error", err)
	fmt.Fprintf(os.Stderr, "Error: %s: %v\n", msg, err)
	os.Exit(1)
}

func main() {
	modelFlag := flag.String("model", "", "Model to use (default: qwen3:32b)")
	modelShort := flag.String("m", "", "Model to use (short form)")
	baseURL := flag.String("url", "", "API base URL (default: http://localhost:11434 for Ollama)")
	apiType := flag.String("api-type", "", "API type: ollama, openai, openai-compatible (auto-detected if not specified)")
	configFile := flag.String("config", "", "Configuration file path (default: llm.env)")
	bearerToken := flag.String("token", "", "Bearer token for authentication")
	listModels := flag.Bool("list-models", false, "List available models and exit")
	debugFlag := flag.Bool("debug", false, "Enable debug logging to stderr")
	conciseFlag := flag.Bool("concise", false, "Start in concise mode (brief responses)")
	conciseShort := flag.Bool("c", false, "Start in concise mode (short form)")
	createConfig := flag.Bool("create-config", false, "Create example configuration file")
	versionFlag := flag.Bool("version", false, "Print version and exit")

	flag.Parse()

	if *versionFlag {
		fmt.Printf("loki-code %s\n", version)
		return
	}

	logFile := setupLogger(*debugFlag)
	if logFile != nil {
		defer logFile.Close()
	}
	slog.Debug("Application started", "debug_enabled", *debugFlag)

	if *createConfig {
		configPath := "llm.env"
		if *configFile != "" {
			configPath = *configFile
		}
		if err := config.CreateExampleConfig(configPath); err != nil {
			fatal("Error creating config file", err)
		}
		fmt.Printf("✓ Created example configuration file: %s\n", configPath)
		fmt.Println("Edit the file with your API settings and run loki-code again.")
		return
	}

	cfg, err := config.LoadConfig(*configFile)
	if err != nil {
		slog.Error("Error loading configuration", "error", err)
		fmt.Fprintln(os.Stderr, "Run 'loki-code --create-config' to create an example configuration file.")
		os.Exit(1)
	}

	if *modelFlag != "" {
		cfg.ModelName = *modelFlag
	} else if *modelShort != "" {
		cfg.ModelName = *modelShort
	}
	if *baseURL != "" {
		cfg.BaseURL = *baseURL
	}
	if *apiType != "" {
		cfg.APIType = *apiType
	}
	if *bearerToken != "" {
		cfg.BearerToken = *bearerToken
	}
	if *debugFlag {
		cfg.Debug = true
	}

	// Activate stderr tee when debug is set via env/config but not via --debug.
	if cfg.Debug && !*debugFlag {
		enableStderrTee(logFile)
	}

	if err := config.ValidateAndFixConfig(&cfg); err != nil {
		fatal("Configuration error", err)
	}

	cfg.Truncator = tools.SmartTruncate

	var v view.View
	tuiView, tuiErr := view.NewTUIView()
	if tuiErr != nil {
		slog.Warn("TUI unavailable, falling back to CLI", "reason", tuiErr)
		v = view.NewCLIView()
	} else {
		v = tuiView
		// TUI mode: drop the stderr tee so raw log lines don't bleed into the
		// alt-screen renderer. Output still goes to logs/loki-code.log.
		if cfg.Debug && logFile != nil {
			slog.SetDefault(newLogger(logFile))
		}
	}

	runner := tools.NewToolRunner(v.Confirm, v.ShowDiffAndConfirm)

	if *listModels {
		fmt.Println("Available models:")
		if err := clients.ListModels(cfg, os.Stdout, os.Stderr); err != nil {
			fatal("Error listing models", err)
		}
		return
	}

	if v.IsCLI() {
		fmt.Println("Loki Code - AI Coding Agent")
		fmt.Printf("Connecting to %s API (%s)...\n", cfg.APIType, cfg.ModelName)
	}

	ctxMgr := session.NewContextManager(4000, tools.GetAvailableTools)

	client, err := clients.CreateClient(cfg, ctxMgr, runner.Execute, tools.GetAvailableTools)
	if err != nil {
		fatal("Error creating client", err)
	}

	if contextWindow, err := client.DetectContextWindow(); err == nil {
		ctxMgr.SetMaxTokens(contextWindow)
		if v.IsCLI() {
			fmt.Printf("✓ Detected context window: %d tokens (auto-compact at 75%%)\n", contextWindow)
		}
	} else if v.IsCLI() {
		fmt.Fprintf(os.Stderr, "⚠️  Could not detect context window: %v\n", err)
		fmt.Println("✓ Using default context limit: 4,000 tokens")
	}

	if *conciseFlag || *conciseShort {
		client.EnableConciseMode()
	}

	if err := agent.Run(context.Background(), client, v); err != nil {
		fatal("Agent error", err)
	}
}
