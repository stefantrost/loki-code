package main

import (
	"flag"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"loki-code/clients"
	"loki-code/internal/agent"
	"loki-code/internal/config"
	"loki-code/internal/session"
	"loki-code/internal/tools"
	"loki-code/internal/view"
)

// setupLogger opens logs/loki-code.log (creating the directory if needed) and
// configures slog to write there.  The log file is at DEBUG level regardless of
// the debug flag so every run is fully auditable; the debug flag additionally
// tees output to stderr for CLI users who want live visibility.
// Returns the open log file so the caller can close it on exit (nil on error).
func setupLogger(debug bool) *os.File {
	if err := os.MkdirAll("logs", 0750); err != nil {
		// Can't create logs dir — fall back to stderr only.
		slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
			Level: slog.LevelDebug,
		})))
		return nil
	}

	lf, err := os.OpenFile(filepath.Join("logs", "loki-code.log"),
		os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
			Level: slog.LevelDebug,
		})))
		return nil
	}

	// Always capture DEBUG in the file; optionally tee to stderr in debug mode.
	// The tee is suppressed later for TUI mode (alt-screen would show raw log
	// lines) — see the view-selection block in main().
	var w io.Writer = lf
	if debug {
		w = io.MultiWriter(lf, os.Stderr)
	}
	slog.SetDefault(slog.New(slog.NewTextHandler(w, &slog.HandlerOptions{
		Level: slog.LevelDebug,
	})))
	return lf
}

func main() {
	modelFlag := flag.String("model", "", "Model to use (default: qwen3:32b)")
	modelShort := flag.String("m", "", "Model to use (short form)")
	baseURL := flag.String("url", "", "API base URL (default: http://localhost:11434 for Ollama)")
	apiType := flag.String("api-type", "", "API type: ollama, openai, openai-compatible (auto-detected if not specified)")
	configFile := flag.String("config", "", "Configuration file path (default: llm.env)")
	bearerToken := flag.String("token", "", "Bearer token for authentication")
	listModels := flag.Bool("list-models", false, "List available models and exit")
	debugFlag := flag.Bool("debug", false, "Enable debug logging")
	conciseFlag := flag.Bool("concise", false, "Start in concise mode (brief responses)")
	conciseShort := flag.Bool("c", false, "Start in concise mode (short form)")
	createConfig := flag.Bool("create-config", false, "Create example configuration file")

	flag.Parse()

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
			slog.Error("Error creating config file", "error", err)
			os.Exit(1)
		}

		fmt.Printf("✓ Created example configuration file: %s\n", configPath)
		fmt.Println("Edit the file with your API settings and run loki-code again.")
		os.Exit(0)
	}

	cfg, err := config.LoadConfig(*configFile)
	if err != nil {
		slog.Error("Error loading configuration", "error", err)
		fmt.Println("Run 'loki-code --create-config' to create an example configuration file.")
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

	if err := config.ValidateAndFixConfig(&cfg); err != nil {
		slog.Error("Configuration error", "error", err)
		os.Exit(1)
	}

	// Select view early — before any stdout prints — so that TUI mode enters
	// alt-screen before writing anything to the terminal.  Startup messages
	// printed to the primary buffer before alt-screen would bleed back into
	// the user's shell after the TUI session ends, and in some terminal
	// emulators can cause a phantom second input row.
	var v view.View
	tuiView, tuiErr := view.NewTUIView()
	if tuiErr != nil {
		slog.Warn("TUI unavailable, falling back to CLI", "reason", tuiErr)
		v = view.NewCLIView()
	} else {
		v = tuiView
		// TUI mode: drop the stderr tee (if --debug added one) so raw log lines
		// don't bleed into the alt-screen renderer.  All output still goes to
		// logs/loki-code.log — use `tail -f logs/loki-code.log` to watch live.
		if *debugFlag && logFile != nil {
			slog.SetDefault(slog.New(slog.NewTextHandler(logFile, &slog.HandlerOptions{
				Level: slog.LevelDebug,
			})))
		}
	}
	_, isCLI := v.(*view.CLIView)

	// Wire the view's confirm/diff methods into the tools package so that
	// mutating tools (create_file, update_file, delete_file) use the right
	// mechanism for the active UI.  In TUI mode this suspends alt-screen
	// around each prompt; in CLI mode v.Confirm and v.ShowDiffAndConfirm
	// delegate to the same ui.* helpers the tools package used by default,
	// so behaviour is unchanged for CLI users.
	tools.SetConfirmHooks(v.Confirm, v.ShowDiffAndConfirm)

	if *listModels {
		fmt.Println("Available models:")
		if strings.ToLower(cfg.APIType) == "ollama" {
			cmd := exec.Command("ollama", "list")
			cmd.Stdout = os.Stdout
			cmd.Stderr = os.Stderr
			err := cmd.Run()
			if err != nil {
				slog.Error("Error running ollama list", "error", err)
				fmt.Println("Make sure Ollama is installed and running")
				os.Exit(1)
			}
		} else {
			fmt.Printf("Model listing not supported for API type: %s\n", cfg.APIType)
			fmt.Printf("Current configured model: %s\n", cfg.ModelName)
		}
		os.Exit(0)
	}

	// In CLI mode print startup diagnostics; TUI mode keeps the terminal clean.
	if isCLI {
		fmt.Println("Loki Code - AI Coding Agent")
		fmt.Printf("Connecting to %s API (%s)...\n", cfg.APIType, cfg.ModelName)
	}

	// Create context manager with tool provider callback
	ctxMgr := session.NewContextManager(4000, tools.GetAvailableTools)

	// Create client with callbacks
	client, err := clients.CreateClient(cfg, ctxMgr, tools.ExecuteToolWithPlanMode, tools.GetAvailableTools)
	if err != nil {
		slog.Error("Error creating client", "error", err)
		os.Exit(1)
	}
	client.SetTruncator(tools.SmartTruncate)

	// Detect and set context window
	if contextWindow, err := client.DetectContextWindow(); err == nil {
		ctxMgr.SetMaxTokens(contextWindow)
		if isCLI {
			fmt.Printf("✓ Detected context window: %d tokens (auto-compact at 75%%)\n", contextWindow)
		}
	} else {
		if isCLI {
			fmt.Printf("⚠️ Could not detect context window: %v\n", err)
			fmt.Printf("✓ Using default context limit: 4,000 tokens\n")
		}
	}

	if *conciseFlag || *conciseShort {
		client.EnableConciseMode()
	}

	if err := agent.Run(client, v); err != nil {
		slog.Error("Agent error", "error", err)
		os.Exit(1)
	}
}
