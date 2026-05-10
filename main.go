package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"strings"
	"syscall"
	
	"loki-code/clients"
)

func main() {
	// Command line flags
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

	fmt.Println("Loki Code - AI Coding Agent")
	
	// Handle --create-config flag
	if *createConfig {
		configPath := "llm.env"
		if *configFile != "" {
			configPath = *configFile
		}
		
		if err := CreateExampleConfig(configPath); err != nil {
			fmt.Printf("Error creating config file: %v\n", err)
			os.Exit(1)
		}
		
		fmt.Printf("✓ Created example configuration file: %s\n", configPath)
		fmt.Println("Edit the file with your API settings and run loki-code again.")
		os.Exit(0)
	}
	
	// Load configuration
	config, err := LoadConfig(*configFile)
	if err != nil {
		fmt.Printf("Error loading configuration: %v\n", err)
		fmt.Println("Run 'loki-code --create-config' to create an example configuration file.")
		os.Exit(1)
	}
	
	// Override config with command line flags (highest priority)
	if *modelFlag != "" {
		config.ModelName = *modelFlag
	} else if *modelShort != "" {
		config.ModelName = *modelShort
	}
	
	if *baseURL != "" {
		config.BaseURL = *baseURL
	}
	
	if *apiType != "" {
		config.APIType = *apiType
	}
	
	if *bearerToken != "" {
		config.BearerToken = *bearerToken
	}
	
	if *debugFlag {
		config.Debug = true
	}
	
	// Validate and fix configuration
	if err := ValidateAndFixConfig(&config); err != nil {
		fmt.Printf("Configuration error: %v\n", err)
		os.Exit(1)
	}
	
	// Print configuration
	PrintConfig(config)
	
	// Handle --list-models flag
	if *listModels {
		fmt.Println("Available models:")
		if strings.ToLower(config.APIType) == "ollama" {
			cmd := exec.Command("ollama", "list")
			cmd.Stdout = os.Stdout
			cmd.Stderr = os.Stderr
			err := cmd.Run()
			if err != nil {
				fmt.Printf("Error running ollama list: %v\n", err)
				fmt.Println("Make sure Ollama is installed and running")
				os.Exit(1)
			}
		} else {
			fmt.Printf("Model listing not supported for API type: %s\n", config.APIType)
			fmt.Printf("Current configured model: %s\n", config.ModelName)
		}
		os.Exit(0)
	}
	
	fmt.Printf("Connecting to %s API (%s)...\n", config.APIType, config.ModelName)

	// Create client using factory
	client, err := clients.CreateClient(config)
	if err != nil {
		fmt.Printf("Error creating client: %v\n", err)
		os.Exit(1)
	}
	
	// Set concise mode if flag is provided
	if *conciseFlag || *conciseShort {
		client.EnableConciseMode()
	}
	
	fmt.Println("Type 'exit', 'quit' to stop, '/plan' to enter plan mode, '/execute' to exit plan mode")
	fmt.Println("Commands: /stats, /clear, /compact, /concise, /verbose, /mode")
	fmt.Println("Tasks: /task [description], /task (show current), /complete")
	fmt.Println("Press Ctrl+C during response to interrupt (or at prompt to exit)")
	fmt.Println("----------------------------------------")

	// Handle Ctrl+C gracefully
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-sigChan
		if client.IsResponseActive() {
			fmt.Println("\n^C [Response interrupted]")
			client.Interrupt()
		} else {
			fmt.Println("\nGoodbye!")
			os.Exit(0)
		}
	}()

	scanner := bufio.NewScanner(os.Stdin)

	for {
		prompt := "\n> "
		if client.IsInPlanMode() {
			prompt = "\n[PLAN] > "
		}
		if client.IsInConciseMode() {
			prompt = strings.Replace(prompt, "> ", "[CONCISE] > ", 1)
		}
		fmt.Print(prompt)
		
		if !scanner.Scan() {
			break
		}

		input := strings.TrimSpace(scanner.Text())
		
		if input == "" {
			continue
		}

		if input == "exit" || input == "quit" {
			fmt.Println("Goodbye!")
			break
		}

		if input == "/clear" {
			client.ClearContext()
			fmt.Println("Context cleared!")
			continue
		}

		if input == "/stats" {
			tokens, messages, maxTokens := client.GetStats()
			mode := "Execute"
			if client.IsInPlanMode() {
				mode = "Plan"
			}
			responseMode := "Verbose"
			if client.IsInConciseMode() {
				responseMode = "Concise"
			}
			
			activeTask := client.GetActiveTask()
			taskInfo := "None"
			if activeTask != "" {
				taskInfo = "Active"
			}
			
			fmt.Printf("Context Stats: %d/%d tokens, %d messages | Mode: %s | Response: %s | Task: %s\n", 
				tokens, maxTokens, messages, mode, responseMode, taskInfo)
			continue
		}

		if input == "/concise" {
			if client.IsInConciseMode() {
				fmt.Println("Already in concise mode!")
				continue
			}
			client.EnableConciseMode()
			fmt.Println("📝 Switched to concise mode - responses will be brief and to-the-point")
			continue
		}

		if input == "/verbose" {
			if !client.IsInConciseMode() {
				fmt.Println("Already in verbose mode!")
				continue
			}
			client.DisableConciseMode()
			fmt.Println("📝 Switched to verbose mode - responses will include detailed explanations")
			continue
		}

		if input == "/mode" {
			mode := "verbose"
			if client.IsInConciseMode() {
				mode = "concise"
			}
			planMode := ""
			if client.IsInPlanMode() {
				planMode = " (plan mode active)"
			}
			fmt.Printf("Current response mode: %s%s\n", mode, planMode)
			continue
		}

		if input == "/task" {
			activeTask := client.GetActiveTask()
			if activeTask != "" {
				fmt.Printf("🎯 Current task: %s\n", activeTask)
			} else {
				fmt.Println("No active task")
			}
			continue
		}

		if input == "/complete" {
			activeTask := client.GetActiveTask()
			if activeTask != "" {
				client.CompleteCurrentTask()
				fmt.Println("✅ Task marked as complete")
			} else {
				fmt.Println("No active task to complete")
			}
			continue
		}

		if strings.HasPrefix(input, "/task ") {
			newTask := strings.TrimPrefix(input, "/task ")
			if strings.TrimSpace(newTask) != "" {
				client.SetActiveTask(newTask)
			} else {
				fmt.Println("Please specify a task: /task <description>")
			}
			continue
		}

		if input == "/compact" {
			if !client.CanCompact() {
				fmt.Println("Context not ready for compacting (need 60%+ token usage)")
				continue
			}
			fmt.Println("Compacting conversation context...")
			if err := client.CompactContext(); err != nil {
				fmt.Printf("Compacting failed: %v\n", err)
			}
			continue
		}

		if input == "/plan" {
			if client.IsInPlanMode() {
				fmt.Println("Already in plan mode!")
				continue
			}
			client.EnablePlanMode()
			fmt.Println("🎯 Plan Mode Activated!")
			fmt.Println("You can now create execution plans. Only read operations are allowed.")
			fmt.Println("Use '/execute' to exit plan mode and enable all tools.")
			continue
		}

		if input == "/execute" {
			if !client.IsInPlanMode() {
				fmt.Println("Not in plan mode!")
				continue
			}
			client.DisablePlanMode()
			fmt.Println("⚡ Execute Mode Activated!")
			fmt.Println("All tools are now available for execution.")
			continue
		}

		fmt.Print("Assistant: ")
		if err := client.StreamChat(input); err != nil {
			fmt.Printf("Error: %v\n", err)
		}
	}

	if err := scanner.Err(); err != nil {
		fmt.Printf("Error reading input: %v\n", err)
	}
}