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
	"time"
)

func main() {
	// Command line flags
	modelFlag := flag.String("model", "", "Model to use (default: qwen3:32b)")
	modelShort := flag.String("m", "", "Model to use (short form)")
	ollamaURL := flag.String("url", "http://localhost:11434", "Ollama server URL")
	listModels := flag.Bool("list-models", false, "List available models and exit")
	debugFlag := flag.Bool("debug", false, "Enable debug logging")
	conciseFlag := flag.Bool("concise", false, "Start in concise mode (brief responses)")
	conciseShort := flag.Bool("c", false, "Start in concise mode (short form)")
	
	flag.Parse()

	// Determine model to use (priority: flag > env > default)
	modelName := "qwen3:32b" // default
	
	// Check environment variable
	if envModel := os.Getenv("LOKI_MODEL"); envModel != "" {
		modelName = envModel
	}
	
	// Check command line flags (highest priority)
	if *modelFlag != "" {
		modelName = *modelFlag
	} else if *modelShort != "" {
		modelName = *modelShort
	}

	fmt.Println("Loki Code - AI Coding Agent")
	
	// Handle --list-models flag
	if *listModels {
		fmt.Println("Available models:")
		cmd := exec.Command("ollama", "list")
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		err := cmd.Run()
		if err != nil {
			fmt.Printf("Error running ollama list: %v\n", err)
			fmt.Println("Make sure Ollama is installed and running")
			os.Exit(1)
		}
		os.Exit(0)
	}
	
	fmt.Printf("Connecting to Ollama (%s)...\n", modelName)

	client := NewOllamaClient(*ollamaURL, modelName)
	client.SetDebug(*debugFlag)
	
	// Set concise mode if flag is provided
	if *conciseFlag || *conciseShort {
		client.SetConciseMode(true)
	}
	
	fmt.Println("Type 'exit', 'quit' to stop, '/plan' to enter plan mode, '/execute' to exit plan mode")
	fmt.Println("Commands: /stats, /clear, /compact, /concise, /verbose, /mode")
	fmt.Println("Tasks: /task [description], /task (show current), /complete")
	fmt.Println("Press Ctrl+C during response to interrupt (or at prompt to exit)")
	fmt.Println("----------------------------------------")

	// Handle Ctrl+C gracefully
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-c
		if client.IsResponseActive() {
			fmt.Println("\n^C [Response interrupted]")
			client.InterruptResponse()
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
			tokens, messages, maxTokens := client.GetContextStats()
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
			if activeTask != nil {
				elapsed := time.Since(activeTask.CreatedAt)
				taskInfo = fmt.Sprintf("Active (%s)", elapsed.Truncate(time.Second))
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
			client.SetConciseMode(true)
			fmt.Println("📝 Switched to concise mode - responses will be brief and to-the-point")
			continue
		}

		if input == "/verbose" {
			if !client.IsInConciseMode() {
				fmt.Println("Already in verbose mode!")
				continue
			}
			client.SetConciseMode(false)
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
			if activeTask != nil {
				fmt.Printf("🎯 Current task: %s\n", activeTask.Goal)
				fmt.Printf("   Created: %s | Status: %s\n", 
					activeTask.CreatedAt.Format("15:04:05"), activeTask.Status)
			} else {
				fmt.Println("No active task")
			}
			continue
		}

		if input == "/complete" {
			activeTask := client.GetActiveTask()
			if activeTask != nil {
				client.CompleteCurrentTask("Manually marked as complete")
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