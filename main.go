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
)

func main() {
	// Command line flags
	modelFlag := flag.String("model", "", "Model to use (default: qwen3:32b)")
	modelShort := flag.String("m", "", "Model to use (short form)")
	ollamaURL := flag.String("url", "http://localhost:11434", "Ollama server URL")
	listModels := flag.Bool("list-models", false, "List available models and exit")
	
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
	
	fmt.Println("Type 'exit', 'quit' to stop, '/plan' to enter plan mode, '/execute' to exit plan mode, or press Ctrl+C")
	fmt.Println("----------------------------------------")

	// Handle Ctrl+C gracefully
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-c
		fmt.Println("\nGoodbye!")
		os.Exit(0)
	}()

	scanner := bufio.NewScanner(os.Stdin)

	for {
		if client.IsInPlanMode() {
			fmt.Print("\n[PLAN] > ")
		} else {
			fmt.Print("\n> ")
		}
		
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
			fmt.Printf("Context Stats: %d/%d tokens, %d messages | Mode: %s\n", tokens, maxTokens, messages, mode)
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