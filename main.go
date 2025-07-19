package main

import (
	"bufio"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"
)

func main() {
	fmt.Println("Loki Code - AI Coding Agent")
	fmt.Println("Connecting to Ollama (qwen3:32b)...")
	fmt.Println("Type 'exit', 'quit' to stop, '/clear' to clear context, '/compact' to compress context, or press Ctrl+C")
	fmt.Println("----------------------------------------")

	client := NewOllamaClient("http://localhost:11434")

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
		fmt.Print("\n> ")
		
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
			fmt.Printf("Context Stats: %d/%d tokens, %d messages\n", tokens, maxTokens, messages)
			continue
		}

		if input == "/compact" {
			if !client.CanCompact() {
				fmt.Println("Not enough messages to compact (need at least 8 messages)")
				continue
			}
			fmt.Println("Compacting conversation context...")
			if err := client.CompactContext(); err != nil {
				fmt.Printf("Compacting failed: %v\n", err)
			}
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