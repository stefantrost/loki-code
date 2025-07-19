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
	fmt.Println("Type 'exit' or 'quit' to stop, or press Ctrl+C")
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

		fmt.Print("Assistant: ")
		if err := client.StreamChat(input); err != nil {
			fmt.Printf("Error: %v\n", err)
		}
	}

	if err := scanner.Err(); err != nil {
		fmt.Printf("Error reading input: %v\n", err)
	}
}