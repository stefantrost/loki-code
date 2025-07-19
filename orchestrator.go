package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

type OllamaClient struct {
	baseURL string
	client  *http.Client
}

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatRequest struct {
	Model    string        `json:"model"`
	Messages []ChatMessage `json:"messages"`
	Stream   bool          `json:"stream"`
}

type ChatResponse struct {
	Model     string      `json:"model"`
	CreatedAt time.Time   `json:"created_at"`
	Message   ChatMessage `json:"message"`
	Done      bool        `json:"done"`
}

func NewOllamaClient(baseURL string) *OllamaClient {
	return &OllamaClient{
		baseURL: baseURL,
		client: &http.Client{
			Timeout: 0, // No timeout for streaming responses
		},
	}
}

func (c *OllamaClient) StreamChat(userInput string) error {
	request := ChatRequest{
		Model: "qwen3:32b",
		Messages: []ChatMessage{
			{
				Role:    "user",
				Content: userInput,
			},
		},
		Stream: true,
	}

	jsonData, err := json.Marshal(request)
	if err != nil {
		return fmt.Errorf("error marshaling request: %v", err)
	}

	resp, err := c.client.Post(c.baseURL+"/api/chat", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("error making request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("API returned status: %s", resp.Status)
	}

	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}

		var chatResponse ChatResponse
		if err := json.Unmarshal([]byte(line), &chatResponse); err != nil {
			continue
		}

		if chatResponse.Message.Content != "" {
			fmt.Print(chatResponse.Message.Content)
		}

		if chatResponse.Done {
			fmt.Println()
			break
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("error reading response: %v", err)
	}

	return nil
}