package tools

import (
	"fmt"
	"log/slog"
	"net"
	"net/url"
	"os/exec"
	"strings"
	"time"
)

// CIDR blocks that http_request must refuse: private + loopback + link-local
// + multicast + CGNAT for v4, plus loopback + ULA + link-local + multicast for v6.
var blockedCIDRs = func() []*net.IPNet {
	blocks := []string{
		"10.0.0.0/8",
		"172.16.0.0/12",
		"192.168.0.0/16",
		"127.0.0.0/8",
		"169.254.0.0/16",
		"100.64.0.0/10",
		"0.0.0.0/8",
		"224.0.0.0/4",
		"::1/128",
		"fc00::/7",
		"fe80::/10",
		"ff00::/8",
	}
	nets := make([]*net.IPNet, 0, len(blocks))
	for _, b := range blocks {
		if _, n, err := net.ParseCIDR(b); err == nil {
			nets = append(nets, n)
		}
	}
	return nets
}()

// sensitiveHeaderNames identifies headers whose values should never be logged.
var sensitiveHeaderNames = map[string]struct{}{
	"authorization":       {},
	"proxy-authorization": {},
	"cookie":              {},
	"set-cookie":          {},
	"x-api-key":           {},
	"api-key":             {},
}

func executeCurl(args map[string]interface{}) (string, error) {
	slog.Debug("executeCurl called", "arguments", args)

	rawURL, ok := args["url"].(string)
	if !ok {
		slog.Error("Invalid url argument type", "expected", "string", "got", fmt.Sprintf("%T", args["url"]))
		return "", fmt.Errorf("url argument is required and must be a string")
	}

	slog.Debug("Validating URL", "url", rawURL)
	if err := validateURL(rawURL); err != nil {
		slog.Error("URL validation failed", "url", rawURL, "error", err)
		return "", err
	}

	method := "GET"
	if methodArg, hasMethod := args["method"].(string); hasMethod {
		method = strings.ToUpper(methodArg)
		slog.Debug("HTTP method specified", "method", method)
		if !isValidHTTPMethod(method) {
			slog.Error("Invalid HTTP method", "method", method)
			return "", fmt.Errorf("invalid HTTP method: %s", methodArg)
		}
	}

	timeout := 10
	if timeoutArg, hasTimeout := args["timeout"].(float64); hasTimeout {
		slog.Debug("Timeout specified", "timeout", timeoutArg)
		if timeoutArg > 30 {
			slog.Error("Timeout exceeds maximum", "timeout", timeoutArg, "max", 30)
			return "", fmt.Errorf("timeout cannot exceed 30 seconds")
		}
		if timeoutArg < 1 {
			slog.Error("Timeout below minimum", "timeout", timeoutArg, "min", 1)
			return "", fmt.Errorf("timeout must be at least 1 second")
		}
		timeout = int(timeoutArg)
	}

	followRedirects := true
	if followArg, hasFollow := args["follow_redirects"].(bool); hasFollow {
		followRedirects = followArg
		slog.Debug("Follow redirects specified", "follow", followRedirects)
	}

	curlArgs := []string{
		"--silent",
		"--show-error",
		"--max-time", fmt.Sprintf("%d", timeout),
		"--request", method,
	}
	slog.Debug("Building curl command", "args", curlArgs)

	if followRedirects {
		curlArgs = append(curlArgs, "--location")
	}

	if headersArg, hasHeaders := args["headers"]; hasHeaders {
		slog.Debug("Processing headers", "header_keys", headerKeyNames(headersArg))
		if headers, ok := headersArg.(map[string]interface{}); ok {
			for key, value := range headers {
				headerStr := fmt.Sprintf("%s: %v", key, value)
				slog.Debug("Adding header", "header", redactHeader(headerStr))
				if err := validateHeader(headerStr); err != nil {
					slog.Error("Invalid header", "header", headerStr, "error", err)
					return "", fmt.Errorf("invalid header %s: %v", key, err)
				}
				curlArgs = append(curlArgs, "--header", headerStr)
			}
		}
	}

	if dataArg, hasData := args["data"].(string); hasData && dataArg != "" {
		if method == "GET" || method == "HEAD" {
			slog.Error("Cannot send data with GET/HEAD method", "method", method)
			return "", fmt.Errorf("cannot send data with %s method", method)
		}
		curlArgs = append(curlArgs, "--data", dataArg)
		slog.Debug("Request data added", "data_length", len(dataArg))
	}

	curlArgs = append(curlArgs, rawURL)
	slog.Debug("Executing curl command", "url", rawURL, "method", method, "args", curlArgs)

	cmd := exec.Command("curl", curlArgs...)
	cmd.Dir = "."

	done := make(chan error, 1)
	var output []byte
	var err error

	go func() {
		output, err = cmd.CombinedOutput()
		done <- err
	}()

	select {
	case cmdErr := <-done:
		outputStr := string(output)
		slog.Debug("Curl command completed", "error", cmdErr, "output_length", len(outputStr))

		if cmdErr != nil {
			slog.Error("Curl command failed", "error", cmdErr, "output", outputStr)
			return "", fmt.Errorf("curl command failed: %v\nOutput: %s", cmdErr, outputStr)
		}

		if len(outputStr) > 10000 {
			outputStr = outputStr[:10000] + "\n... (output truncated at 10,000 characters)"
		}

		result := formatHTTPResponse(method, rawURL, outputStr)
		slog.Debug("HTTP request completed successfully", "url", rawURL, "method", method, "result_length", len(result))
		return result, nil

	case <-time.After(time.Duration(timeout+5) * time.Second):
		slog.Error("HTTP request timed out", "url", rawURL, "timeout", timeout)
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
		}
		return "", fmt.Errorf("HTTP request timed out after %d seconds", timeout)
	}
}

func validateURL(rawURL string) error {
	u, err := url.Parse(rawURL)
	if err != nil {
		return fmt.Errorf("invalid URL: %v", err)
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return fmt.Errorf("URL must start with http:// or https://")
	}

	host := strings.ToLower(u.Hostname())
	if host == "" {
		return fmt.Errorf("URL has no host")
	}

	// Reject obvious internal hostnames before any name lookup.
	if host == "localhost" || strings.HasSuffix(host, ".localhost") ||
		strings.HasSuffix(host, ".local") || strings.HasSuffix(host, ".internal") {
		return fmt.Errorf("access to internal/localhost addresses is not allowed")
	}

	// If the host is a literal IP, check it directly. For names, resolve and
	// reject if any answer falls in a blocked range (defense against DNS
	// rebinding to a private address).
	var ips []net.IP
	if ip := net.ParseIP(host); ip != nil {
		ips = []net.IP{ip}
	} else {
		resolved, lookupErr := net.LookupIP(host)
		if lookupErr != nil {
			// Let the HTTP client surface the real error; don't block on DNS failure.
			return nil
		}
		ips = resolved
	}

	for _, ip := range ips {
		for _, block := range blockedCIDRs {
			if block.Contains(ip) {
				return fmt.Errorf("access to internal/localhost addresses is not allowed")
			}
		}
	}

	return nil
}

func isValidHTTPMethod(method string) bool {
	validMethods := map[string]bool{
		"GET": true, "POST": true, "PUT": true, "DELETE": true,
		"PATCH": true, "HEAD": true, "OPTIONS": true,
	}
	return validMethods[method]
}

func redactHeader(headerStr string) string {
	idx := strings.Index(headerStr, ":")
	if idx <= 0 {
		return headerStr
	}
	name := strings.ToLower(strings.TrimSpace(headerStr[:idx]))
	if _, sensitive := sensitiveHeaderNames[name]; sensitive {
		return headerStr[:idx] + ": [REDACTED]"
	}
	return headerStr
}

func headerKeyNames(headersArg interface{}) []string {
	headers, ok := headersArg.(map[string]interface{})
	if !ok {
		return nil
	}
	names := make([]string, 0, len(headers))
	for k := range headers {
		names = append(names, k)
	}
	return names
}

func validateHeader(header string) error {
	if !strings.Contains(header, ":") {
		return fmt.Errorf("header must be in 'Key: Value' format")
	}

	if strings.Contains(header, "\n") || strings.Contains(header, "\r") {
		return fmt.Errorf("headers cannot contain newline characters")
	}

	return nil
}

func formatHTTPResponse(method, rawURL, response string) string {
	var result strings.Builder

	fmt.Fprintf(&result, "🌐 HTTP %s Request\n", method)
	fmt.Fprintf(&result, "URL: %s\n", rawURL)
	result.WriteString(strings.Repeat("=", 50) + "\n\n")

	if strings.TrimSpace(response) == "" {
		result.WriteString("(Empty response)\n")
	} else {
		result.WriteString(response)
	}

	return result.String()
}
