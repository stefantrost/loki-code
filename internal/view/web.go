package view

import "fmt"

// WebView is a placeholder for a future HTTP + WebSocket view.
// It satisfies the View interface but returns an error on Run.
//
// Design notes: Run would start an HTTP server and wait for a WebSocket
// connection; WriteToken would send {type:"token"} frames; CommitMessage
// would send {type:"message", rendered: <html>} frames; ShowDiffAndConfirm
// would send the diff as a frame and await a {type:"confirm"} response.
type WebView struct{}

// NewWebView returns an unimplemented WebView stub.
func NewWebView() *WebView { return &WebView{} }

func (v *WebView) ReadInput(_ string) (string, error) {
	return "", fmt.Errorf("web view: not implemented")
}
func (v *WebView) WriteToken(_ string)       {}
func (v *WebView) CommitMessage(_, _ string) {}
func (v *WebView) WriteSystem(_ string)      {}
func (v *WebView) ShowDiffAndConfirm(_, _, _ string) (bool, error) {
	return false, fmt.Errorf("web view: not implemented")
}
func (v *WebView) Confirm(_ string) (bool, error) {
	return false, fmt.Errorf("web view: not implemented")
}
func (v *WebView) BeginStream()         {}
func (v *WebView) WriteThinking(_ string) {}
func (v *WebView) UpdateStatus(_ Status) {}
func (v *WebView) Run(_ func()) error    { return fmt.Errorf("web view: not yet implemented") }
func (v *WebView) Stop()                 {}
func (v *WebView) IsCLI() bool           { return false }
