// Package llm defines a vendor-neutral contract between an app's
// tool-calling loop and an LLM backend. Callers work only with the types in
// this file — never with a specific vendor's SDK types — so adding a
// provider is a new adapter package (see anthropic and openai), not a change
// to the caller's loop.
package llm

import (
	"context"
	"encoding/json"
)

type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

// ToolUse is a model-issued request to call a tool.
type ToolUse struct {
	ID    string
	Name  string
	Input json.RawMessage
}

// ToolResult is the outcome of executing a ToolUse, fed back to the model.
type ToolResult struct {
	ToolUseID string
	Content   string
	IsError   bool
}

// Message is one turn of conversation history. A single turn carries either
// plain text, a tool use (assistant asked to call a tool), or a tool result
// (the answer being fed back) — never more than one of these three.
type Message struct {
	Role       Role
	Text       string
	ToolUse    *ToolUse
	ToolResult *ToolResult
}

// ToolSpec describes one callable tool. InputSchema is a JSON Schema object,
// vendor-neutral — each adapter translates it into its own wire format.
type ToolSpec struct {
	Name        string
	Description string
	InputSchema json.RawMessage
}

type EventKind string

const (
	EventDelta    EventKind = "delta"
	EventToolCall EventKind = "tool_call"
	EventDone     EventKind = "done"
	EventError    EventKind = "error"
)

// Usage is the token count of one model turn, for metering. Input counts
// uncached prompt tokens; prompt-cache writes and reads are reported apart
// because providers price them differently.
type Usage struct {
	Input      int
	Output     int
	CacheWrite int
	CacheRead  int
}

// Add sums u and v, for totalling several turns.
func (u Usage) Add(v Usage) Usage {
	return Usage{u.Input + v.Input, u.Output + v.Output, u.CacheWrite + v.CacheWrite, u.CacheRead + v.CacheRead}
}

// Event is one unit of streamed output from a provider.
type Event struct {
	Kind    EventKind
	Text    string   // set on EventDelta: an incremental chunk of assistant text
	ToolUse *ToolUse // set on EventToolCall
	Usage   *Usage   // set on EventDone when the provider reported it
	Err     error    // set on EventError; normalized, see errors.go
}

// ChatRequest is one turn of the tool-calling loop sent to a provider.
type ChatRequest struct {
	System   string
	Messages []Message
	Tools    []ToolSpec
	Model    string
	// ToolChoice, when set, names the one tool the model must call this
	// turn instead of answering in text — for structured extraction, where
	// the tool's input is the result. Empty leaves the choice to the model.
	// It must name one of Tools.
	ToolChoice string
	// CacheSystem asks the provider to cache the prompt through System
	// (tools and system prompt), for callers that resend a large, unchanged
	// System across several requests. Providers that cache prefixes on their
	// own (OpenAI) ignore it; prompts below the provider's minimum aren't
	// cached either way.
	CacheSystem bool
}

// Provider streams a single model turn. Implementations must close the
// returned channel after emitting a terminal EventDone or EventError.
type Provider interface {
	StreamChat(ctx context.Context, req ChatRequest) (<-chan Event, error)
}

// Verifier checks, without spending tokens, that the adapter's API key is
// accepted and can use model: for validating a key and model when they're
// saved. A rejected key comes back as an *Error of KindAuth, an unknown model
// as KindInvalidReq.
type Verifier interface {
	Verify(ctx context.Context, model string) error
}
