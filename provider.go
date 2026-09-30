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

// Event is one unit of streamed output from a provider.
type Event struct {
	Kind    EventKind
	Text    string   // set on EventDelta: an incremental chunk of assistant text
	ToolUse *ToolUse // set on EventToolCall
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
}

// Provider streams a single model turn. Implementations must close the
// returned channel after emitting a terminal EventDone or EventError.
type Provider interface {
	StreamChat(ctx context.Context, req ChatRequest) (<-chan Event, error)
}

// Verifier checks that the adapter's API key is accepted without spending
// tokens, for validating a key when it's saved. A rejected key comes back as
// an *Error of KindAuth.
type Verifier interface {
	Verify(ctx context.Context) error
}
