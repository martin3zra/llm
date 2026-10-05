package openai

import (
	"encoding/json"
	"net/http"
	"testing"

	sdk "github.com/openai/openai-go"

	"github.com/martin3zra/llm"
)

// marshalOne mirrors the Anthropic adapter's test helper: assert on the
// actual wire shape rather than reaching into SDK struct internals.
func marshalOne(t *testing.T, v any) map[string]any {
	t.Helper()
	raw, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("unmarshal into map: %v (raw: %s)", err, raw)
	}
	return m
}

func TestToOpenAIMessages_InjectsSystemPromptFirst(t *testing.T) {
	out := toOpenAIMessages("be helpful", []llm.Message{
		{Role: llm.RoleUser, Text: "hi"},
	})
	if len(out) != 2 {
		t.Fatalf("got %d messages, want 2 (system + user)", len(out))
	}
	m := marshalOne(t, out[0])
	if m["role"] != "system" || m["content"] != "be helpful" {
		t.Errorf("first message = %+v, want the system prompt", m)
	}
}

func TestToOpenAIMessages_OmitsSystemMessageWhenEmpty(t *testing.T) {
	out := toOpenAIMessages("", []llm.Message{{Role: llm.RoleUser, Text: "hi"}})
	if len(out) != 1 {
		t.Fatalf("got %d messages, want 1 (no system message)", len(out))
	}
}

func TestToOpenAIMessages_PlainUserAndAssistantText(t *testing.T) {
	out := toOpenAIMessages("", []llm.Message{
		{Role: llm.RoleUser, Text: "what's the status?"},
		{Role: llm.RoleAssistant, Text: "let me check"},
	})
	if m := marshalOne(t, out[0]); m["role"] != "user" {
		t.Errorf("role = %v, want user", m["role"])
	}
	if m := marshalOne(t, out[1]); m["role"] != "assistant" {
		t.Errorf("role = %v, want assistant", m["role"])
	}
}

func TestToOpenAIMessages_ToolUseBecomesAssistantToolCall(t *testing.T) {
	out := toOpenAIMessages("", []llm.Message{
		{ToolUse: &llm.ToolUse{ID: "call_1", Name: "get_payroll_period_status", Input: json.RawMessage(`{"period_uuid":"abc"}`)}},
	})
	m := marshalOne(t, out[0])
	if m["role"] != "assistant" {
		t.Errorf("role = %v, want assistant", m["role"])
	}
	toolCalls, ok := m["tool_calls"].([]any)
	if !ok || len(toolCalls) != 1 {
		t.Fatalf("tool_calls = %v, want a single entry", m["tool_calls"])
	}
	call := toolCalls[0].(map[string]any)
	if call["id"] != "call_1" {
		t.Errorf("id = %v, want call_1", call["id"])
	}
	fn := call["function"].(map[string]any)
	if fn["name"] != "get_payroll_period_status" {
		t.Errorf("function.name = %v", fn["name"])
	}
	if fn["arguments"] != `{"period_uuid":"abc"}` {
		t.Errorf("function.arguments = %v", fn["arguments"])
	}
}

// TestToOpenAIMessages_ToolResultUsesToolRole documents a real API
// difference from Anthropic: OpenAI has a distinct "tool" role, so unlike
// the Anthropic adapter this does NOT get folded into a user turn.
func TestToOpenAIMessages_ToolResultUsesToolRole(t *testing.T) {
	out := toOpenAIMessages("", []llm.Message{
		{ToolResult: &llm.ToolResult{ToolUseID: "call_1", Content: `{"status":"draft"}`}},
	})
	m := marshalOne(t, out[0])
	if m["role"] != "tool" {
		t.Errorf("role = %v, want tool (OpenAI has a distinct tool role)", m["role"])
	}
	if m["tool_call_id"] != "call_1" {
		t.Errorf("tool_call_id = %v, want call_1", m["tool_call_id"])
	}
	if m["content"] != `{"status":"draft"}` {
		t.Errorf("content = %v", m["content"])
	}
}

func TestToOpenAITools_MapsNameDescriptionAndSchema(t *testing.T) {
	specs := []llm.ToolSpec{
		{
			Name:        "get_payroll_period_status",
			Description: "look up a period",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"period_uuid":{"type":"string"}},"required":["period_uuid"]}`),
		},
	}
	out := toOpenAITools(specs)
	if len(out) != 1 {
		t.Fatalf("got %d tools, want 1", len(out))
	}
	m := marshalOne(t, out[0])
	if m["type"] != "function" {
		t.Errorf("type = %v, want function", m["type"])
	}
	fn := m["function"].(map[string]any)
	if fn["name"] != "get_payroll_period_status" {
		t.Errorf("name = %v", fn["name"])
	}
	if fn["description"] != "look up a period" {
		t.Errorf("description = %v", fn["description"])
	}
	params := fn["parameters"].(map[string]any)
	required, _ := params["required"].([]any)
	if len(required) != 1 || required[0] != "period_uuid" {
		t.Errorf("parameters.required = %v, want [period_uuid]", params["required"])
	}
}

func TestNormalizeError_MapsStatusCodesToKind(t *testing.T) {
	cases := []struct {
		status int
		want   llm.Kind
	}{
		{http.StatusUnauthorized, llm.KindAuth},
		{http.StatusForbidden, llm.KindAuth},
		{http.StatusTooManyRequests, llm.KindRateLimit},
		{http.StatusBadRequest, llm.KindInvalidReq},
		{http.StatusNotFound, llm.KindInvalidReq},
		{http.StatusInternalServerError, llm.KindUnavailable},
	}

	for _, c := range cases {
		apiErr := &sdk.Error{StatusCode: c.status}
		got := normalizeError(apiErr)
		if got.Kind != c.want {
			t.Errorf("status %d: kind = %q, want %q", c.status, got.Kind, c.want)
		}
	}
}

func TestNormalizeError_NonAPIErrorIsUnknown(t *testing.T) {
	got := normalizeError(plainError{})
	if got.Kind != llm.KindUnknown {
		t.Errorf("kind = %q, want unknown", got.Kind)
	}
}

type plainError struct{}

func (plainError) Error() string { return "context deadline exceeded" }

func TestToOpenAIParams_AsksForUsage(t *testing.T) {
	params := toOpenAIParams(llm.ChatRequest{Model: "gpt-x"})
	if !params.StreamOptions.IncludeUsage.Valid() || !params.StreamOptions.IncludeUsage.Value {
		t.Fatal("stream_options.include_usage isn't set; streams would report no usage")
	}
}
