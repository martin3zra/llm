package anthropic

import (
	"encoding/json"
	"net/http"
	"testing"

	sdk "github.com/anthropics/anthropic-sdk-go"

	"github.com/martin3zra/llm"
)

// marshalOne is a test helper: MessageParam's exported fields are behind a
// custom MarshalJSON, so asserting on the wire shape is more robust (and
// more honest about what the API actually receives) than reaching into the
// SDK struct's internals.
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

func TestToAnthropicMessages_PlainUserText(t *testing.T) {
	out := toAnthropicMessages([]llm.Message{
		{Role: llm.RoleUser, Text: "what's the status?"},
	})
	if len(out) != 1 {
		t.Fatalf("got %d messages, want 1", len(out))
	}
	m := marshalOne(t, out[0])
	if m["role"] != "user" {
		t.Errorf("role = %v, want user", m["role"])
	}
}

func TestToAnthropicMessages_PlainAssistantText(t *testing.T) {
	out := toAnthropicMessages([]llm.Message{
		{Role: llm.RoleAssistant, Text: "here's the answer"},
	})
	m := marshalOne(t, out[0])
	if m["role"] != "assistant" {
		t.Errorf("role = %v, want assistant", m["role"])
	}
}

func TestToAnthropicMessages_ToolUseBecomesAssistantToolUseBlock(t *testing.T) {
	out := toAnthropicMessages([]llm.Message{
		{ToolUse: &llm.ToolUse{ID: "tu_1", Name: "get_payroll_period_status", Input: json.RawMessage(`{"period_uuid":"abc"}`)}},
	})
	m := marshalOne(t, out[0])
	if m["role"] != "assistant" {
		t.Errorf("role = %v, want assistant", m["role"])
	}
	content, ok := m["content"].([]any)
	if !ok || len(content) != 1 {
		t.Fatalf("content = %v, want a single block", m["content"])
	}
	block := content[0].(map[string]any)
	if block["type"] != "tool_use" || block["id"] != "tu_1" || block["name"] != "get_payroll_period_status" {
		t.Errorf("tool_use block = %+v", block)
	}
}

// TestToAnthropicMessages_ToolResultIsAlwaysUserRole documents a real API
// constraint, not just this code's choice: the Messages API has no
// distinct "tool" role, so a tool result must be fed back as a user turn.
func TestToAnthropicMessages_ToolResultIsAlwaysUserRole(t *testing.T) {
	out := toAnthropicMessages([]llm.Message{
		{ToolResult: &llm.ToolResult{ToolUseID: "tu_1", Content: `{"status":"draft"}`, IsError: false}},
	})
	m := marshalOne(t, out[0])
	if m["role"] != "user" {
		t.Errorf("role = %v, want user (Anthropic has no tool role)", m["role"])
	}
	block := m["content"].([]any)[0].(map[string]any)
	if block["type"] != "tool_result" || block["tool_use_id"] != "tu_1" {
		t.Errorf("tool_result block = %+v", block)
	}
}

func TestToAnthropicMessages_ToolResultErrorFlagPropagates(t *testing.T) {
	out := toAnthropicMessages([]llm.Message{
		{ToolResult: &llm.ToolResult{ToolUseID: "tu_1", Content: "boom", IsError: true}},
	})
	m := marshalOne(t, out[0])
	block := m["content"].([]any)[0].(map[string]any)
	if isErr, _ := block["is_error"].(bool); !isErr {
		t.Errorf("is_error = %v, want true", block["is_error"])
	}
}

func TestToAnthropicTools_MapsNameDescriptionAndSchema(t *testing.T) {
	specs := []llm.ToolSpec{
		{
			Name:        "get_payroll_period_status",
			Description: "look up a period",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"period_uuid":{"type":"string"}},"required":["period_uuid"]}`),
		},
	}
	out := toAnthropicTools(specs)
	if len(out) != 1 {
		t.Fatalf("got %d tools, want 1", len(out))
	}
	m := marshalOne(t, out[0])
	if m["name"] != "get_payroll_period_status" {
		t.Errorf("name = %v", m["name"])
	}
	if m["description"] != "look up a period" {
		t.Errorf("description = %v", m["description"])
	}
	schema, ok := m["input_schema"].(map[string]any)
	if !ok {
		t.Fatalf("input_schema missing or wrong shape: %v", m["input_schema"])
	}
	required, _ := schema["required"].([]any)
	if len(required) != 1 || required[0] != "period_uuid" {
		t.Errorf("input_schema.required = %v, want [period_uuid]", schema["required"])
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
