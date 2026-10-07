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

func TestToOpenAIParams_SystemPromptBecomesInstructions(t *testing.T) {
	m := marshalOne(t, toOpenAIParams(llm.ChatRequest{
		Model:    "gpt-x",
		System:   "be helpful",
		Messages: []llm.Message{{Role: llm.RoleUser, Text: "hi"}},
	}))
	if m["instructions"] != "be helpful" {
		t.Errorf("instructions = %v, want the system prompt", m["instructions"])
	}
	if input := m["input"].([]any); len(input) != 1 {
		t.Errorf("got %d input items, want 1 (the system prompt isn't one)", len(input))
	}
}

func TestToOpenAIParams_OmitsInstructionsWhenEmpty(t *testing.T) {
	m := marshalOne(t, toOpenAIParams(llm.ChatRequest{Model: "gpt-x", Messages: []llm.Message{{Role: llm.RoleUser, Text: "hi"}}}))
	if _, ok := m["instructions"]; ok {
		t.Errorf("instructions = %v, want it left out", m["instructions"])
	}
}

func TestToOpenAIParams_DoesNotStoreResponses(t *testing.T) {
	m := marshalOne(t, toOpenAIParams(llm.ChatRequest{Model: "gpt-x"}))
	if m["store"] != false {
		t.Errorf("store = %v, want false", m["store"])
	}
}

func TestToOpenAIInput_PlainUserAndAssistantText(t *testing.T) {
	out := toOpenAIInput([]llm.Message{
		{Role: llm.RoleUser, Text: "what's the status?"},
		{Role: llm.RoleAssistant, Text: "let me check"},
	})
	if m := marshalOne(t, out[0]); m["role"] != "user" || m["content"] != "what's the status?" {
		t.Errorf("first item = %+v, want the user's text", m)
	}
	if m := marshalOne(t, out[1]); m["role"] != "assistant" || m["content"] != "let me check" {
		t.Errorf("second item = %+v, want the assistant's text", m)
	}
}

func TestToOpenAIInput_ToolUseBecomesFunctionCall(t *testing.T) {
	out := toOpenAIInput([]llm.Message{
		{ToolUse: &llm.ToolUse{ID: "call_1", Name: "get_payroll_period_status", Input: json.RawMessage(`{"period_uuid":"abc"}`)}},
	})
	m := marshalOne(t, out[0])
	if m["type"] != "function_call" {
		t.Errorf("type = %v, want function_call", m["type"])
	}
	if m["call_id"] != "call_1" || m["name"] != "get_payroll_period_status" {
		t.Errorf("call_id, name = %v, %v", m["call_id"], m["name"])
	}
	if m["arguments"] != `{"period_uuid":"abc"}` {
		t.Errorf("arguments = %v", m["arguments"])
	}
	// No item id: with store off there's no stored item for one to name.
	if _, ok := m["id"]; ok {
		t.Errorf("id = %v, want it left out", m["id"])
	}
}

// TestToOpenAIInput_ToolResultIsFunctionCallOutput documents a real API
// difference from Anthropic: the result is its own item type, so unlike the
// Anthropic adapter it does NOT get folded into a user turn.
func TestToOpenAIInput_ToolResultIsFunctionCallOutput(t *testing.T) {
	out := toOpenAIInput([]llm.Message{
		{ToolResult: &llm.ToolResult{ToolUseID: "call_1", Content: `{"status":"draft"}`}},
	})
	m := marshalOne(t, out[0])
	if m["type"] != "function_call_output" {
		t.Errorf("type = %v, want function_call_output", m["type"])
	}
	if m["call_id"] != "call_1" {
		t.Errorf("call_id = %v, want call_1", m["call_id"])
	}
	if m["output"] != `{"status":"draft"}` {
		t.Errorf("output = %v", m["output"])
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
	if m["name"] != "get_payroll_period_status" {
		t.Errorf("name = %v", m["name"])
	}
	if m["description"] != "look up a period" {
		t.Errorf("description = %v", m["description"])
	}
	if m["strict"] != false {
		t.Errorf("strict = %v, want false", m["strict"])
	}
	params := m["parameters"].(map[string]any)
	required, _ := params["required"].([]any)
	if len(required) != 1 || required[0] != "period_uuid" {
		t.Errorf("parameters.required = %v, want [period_uuid]", params["required"])
	}
}

func TestToOpenAITools_MalformedSchemaFallsBackToNoParameters(t *testing.T) {
	m := marshalOne(t, toOpenAITools([]llm.ToolSpec{{Name: "ping", InputSchema: json.RawMessage(`not json`)}})[0])
	params, ok := m["parameters"].(map[string]any)
	if !ok || params["type"] != "object" {
		t.Errorf("parameters = %v, want an empty object schema", m["parameters"])
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

func TestNormalizeError_RejectedRequestKeepsOpenAIsReason(t *testing.T) {
	got := normalizeError(&sdk.Error{StatusCode: http.StatusBadRequest, Message: "Unsupported parameter: 'temperature'."})
	if want := "OpenAI rejected the request: Unsupported parameter: 'temperature'."; got.Message != want {
		t.Errorf("message = %q, want %q", got.Message, want)
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
