package openai

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/openai/openai-go/option"

	"github.com/martin3zra/llm"
)

func TestToOpenAIParams_ToolChoiceForcesNamedFunction(t *testing.T) {
	m := marshalOne(t, toOpenAIParams(llm.ChatRequest{
		Model:      "gpt-5-mini",
		Messages:   []llm.Message{{Role: llm.RoleUser, Text: "800 barbería"}},
		Tools:      []llm.ToolSpec{{Name: "record_expense", InputSchema: []byte(`{"type":"object"}`)}},
		ToolChoice: "record_expense",
	}))
	choice, ok := m["tool_choice"].(map[string]any)
	if !ok {
		t.Fatalf("tool_choice missing: %v", m)
	}
	fn, _ := choice["function"].(map[string]any)
	if choice["type"] != "function" || fn["name"] != "record_expense" {
		t.Errorf("tool_choice = %v, want {type: function, function: {name: record_expense}}", choice)
	}
}

func TestToOpenAIParams_NoToolChoiceByDefault(t *testing.T) {
	m := marshalOne(t, toOpenAIParams(llm.ChatRequest{
		Model:    "gpt-5-mini",
		Messages: []llm.Message{{Role: llm.RoleUser, Text: "hi"}},
	}))
	if _, ok := m["tool_choice"]; ok {
		t.Errorf("tool_choice = %v, want it left out", m["tool_choice"])
	}
}

func verifyAgainst(t *testing.T, status int, body string) error {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/models/gpt-5-mini" {
			t.Errorf("path = %s, want /models/gpt-5-mini", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	defer srv.Close()
	a := New("sk-test", option.WithBaseURL(srv.URL), option.WithMaxRetries(0))
	return a.Verify(context.Background(), "gpt-5-mini")
}

func TestVerify_AcceptedKey(t *testing.T) {
	if err := verifyAgainst(t, http.StatusOK, `{"id":"gpt-5-mini","object":"model","created":1754000000,"owned_by":"openai"}`); err != nil {
		t.Fatalf("Verify = %v, want nil", err)
	}
}

func TestVerify_RejectedKeyIsKindAuth(t *testing.T) {
	err := verifyAgainst(t, http.StatusUnauthorized, `{"error":{"message":"Incorrect API key provided","type":"invalid_request_error","code":"invalid_api_key"}}`)
	var perr *llm.Error
	if !errors.As(err, &perr) || perr.Kind != llm.KindAuth {
		t.Fatalf("Verify = %v, want an *llm.Error of KindAuth", err)
	}
}

var _ llm.Verifier = (*Adapter)(nil)

func TestVerify_UnknownModelIsKindInvalidReq(t *testing.T) {
	err := verifyAgainst(t, http.StatusNotFound, `{"error":{"message":"The model 'gpt-5-mini' does not exist","type":"invalid_request_error","code":"model_not_found"}}`)
	var perr *llm.Error
	if !errors.As(err, &perr) || perr.Kind != llm.KindInvalidReq {
		t.Fatalf("Verify = %v, want an *llm.Error of KindInvalidReq", err)
	}
}
