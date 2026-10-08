package anthropic

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/anthropics/anthropic-sdk-go/option"

	"github.com/martin3zra/llm"
)

func TestToAnthropicParams_ToolChoiceForcesNamedTool(t *testing.T) {
	m := marshalOne(t, toAnthropicParams(llm.ChatRequest{
		Model:      "claude-haiku-4-5",
		Messages:   []llm.Message{{Role: llm.RoleUser, Text: "800 barbería"}},
		Tools:      []llm.ToolSpec{{Name: "record_expense", InputSchema: []byte(`{"type":"object"}`)}},
		ToolChoice: "record_expense",
	}))
	choice, ok := m["tool_choice"].(map[string]any)
	if !ok {
		t.Fatalf("tool_choice missing: %v", m)
	}
	if choice["type"] != "tool" || choice["name"] != "record_expense" {
		t.Errorf("tool_choice = %v, want {type: tool, name: record_expense}", choice)
	}
}

func TestToAnthropicParams_NoToolChoiceByDefault(t *testing.T) {
	m := marshalOne(t, toAnthropicParams(llm.ChatRequest{
		Model:    "claude-haiku-4-5",
		Messages: []llm.Message{{Role: llm.RoleUser, Text: "hi"}},
	}))
	if _, ok := m["tool_choice"]; ok {
		t.Errorf("tool_choice = %v, want it left out", m["tool_choice"])
	}
}

func verifyAgainst(t *testing.T, status int, body string) error {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models/claude-haiku-4-5" {
			t.Errorf("path = %s, want /v1/models/claude-haiku-4-5", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	defer srv.Close()
	a := New("sk-ant-test", option.WithBaseURL(srv.URL), option.WithMaxRetries(0))
	return a.Verify(context.Background(), "claude-haiku-4-5")
}

func TestVerify_AcceptedKey(t *testing.T) {
	if err := verifyAgainst(t, http.StatusOK, `{"type":"model","id":"claude-haiku-4-5","display_name":"Claude Haiku 4.5","created_at":"2025-10-01T00:00:00Z"}`); err != nil {
		t.Fatalf("Verify = %v, want nil", err)
	}
}

func TestVerify_RejectedKeyIsKindAuth(t *testing.T) {
	err := verifyAgainst(t, http.StatusUnauthorized, `{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}`)
	var perr *llm.Error
	if !errors.As(err, &perr) || perr.Kind != llm.KindAuth {
		t.Fatalf("Verify = %v, want an *llm.Error of KindAuth", err)
	}
}

var _ llm.Verifier = (*Adapter)(nil)

func TestVerify_UnknownModelIsKindInvalidReq(t *testing.T) {
	err := verifyAgainst(t, http.StatusNotFound, `{"type":"error","error":{"type":"not_found_error","message":"model: claude-haiku-4-5"}}`)
	var perr *llm.Error
	if !errors.As(err, &perr) || perr.Kind != llm.KindInvalidReq {
		t.Fatalf("Verify = %v, want an *llm.Error of KindInvalidReq", err)
	}
}

func TestToAnthropicParams_CacheSystemMarksTheSystemBlock(t *testing.T) {
	for _, cache := range []bool{false, true} {
		m := marshalOne(t, toAnthropicParams(llm.ChatRequest{
			Model:       "claude-haiku-4-5",
			System:      "a long, stable prompt",
			Messages:    []llm.Message{{Role: llm.RoleUser, Text: "hi"}},
			CacheSystem: cache,
		}))
		system := m["system"].([]any)[0].(map[string]any)
		cc, ok := system["cache_control"].(map[string]any)
		if ok != cache || cache && cc["type"] != "ephemeral" {
			t.Errorf("CacheSystem %v: system block = %v", cache, system)
		}
	}
}
