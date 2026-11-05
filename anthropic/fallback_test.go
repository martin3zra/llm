package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/anthropics/anthropic-sdk-go/option"

	"github.com/martin3zra/llm"
)

const toolUseSSE = `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[],"stop_reason":null,"usage":{"input_tokens":3,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tu_1","name":"record_notes","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"a\":1}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}

event: message_stop
data: {"type":"message_stop"}

`

// fakeNoForcedChoice behaves like a model that rejects tool_choice "tool".
func fakeNoForcedChoice(t *testing.T) (*httptest.Server, func() []map[string]any) {
	var mu sync.Mutex
	var bodies []map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		mu.Lock()
		bodies = append(bodies, body)
		mu.Unlock()
		if choice, _ := body["tool_choice"].(map[string]any); choice["type"] == "tool" {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"type":"error","error":{"type":"invalid_request_error","message":"tool_choice: type \"tool\" and \"any\" are not supported for this model."}}`))
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(toolUseSSE))
	}))
	t.Cleanup(srv.Close)
	return srv, func() []map[string]any { mu.Lock(); defer mu.Unlock(); return bodies }
}

func collect(t *testing.T, a *Adapter, model string) []llm.Event {
	t.Helper()
	ch, err := a.StreamChat(context.Background(), llm.ChatRequest{
		Model:      model,
		System:     "Write notes.",
		Messages:   []llm.Message{{Role: llm.RoleUser, Text: "tables"}},
		Tools:      []llm.ToolSpec{{Name: "record_notes", InputSchema: []byte(`{"type":"object"}`)}},
		ToolChoice: "record_notes",
	})
	if err != nil {
		t.Fatal(err)
	}
	var evs []llm.Event
	for e := range ch {
		evs = append(evs, e)
	}
	return evs
}

func TestStreamChat_FallsBackWhenForcedToolChoiceIsRejected(t *testing.T) {
	srv, bodies := fakeNoForcedChoice(t)
	a := New("sk-ant-test", option.WithBaseURL(srv.URL), option.WithMaxRetries(0))
	model := "claude-test-no-forced-" + t.Name()

	evs := collect(t, a, model)
	var call *llm.ToolUse
	for _, e := range evs {
		if e.Kind == llm.EventError {
			t.Fatalf("got error event: %v", e.Err)
		}
		if e.Kind == llm.EventToolCall {
			call = e.ToolUse
		}
	}
	if call == nil || call.Name != "record_notes" || string(call.Input) != `{"a":1}` {
		t.Fatalf("tool call = %+v, want record_notes {\"a\":1}", call)
	}
	got := bodies()
	if len(got) != 2 {
		t.Fatalf("requests = %d, want 2 (forced, then auto)", len(got))
	}
	retry := got[1]
	if choice := retry["tool_choice"].(map[string]any); choice["type"] != "auto" {
		t.Errorf("retry tool_choice = %v, want auto", choice)
	}
	system := retry["system"].([]any)
	if len(system) != 2 || !strings.Contains(system[1].(map[string]any)["text"].(string), "record_notes") {
		t.Errorf("retry system = %v, want an instruction naming record_notes", system)
	}

	// The model is remembered: the next call goes straight to auto.
	collect(t, a, model)
	if n := len(bodies()); n != 3 {
		t.Errorf("requests after second call = %d, want 3", n)
	}
}

func TestNormalizeError_BadRequestCarriesAnthropicMessage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"invalid_request_error","message":"max_tokens: too large"}}`))
	}))
	defer srv.Close()
	a := New("sk-ant-test", option.WithBaseURL(srv.URL), option.WithMaxRetries(0))
	ch, _ := a.StreamChat(context.Background(), llm.ChatRequest{Model: "m", Messages: []llm.Message{{Role: llm.RoleUser, Text: "hi"}}})
	for e := range ch {
		var perr *llm.Error
		if e.Kind == llm.EventError {
			if !errors.As(e.Err, &perr) || perr.Kind != llm.KindInvalidReq || !strings.Contains(perr.Error(), "max_tokens: too large") {
				t.Fatalf("error = %v, want KindInvalidReq carrying Anthropic's message", e.Err)
			}
			return
		}
	}
	t.Fatal("no error event")
}
