package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/openai/openai-go/v3/option"

	"github.com/martin3zra/llm"
)

// streamFrom runs StreamChat against a local server that replies with the
// given Responses stream events, and collects what comes out.
func streamFrom(t *testing.T, events ...string) []llm.Event {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/responses" {
			t.Errorf("path = %s, want /responses", r.URL.Path)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for _, e := range events {
			if e == keepAlive {
				fmt.Fprint(w, ": keep-alive\n\n")
				continue
			}
			var head struct{ Type string }
			_ = json.Unmarshal([]byte(e), &head)
			fmt.Fprintf(w, "event: %s\ndata: %s\n\n", head.Type, e)
		}
	}))
	defer srv.Close()

	a := New("sk-test", option.WithBaseURL(srv.URL), option.WithMaxRetries(0))
	ch, err := a.StreamChat(context.Background(), llm.ChatRequest{
		Model:    "gpt-6-luna",
		Messages: []llm.Message{{Role: llm.RoleUser, Text: "hi"}},
	})
	if err != nil {
		t.Fatalf("StreamChat: %v", err)
	}
	var got []llm.Event
	for ev := range ch {
		got = append(got, ev)
	}
	return got
}

// keepAlive stands for the SSE comment OpenAI sends while the model thinks.
const keepAlive = "keep-alive"

// A keep-alive comment is not an event. openai-go v1 handed it on as one
// with no data and failed with "unexpected end of JSON input", which
// dropped the whole response.
func TestStreamChat_KeepAliveIsSkipped(t *testing.T) {
	got := streamFrom(t,
		keepAlive,
		`{"type":"response.output_item.done","output_index":0,"sequence_number":1,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"record_notes","arguments":"{}","status":"completed"}}`,
		keepAlive,
		`{"type":"response.completed","sequence_number":2,"response":{"id":"resp_1","status":"completed","usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}`,
	)
	var call bool
	for _, ev := range got {
		switch ev.Kind {
		case llm.EventError:
			t.Fatalf("unexpected error: %v", ev.Err)
		case llm.EventToolCall:
			call = true
		}
	}
	if !call {
		t.Error("no tool call")
	}
}

func TestStreamChat_TextToolCallAndUsage(t *testing.T) {
	got := streamFrom(t,
		`{"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"Hel","sequence_number":1}`,
		`{"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"lo","sequence_number":2}`,
		`{"type":"response.output_item.done","output_index":1,"sequence_number":3,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"record_notes","arguments":"{\"tables\":[]}","status":"completed"}}`,
		`{"type":"response.completed","sequence_number":4,"response":{"id":"resp_1","status":"completed","usage":{"input_tokens":120,"input_tokens_details":{"cached_tokens":20},"output_tokens":30,"output_tokens_details":{"reasoning_tokens":10},"total_tokens":150}}}`,
	)

	var text strings.Builder
	var call *llm.ToolUse
	var done *llm.Event
	for i, ev := range got {
		switch ev.Kind {
		case llm.EventDelta:
			text.WriteString(ev.Text)
		case llm.EventToolCall:
			call = ev.ToolUse
		case llm.EventDone:
			done = &got[i]
		case llm.EventError:
			t.Fatalf("unexpected error: %v", ev.Err)
		}
	}
	if text.String() != "Hello" {
		t.Errorf("text = %q, want Hello", text.String())
	}
	if call == nil || call.ID != "call_1" || call.Name != "record_notes" || string(call.Input) != `{"tables":[]}` {
		t.Errorf("tool call = %+v, want call_1 record_notes with its arguments", call)
	}
	if done == nil || done.Usage == nil {
		t.Fatal("no done event with usage")
	}
	if want := (llm.Usage{Input: 100, Output: 30, CacheRead: 20}); *done.Usage != want {
		t.Errorf("usage = %+v, want %+v", *done.Usage, want)
	}
	if last := got[len(got)-1]; last.Kind != llm.EventDone {
		t.Errorf("last event = %s, want done", last.Kind)
	}
}

func TestStreamChat_FailedResponseIsAnError(t *testing.T) {
	got := streamFrom(t,
		`{"type":"response.failed","sequence_number":1,"response":{"id":"resp_1","status":"failed","error":{"code":"rate_limit_exceeded","message":"slow down"}}}`,
	)
	last := got[len(got)-1]
	var perr *llm.Error
	if last.Kind != llm.EventError || !errors.As(last.Err, &perr) || perr.Kind != llm.KindRateLimit {
		t.Fatalf("last event = %+v, want an error of KindRateLimit", last)
	}
}

func TestStreamChat_ErrorEventIsAnError(t *testing.T) {
	got := streamFrom(t, `{"type":"error","sequence_number":1,"code":"server_error","message":"boom","param":null}`)
	last := got[len(got)-1]
	var perr *llm.Error
	if last.Kind != llm.EventError || !errors.As(last.Err, &perr) || perr.Message != "boom" {
		t.Fatalf("last event = %+v, want an error saying boom", last)
	}
}

func TestStreamChat_InsufficientQuotaIsNoCredit(t *testing.T) {
	got := streamFrom(t,
		`{"type":"response.failed","sequence_number":1,"response":{"id":"resp_1","status":"failed","error":{"code":"insufficient_quota","message":"You exceeded your current quota"}}}`,
	)
	last := got[len(got)-1]
	var perr *llm.Error
	if last.Kind != llm.EventError || !errors.As(last.Err, &perr) || perr.Kind != llm.KindNoCredit {
		t.Fatalf("last event = %+v, want an error of KindNoCredit", last)
	}
}
