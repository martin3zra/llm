package agent

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/martin3zra/llm"
)

// scripted replays one list of events per StreamChat call and records the
// requests it got.
type scripted struct {
	turns [][]llm.Event
	reqs  []llm.ChatRequest
}

func (s *scripted) StreamChat(_ context.Context, req llm.ChatRequest) (<-chan llm.Event, error) {
	s.reqs = append(s.reqs, req)
	var evs []llm.Event
	if len(s.turns) > 0 {
		evs, s.turns = s.turns[0], s.turns[1:]
	}
	ch := make(chan llm.Event, len(evs))
	for _, e := range evs {
		ch <- e
	}
	close(ch)
	return ch, nil
}

type echoTool struct {
	err   error
	delay time.Duration
}

func (echoTool) Spec() llm.ToolSpec {
	return llm.ToolSpec{Name: "echo", InputSchema: json.RawMessage(`{"type":"object"}`)}
}

func (e echoTool) Call(ctx context.Context, input json.RawMessage) (Result, error) {
	if e.delay > 0 {
		select {
		case <-time.After(e.delay):
		case <-ctx.Done():
			return Result{}, ctx.Err()
		}
	}
	if e.err != nil {
		return Result{}, e.err
	}
	return Result{Content: "echo " + string(input), Payload: map[string]int{"rows": 3}}, nil
}

func use(id string) llm.Event {
	return llm.Event{Kind: llm.EventToolCall, ToolUse: &llm.ToolUse{ID: id, Name: "echo", Input: json.RawMessage(`{"q":1}`)}}
}

func done(in, out int) llm.Event {
	return llm.Event{Kind: llm.EventDone, Usage: &llm.Usage{Input: in, Output: out}}
}

func collect(ch <-chan Event) []Event {
	var evs []Event
	for e := range ch {
		evs = append(evs, e)
	}
	return evs
}

func TestRunToolLoop(t *testing.T) {
	p := &scripted{turns: [][]llm.Event{
		{{Kind: llm.EventDelta, Text: "Let me check."}, use("a"), done(100, 10)},
		{{Kind: llm.EventDelta, Text: "There are "}, {Kind: llm.EventDelta, Text: "3."}, done(150, 5)},
	}}
	history := []llm.Message{{Role: llm.RoleUser, Text: "how many?"}}
	evs := collect(Run(context.Background(), p, Request{
		System: "sys", History: history, Model: "m", Tools: []Tool{echoTool{}},
	}))

	var kinds []EventKind
	for _, e := range evs {
		kinds = append(kinds, e.Kind)
	}
	want := []EventKind{EventDelta, EventToolStart, EventToolEnd, EventDelta, EventDelta, EventDone}
	if len(kinds) != len(want) {
		t.Fatalf("events = %v", kinds)
	}
	for i := range want {
		if kinds[i] != want[i] {
			t.Fatalf("events = %v, want %v", kinds, want)
		}
	}

	end := evs[2]
	if end.Result.Content != `echo {"q":1}` || end.Result.Payload == nil {
		t.Errorf("tool_end = %+v", end.Result)
	}

	last := evs[len(evs)-1]
	if last.Usage != (llm.Usage{Input: 250, Output: 15}) {
		t.Errorf("usage = %+v", last.Usage)
	}
	// assistant text, tool use, tool result, final text.
	if len(last.Messages) != 4 || last.Messages[0].Text != "Let me check." || last.Messages[1].ToolUse.ID != "a" ||
		last.Messages[2].ToolResult.ToolUseID != "a" || last.Messages[3].Text != "There are 3." {
		t.Errorf("messages = %+v", last.Messages)
	}
	if len(history) != 1 {
		t.Error("Run modified the caller's history")
	}
	// The second request carried the tool round-trip; the payload never did.
	if got := p.reqs[1].Messages; len(got) != 4 || got[3].ToolResult == nil {
		t.Errorf("second request messages = %+v", got)
	}
	if p.reqs[0].System != "sys" || p.reqs[0].Model != "m" || len(p.reqs[0].Tools) != 1 {
		t.Errorf("first request = %+v", p.reqs[0])
	}
}

func TestRunToolErrorsGoBackToTheModel(t *testing.T) {
	for name, tools := range map[string][]Tool{
		"failing tool": {echoTool{err: errors.New("boom")}},
		"unknown tool": nil,
		"timeout":      {echoTool{delay: time.Second}},
	} {
		p := &scripted{turns: [][]llm.Event{{use("a"), done(1, 1)}, {{Kind: llm.EventDelta, Text: "sorry"}, done(1, 1)}}}
		evs := collect(Run(context.Background(), p, Request{
			History: []llm.Message{{Role: llm.RoleUser, Text: "x"}}, Tools: tools,
			Limits: Limits{ToolTimeout: 20 * time.Millisecond},
		}))
		if evs[len(evs)-1].Kind != EventDone {
			t.Fatalf("%s: last event = %+v", name, evs[len(evs)-1])
		}
		res := p.reqs[1].Messages[2].ToolResult
		if res == nil || !res.IsError {
			t.Errorf("%s: tool result = %+v", name, res)
		}
	}
}

func TestRunStopsAfterMaxIter(t *testing.T) {
	p := &scripted{turns: [][]llm.Event{{use("a"), done(1, 1)}, {use("b"), done(1, 1)}, {use("c"), done(1, 1)}}}
	evs := collect(Run(context.Background(), p, Request{
		History: []llm.Message{{Role: llm.RoleUser, Text: "x"}}, Tools: []Tool{echoTool{}}, Limits: Limits{MaxIter: 2},
	}))
	last := evs[len(evs)-1]
	if last.Kind != EventError || !errors.Is(last.Err, ErrTooManySteps) || len(p.reqs) != 2 {
		t.Fatalf("last = %+v after %d requests", last, len(p.reqs))
	}
}

func TestRunPassesProviderErrors(t *testing.T) {
	boom := llm.NewError(llm.KindRateLimit, "slow down", nil)
	p := &scripted{turns: [][]llm.Event{{{Kind: llm.EventDelta, Text: "par"}, {Kind: llm.EventError, Err: boom}}}}
	evs := collect(Run(context.Background(), p, Request{History: []llm.Message{{Role: llm.RoleUser, Text: "x"}}}))
	last := evs[len(evs)-1]
	if last.Kind != EventError || !errors.Is(last.Err, boom) {
		t.Fatalf("last = %+v", last)
	}
}

func TestRunStopsWhenCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	p := &scripted{turns: [][]llm.Event{{{Kind: llm.EventDelta, Text: "a"}, {Kind: llm.EventDelta, Text: "b"}, done(1, 1)}}}
	ch := Run(ctx, p, Request{History: []llm.Message{{Role: llm.RoleUser, Text: "x"}}})
	<-ch // first delta
	cancel()
	for range ch {
	}
	// The channel closed without the caller draining every event: no leak.
}
