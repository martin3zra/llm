// Package agent runs one user turn of a tool-calling conversation over any
// llm.Provider: stream the model's answer, run the tools it asks for, feed
// the results back, and repeat until it answers without calling a tool.
//
// The caller owns storage. Run takes the conversation so far and reports the
// messages the turn added in its done event, so apps with different
// conversation stores can share the loop.
package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/martin3zra/llm"
)

// Tool is one callable tool.
type Tool interface {
	Spec() llm.ToolSpec
	Call(ctx context.Context, input json.RawMessage) (Result, error)
}

// Result is a tool's outcome.
type Result struct {
	Content string // what the model sees
	IsError bool
	// Payload is optional out-of-band data for the caller (a result set id, a
	// chart). It's passed through on the tool_end event and never sent to the
	// model.
	Payload any
}

type Limits struct {
	MaxIter     int           // model round-trips per turn
	ToolTimeout time.Duration // per tool call
}

var DefaultLimits = Limits{MaxIter: 8, ToolTimeout: 10 * time.Second}

type Request struct {
	System string
	// History is the conversation so far, ending with the new user message.
	History []llm.Message
	Model   string
	Tools   []Tool
	Limits  Limits // zero fields fall back to DefaultLimits
}

type EventKind string

const (
	EventDelta     EventKind = "delta"      // Text
	EventToolStart EventKind = "tool_start" // ToolUse
	EventToolEnd   EventKind = "tool_end"   // ToolUse, Result
	EventDone      EventKind = "done"       // Messages, Usage
	EventError     EventKind = "error"      // Err
)

type Event struct {
	Kind    EventKind
	Text    string
	ToolUse *llm.ToolUse
	Result  *Result
	// Messages are the ones this turn appended after Request.History: the
	// assistant's text, its tool uses and the tool results, in order.
	Messages []llm.Message
	Usage    llm.Usage // summed over every model round-trip in the turn
	Err      error
}

// ErrTooManySteps means the model kept calling tools past Limits.MaxIter.
var ErrTooManySteps = errors.New("agent: too many tool-calling steps")

// Run executes one turn. The channel ends with exactly one done or error
// event and is then closed. Cancelling ctx stops the turn.
func Run(ctx context.Context, p llm.Provider, req Request) <-chan Event {
	out := make(chan Event)
	go func() {
		defer close(out)
		send := func(e Event) bool {
			select {
			case out <- e:
				return true
			case <-ctx.Done():
				return false
			}
		}

		lim := req.Limits
		if lim.MaxIter <= 0 {
			lim.MaxIter = DefaultLimits.MaxIter
		}
		if lim.ToolTimeout <= 0 {
			lim.ToolTimeout = DefaultLimits.ToolTimeout
		}

		specs := make([]llm.ToolSpec, len(req.Tools))
		byName := make(map[string]Tool, len(req.Tools))
		for i, t := range req.Tools {
			specs[i] = t.Spec()
			byName[specs[i].Name] = t
		}

		history := append([]llm.Message(nil), req.History...)
		start := len(history)
		var usage llm.Usage

		for range lim.MaxIter {
			events, err := p.StreamChat(ctx, llm.ChatRequest{
				System: req.System, Messages: history, Tools: specs, Model: req.Model,
			})
			if err != nil {
				send(Event{Kind: EventError, Err: err})
				return
			}

			var text string
			var calls []*llm.ToolUse
			for ev := range events {
				switch ev.Kind {
				case llm.EventDelta:
					text += ev.Text
					if !send(Event{Kind: EventDelta, Text: ev.Text}) {
						return
					}
				case llm.EventToolCall:
					calls = append(calls, ev.ToolUse)
				case llm.EventDone:
					if ev.Usage != nil {
						usage = usage.Add(*ev.Usage)
					}
				case llm.EventError:
					send(Event{Kind: EventError, Err: ev.Err})
					return
				}
			}
			if ctx.Err() != nil {
				return
			}

			if text != "" {
				history = append(history, llm.Message{Role: llm.RoleAssistant, Text: text})
			}
			if len(calls) == 0 {
				send(Event{Kind: EventDone, Messages: history[start:], Usage: usage})
				return
			}

			for _, call := range calls {
				history = append(history, llm.Message{Role: llm.RoleAssistant, ToolUse: call})
				if !send(Event{Kind: EventToolStart, ToolUse: call}) {
					return
				}
				res := callTool(ctx, byName[call.Name], call, lim.ToolTimeout)
				history = append(history, llm.Message{Role: llm.RoleUser, ToolResult: &llm.ToolResult{
					ToolUseID: call.ID, Content: res.Content, IsError: res.IsError,
				}})
				if !send(Event{Kind: EventToolEnd, ToolUse: call, Result: &res}) {
					return
				}
			}
		}
		send(Event{Kind: EventError, Err: ErrTooManySteps})
	}()
	return out
}

func callTool(ctx context.Context, t Tool, call *llm.ToolUse, timeout time.Duration) Result {
	if t == nil {
		return Result{Content: fmt.Sprintf("error: unknown tool %q", call.Name), IsError: true}
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	res, err := t.Call(ctx, call.Input)
	if err != nil {
		return Result{Content: "error: " + err.Error(), IsError: true}
	}
	return res
}
