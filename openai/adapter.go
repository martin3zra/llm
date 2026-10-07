// Package openai adapts the Responses API to the vendor-neutral
// llm.Provider interface. It targets Responses rather than Chat Completions
// because newer reasoning models (gpt-6-luna, for one) refuse function
// tools on Chat Completions unless reasoning is switched off.
package openai

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"

	sdk "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/responses"

	"github.com/martin3zra/llm"
)

type Adapter struct {
	client sdk.Client
}

// New builds an adapter scoped to a single caller's API key. Adapters are
// cheap — one is created per chat request from the caller's own key,
// never shared or cached across users.
//
// Unlike the Anthropic SDK, openai-go has no "without environment
// defaults" option — NewClient always reads OPENAI_API_KEY/OPENAI_BASE_URL
// first. option.WithAPIKey below is applied after those environment
// defaults and overrides them for this client instance, so a box-wide
// OPENAI_API_KEY (there shouldn't be one, but if there ever is) can't leak
// into a per-user request.
//
// opts are applied after the key (tests use them for option.WithBaseURL).
func New(apiKey string, opts ...option.RequestOption) *Adapter {
	opts = append([]option.RequestOption{option.WithAPIKey(apiKey)}, opts...)
	return &Adapter{client: sdk.NewClient(opts...)}
}

func (a *Adapter) StreamChat(ctx context.Context, req llm.ChatRequest) (<-chan llm.Event, error) {
	params := toOpenAIParams(req)

	stream := a.client.Responses.NewStreaming(ctx, params)

	out := make(chan llm.Event)
	go func() {
		defer close(out)
		defer stream.Close()

		var calls []llm.ToolUse
		var usage *llm.Usage
		for stream.Next() {
			ev := stream.Current()
			switch ev.Type {
			case "response.output_text.delta":
				if d := ev.AsResponseOutputTextDelta().Delta; d != "" {
					out <- llm.Event{Kind: llm.EventDelta, Text: d}
				}
			case "response.output_item.done":
				if item := ev.AsResponseOutputItemDone().Item; item.Type == "function_call" {
					calls = append(calls, llm.ToolUse{
						ID:    item.CallID,
						Name:  item.Name,
						Input: json.RawMessage(item.Arguments),
					})
				}
			case "response.completed", "response.incomplete":
				// An incomplete response (output token cap, content filter)
				// still ends the turn normally, as a truncated answer.
				usage = toUsage(ev.Response.Usage)
			case "response.failed":
				e := ev.AsResponseFailed().Response.Error
				out <- llm.Event{Kind: llm.EventError, Err: streamError(string(e.Code), e.Message)}
				return
			case "error":
				e := ev.AsError()
				out <- llm.Event{Kind: llm.EventError, Err: streamError(e.Code, e.Message)}
				return
			}
		}

		if err := stream.Err(); err != nil {
			out <- llm.Event{Kind: llm.EventError, Err: normalizeError(err)}
			return
		}

		for i := range calls {
			out <- llm.Event{Kind: llm.EventToolCall, ToolUse: &calls[i]}
		}
		out <- llm.Event{Kind: llm.EventDone, Usage: usage}
	}()

	return out, nil
}

func toUsage(u responses.ResponseUsage) *llm.Usage {
	if u.TotalTokens == 0 {
		return nil
	}
	cached := int(u.InputTokensDetails.CachedTokens)
	return &llm.Usage{
		Input:     int(u.InputTokens) - cached,
		Output:    int(u.OutputTokens),
		CacheRead: cached,
	}
}

// toOpenAIParams maps one ChatRequest onto the Responses request.
func toOpenAIParams(req llm.ChatRequest) responses.ResponseNewParams {
	params := responses.ResponseNewParams{
		Model: req.Model,
		Input: responses.ResponseNewParamsInputUnion{OfInputItemList: toOpenAIInput(req.Messages)},
		Tools: toOpenAITools(req.Tools),
		// Chat Completions kept nothing by default; Responses stores every
		// response on OpenAI's side unless told not to. The caller keeps
		// the history, so there's nothing to gain from storing it.
		Store: sdk.Bool(false),
	}
	if req.System != "" {
		params.Instructions = sdk.String(req.System)
	}
	if req.ToolChoice != "" {
		params.ToolChoice = responses.ResponseNewParamsToolChoiceUnion{
			OfFunctionTool: &responses.ToolChoiceFunctionParam{Name: req.ToolChoice},
		}
	}
	return params
}

// Verify looks up model: free, no tokens. It fails with 401 for a bad or
// revoked key, and 404 for a model the key can't use.
func (a *Adapter) Verify(ctx context.Context, model string) error {
	if _, err := a.client.Models.Get(ctx, model); err != nil {
		return normalizeError(err)
	}
	return nil
}

// toOpenAIInput maps history onto Responses input items. The system prompt
// isn't one of them: Responses takes it as top-level instructions.
func toOpenAIInput(msgs []llm.Message) responses.ResponseInputParam {
	out := make(responses.ResponseInputParam, 0, len(msgs))
	for _, m := range msgs {
		switch {
		case m.ToolResult != nil:
			// Results are their own item type, matched to the call by
			// call_id, unlike Anthropic which feeds them back as a user turn.
			out = append(out, responses.ResponseInputItemParamOfFunctionCallOutput(m.ToolResult.ToolUseID, m.ToolResult.Content))
		case m.ToolUse != nil:
			out = append(out, responses.ResponseInputItemParamOfFunctionCall(string(m.ToolUse.Input), m.ToolUse.ID, m.ToolUse.Name))
		case m.Role == llm.RoleAssistant:
			out = append(out, responses.ResponseInputItemParamOfMessage(m.Text, responses.EasyInputMessageRoleAssistant))
		default:
			out = append(out, responses.ResponseInputItemParamOfMessage(m.Text, responses.EasyInputMessageRoleUser))
		}
	}
	return out
}

func toOpenAITools(specs []llm.ToolSpec) []responses.ToolUnionParam {
	out := make([]responses.ToolUnionParam, 0, len(specs))
	for _, spec := range specs {
		var schema map[string]any
		// Malformed schemas are a programmer error in the caller's tool list, not a
		// runtime condition to recover from — an empty parameter list is
		// an acceptable degraded fallback rather than crashing the request.
		if json.Unmarshal(spec.InputSchema, &schema) != nil || schema == nil {
			schema = map[string]any{"type": "object", "properties": map[string]any{}}
		}

		fn := &responses.FunctionToolParam{
			Name:       spec.Name,
			Parameters: schema,
			// Strict mode wants every property required and
			// additionalProperties false; callers' schemas aren't written
			// that way.
			Strict: sdk.Bool(false),
		}
		if spec.Description != "" {
			fn.Description = sdk.String(spec.Description)
		}
		out = append(out, responses.ToolUnionParam{OfFunction: fn})
	}
	return out
}

// normalizeError sorts an HTTP failure by status. For a rejected request it
// keeps OpenAI's own reason: it names the parameter at fault and never
// contains the key.
func normalizeError(err error) *llm.Error {
	var apiErr *sdk.Error
	if errors.As(err, &apiErr) {
		switch apiErr.StatusCode {
		case http.StatusUnauthorized, http.StatusForbidden:
			return llm.NewError(llm.KindAuth, "OpenAI rejected the configured API key", err)
		case http.StatusTooManyRequests:
			return llm.NewError(llm.KindRateLimit, "OpenAI is rate-limiting this request", err)
		case http.StatusBadRequest, http.StatusNotFound:
			msg := "OpenAI rejected the request"
			if apiErr.Message != "" {
				msg += ": " + apiErr.Message
			}
			return llm.NewError(llm.KindInvalidReq, msg, err)
		}
		return llm.NewError(llm.KindUnavailable, "OpenAI API error", err)
	}
	return llm.NewError(llm.KindUnknown, err.Error(), err)
}

// streamError sorts a failure reported inside the stream, after the HTTP
// request itself succeeded.
func streamError(code, message string) *llm.Error {
	if message == "" {
		message = "OpenAI failed to finish the response"
	}
	cause := errors.New(code + ": " + message)
	if code == "rate_limit_exceeded" {
		return llm.NewError(llm.KindRateLimit, "OpenAI is rate-limiting this request", cause)
	}
	return llm.NewError(llm.KindUnavailable, message, cause)
}
