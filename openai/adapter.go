// Package openai adapts the Chat Completions API to the vendor-neutral
// llm.Provider interface. Deliberately targets Chat Completions, not
// the newer Responses API — Chat Completions' tool-calling shape and
// streaming semantics map onto llm.Provider with less translation,
// and it remains fully supported.
package openai

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"

	sdk "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared"

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

	stream := a.client.Chat.Completions.NewStreaming(ctx, params)

	out := make(chan llm.Event)
	go func() {
		defer close(out)
		defer stream.Close()

		acc := sdk.ChatCompletionAccumulator{}
		var usage *llm.Usage
		for stream.Next() {
			chunk := stream.Current()
			acc.AddChunk(chunk)

			// With include_usage, the last chunk carries the totals.
			if chunk.Usage.TotalTokens > 0 {
				cached := int(chunk.Usage.PromptTokensDetails.CachedTokens)
				usage = &llm.Usage{
					Input:     int(chunk.Usage.PromptTokens) - cached,
					Output:    int(chunk.Usage.CompletionTokens),
					CacheRead: cached,
				}
			}

			if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
				out <- llm.Event{Kind: llm.EventDelta, Text: chunk.Choices[0].Delta.Content}
			}
		}

		if err := stream.Err(); err != nil {
			out <- llm.Event{Kind: llm.EventError, Err: normalizeError(err)}
			return
		}

		if len(acc.Choices) > 0 {
			for _, tc := range acc.Choices[0].Message.ToolCalls {
				out <- llm.Event{
					Kind: llm.EventToolCall,
					ToolUse: &llm.ToolUse{
						ID:    tc.ID,
						Name:  tc.Function.Name,
						Input: json.RawMessage(tc.Function.Arguments),
					},
				}
			}
		}

		out <- llm.Event{Kind: llm.EventDone, Usage: usage}
	}()

	return out, nil
}

// toOpenAIParams maps one ChatRequest onto the Chat Completions request.
func toOpenAIParams(req llm.ChatRequest) sdk.ChatCompletionNewParams {
	params := sdk.ChatCompletionNewParams{
		Model:    sdk.ChatModel(req.Model),
		Messages: toOpenAIMessages(req.System, req.Messages),
		Tools:    toOpenAITools(req.Tools),
		// Without this the stream never reports token usage.
		StreamOptions: sdk.ChatCompletionStreamOptionsParam{IncludeUsage: sdk.Bool(true)},
	}
	if req.ToolChoice != "" {
		params.ToolChoice = sdk.ChatCompletionToolChoiceOptionParamOfChatCompletionNamedToolChoice(
			sdk.ChatCompletionNamedToolChoiceFunctionParam{Name: req.ToolChoice},
		)
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

// toOpenAIMessages injects the system prompt as the first message — Chat
// Completions has a dedicated system role, unlike Anthropic's separate
// top-level System param.
func toOpenAIMessages(system string, msgs []llm.Message) []sdk.ChatCompletionMessageParamUnion {
	out := make([]sdk.ChatCompletionMessageParamUnion, 0, len(msgs)+1)
	if system != "" {
		out = append(out, sdk.SystemMessage(system))
	}

	for _, m := range msgs {
		switch {
		case m.ToolResult != nil:
			// OpenAI has a distinct "tool" role, unlike Anthropic which
			// feeds tool results back as a user turn.
			out = append(out, sdk.ToolMessage(m.ToolResult.Content, m.ToolResult.ToolUseID))
		case m.ToolUse != nil:
			out = append(out, sdk.ChatCompletionMessageParamUnion{
				OfAssistant: &sdk.ChatCompletionAssistantMessageParam{
					ToolCalls: []sdk.ChatCompletionMessageToolCallParam{
						{
							ID: m.ToolUse.ID,
							Function: sdk.ChatCompletionMessageToolCallFunctionParam{
								Name:      m.ToolUse.Name,
								Arguments: string(m.ToolUse.Input),
							},
						},
					},
				},
			})
		case m.Role == llm.RoleAssistant:
			out = append(out, sdk.AssistantMessage(m.Text))
		default:
			out = append(out, sdk.UserMessage(m.Text))
		}
	}
	return out
}

func toOpenAITools(specs []llm.ToolSpec) []sdk.ChatCompletionToolParam {
	out := make([]sdk.ChatCompletionToolParam, 0, len(specs))
	for _, spec := range specs {
		var schema shared.FunctionParameters
		// Malformed schemas are a programmer error in the caller's tool list, not a
		// runtime condition to recover from — an empty parameter list is
		// an acceptable degraded fallback rather than crashing the request.
		_ = json.Unmarshal(spec.InputSchema, &schema)

		out = append(out, sdk.ChatCompletionToolParam{
			Function: shared.FunctionDefinitionParam{
				Name:        spec.Name,
				Description: sdk.String(spec.Description),
				Parameters:  schema,
			},
		})
	}
	return out
}

func normalizeError(err error) *llm.Error {
	var apiErr *sdk.Error
	if errors.As(err, &apiErr) {
		switch apiErr.StatusCode {
		case http.StatusUnauthorized, http.StatusForbidden:
			return llm.NewError(llm.KindAuth, "OpenAI rejected the configured API key", err)
		case http.StatusTooManyRequests:
			return llm.NewError(llm.KindRateLimit, "OpenAI is rate-limiting this request", err)
		case http.StatusBadRequest, http.StatusNotFound:
			return llm.NewError(llm.KindInvalidReq, "OpenAI rejected the request (check model name)", err)
		}
		return llm.NewError(llm.KindUnavailable, "OpenAI API error", err)
	}
	return llm.NewError(llm.KindUnknown, err.Error(), err)
}
