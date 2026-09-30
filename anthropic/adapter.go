// Package anthropic adapts the Claude Messages API to the vendor-neutral
// llm.Provider interface.
package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"

	"github.com/martin3zra/llm"
)

const defaultMaxTokens = 4096

type Adapter struct {
	client sdk.Client
}

// New builds an adapter scoped to a single caller's API key. Adapters are
// cheap — one is created per chat request from the caller's own key,
// never shared or cached across users.
//
// opts are applied after the key (tests use them for option.WithBaseURL).
func New(apiKey string, opts ...option.RequestOption) *Adapter {
	opts = append([]option.RequestOption{
		option.WithAPIKey(apiKey),
		option.WithoutEnvironmentDefaults(), // never fall back to a box-wide ANTHROPIC_API_KEY
	}, opts...)
	return &Adapter{client: sdk.NewClient(opts...)}
}

func (a *Adapter) StreamChat(ctx context.Context, req llm.ChatRequest) (<-chan llm.Event, error) {
	params := toAnthropicParams(req)

	stream := a.client.Messages.NewStreaming(ctx, params)

	out := make(chan llm.Event)
	go func() {
		defer close(out)
		defer stream.Close()

		acc := sdk.Message{}
		for stream.Next() {
			event := stream.Current()
			if err := acc.Accumulate(event); err != nil {
				out <- llm.Event{Kind: llm.EventError, Err: normalizeError(err)}
				return
			}

			if event.Type == "content_block_delta" && event.Delta.Type == "text_delta" {
				out <- llm.Event{Kind: llm.EventDelta, Text: event.Delta.Text}
			}
		}

		if err := stream.Err(); err != nil {
			out <- llm.Event{Kind: llm.EventError, Err: normalizeError(err)}
			return
		}

		for _, block := range acc.Content {
			if block.Type != "tool_use" {
				continue
			}
			out <- llm.Event{
				Kind: llm.EventToolCall,
				ToolUse: &llm.ToolUse{
					ID:    block.ID,
					Name:  block.Name,
					Input: json.RawMessage(block.Input),
				},
			}
		}

		out <- llm.Event{Kind: llm.EventDone}
	}()

	return out, nil
}

// toAnthropicParams maps one ChatRequest onto the Messages API request.
func toAnthropicParams(req llm.ChatRequest) sdk.MessageNewParams {
	params := sdk.MessageNewParams{
		Model:     sdk.Model(req.Model),
		MaxTokens: defaultMaxTokens,
		System: []sdk.TextBlockParam{
			{Text: req.System},
		},
		Messages: toAnthropicMessages(req.Messages),
		Tools:    toAnthropicTools(req.Tools),
	}
	if req.ToolChoice != "" {
		params.ToolChoice = sdk.ToolChoiceParamOfTool(req.ToolChoice)
	}
	return params
}

// Verify lists one model: free, no tokens, and it fails with 401/403 for a
// bad or disabled key.
func (a *Adapter) Verify(ctx context.Context) error {
	_, err := a.client.Models.List(ctx, sdk.ModelListParams{Limit: sdk.Int(1)})
	if err != nil {
		return normalizeError(err)
	}
	return nil
}

func toAnthropicMessages(msgs []llm.Message) []sdk.MessageParam {
	out := make([]sdk.MessageParam, 0, len(msgs))
	for _, m := range msgs {
		switch {
		case m.ToolResult != nil:
			// Tool results are always fed back as a user-role turn per the
			// Messages API — Anthropic doesn't have a distinct "tool" role.
			out = append(out, sdk.NewUserMessage(
				sdk.NewToolResultBlock(m.ToolResult.ToolUseID, m.ToolResult.Content, m.ToolResult.IsError),
			))
		case m.ToolUse != nil:
			out = append(out, sdk.NewAssistantMessage(sdk.ContentBlockParamUnion{
				OfToolUse: &sdk.ToolUseBlockParam{
					ID:    m.ToolUse.ID,
					Name:  m.ToolUse.Name,
					Input: json.RawMessage(m.ToolUse.Input),
				},
			}))
		case m.Role == llm.RoleAssistant:
			out = append(out, sdk.NewAssistantMessage(sdk.NewTextBlock(m.Text)))
		default:
			out = append(out, sdk.NewUserMessage(sdk.NewTextBlock(m.Text)))
		}
	}
	return out
}

func toAnthropicTools(specs []llm.ToolSpec) []sdk.ToolUnionParam {
	out := make([]sdk.ToolUnionParam, 0, len(specs))
	for _, spec := range specs {
		var schema struct {
			Properties any      `json:"properties"`
			Required   []string `json:"required"`
		}
		// Malformed schemas are a programmer error in the caller's tool list, not a
		// runtime condition to recover from — zero-value schema is an
		// acceptable degraded fallback rather than crashing the request.
		_ = json.Unmarshal(spec.InputSchema, &schema)

		out = append(out, sdk.ToolUnionParam{
			OfTool: &sdk.ToolParam{
				Name:        spec.Name,
				Description: sdk.String(spec.Description),
				InputSchema: sdk.ToolInputSchemaParam{
					Properties: schema.Properties,
					Required:   schema.Required,
				},
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
			return llm.NewError(llm.KindAuth, "Anthropic rejected the configured API key", err)
		case http.StatusTooManyRequests:
			return llm.NewError(llm.KindRateLimit, "Anthropic is rate-limiting this request", err)
		case http.StatusBadRequest, http.StatusNotFound:
			return llm.NewError(llm.KindInvalidReq, "Anthropic rejected the request (check model name)", err)
		}
		return llm.NewError(llm.KindUnavailable, "Anthropic API error", err)
	}
	return llm.NewError(llm.KindUnknown, err.Error(), err)
}
