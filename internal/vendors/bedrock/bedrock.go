package bedrock

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/aws/signer/v4"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/baalimago/clai/internal/models"
	pub_models "github.com/baalimago/clai/pkg/text/models"
	"github.com/baalimago/go_away_boilerplate/pkg/ancli"
	"github.com/baalimago/go_away_boilerplate/pkg/misc"
)

const defaultServiceName = "bedrock"

var Default = Bedrock{
	Model:       "anthropic.claude-3-5-sonnet-20240620-v1:0",
	Region:      "",
	Endpoint:    "",
	MaxTokens:   intPtr(1024),
	Temperature: floatPtr(0.7),
	TopP:        floatPtr(1.0),
}

type Bedrock struct {
	Model       string   `json:"model"`
	Region      string   `json:"region"`
	Endpoint    string   `json:"endpoint"`
	MaxTokens   *int     `json:"max_tokens"`
	Temperature *float64 `json:"temperature"`
	TopP        *float64 `json:"top_p"`

	client      *http.Client                 `json:"-"`
	creds       aws.CredentialsProvider      `json:"-"`
	signer      *v4.Signer                   `json:"-"`
	debug       bool                         `json:"-"`
	toolSpecs   []pub_models.Specification   `json:"-"`
	toolChoice  *bedrockToolChoice           `json:"-"`
	regionFinal string                       `json:"-"`
}

func intPtr(v int) *int {
	return &v
}

func floatPtr(v float64) *float64 {
	return &v
}

func (b *Bedrock) Setup() error {
	var optFns []func(*config.LoadOptions) error
	if b.Region != "" {
		optFns = append(optFns, config.WithRegion(b.Region))
	}
	cfg, err := config.LoadDefaultConfig(context.Background(), optFns...)
	if err != nil {
		return fmt.Errorf("failed to load aws config: %w", err)
	}

	b.client = &http.Client{}
	b.creds = cfg.Credentials
	b.signer = v4.NewSigner()
	b.regionFinal = cfg.Region
	if b.regionFinal == "" {
		return fmt.Errorf("bedrock region not set (configure AWS region or set Region in bedrock config)")
	}

	if misc.Truthy(os.Getenv("DEBUG")) || misc.Truthy(os.Getenv("BEDROCK_DEBUG")) {
		b.debug = true
	}

	// Default to auto tool choice when tools are present.
	b.toolChoice = &bedrockToolChoice{Auto: &struct{}{}}
	return nil
}

func (b *Bedrock) RegisterTool(tool pub_models.LLMTool) {
	b.toolSpecs = append(b.toolSpecs, tool.Specification())
}

func (b *Bedrock) StreamCompletions(ctx context.Context, chat pub_models.Chat) (chan models.CompletionEvent, error) {
	if b.client == nil || b.signer == nil || b.creds == nil {
		return nil, fmt.Errorf("bedrock not initialized, call Setup first")
	}

	outChan := make(chan models.CompletionEvent)
	go func() {
		defer close(outChan)

		resp, err := b.converse(ctx, chat)
		if err != nil {
			outChan <- err
			return
		}

		for _, block := range resp.Output.Message.Content {
			switch {
			case block.Text != "":
				outChan <- block.Text
			case block.ToolUse != nil:
				call := b.toolUseToCall(*block.ToolUse)
				outChan <- call
			}
		}
		outChan <- models.StopEvent{}
	}()

	return outChan, nil
}

func (b *Bedrock) converse(ctx context.Context, chat pub_models.Chat) (*converseResponse, error) {
	reqBody, err := b.buildRequest(chat)
	if err != nil {
		return nil, fmt.Errorf("failed to build bedrock request: %w", err)
	}

	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to encode bedrock request: %w", err)
	}

	req, err := b.newSignedRequest(ctx, bodyBytes)
	if err != nil {
		return nil, fmt.Errorf("failed to sign bedrock request: %w", err)
	}

	resp, err := b.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute bedrock request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read bedrock response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("bedrock error: %v, body: %s", resp.Status, string(respBody))
	}

	if b.debug {
		ancli.PrintOK(fmt.Sprintf("bedrock response: %s\n", string(respBody)))
	}

	var parsed converseResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return nil, fmt.Errorf("failed to parse bedrock response: %w", err)
	}
	return &parsed, nil
}

func (b *Bedrock) buildRequest(chat pub_models.Chat) (*converseRequest, error) {
	systemBlocks := buildSystemBlocks(chat)
	messages, err := b.buildMessages(chat)
	if err != nil {
		return nil, err
	}

	req := &converseRequest{
		Messages: messages,
		System:   systemBlocks,
	}

	if b.MaxTokens != nil || b.Temperature != nil || b.TopP != nil {
		req.InferenceConfig = &bedrockInferenceConfig{
			MaxTokens:   b.MaxTokens,
			Temperature: b.Temperature,
			TopP:        b.TopP,
		}
	}

	if len(b.toolSpecs) > 0 {
		req.ToolConfig = &bedrockToolConfig{
			Tools:      toBedrockTools(b.toolSpecs),
			ToolChoice: b.toolChoice,
		}
	}

	return req, nil
}

func (b *Bedrock) buildMessages(chat pub_models.Chat) ([]bedrockMessage, error) {
	messages := make([]bedrockMessage, 0, len(chat.Messages))
	for _, msg := range chat.Messages {
		if msg.Role == "system" {
			continue
		}

		role := msg.Role
		if role == "tool" {
			role = "user"
		}

		content := make([]bedrockContentBlock, 0, 1)
		if msg.Content != "" {
			content = append(content, bedrockContentBlock{Text: msg.Content})
		}
		for _, part := range msg.ContentParts {
			if part.Text != "" {
				content = append(content, bedrockContentBlock{Text: part.Text})
			}
		}

		if msg.Role == "assistant" && len(msg.ToolCalls) > 0 {
			content = make([]bedrockContentBlock, 0, len(msg.ToolCalls))
			for _, call := range msg.ToolCalls {
				content = append(content, bedrockContentBlock{
					ToolUse: &bedrockToolUse{
						ToolUseID: call.ID,
						Name:      call.Name,
						Input:     call.Inputs,
					},
				})
			}
		}

		if msg.Role == "tool" {
			content = []bedrockContentBlock{
				{
					ToolResult: &bedrockToolResult{
						ToolUseID: msg.ToolCallID,
						Content: []bedrockContentBlock{
							{Text: msg.Content},
						},
						Status: "success",
					},
				},
			}
		}

		if len(content) == 0 {
			continue
		}

		messages = append(messages, bedrockMessage{
			Role:    role,
			Content: content,
		})
	}
	return messages, nil
}

func (b *Bedrock) toolUseToCall(toolUse bedrockToolUse) pub_models.Call {
	var inputs *pub_models.Input
	if toolUse.Input != nil {
		inputs = toolUse.Input
	}
	return pub_models.Call{
		ID:       toolUse.ToolUseID,
		Name:     toolUse.Name,
		Type:     "function",
		Inputs:   inputs,
		Function: pub_models.Specification{Name: toolUse.Name},
	}
}

func buildSystemBlocks(chat pub_models.Chat) []bedrockSystemBlock {
	var systemParts []string
	for _, msg := range chat.Messages {
		if msg.Role == "system" && msg.Content != "" {
			systemParts = append(systemParts, msg.Content)
		}
	}
	if len(systemParts) == 0 {
		return nil
	}
	return []bedrockSystemBlock{{Text: strings.Join(systemParts, "\n\n")}}
}

func (b *Bedrock) newSignedRequest(ctx context.Context, body []byte) (*http.Request, error) {
	endpoint := strings.TrimRight(b.Endpoint, "/")
	if endpoint == "" {
		endpoint = fmt.Sprintf("https://bedrock-runtime.%s.amazonaws.com", b.regionFinal)
	}
	modelID := url.PathEscape(b.Model)
	reqURL := fmt.Sprintf("%s/model/%s/converse", endpoint, modelID)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, reqURL, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create bedrock request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	hash := sha256.Sum256(body)
	payloadHash := hex.EncodeToString(hash[:])

	creds, err := b.creds.Retrieve(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve aws credentials: %w", err)
	}

	err = b.signer.SignHTTP(ctx, creds, req, payloadHash, defaultServiceName, b.regionFinal, time.Now())
	if err != nil {
		return nil, fmt.Errorf("failed to sign bedrock request: %w", err)
	}

	return req, nil
}

type converseRequest struct {
	Messages       []bedrockMessage         `json:"messages,omitempty"`
	System         []bedrockSystemBlock     `json:"system,omitempty"`
	InferenceConfig *bedrockInferenceConfig `json:"inferenceConfig,omitempty"`
	ToolConfig     *bedrockToolConfig       `json:"toolConfig,omitempty"`
}

type bedrockInferenceConfig struct {
	MaxTokens   *int     `json:"maxTokens,omitempty"`
	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"topP,omitempty"`
}

type bedrockToolConfig struct {
	Tools      []bedrockTool     `json:"tools,omitempty"`
	ToolChoice *bedrockToolChoice `json:"toolChoice,omitempty"`
}

type bedrockToolChoice struct {
	Auto *struct{}       `json:"auto,omitempty"`
	Any  *struct{}       `json:"any,omitempty"`
	Tool *bedrockToolRef `json:"tool,omitempty"`
}

type bedrockToolRef struct {
	Name string `json:"name"`
}

type bedrockTool struct {
	ToolSpec bedrockToolSpec `json:"toolSpec"`
}

type bedrockToolSpec struct {
	Name        string             `json:"name"`
	Description string             `json:"description,omitempty"`
	InputSchema bedrockInputSchema `json:"inputSchema"`
}

type bedrockInputSchema struct {
	JSON *pub_models.InputSchema `json:"json,omitempty"`
}

func toBedrockTools(specs []pub_models.Specification) []bedrockTool {
	tools := make([]bedrockTool, 0, len(specs))
	for _, spec := range specs {
		tools = append(tools, bedrockTool{
			ToolSpec: bedrockToolSpec{
				Name:        spec.Name,
				Description: spec.Description,
				InputSchema: bedrockInputSchema{JSON: spec.Inputs},
			},
		})
	}
	return tools
}

type bedrockSystemBlock struct {
	Text string `json:"text,omitempty"`
}

type bedrockMessage struct {
	Role    string               `json:"role"`
	Content []bedrockContentBlock `json:"content"`
}

type bedrockContentBlock struct {
	Text       string              `json:"text,omitempty"`
	ToolUse    *bedrockToolUse     `json:"toolUse,omitempty"`
	ToolResult *bedrockToolResult  `json:"toolResult,omitempty"`
}

type bedrockToolUse struct {
	ToolUseID string           `json:"toolUseId,omitempty"`
	Name      string           `json:"name,omitempty"`
	Input     *pub_models.Input `json:"input,omitempty"`
}

type bedrockToolResult struct {
	ToolUseID string               `json:"toolUseId,omitempty"`
	Content   []bedrockContentBlock `json:"content,omitempty"`
	Status    string               `json:"status,omitempty"`
}

type converseResponse struct {
	Output struct {
		Message bedrockMessage `json:"message"`
	} `json:"output"`
	StopReason string `json:"stopReason,omitempty"`
}
