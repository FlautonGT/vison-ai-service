package aiclient

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"strconv"
	"time"
)

// Client calls the Python AI inference service.
type Client struct {
	baseURL         string
	aiServiceSecret string
	httpClient      *http.Client
}

type ValidationOptions struct {
	ValidateAttributes bool
	ValidateQuality    bool
}

// AIServiceError maps Python error payload.
type AIServiceError struct {
	StatusCode int
	Code       string
	Message    string
	Detail     string
}

func (e *AIServiceError) Error() string {
	return fmt.Sprintf("[%s] %s", e.Code, e.Detail)
}

func New(baseURL string) *Client {
	return NewWithSecret(baseURL, "")
}

func NewWithSecret(baseURL, aiServiceSecret string) *Client {
	return &Client{
		baseURL:         baseURL,
		aiServiceSecret: aiServiceSecret,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

func (c *Client) SetServiceSecret(secret string) {
	c.aiServiceSecret = secret
}

func (c *Client) Liveness(ctx context.Context, imageData []byte, validation ValidationOptions) (map[string]interface{}, error) {
	return c.postSingleImage(ctx, "/api/face/liveness", "image", imageData, nil, validation)
}

func (c *Client) Deepfake(ctx context.Context, imageData []byte, validation ValidationOptions) (map[string]interface{}, error) {
	return c.postSingleImage(ctx, "/api/face/deepfake", "image", imageData, nil, validation)
}

func (c *Client) Analyze(ctx context.Context, imageData []byte, validation ValidationOptions) (map[string]interface{}, error) {
	return c.postSingleImage(ctx, "/api/face/analyze", "image", imageData, nil, validation)
}

func (c *Client) Compare(
	ctx context.Context,
	sourceData, targetData []byte,
	threshold *float64,
	validation ValidationOptions,
) (map[string]interface{}, error) {
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	sourcePart, _ := writer.CreateFormFile("sourceImage", "source.jpg")
	sourcePart.Write(sourceData)

	targetPart, _ := writer.CreateFormFile("targetImage", "target.jpg")
	targetPart.Write(targetData)

	if threshold != nil {
		writer.WriteField("similarityThreshold", strconv.FormatFloat(*threshold, 'f', 2, 64))
	}
	writeValidationFields(writer, validation)
	writer.Close()

	return c.doRequest(ctx, "POST", "/api/face/compare", body, writer.FormDataContentType())
}

// Embed extracts 512-dim embedding from one face image.
func (c *Client) Embed(
	ctx context.Context,
	imageData []byte,
	validation ValidationOptions,
) (map[string]interface{}, error) {
	return c.postSingleImage(ctx, "/api/face/embed", "image", imageData, nil, validation)
}

// Similarity compares image embedding with stored embedding sent by Go backend.
func (c *Client) Similarity(
	ctx context.Context,
	imageData []byte,
	storedEmbedding []float64,
	validation ValidationOptions,
) (map[string]interface{}, error) {
	embJSON, err := json.Marshal(storedEmbedding)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal stored embedding: %w", err)
	}

	fields := map[string]string{
		"embeddingStored": string(embJSON),
	}
	return c.postSingleImage(ctx, "/api/face/similarity", "image", imageData, fields, validation)
}

func (c *Client) postSingleImage(
	ctx context.Context,
	path, fieldName string,
	imageData []byte,
	extraFields map[string]string,
	validation ValidationOptions,
) (map[string]interface{}, error) {
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	part, _ := writer.CreateFormFile(fieldName, fieldName+".jpg")
	part.Write(imageData)

	for k, v := range extraFields {
		writer.WriteField(k, v)
	}
	writeValidationFields(writer, validation)
	writer.Close()

	return c.doRequest(ctx, "POST", path, body, writer.FormDataContentType())
}

func writeValidationFields(writer *multipart.Writer, v ValidationOptions) {
	if v.ValidateAttributes {
		writer.WriteField("validateAttributes", "true")
	}
	if v.ValidateQuality {
		writer.WriteField("validateQuality", "true")
	}
}

func (c *Client) doRequest(
	ctx context.Context,
	method, path string,
	body io.Reader,
	contentType string,
) (map[string]interface{}, error) {
	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, body)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", contentType)
	if c.aiServiceSecret != "" {
		req.Header.Set("X-AI-Service-Key", c.aiServiceSecret)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("AI service request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read AI service response: %w", err)
	}

	var result map[string]interface{}
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("failed to parse AI service response: %w", err)
	}

	if resp.StatusCode >= 400 {
		errObj, _ := result["error"].(map[string]interface{})
		code, _ := errObj["code"].(string)
		detail, _ := errObj["detail"].(string)
		message, _ := result["message"].(string)
		return nil, &AIServiceError{
			StatusCode: resp.StatusCode,
			Code:       code,
			Message:    message,
			Detail:     detail,
		}
	}

	return result, nil
}

// ParseEmbedding converts embed response["embedding"] into []float64.
func ParseEmbedding(result map[string]interface{}) ([]float64, error) {
	raw, ok := result["embedding"].([]interface{})
	if !ok {
		return nil, fmt.Errorf("embedding field missing or invalid")
	}

	embedding := make([]float64, len(raw))
	for i, value := range raw {
		number, ok := value.(float64)
		if !ok {
			return nil, fmt.Errorf("embedding[%d] is not a number", i)
		}
		embedding[i] = number
	}
	return embedding, nil
}
