/**
 * Type definitions for MyAI frontend
 */

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  isStreaming?: boolean;
}

export interface GenerateRequest {
  prompt: string;
  max_new_tokens?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  stream?: boolean;
}

export interface GenerateResponse {
  request_id: string;
  text: string;
  tokens: number;
  latency: number;
  model: string;
}

export interface User {
  username: string;
  email?: string;
  full_name?: string;
  is_admin: boolean;
}

export interface AuthToken {
  access_token: string;
  token_type: string;
  expires_in: number;
}

export interface ModelInfo {
  status: string;
  model_loaded: boolean;
  model_info: {
    model_name: string;
    device: string;
    quantized: boolean;
    num_parameters?: number;
  };
}

export interface ChatSettings {
  maxTokens: number;
  temperature: number;
  topP: number;
  topK: number;
  stream: boolean;
}

export interface ErrorResponse {
  error: string;
  detail?: string;
}
