/**
 * API service for communicating with MyAI backend
 */

import axios, { AxiosInstance } from 'axios';
import type { GenerateRequest, GenerateResponse, AuthToken, User, ModelInfo } from '../types';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

class APIService {
  private client: AxiosInstance;
  private token: string | null = null;

  constructor() {
    this.client = axios.create({
      baseURL: API_URL,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Add request interceptor to include auth token
    this.client.interceptors.request.use((config) => {
      if (this.token) {
        config.headers.Authorization = `Bearer ${this.token}`;
      }
      return config;
    });

    // Load token from localStorage
    const savedToken = localStorage.getItem('auth_token');
    if (savedToken) {
      this.token = savedToken;
    }
  }

  setToken(token: string) {
    this.token = token;
    localStorage.setItem('auth_token', token);
  }

  clearToken() {
    this.token = null;
    localStorage.removeItem('auth_token');
  }

  async login(username: string, password: string): Promise<AuthToken> {
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);

    const response = await this.client.post<AuthToken>('/token', formData, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    });

    this.setToken(response.data.access_token);
    return response.data;
  }

  async getCurrentUser(): Promise<User> {
    const response = await this.client.get<User>('/users/me');
    return response.data;
  }

  async generate(request: GenerateRequest): Promise<GenerateResponse> {
    const response = await this.client.post<GenerateResponse>('/generate', request);
    return response.data;
  }

  async getHealth(): Promise<ModelInfo> {
    const response = await this.client.get<ModelInfo>('/health');
    return response.data;
  }

  async getModelInfo(): Promise<any> {
    const response = await this.client.get('/admin/model/info');
    return response.data;
  }

  async loadModel(): Promise<any> {
    const response = await this.client.post('/admin/model/load');
    return response.data;
  }

  async unloadModel(): Promise<any> {
    const response = await this.client.post('/admin/model/unload');
    return response.data;
  }

  createWebSocket(onMessage: (token: string) => void, onError: (error: any) => void, onClose: () => void): WebSocket {
    const wsUrl = (import.meta.env.VITE_WS_URL || 'ws://localhost:8000') + '/ws/generate';
    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.token) {
          onMessage(data.token);
        } else if (data.done) {
          ws.close();
        } else if (data.error) {
          onError(new Error(data.error));
          ws.close();
        }
      } catch (error) {
        onError(error);
      }
    };

    ws.onerror = (event) => {
      onError(new Error('WebSocket error'));
    };

    ws.onclose = () => {
      onClose();
    };

    return ws;
  }
}

export const apiService = new APIService();
