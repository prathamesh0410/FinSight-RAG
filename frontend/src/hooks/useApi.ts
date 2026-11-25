import { useState, useCallback } from 'react';
import type { QueryRequest, QueryResponse, ModelsResponse, HealthResponse } from '../types/api';

const API_BASE = '/api/v1';

export function useApi() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const query = useCallback(async (request: QueryRequest): Promise<QueryResponse | null> => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      return await response.json();
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Unknown error';
      setError(msg);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const getModels = useCallback(async (): Promise<ModelsResponse | null> => {
    try {
      const response = await fetch(`${API_BASE}/models`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (err) {
      console.error('Failed to fetch models:', err);
      return null;
    }
  }, []);

  const healthCheck = useCallback(async (): Promise<HealthResponse | null> => {
    try {
      const response = await fetch('/health');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (err) {
      console.error('Health check failed:', err);
      return null;
    }
  }, []);

  return { query, getModels, healthCheck, loading, error, setError };
}