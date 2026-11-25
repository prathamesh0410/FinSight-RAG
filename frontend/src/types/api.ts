export interface Citation {
  source_id: number;
  title: string;
  url: string;
  regulator: string;
  type: string;
  date: string | null;
  snippet: string;
}

export interface QueryRequest {
  query: string;
  top_k?: number;
  model?: string;
}

export interface QueryResponse {
  query: string;
  answer: string;
  citations: Citation[];
  confidence: number;
  response_time_ms: number;
  sources_checked: number;
}

export interface Model {
  id: string;
  name: string;
  free: boolean;
}

export interface ModelsResponse {
  models: Model[];
  default: string;
}

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
}