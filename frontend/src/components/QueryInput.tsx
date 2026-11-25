import { useState, useEffect, FormEvent, ChangeEvent } from 'react';
import type { Model } from '../types/api';

interface QueryInputProps {
  onSubmit: (query: string, model?: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

export function QueryInput({ onSubmit, disabled, placeholder }: QueryInputProps) {
  const [query, setQuery] = useState('');
  const [model, setModel] = useState<string>('');
  const [models, setModels] = useState<Model[]>([]);
  const [loadingModels, setLoadingModels] = useState(true);

  useEffect(() => {
    fetch('/api/v1/models')
      .then(r => r.json())
      .then(data => {
        setModels(data.models);
        setModel(data.default);
        setLoadingModels(false);
      })
      .catch(() => setLoadingModels(false));
  }, []);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!query.trim() || disabled) return;
    onSubmit(query.trim(), model || undefined);
    setQuery('');
  };

  return (
    <form onSubmit={handleSubmit} className="w-full">
      <div className="flex gap-3 items-end">
        <div className="flex-1">
          <label htmlFor="query" className="sr-only">Search query</label>
          <input
            id="query"
            type="text"
            value={query}
            onChange={(e: ChangeEvent<HTMLInputElement>) => setQuery(e.target.value)}
            placeholder={placeholder || "Ask a question... e.g., 'What are RBI norms on digital lending co-lending?'"}
            disabled={disabled}
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent outline-none transition-all disabled:bg-gray-100"
            autoFocus
          />
        </div>

        <div className="w-64 shrink-0">
          <label htmlFor="model" className="sr-only">AI Model</label>
          <select
            id="model"
            value={model}
            onChange={(e: ChangeEvent<HTMLSelectElement>) => setModel(e.target.value)}
            disabled={disabled || loadingModels}
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent outline-none bg-white text-sm disabled:bg-gray-100"
          >
            {loadingModels ? (
              <option value="" disabled>Loading models...</option>
            ) : (
              models.map(m => (
                <option key={m.id} value={m.id}>
                  {m.name} {m.free && '— free'}
                </option>
              ))
            )}
          </select>
        </div>

        <button
          type="submit"
          disabled={disabled || !query.trim()}
          className="px-6 py-3 bg-green-600 text-white rounded-lg font-medium hover:bg-green-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap"
        >
          {disabled ? 'Searching...' : 'Ask'}
        </button>
      </div>
    </form>
  );
}