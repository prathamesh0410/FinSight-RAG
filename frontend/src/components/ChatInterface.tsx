import { useState, useRef, useEffect } from 'react';
import { QueryInput } from './QueryInput';
import { AnswerDisplay } from './AnswerDisplay';
import { useApi } from '../hooks/useApi';
import type { QueryResponse } from '../types/api';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  query: string;
  response: QueryResponse | null;
  loading: boolean;
  error: string | null;
  model?: string;
}

// Example questions to help users get started
const EXAMPLE_QUESTIONS = [
  { category: 'RBI', questions: [
    'What are the latest RBI KYC norms for customer due diligence?',
    'Explain RBI compromise settlement framework for stressed assets',
    'What are RBI large exposure limits for banks?',
    'RBI fraud risk management guidelines for commercial banks',
    'RBI digital lending co-lending norms',
  ]},
  { category: 'Basel / Capital', questions: [
    'What are Basel III capital requirements?',
    'Explain Basel III output floor and risk-weighted assets',
    'Basel FRTB market risk framework',
  ]},
  { category: 'NPA / Provisioning', questions: [
    'RBI NPA provisioning and asset classification norms',
    'RBI divergence in asset classification disclosure',
    'Technical write-offs vs compromise settlement',
  ]},
];

export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [showExamples, setShowExamples] = useState(true);
  const { query: apiQuery, healthCheck, loading: apiLoading } = useApi();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    healthCheck().then(h => {
      if (!h) {
        setMessages(prev => [...prev, {
          id: 'error',
          role: 'assistant',
          query: '',
          response: null,
          loading: false,
          error: 'Backend unavailable. Make sure the server is running on http://localhost:8000',
        }]);
      }
    });
  }, [healthCheck]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    inputRef.current?.focus();
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSearch = async (q: string, model?: string) => {
    setShowExamples(false);

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      query: q,
      response: null,
      loading: false,
      error: null,
      model,
    };

    const assistantMessage: Message = {
      id: crypto.randomUUID(),
      role: 'assistant',
      query: '',
      response: null,
      loading: true,
      error: null,
    };

    setMessages(prev => [...prev, userMessage, assistantMessage]);

    try {
      const result = await apiQuery({ query: q, top_k: 5, model });
      setMessages(prev => prev.map(m =>
        m.id === assistantMessage.id
          ? { ...m, response: result || null, loading: false, error: result ? null : 'No response received' }
          : m
      ));
    } catch (err) {
      setMessages(prev => prev.map(m =>
        m.id === assistantMessage.id
          ? { ...m, loading: false, error: err instanceof Error ? err.message : 'Search failed' }
          : m
      ));
    }
  };

  const handleExampleClick = (question: string) => {
    handleSearch(question);
  };

  const handleNewChat = () => {
    setMessages([]);
    setShowExamples(true);
  };

  const renderMessage = (msg: Message) => {
    if (msg.role === 'user') {
      return (
        <div key={msg.id} className="flex justify-end mb-4 animate-slide-in">
          <div className="max-w-[80%] bg-green-600 text-white rounded-2xl rounded-br-md px-5 py-3 shadow-sm">
            <p className="text-sm">{msg.query}</p>
          </div>
        </div>
      );
    }

    return (
      <div key={msg.id} className="flex justify-start mb-4 animate-slide-in">
        <div className="max-w-[90%] w-full">
          {msg.loading && (
            <div className="bg-white border border-gray-200 rounded-2xl rounded-bl-md px-5 py-4 shadow-sm animate-pulse space-y-3">
              <div className="h-4 bg-gray-200 rounded w-1/2"></div>
              <div className="h-4 bg-gray-200 rounded w-3/4"></div>
              <div className="h-4 bg-gray-200 rounded w-full"></div>
              <div className="h-4 bg-gray-200 rounded w-2/3"></div>
            </div>
          )}
          {msg.error && (
            <div className="bg-red-50 border border-red-200 text-red-800 rounded-2xl rounded-bl-md px-5 py-3 shadow-sm">
              <div className="flex items-center gap-2">
                <svg className="w-5 h-5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
                <p className="font-medium text-sm">Unable to get answer</p>
              </div>
              <p className="text-sm mt-2">{msg.error}</p>
              <p className="text-xs text-red-600 mt-2">The AI model is rate-limited (free tier). Citations below still work.</p>
            </div>
          )}
          {msg.response && !msg.loading && (
            <AnswerDisplay response={msg.response} loading={false} error={null} />
          )}
          {(!msg.response && !msg.loading && !msg.error) && (
            <div className="bg-gray-50 border border-gray-200 rounded-2xl rounded-bl-md px-5 py-3 text-center text-gray-500 text-sm">
              No results found. Try different keywords.
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="flex flex-col h-[calc(100vh-140px)] min-h-[550px] bg-gray-50">
      {/* Header with clear purpose */}
      <div className="bg-white border-b border-gray-200 px-4 py-4 sm:px-6">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 bg-green-100 rounded-xl flex items-center justify-center">
              <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">FinSight RAG</h1>
              <p className="text-sm text-gray-500">Ask questions about banking regulations — get cited answers from live RBI, SEBI, Basel sources</p>
            </div>
          </div>

          {/* How it works - simple 3 steps */}
          <div className="grid grid-cols-3 gap-3 text-center text-xs text-gray-600">
            <div className="flex flex-col items-center gap-1 p-2 bg-gray-50 rounded-lg">
              <span className="w-6 h-6 bg-green-100 text-green-600 rounded-full flex items-center justify-center text-xs font-bold">1</span>
              <span>Ask naturally</span>
            </div>
            <div className="flex flex-col items-center gap-1 p-2 bg-gray-50 rounded-lg">
              <span className="w-6 h-6 bg-green-100 text-green-600 rounded-full flex items-center justify-center text-xs font-bold">2</span>
              <span>Live search</span>
            </div>
            <div className="flex flex-col items-center gap-1 p-2 bg-gray-50 rounded-lg">
              <span className="w-6 h-6 bg-green-100 text-green-600 rounded-full flex items-center justify-center text-xs font-bold">3</span>
              <span>Cited answer</span>
            </div>
          </div>
        </div>
      </div>

      {/* Chat History */}
      <div className="flex-1 overflow-y-auto px-4 pb-4" role="log" aria-live="polite">
        <div className="max-w-4xl mx-auto">
          {messages.length === 0 && showExamples && (
            <div className="py-12 space-y-6 animate-fade-in">
              <div className="text-center">
                <svg className="w-16 h-16 mx-auto text-gray-300 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
                <h2 className="text-xl font-semibold text-gray-900 mb-2">What would you like to know?</h2>
                <p className="text-gray-500">Type your question or click an example below</p>
              </div>

              <div className="space-y-4">
                {EXAMPLE_QUESTIONS.map((cat, i) => (
                  <div key={i} className="bg-white border border-gray-200 rounded-xl p-4 hover:border-green-300 transition-colors">
                    <h3 className="text-xs font-semibold text-green-700 uppercase tracking-wider mb-3">{cat.category}</h3>
                    <div className="flex flex-wrap gap-2">
                      {cat.questions.map((q, j) => (
                        <button
                          key={j}
                          onClick={() => handleExampleClick(q)}
                          className="text-left px-3 py-2 bg-gray-50 hover:bg-green-50 border border-gray-200 hover:border-green-200 rounded-lg text-sm text-gray-700 hover:text-green-800 transition-all whitespace-normal"
                        >
                          {q}
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>

              <div className="text-center text-xs text-gray-400 pt-4 border-t border-gray-200">
                <p>💡 Try: "RBI KYC amendment 2024", "Basel III capital buffer", "NPA provisioning norms", "digital lending guidelines"</p>
              </div>
            </div>
          )}

          {messages.length > 0 && (
            <div className="space-y-3 animate-fade-in">
              {messages.map(renderMessage)}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
      </div>

      {/* Input Area - Fixed at bottom */}
      <div className="bg-white border-t border-gray-200">
        <div className="max-w-4xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between gap-3 mb-3">
            <div className="flex items-center gap-3 text-xs text-gray-500 flex-wrap">
              <span className="flex items-center gap-1 px-2 py-1 bg-green-50 text-green-700 rounded-full">
                <span className="w-1.5 h-1.5 rounded-full bg-green-500"></span>
                Live web search
              </span>
              <span className="flex items-center gap-1 px-2 py-1 bg-green-50 text-green-700 rounded-full">
                <span className="w-1.5 h-1.5 rounded-full bg-green-500"></span>
                FTS5 SQLite
              </span>
              <span className="flex items-center gap-1 px-2 py-1 bg-green-50 text-green-700 rounded-full">
                <span className="w-1.5 h-1.5 rounded-full bg-green-500"></span>
                OpenRouter LLM
              </span>
            </div>
            <button
              onClick={handleNewChat}
              disabled={messages.length === 0}
              className="text-xs font-medium text-gray-500 hover:text-green-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap px-3 py-1.5 border border-gray-200 rounded-lg hover:bg-gray-50"
            >
              + New Chat
            </button>
          </div>
          <QueryInput
            onSubmit={handleSearch}
            disabled={apiLoading || messages.some(m => m.loading)}
            placeholder="Type your question... e.g., 'What are RBI norms for digital lending co-lending?'"
          />
          <p className="text-xs text-gray-400 text-center mt-3">
            Press <kbd className="px-1.5 py-0.5 bg-gray-100 rounded text-gray-600 border">Enter</kbd> to search •
            Model rate limits may apply on free tier
          </p>
        </div>
      </div>
    </div>
  );
}