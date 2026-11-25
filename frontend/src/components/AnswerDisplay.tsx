import type { QueryResponse } from '../types/api';

interface AnswerDisplayProps {
  response: QueryResponse | null;
  loading: boolean;
  error: string | null;
}

export function AnswerDisplay({ response, loading, error }: AnswerDisplayProps) {
  if (loading) {
    return (
      <div className="animate-pulse space-y-4">
        <div className="h-8 bg-gray-200 rounded w-3/4"></div>
        <div className="h-32 bg-gray-200 rounded"></div>
        <div className="h-8 bg-gray-200 rounded w-1/2"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 text-red-800 p-4 rounded-lg" role="alert">
        <p className="font-medium">Error</p>
        <p className="text-sm mt-1">{error}</p>
      </div>
    );
  }

  if (!response) return null;

  const renderAnswer = (answer: string) => {
    const parts = answer.split(/(\[\d+\])/);
    return (
      <div className="prose prose-sm max-w-none text-gray-800">
        {parts.map((part, i) => {
          const match = part.match(/^\[(\d+)\]$/);
          if (match) {
            return (
              <sup key={i} className="text-green-600 font-semibold cursor-pointer hover:underline"
                title={`Source ${match[1]}`}>
                [{match[1]}]
              </sup>
            );
          }
          return <span key={i}>{part}</span>;
        })}
      </div>
    );
  };

  return (
    <div className="space-y-6">
      <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Answer</h3>
          <div className="flex items-center gap-4 text-sm text-gray-500">
            <span className="flex items-center gap-1">
              <span className={`w-2 h-2 rounded-full ${
                response.confidence > 0.7 ? 'bg-green-500' :
                response.confidence > 0.4 ? 'bg-yellow-500' : 'bg-red-500'
              }`}></span>
              Confidence: {Math.round(response.confidence * 100)}%
            </span>
            <span>⏱ {response.response_time_ms}ms</span>
            <span>📄 {response.sources_checked} sources</span>
          </div>
        </div>
        {renderAnswer(response.answer)}
      </div>

      {response.citations.length > 0 && (
        <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Sources & Citations</h3>
          <div className="space-y-4">
            {response.citations.map((citation: typeof response.citations[0]) => (
              <div key={citation.source_id} className="border-l-4 border-green-500 pl-4 py-2 hover:bg-gray-50 transition-colors">
                <div className="flex items-start gap-2">
                  <span className="text-green-600 font-bold text-sm mt-0.5">[{citation.source_id}]</span>
                  <div className="flex-1 min-w-0">
                    <a
                      href={citation.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-600 hover:underline font-medium break-all"
                    >
                      {citation.title}
                    </a>
                    <div className="flex flex-wrap gap-3 mt-1 text-xs text-gray-600">
                      <span className="px-2 py-0.5 bg-gray-100 rounded">{citation.regulator}</span>
                      <span className="px-2 py-0.5 bg-gray-100 rounded capitalize">{citation.type}</span>
                      {citation.date && <span className="px-2 py-0.5 bg-gray-100 rounded">{citation.date}</span>}
                    </div>
                    <p className="mt-2 text-sm text-gray-700 line-clamp-2">{citation.snippet}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}