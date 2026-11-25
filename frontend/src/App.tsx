import { ChatInterface } from './components/ChatInterface';

function App() {
  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <header className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <svg className="w-8 h-8 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <div>
                <h1 className="text-xl font-bold text-gray-900">FinSight RAG</h1>
                <p className="text-xs text-gray-500">Banking Regulatory Intelligence</p>
              </div>
            </div>
            <div className="hidden sm:flex items-center gap-4 text-sm text-gray-600">
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-green-500"></span>
                Live Web Search
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-green-500"></span>
                FTS5 SQLite
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-green-500"></span>
                OpenRouter LLM
              </span>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-8 sm:px-6 lg:px-8 flex-1 w-full">
        <div className="w-full">
          <ChatInterface />
        </div>
      </main>

      <footer className="mt-8 text-center text-sm text-gray-500 border-t border-gray-200 pt-4">
        <p>Powered by FastAPI • SQLite FTS5 • OpenRouter • React + Vite + TypeScript</p>
        <p className="mt-1">No PDF storage — live web fetching with citations</p>
      </footer>
    </div>
  );
}

export default App;