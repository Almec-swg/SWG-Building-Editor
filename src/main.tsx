import { Component, StrictMode, type ErrorInfo, type ReactNode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'

interface ErrorBoundaryState {
  hasError: boolean
  message: string
}

class ErrorBoundary extends Component<{ children: ReactNode }, ErrorBoundaryState> {
  public constructor(props: { children: ReactNode }) {
    super(props)
    this.state = { hasError: false, message: '' }
  }

  public static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, message: error.message }
  }

  public componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error('App crashed', error, info)
  }

  public render(): ReactNode {
    if (this.state.hasError) {
      return (
        <div
          style={{
            minHeight: '100vh',
            background: '#1a1c21',
            color: '#ffebcf',
            padding: '24px',
            fontFamily: 'Segoe UI, sans-serif',
          }}
        >
          <h1 style={{ marginTop: 0 }}>SWG Building Editor failed to load</h1>
          <p style={{ marginBottom: '8px' }}>Runtime error:</p>
          <pre
            style={{
              background: '#111216',
              border: '1px solid #3a3d46',
              borderRadius: '8px',
              padding: '12px',
              overflowX: 'auto',
              whiteSpace: 'pre-wrap',
            }}
          >
            {this.state.message || 'Unknown error'}
          </pre>
          <p>Open DevTools console for full stack trace.</p>
        </div>
      )
    }

    return this.props.children
  }
}

const rootElement = document.getElementById('root')

if (!rootElement) {
  throw new Error('Missing #root element in index.html')
}

const root = createRoot(rootElement)

function renderStartupFailure(message: string): void {
  root.render(
    <div
      style={{
        minHeight: '100vh',
        background: '#1a1c21',
        color: '#ffebcf',
        padding: '24px',
        fontFamily: 'Segoe UI, sans-serif',
      }}
    >
      <h1 style={{ marginTop: 0 }}>SWG Building Editor could not start</h1>
      <p style={{ marginBottom: '8px' }}>Startup error:</p>
      <pre
        style={{
          background: '#111216',
          border: '1px solid #3a3d46',
          borderRadius: '8px',
          padding: '12px',
          overflowX: 'auto',
          whiteSpace: 'pre-wrap',
        }}
      >
        {message}
      </pre>
      <p>Copy this message and send it to continue troubleshooting.</p>
    </div>,
  )
}

void import('./App.tsx')
  .then((module) => {
    const App = module.default
    root.render(
      <StrictMode>
        <ErrorBoundary>
          <App />
        </ErrorBoundary>
      </StrictMode>,
    )
  })
  .catch((error: unknown) => {
    const message = error instanceof Error ? error.stack ?? error.message : String(error)
    console.error('Startup failure', error)
    renderStartupFailure(message)
  })
