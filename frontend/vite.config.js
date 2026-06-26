import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The frontend talks to the backend directly via VITE_API_BASE (default
// http://localhost:8000). We also set up a dev proxy so relative /api calls
// work if you prefer same-origin during development.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        ws: true,
      },
    },
  },
})
