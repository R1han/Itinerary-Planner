import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        // Split the two big vendors out of the app chunk: they change far less often than the
        // app does, so a redeploy does not invalidate 400 kB of cached third-party code.
        manualChunks: {
          map: ['leaflet', 'react-leaflet'],
          markdown: ['react-markdown', 'remark-gfm'],
        },
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      // Everything the API owns, proxied so the browser sees one origin in dev.
      '^/(auth|me|chat|conversations|events|family|itineraries|preferences|health)(/|$)': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
