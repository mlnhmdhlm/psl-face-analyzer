import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  // Relative asset paths also work when deployed to a GitHub Pages project URL.
  base: './',
});
