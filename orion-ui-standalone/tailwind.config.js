/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class',
  content: ['./web/templates/**/*.html'],
  theme: {
    extend: {
      colors: {
        forge: {
          bg:        '#0f0f14',
          card:      '#1a1a24',
          border:    '#2a2a3a',
          accent:    '#6366f1',
          accent2:   '#818cf8',
          highlight: '#22d3ee',
          warn:      '#f59e0b',
          danger:    '#ef4444',
          success:   '#10b981',
          muted:     '#71717a',
        }
      }
    }
  },
  plugins: [],
}
