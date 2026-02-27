# Orion Forge — Website

A modern single-page website for the OrionForge Modular AI Identity Ecosystem.

## Features

- **Animated particle background** with WebGL-free Canvas 2D network
- **7 real agent profiles** — Orion, Elysia, Codex Animus, Vageta, Bender, Veilwalker, Carolina
- **10 runtime tools** from the agent-runtime tool registry
- **Interactive agent detail modals** with traits, values, tools, and technical specs
- **Dashboard preview** showing all 9 UI pages with launch link
- **5-layer identity injection pipeline** visualization
- **Supabase OAuth** — Google, GitHub, Discord, and email/password
- **Scroll-triggered reveal animations** with IntersectionObserver
- **Animated stat counters**
- **Glassmorphism design** with scroll progress indicator
- **Fully responsive** (mobile, tablet, desktop)

## Setup

### Quick Preview

Just open `index.html` in a browser. Everything works without a server.

### Supabase Auth Setup

1. Create a project at [supabase.com](https://supabase.com)
2. In your Supabase dashboard, go to **Authentication > Providers** and enable:
   - Google
   - GitHub
   - Discord
3. Copy your **Project URL** and **Anon Key** from **Settings > API**
4. Edit `index.html` and replace the config at the top of the `<script>` section:

```javascript
const SUPABASE_URL = 'https://YOUR_PROJECT_REF.supabase.co';
const SUPABASE_ANON_KEY = 'YOUR_ANON_KEY';
```

5. In Supabase **Authentication > URL Configuration**, add your website URL to the redirect allow list.

### Deploy to GitHub Pages

```bash
# From the repo root
git subtree push --prefix website origin gh-pages
```

Or on Vercel/Netlify, point to the `website/` directory.

## Customization

- **Colors:** Edit the CSS custom properties in `:root { ... }`
- **Agents:** Edit the `agents` array in the JavaScript section
- **Tools:** Edit the `tools` array in the JavaScript section
- **Branding:** Search for "Orion Forge" and replace as needed
