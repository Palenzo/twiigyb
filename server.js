require('dotenv').config();
const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 5000;

const allowedOrigins = process.env.ALLOWED_ORIGINS
  ? process.env.ALLOWED_ORIGINS.split(',')
  : ['http://localhost:3000', 'http://localhost:3001'];
app.use(cors({ origin: allowedOrigins }));
app.use(express.json());

const MD_FILE = path.join(__dirname, 'ds_interview_guide.md');

// ── Helpers ────────────────────────────────────────────────────────────────

function slugify(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, '')
    .trim()
    .replace(/\s+/g, '-');
}

function extractSubsections(content) {
  const subsections = [];
  const lines = content.split('\n');
  let inFence = false;
  lines.forEach((line) => {
    if (/^```|^~~~/.test(line)) { inFence = !inFence; return; }
    if (inFence) return;
    const m = line.match(/^## (.+)$/);
    if (m) {
      const title = m[1].trim();
      subsections.push({ title, anchor: slugify(title) });
    }
  });
  return subsections;
}

function parseMarkdown(content) {
  const lines = content.split('\n');
  const sections = [];
  let currentTitle = null;
  let currentLines = [];
  let inFence = false;

  for (const line of lines) {
    if (/^```|^~~~/.test(line)) { inFence = !inFence; }

    if (!inFence && /^# [^#]/.test(line)) {
      // New H1 found
      if (currentTitle !== null) {
        const body = currentLines.join('\n').trim();
        sections.push({
          title: currentTitle,
          slug: slugify(currentTitle),
          content: body,
          subsections: extractSubsections(body),
        });
      }
      currentTitle = line.replace(/^# /, '').trim();
      currentLines = [];
    } else {
      currentLines.push(line);
    }
  }

  // Last section
  if (currentTitle !== null) {
    const body = currentLines.join('\n').trim();
    sections.push({
      title: currentTitle,
      slug: slugify(currentTitle),
      content: body,
      subsections: extractSubsections(body),
    });
  }

  return sections;
}

// Skip the document title and raw TOC
const SKIP_SLUGS = new Set([
  'complete-data-science-interview-preparation-guide',
  'table-of-contents',
]);

let _sections = null;
function getSections() {
  if (!_sections) {
    const raw = fs.readFileSync(MD_FILE, 'utf8');
    _sections = parseMarkdown(raw).filter((s) => !SKIP_SLUGS.has(s.slug));
  }
  return _sections;
}

// ── Routes ─────────────────────────────────────────────────────────────────

// GET /api/health  →  keepalive probe
app.get('/api/health', (_req, res) => res.json({ status: 'ok' }));

// GET /api/sections  →  lightweight list (no content)
app.get('/api/sections', (_req, res) => {
  try {
    const sections = getSections();
    res.json(sections.map(({ title, slug, subsections }) => ({ title, slug, subsections })));
  } catch (err) {
    console.error('getSections error:', err);
    res.status(500).json({ error: 'Failed to load sections' });
  }
});

// GET /api/sections/:slug  →  full section incl. content
app.get('/api/sections/:slug', (req, res) => {
  try {
    const section = getSections().find((s) => s.slug === req.params.slug);
    if (!section) return res.status(404).json({ error: 'Section not found' });
    res.json(section);
  } catch (err) {
    console.error('getSection error:', err);
    res.status(500).json({ error: 'Failed to load section' });
  }
});

// Production static serving
if (process.env.NODE_ENV === 'production') {
  app.use(express.static(path.join(__dirname, '../frontend/build')));
  app.get('*', (_req, res) =>
    res.sendFile(path.join(__dirname, '../frontend/build/index.html'))
  );
}

app.listen(PORT, () => {
  console.log(`🚀  Backend listening on http://localhost:${PORT}`);

  // Self-ping every 14 min to prevent Render free tier from sleeping
  if (process.env.NODE_ENV === 'production' && process.env.RENDER_EXTERNAL_URL) {
    const PING_URL = `${process.env.RENDER_EXTERNAL_URL}/api/health`;
    setInterval(() => {
      fetch(PING_URL).catch((e) => console.warn('Self-ping failed:', e.message));
    }, 14 * 60 * 1000);
    console.log(`⏰  Self-ping active → ${PING_URL}`);
  }
});
