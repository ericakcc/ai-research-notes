---
name: arxiv-paper-reader
description: Read and analyze arXiv papers directly within Claude Code. Use when the user asks to read, summarize, or analyze an arXiv paper, or when working with paper URLs from arxiv.org.
---

# arXiv Paper Reader

## Overview

This skill enables Claude to read and analyze arXiv research papers by converting arXiv URLs to their HTML versions and fetching the full content. No external dependencies required — uses Claude's built-in WebFetch tool.

## When to Use

- User shares an arXiv URL (abstract or PDF) and asks to read/summarize it
- User asks to analyze a paper from arXiv by ID (e.g. "read 2402.13616")
- User asks to compare multiple arXiv papers
- Working on research notes and need to extract key information from a paper

## URL Conversion Rules

arXiv papers have multiple URL formats. Always convert to the **HTML version** for best readability:

| Input Format | Example | Convert To |
|-------------|---------|------------|
| Abstract page | `https://arxiv.org/abs/2402.13616` | `https://arxiv.org/html/2402.13616` |
| PDF link | `https://arxiv.org/pdf/2402.13616` | `https://arxiv.org/html/2402.13616` |
| PDF with .pdf | `https://arxiv.org/pdf/2402.13616.pdf` | `https://arxiv.org/html/2402.13616` |
| Just an ID | `2402.13616` | `https://arxiv.org/html/2402.13616` |
| With version | `https://arxiv.org/abs/2402.13616v2` | `https://arxiv.org/html/2402.13616v2` |

**Rule**: Replace `/abs/` or `/pdf/` with `/html/` and remove `.pdf` suffix if present.

**Fallback**: If the HTML version returns an error (not all papers have HTML), fall back to the abstract page (`/abs/`) and use WebFetch to extract what's available.

## Workflow

### Reading a Single Paper

1. Convert the URL to HTML format using the rules above
2. Use `WebFetch` to fetch the HTML page
3. Extract and present:
   - **Title**
   - **Authors**
   - **Abstract**
   - **Key sections** (Introduction, Method, Results, Conclusion)
   - **Key figures/tables** descriptions if mentioned in text

### Summarizing a Paper

When asked to summarize, produce a structured summary:

```markdown
## Paper Summary: [Title]

**Authors**: [list]
**arXiv**: [ID with link]
**Year**: [year]

### Core Contribution
[1-2 sentences: what is the main novel contribution?]

### Method
[3-5 bullet points: how does it work?]

### Key Results
[2-3 bullet points: main experimental findings]

### Limitations
[1-2 bullet points if discussed]

### Relevance
[How this relates to the user's current research context]
```

### Comparing Papers

When asked to compare multiple papers, create a comparison table:

```markdown
| Aspect | Paper A | Paper B |
|--------|---------|---------|
| Core idea | ... | ... |
| Architecture | ... | ... |
| Dataset | ... | ... |
| Key metric | ... | ... |
| Strength | ... | ... |
| Weakness | ... | ... |
```

## Integration with Research Notes

When working within the `ai-research-notes` project, after reading a paper:

1. Ask the user if they want to save notes to the appropriate `topics/<topic>/notes/<phase>/` directory
2. Use the summary format above as the note template
3. Update `resources/learning_log.md` to check off the "First pass" item for that paper

## Tips

- For math-heavy papers, the HTML version preserves LaTeX rendering better than PDF extraction
- If a paper has supplementary material, check for appendix sections in the HTML
- When the user provides just a paper name (e.g. "read the YOLOv9 paper"), search for the arXiv ID in the project's README.md or notes first before asking
