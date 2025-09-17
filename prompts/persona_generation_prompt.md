# Persona Generation Prompt (Profile + Resume → JSON)

## Objective
Generate two concise, high‑quality fields from a person’s profile and resume texts:
- needs (40–60 words)
- personality (40–60 words)

Return a single JSON object with exactly three keys: id, needs, personality.

## Input Format
You will receive a single message containing three sections:

ID: <string>
PROFILE:
<raw profile text>

RESUME:
<raw resume text>

The profile and resume are free‑form text as collected. They may overlap or differ in detail.

## Output Schema (strict)
Return ONLY a JSON object (no code fences, no commentary):
{
  "id": "<string>",
  "needs": "<40–60 words>",
  "personality": "<40–60 words>"
}

## Guidance
- Use BOTH PROFILE and RESUME; if one is sparse, rely more on the other.
- needs: summarize current objectives and constraints (what they seek, key challenges, timing, context). Be specific and practical.
- personality: describe communication style, collaboration traits, decision patterns, values, and work habits. Avoid clichés and fluff.
- Prefer concrete signals (domains, projects, scale, metrics) over buzzwords.
- Style: one paragraph per field, plain sentences, professional tone, no lists.
- Stay faithful to the inputs; avoid speculation. If uncertain, infer cautiously based on consistent signals.

## Quality Checklist (apply before finalizing)
1. Each field is 40–60 words, single paragraph, no lists.
2. Grounded in PROFILE and RESUME; no invented facts.
3. needs = concrete aims and constraints; personality = style and values.
4. Output is valid JSON with keys exactly: id, needs, personality.
