// Bürgerkonto 2030 — Cloudflare Worker
//
// One Worker serves the demo page (static asset from ./public) and the assistant's two model routes:
//
//   POST /api/llm     { prompt, tier }                      -> the model's reply, parsed as JSON
//                     used by the page for the interpretation layer (free text -> catalog procedures,
//                     clarifying questions, answers about a running case) and for step derivation
//                     when no official page is fetched.
//   POST /api/lookup  { query, lang, profile, sources }     -> steps derived from live-fetched official pages
//                     1. candidate pages: the page's curated index (ids -> oesterreich.gv.at URLs) plus,
//                        if SEARCH_URL is configured, the site search of oesterreich.gv.at
//                     2. the top pages are fetched server-side (the browser can't: CSP) and reduced to text
//                     3. the model derives the steps FROM THAT TEXT and may only cite the fetched URLs
//   GET  /api/health                                        -> { ok, provider, model }
//
// Model provider (see wrangler.toml / README):
//   - ANTHROPIC_API_KEY secret set  -> Anthropic Messages API (claude-haiku-5-5 for "quick", claude-sonnet-5-5 otherwise)
//   - otherwise the Workers AI binding (AI_MODEL, default @cf/meta/llama-3.3-70b-instruct-fp8-fast)
//   MODEL_PROVIDER = "auto" | "anthropic" | "workers-ai" forces one.
//
// Nothing personal is stored: requests are not logged by this code, and the page only sends an
// anonymised profile summary (citizenship, city, flags) — never names, numbers or addresses.

const SITE = "https://www.oesterreich.gv.at";
const MAX_PAGES = 3;              // official pages fetched per lookup
const MAX_CHARS_PER_PAGE = 6000;  // text kept per page
const MAX_PROMPT = 60000;         // characters accepted on /api/llm
const MAX_BODY = 200000;          // bytes accepted per request
const MAX_TOKENS = 2500;

export default {
  async fetch(req, env) {
    const url = new URL(req.url);
    if (!url.pathname.startsWith("/api/")) {
      // static assets are served before the Worker runs; this only answers unknown paths
      return env.ASSETS ? env.ASSETS.fetch(req) : new Response("Not found", { status: 404 });
    }
    if (req.method === "OPTIONS") return cors(new Response(null, { status: 204 }));
    if (url.pathname === "/api/health") {
      let provider = null, model = null;
      try { provider = providerOf(env); model = modelName(env, provider, "quick"); } catch (e) { /* unconfigured */ }
      return cors(json({ ok: !!provider, provider, model, lookup: !!env.SEARCH_URL ? "site search + index" : "index only" }));
    }
    if (req.method !== "POST") return cors(json({ error: "POST only" }, 405));
    if (Number(req.headers.get("content-length") || 0) > MAX_BODY) return cors(json({ error: "body too large" }, 413));
    let body;
    try { body = await req.json(); } catch { return cors(json({ error: "bad json" }, 400)); }

    try {
      if (url.pathname === "/api/llm") {
        const prompt = String(body.prompt || "");
        const tier = body.tier === "default" || body.tier === "complex" ? body.tier : "quick";
        if (!prompt.trim()) return cors(json({ error: "empty prompt" }, 400));
        if (prompt.length > MAX_PROMPT) return cors(json({ error: "prompt too long" }, 413));
        const text = await callModel(env, prompt, tier);
        return cors(json(parseJson(text)));
      }
      if (url.pathname === "/api/lookup") {
        const query = String(body.query || "").slice(0, 2000);
        const lang = body.lang === "de" ? "de" : "en";
        const profile = body.profile && typeof body.profile === "object" ? body.profile : {};
        const sources = body.sources && typeof body.sources === "object" ? body.sources : {};
        if (!query.trim()) return cors(json({ error: "empty query" }, 400));
        const candidates = await findPages(query, sources, env);
        const pages = (await Promise.all(candidates.slice(0, MAX_PAGES).map(fetchText))).filter(Boolean);
        const result = await derive(env, query, lang, profile, pages, sources);
        return cors(json(result));
      }
      return cors(json({ error: "unknown route" }, 404));
    } catch (e) {
      return cors(json({ error: String((e && e.message) || e) }, 502));
    }
  },
};

// ---------- model ----------
function providerOf(env) {
  const want = (env.MODEL_PROVIDER || "auto").toLowerCase();
  if (want === "anthropic" || (want === "auto" && env.ANTHROPIC_API_KEY)) {
    if (!env.ANTHROPIC_API_KEY) throw new Error("MODEL_PROVIDER=anthropic but the ANTHROPIC_API_KEY secret is not set");
    return "anthropic";
  }
  if (!env.AI) throw new Error("no model configured: set the ANTHROPIC_API_KEY secret or keep the [ai] binding in wrangler.toml");
  return "workers-ai";
}
function modelName(env, provider, tier) {
  if (provider === "anthropic") return tier === "quick" ? (env.ANTHROPIC_MODEL_QUICK || "claude-haiku-5-5") : (env.ANTHROPIC_MODEL || "claude-sonnet-5-5");
  return env.AI_MODEL || "@cf/meta/llama-3.3-70b-instruct-fp8-fast";
}
async function callModel(env, prompt, tier = "quick") {
  const provider = providerOf(env);
  const model = modelName(env, provider, tier);
  if (provider === "workers-ai") {
    const out = await env.AI.run(model, {
      messages: [
        { role: "system", content: "You are a precise assistant for Austrian public administration. Reply with exactly one JSON object and nothing else." },
        { role: "user", content: prompt },
      ],
      max_tokens: MAX_TOKENS,
    });
    return typeof out === "string" ? out : (out && out.response) || "";
  }
  const r = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "content-type": "application/json", "x-api-key": env.ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01" },
    body: JSON.stringify({ model, max_tokens: MAX_TOKENS, messages: [{ role: "user", content: prompt }] }),
  });
  if (!r.ok) throw new Error("model call failed: " + r.status + " " + (await r.text()).slice(0, 200));
  const d = await r.json();
  return (d.content || []).map((c) => c.text || "").join("");
}
function parseJson(text) {
  const s = String(text || "").replace(/```(?:json)?/gi, "");
  const a = s.indexOf("{"), b = s.lastIndexOf("}");
  if (a < 0 || b <= a) throw new Error("model did not return JSON");
  return JSON.parse(s.slice(a, b + 1));
}

// ---------- 1. candidate official pages ----------
async function findPages(query, sources, env) {
  const out = [];
  // a) site search, when a search URL template is configured (e.g. "https://www.oesterreich.gv.at/de/suche?q={q}")
  if (env.SEARCH_URL && env.SEARCH_URL.includes("{q}")) {
    try {
      const r = await fetch(env.SEARCH_URL.replace("{q}", encodeURIComponent(query)), { headers: UA, cf: { cacheTtl: 600 } });
      const html = await r.text();
      const re = /href="(?:https:\/\/www\.oesterreich\.gv\.at)?(\/de\/(?:themen|lebenslagen|landingpages)\/[^"#?]+)"/g;
      let m; const seen = new Set();
      while ((m = re.exec(html)) && out.length < 6) {
        const u = SITE + m[1];
        if (!seen.has(u)) { seen.add(u); out.push({ id: null, url: u }); }
      }
    } catch { /* search unavailable: the curated index still works */ }
  }
  // b) the page's curated index: word overlap between the query (plus German equivalents of common
  //    English words) and each source's id, title and URL path
  const words = query.toLowerCase().split(/[^a-zäöüß0-9]+/).filter((w) => w.length > 2);
  const q = [...new Set(words.flatMap((w) => [w, ...(SYN[w] || [])]))].filter((w) => w.length > 3);
  const scored = Object.entries(sources)
    .filter(([, s]) => s && typeof s.u === "string" && /^https:\/\/([a-z0-9-]+\.)*(gv\.at|fsw\.at|sozialministeriumservice\.at)\//.test(s.u))
    .map(([id, s]) => { const hay = (id + " " + (s.t || "") + " " + s.u.replace(/^https?:\/\/[^/]+/, "")).toLowerCase(); return { id, url: s.u, score: q.filter((w) => hay.includes(w)).length }; })
    .filter((x) => x.score > 0)
    .sort((a, b) => b.score - a.score);
  for (const s of scored.slice(0, 3)) if (!out.some((o) => o.url === s.url)) out.push(s);
  return out;
}

// English query words -> German words used in the official titles and URLs
const SYN = { dog: ["hund"], dogs: ["hund"], passport: ["reisepass", "pass"], licence: ["fuehrerschein", "führerschein"], license: ["fuehrerschein", "führerschein"], driving: ["fuehrerschein"],
  car: ["kfz", "fahrzeug", "auto"], vehicle: ["kfz", "fahrzeug"], move: ["umzug", "anmeldung", "meldezettel"], moving: ["umzug", "anmeldung", "meldezettel"], address: ["meldezettel", "anmeldung", "ummeldung"], residence: ["wohnsitz", "meldezettel"],
  register: ["anmeldung", "meldung"], deregister: ["abmeldung"], marriage: ["heirat", "ehe"], married: ["heirat"], wedding: ["heirat"], birth: ["geburt"], baby: ["geburt"], child: ["kind", "kinder"], children: ["kinder"],
  death: ["todesfall", "sterbe"], died: ["todesfall", "sterbe"], inheritance: ["verlassenschaft", "erbe"], kindergarten: ["kindergarten", "kinderbetreuung"], childcare: ["kinderbetreuung", "kindergarten"], name: ["namensaenderung", "namensänderung"],
  insurance: ["versicherung", "e-card", "ecard"], card: ["e-card", "ecard"], care: ["pflege"], carer: ["pflege"], nursing: ["pflege", "hauskrankenpflege"], disability: ["behindertenpass", "behinderung"], leave: ["pflegekarenz", "karenz"],
  parking: ["parkpickerl", "parken"], tax: ["steuer"], unemployed: ["arbeitslos", "ams"], job: ["arbeit", "ams"], citizenship: ["staatsbuergerschaft", "staatsbürgerschaft"], lost: ["verlust", "verloren"], stolen: ["diebstahl"] };

// ---------- 2. fetch + reduce to text ----------
const UA = { "user-agent": "Mozilla/5.0 (compatible; citizen-account-2030 hackathon demo)", accept: "text/html" };
async function fetchText(c) {
  try {
    const r = await fetch(c.url, { headers: UA, cf: { cacheTtl: 3600 } });
    if (!r.ok) return null;
    const html = await r.text();
    const main = (html.match(/<main[\s\S]*?<\/main>/i) || [html])[0];
    const text = main
      .replace(/<script[\s\S]*?<\/script>|<style[\s\S]*?<\/style>|<nav[\s\S]*?<\/nav>/gi, " ")
      .replace(/<[^>]+>/g, " ")
      .replace(/&nbsp;/g, " ").replace(/&amp;/g, "&").replace(/&quot;/g, '"').replace(/&#\d+;/g, " ")
      .replace(/\s+/g, " ").trim().slice(0, MAX_CHARS_PER_PAGE);
    const title = ((html.match(/<title>([^<]*)<\/title>/i) || [])[1] || c.url).replace(/\s*[-|–].*$/, "").trim();
    return { ...c, title, text };
  } catch { return null; }
}

// ---------- 3. derive the steps from the fetched text ----------
const PROFILE_FIELDS = "name, born, citizen, phone, email, passportNum, passportValid, passportBy, licNum, ecardNum, insurer, street, city, since, household, plate, model, employer, rp, rpBorn, rpSvnr, rpAddress, rpInsurer, rpHospital";
async function derive(env, query, lang, profile, pages, sources) {
  const src = pages.map((p, i) => `[${i + 1}] ${p.title} — ${p.url}\n${p.text}`).join("\n\n");
  const prompt = `You derive Austrian public-administration procedures for a citizen-account assistant.
${pages.length ? "Use ONLY the official page texts below. Cite pages by their URL." : "No official page could be fetched; use only well-established rules you are sure of, and prefer \"inquiry\" steps over guessing."} Never invent URLs, fees or deadlines; if the pages don't state something, leave it out or use an "inquiry" step.
Citizen request: ${JSON.stringify(query)}
UI language: ${lang}
Citizen profile (data the state already holds, anonymised): ${JSON.stringify(profile)}
Profile fields that may be marked "review" (already on file): ${PROFILE_FIELDS}
Categories (page): identity, passport, licence, ecard, residence, household, care, vehicle

OFFICIAL PAGES
${src || "(none)"}

Reply with ONLY one JSON object:
{"title_en":"","title_de":"","desc_en":"","desc_de":"","page":"residence","channel":"online","confidence":0.0,"days":0,
 "source_urls":["https://..."],
 "steps":[{"kind":"review|lookup|input|inquiry|external|confirm|info","title_en":"","title_de":"","desc_en":"","desc_de":"",
           "fields":[...],"to_en":"","to_de":"","question_en":"","question_de":"","party_en":"","party_de":"","fee":"free","source_url":"https://..."}]}
Rules: 4-7 steps, chronological; exactly one "confirm". The AGENT does the finding out: never write a step that tells the citizen to find, check, look up, ask or contact anyone for information.
- "lookup": a concrete fact taken from the pages (competent authority and how to apply, deadline, documents, fee), one topic per step, quoted precisely in desc with its source_url. If a third party must supply it (a landlord, a hospital, an insurer), name it in party_en/party_de; the agent requests it and waits.
- "input": only facts the state cannot know; fields are {k,label_en,label_de,type:text|date|select|file,options:"en|de;en|de"}.
- "external": only acts the citizen must perform in person (sign, appear, pay at a counter, hand over an item); if instead another person confirms something (landlord, second parent, employer), set party_en/party_de and the agent requests it.
- "inquiry": when the pages do not state a requirement: to_en/to_de = the authority, question_en/question_de = one precise question; the citizen approves sending it. Use this instead of guessing.
- "channel": "online" only if this can really be submitted online today (an online form or portal of the authority), "appointment" if the citizen must appear in person, "paper" otherwise.
- "review" fields are names from the list above. Plain language (B1), both languages. confidence reflects how well the pages cover the request. If the request is outside Austrian public administration, return {"error":"..."}.`;

  const text = await callModel(env, prompt, "default");
  const j = parseJson(text);
  if (j.error) return j;

  // map URLs back to the page's source ids (so the demo renders its known links); fetched pages become new ids
  const byUrl = {};
  for (const [id, s] of Object.entries(sources)) if (s && s.u) byUrl[s.u] = id;
  const extra = {};
  const idFor = (u) => {
    if (!u) return null;
    if (byUrl[u]) return byUrl[u];
    const p = pages.find((x) => x.url === u);
    if (!p) return null;
    const id = "live_" + Object.keys(extra).length;
    extra[id] = { t: p.title, u };
    byUrl[u] = id;
    return id;
  };
  j.sources = (j.source_urls || []).map(idFor).filter(Boolean);
  for (const s of j.steps || []) { s.source = idFor(s.source_url); delete s.source_url; }
  delete j.source_urls;
  j.new_sources = extra;           // the page merges these into its SOURCES before registering the procedure
  j.fetched = pages.map((p) => p.url);
  return j;
}

const json = (o, status = 200) => new Response(JSON.stringify(o), { status, headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" } });
const cors = (res) => { res.headers.set("access-control-allow-origin", "*"); res.headers.set("access-control-allow-headers", "content-type"); res.headers.set("access-control-allow-methods", "POST, GET, OPTIONS"); return res; };
