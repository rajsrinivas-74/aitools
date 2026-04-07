# 🚀 AI Orchestrator — Executive Newsletter Bot (n8n)

## 📌 Overview

This workflow is an **AI-powered newsletter generation system** built using **n8n**.
It captures user queries from Slack or Telegram, intelligently decides whether to fetch **real-time news (NewsAPI)** or generate **AI-curated content**, and delivers a **clean executive HTML newsletter via email**.

---

## 🧭 End-to-End Flow

1. **Input Capture** → Slack / Telegram message
2. **Topic Extraction** → Parse command and extract topic
3. **AI Guardrail + Routing** → Validate + decide LLM vs Web
4. **Routing (IF Node)**

   * Web → NewsAPI
   * LLM → AI Agent
5. **Normalization + Ranking + Formatting** → Code node
6. **Email Delivery** → Gmail node sends newsletter

---

## 🧠 Architecture

```
Slack / Telegram
        ↓
Code (Extract Topic)
        ↓
AI Guardrail + Routing
        ↓
IF Node
   ┌───────────────┐
   │               │
NewsAPI         AI Agent
   │               │
   └──────→ Code (Normalize + Rank + HTML)
                     ↓
                 Gmail (Send Newsletter)
```

---

## 🧩 Key Components

### 1️⃣ Input Layer

* **Telegram Trigger**
* **Slack Trigger**

Captures user queries like:

```
@n8n_slack newsletter AI trends across world
```

---

### 2️⃣ Topic Extraction (Code Node)

Extracts topic using regex:

```javascript
/^@n8n_slack\s+(newsletter|news\s*letter|news|letter)\s+(.+)/i
```

Output:

```json
{
  "topic": "AI trends across world"
}
```

---

### 3️⃣ AI Guardrail + Routing (AI Agent)

Validates input and decides:

```json
{
  "input": "AI trends across world",
  "guardrail": false,
  "llm": false,
  "web_search": true
}
```

#### Decisions:

* `web_search = true` → NewsAPI
* `llm = true` → AI generation
* `guardrail = true` → block

---

### 4️⃣ Routing Logic (IF Node)

| Condition           | Action        |
| ------------------- | ------------- |
| `web_search = true` | Call NewsAPI  |
| `llm = true`        | Call AI Agent |

---

### 5️⃣ Data Sources

#### 🌐 NewsAPI (HTTP Node)

* Endpoint: `/v2/everything`
* Params:

  * `q` → topic
  * `sortBy` → publishedAt
  * `pageSize` → 5
  * `language` → en

---

#### 🤖 AI Agent (LLM)

Generates structured output similar to NewsAPI:

```json
{
  "status": "ok",
  "articles": [
    {
      "title": "...",
      "description": "...",
      "source": { "name": "Reuters" },
      "url": "...",
      "publishedAt": "2026-03-30T08:00:00Z"
    }
  ]
}
```

---

## ⚙️ Core Engine — Code Node (Normalization + Ranking + HTML)

This is the **heart of the workflow**.

### ✅ Responsibilities

* Parse AI output (string → JSON)
* Normalize AI + NewsAPI responses
* Rank articles (best story selection)
* Generate HTML newsletter

---

## 🧠 Ranking Engine

### 🎯 Factors Used

| Factor               | Weight |
| -------------------- | ------ |
| Recency              | High   |
| Source credibility   | Medium |
| Title quality        | Medium |
| Description richness | Medium |

### 🏆 Source Scoring

| Tier   | Sources                          |
| ------ | -------------------------------- |
| High   | Reuters, BBC, Bloomberg, FT, WSJ |
| Medium | CNBC, TechCrunch                 |
| Low    | Others                           |

---

## 📰 HTML Newsletter Features

* ⭐ Top Story (auto-selected)
* 🟠 Freshness tags:

  * 🔴 Breaking (≤1 day)
  * 🟠 Recent (≤3 days)
  * 🟡 This Week (≤7 days)
  * ⚪ Earlier
* 📅 Date format: `DD Mon`
* 👤 Personalized greeting
* 📊 Clean executive layout
* 📩 Email-friendly design

---

## 📤 Output (Gmail Node)

Uses:

```javascript
{{$json.newsletter.content}}
```

Sends:

* Fully formatted HTML newsletter
* Ready for inbox consumption

---

## 🧪 Example Input

```
@n8n_slack newsletter AI trends across world
```

---

## 🧪 Example Output

* Top Story with ranking
* 4 additional curated articles
* Styled HTML email

---

## ⚠️ Error Handling

* Invalid JSON → fallback response
* Missing articles → empty state handled
* Guardrail violation → blocked early

---

## 🔐 Guardrails

Prevents:

* PII exposure
* Sensitive data requests
* Jailbreak attempts
* Unsafe/NSFW queries

---

## 🚀 Key Strengths

* 🧠 Intelligent decision engine
* 🔀 Dual data sources (AI + API)
* 🧱 Modular architecture
* 📊 Smart ranking system
* 🎯 Production-ready pipeline

---

## 🔧 Setup Instructions

1. Import workflow JSON into n8n
2. Configure credentials:

   * OpenAI API
   * NewsAPI key
   * Gmail OAuth
   * Slack & Telegram tokens
3. Activate workflow
4. Trigger via Slack/Telegram

---

## 📈 Future Enhancements

* 🔁 Deduplication (AI + NewsAPI merge)
* 🎯 Personalization (role-based newsletters)
* 🧠 Confidence scoring
* 📊 Analytics dashboard
* 🌙 Dark/light theme
* 📚 Newsletter archive UI

---

## 💡 Final Insight

This workflow is not just automation —
it is a **mini AI Orchestrator platform**:

* Decision Engine
* Content Engine
* Ranking Engine
* Delivery Engine

---

## 👨‍💻 Author

**AI Orchestrator Initiative**
Enabling intelligent, automated decision-driven workflows.

---

## ⭐ Summary

👉 *User query → AI decides → fetch/generate → rank → format → deliver*

---