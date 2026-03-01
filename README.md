# StGB-Agent

Ein KI-gestützter **strafrechtlicher Analyse-Agent**, der Anklageschriften und andere strafrechtliche Dokumente automatisch im **Gutachtenstil** analysiert. Der Agent kombiniert OCR-Textextraktion, juristische Kommentarliteratur (RAG) und aktuelle Rechtsprechung zu einem umfassenden strafrechtlichen Gutachten.

## Funktionsweise

Der StGB-Agent arbeitet als **Multi-Step-Pipeline** auf Basis von [LangGraph](https://github.com/langchain-ai/langgraph). Ein hochgeladenes PDF durchläuft sechs spezialisierte Verarbeitungsschritte:

```
PDF Upload
    │
    ▼
┌─────────────────────┐
│  1. PDF-Extraktion   │  Mistral OCR extrahiert den Text aus dem PDF
│     (Mistral OCR)    │
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  2. Dokumentanalyse  │  Gemini analysiert den Sachverhalt, identifiziert
│     (Gemini)         │  Straftatbestände und zerlegt die Prüfung in Teilfragen
└────────┬────────────┘
         ▼
┌─────────────────────────────────────────────────────┐
│  Schleife: Für jede Teilfrage (3-7 Fragen)          │
│                                                      │
│  3. RAG-Retrieval ──► StGB/StPO-Kommentare (RAGIE)  │
│  4. Websuche ────────► Aktuelle Rechtsprechung       │
│     (Gemini + Google Search Grounding)               │
│  5. Zwischensynthese ► Teilanalyse im Gutachtenstil  │
│                                                      │
└────────┬────────────────────────────────────────────┘
         ▼
┌─────────────────────┐
│  6. Gesamtgutachten  │  Alle Teilanalysen werden zu einem vollständigen
│     (Gemini)         │  strafrechtlichen Gutachten zusammengeführt
└─────────────────────┘
```

### Die Pipeline im Detail

| Schritt | Node | Beschreibung |
|---------|------|-------------|
| 1 | `extract_pdf` | Extrahiert Text aus dem hochgeladenen PDF mittels **Mistral OCR** (Base64-Upload) |
| 2 | `analyze_document` | **Gemini** identifiziert Dokumenttyp, Angeklagte, relevante Paragraphen und zerlegt die Prüfung in 3-7 juristische Teilfragen |
| 3 | `retrieve_rag` | Für jede Teilfrage werden relevante Passagen aus **StGB/StPO-Kommentaren** über RAGIE abgerufen (Top-6, Reranking) |
| 4 | `search_case_law` | **Gemini mit Google Search Grounding** sucht aktuelle Rechtsprechung (Urteile, Beschlüsse) zur jeweiligen Rechtsfrage |
| 5 | `synthesize_step` | **Gemini** erstellt pro Teilfrage eine Analyse im Gutachtenstil (Obersatz, Definition, Subsumtion, Ergebnis) |
| 6 | `final_synthesis` | Alle Teilanalysen werden zu einem zusammenhängenden **Gesamtgutachten** mit Sachverhalt, strafrechtlicher Würdigung, Konkurrenzen und Gesamtergebnis zusammengeführt |

### Zwei Analyse-Modi

- **Vollanalyse**: PDF hochladen ohne Fragetext — der Agent prüft alle erkennbaren Straftatbestände umfassend
- **Fokussierte Analyse**: PDF hochladen mit konkreter Frage — der Agent konzentriert sich auf die spezifische Rechtsfrage (max. 3 Teilfragen)

## Tech-Stack

| Komponente | Technologie |
|-----------|------------|
| Agent-Framework | [LangGraph](https://github.com/langchain-ai/langgraph) + [LangChain](https://github.com/langchain-ai/langchain) |
| LLM | Google Gemini 3.1 Pro |
| OCR | Mistral OCR |
| RAG | [RAGIE](https://ragie.ai) (StGB/StPO-Kommentare) |
| Websuche | Gemini Google Search Grounding |
| Frontend | [Chainlit](https://chainlit.io) |

## Projektstruktur

```
StGB-Agent/
├── app.py                         # Chainlit-App (Chat-UI + Streaming)
├── agent/
│   ├── graph.py                   # LangGraph-Workflow Definition
│   ├── state.py                   # AgentState TypedDict
│   └── nodes/
│       ├── extract_pdf.py         # PDF-Textextraktion (Mistral OCR)
│       ├── analyze_document.py    # Dokumentanalyse + Teilfragen-Zerlegung
│       ├── retrieve_rag.py        # RAG-Retrieval aus RAGIE
│       ├── search_case_law.py     # Rechtsprechungssuche (Google Search)
│       ├── synthesize_step.py     # Zwischenanalyse pro Teilfrage
│       ├── final_synthesis.py     # Gesamtgutachten-Erstellung
│       └── respond.py             # Antwort-Formatierung
├── prompts/
│   ├── analyze_document.py        # System-Prompts für Dokumentanalyse
│   ├── synthesize.py              # System-Prompt für Gutachtenstil
│   └── final_analysis.py          # System-Prompt für Gesamtgutachten
├── services/
│   ├── gemini_client.py           # Gemini LLM + Google Search Grounding
│   ├── mistral_ocr.py             # Mistral OCR Client
│   └── ragie_client.py            # RAGIE RAG Client
├── utils/
│   └── config.py                  # Umgebungsvariablen-Verwaltung
├── requirements.txt
└── .env.local                     # API-Keys (nicht im Repo)
```

## Installation

### Voraussetzungen

- Python 3.11+
- API-Keys für Gemini, Mistral und RAGIE

### Setup

1. **Repository klonen**

```bash
git clone https://github.com/doedel123/STGB_Bot.git
cd STGB_Bot
```

2. **Virtuelle Umgebung erstellen**

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# oder: venv\Scripts\activate  # Windows
```

3. **Abhängigkeiten installieren**

```bash
pip install -r requirements.txt
```

4. **Umgebungsvariablen konfigurieren**

Erstelle eine `.env.local` Datei im Projektverzeichnis:

```env
GEMINI_API_KEY=dein-gemini-api-key
MISTRAL_API_KEY=dein-mistral-api-key
RAGIE_API_KEY=dein-ragie-api-key
MODEL_NAME=gemini-3.1-pro-preview   # optional, Standard: gemini-3.1-pro-preview
```

## Starten

```bash
chainlit run app.py
```

Die Anwendung startet unter `http://localhost:8000`. Lade ein PDF hoch (z.B. eine Anklageschrift) und der Agent erstellt automatisch ein strafrechtliches Gutachten.

## Nutzung

1. Öffne die Chat-Oberfläche im Browser
2. Lade ein strafrechtliches PDF-Dokument hoch (Anklageschrift, Urteil, Beschluss, etc.)
3. **Optional**: Stelle eine konkrete Frage zum Dokument (z.B. "Liegt ein Rücktritt vom Versuch vor?")
4. Der Agent durchläuft die Pipeline und zeigt den Fortschritt in Echtzeit an
5. Am Ende erhältst du ein vollständiges Gutachten mit Quellenangaben

## Lizenz

Dieses Projekt ist für den privaten Gebrauch bestimmt.
