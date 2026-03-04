# StGB-Agent

Ein KI-gestuetzter **strafrechtlicher Analyse-Agent**, der Anklageschriften und andere strafrechtliche Dokumente im **Gutachtenstil** analysiert und zusaetzlich als **Anklageschrift-Debugger** auf innere Fehler prueft. Der Agent kombiniert OCR-Textextraktion, juristische Kommentarliteratur (RAG) und aktuelle Rechtsprechung zu einem verteidigungsorientierten Ergebnis.

## Funktionsweise

Der StGB-Agent arbeitet als **Multi-Step-Pipeline** auf Basis von [LangGraph](https://github.com/langchain-ai/langgraph). Ein hochgeladenes PDF durchlaeuft folgende Verarbeitungsschritte:

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
│  2. Fakten-Extraktion│ Trennung: Tatsachen vs Behauptungen/Wertungen
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  3. Dokumentanalyse  │ Teilfragen + issues_to_check (Widerspruch/Belege)
└────────┬────────────┘
         ▼
┌─────────────────────────────────────────────────────┐
│  Schleife: Fuer jede Teilfrage (3-7 Fragen)          │
│                                                      │
│  4. RAG-Retrieval ──► StGB/StPO-Kommentare (RAGIE)  │
│  5. Websuche ────────► Aktuelle Rechtsprechung       │
│  6. Zwischensynthese ► Gutachten + Belegmapping      │
│                                                      │
└────────┬────────────────────────────────────────────┘
         ▼
┌─────────────────────┐
│  7. Allegation-Check │  Behauptung→Fakten-Mapping, Support-Staerke,
│                      │  Widersprueche, Zirkelschluss-Hinweise
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  8. Gesamtgutachten  │  Gutachten + Fehler-/Widerspruchsbericht
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  9. Red-Team-Pass    │  verpasste Probleme, fehlende Zitate,
│                      │  ueberkonfidente Aussagen
└─────────────────────┘
```

### Die Pipeline im Detail

| Schritt | Node | Beschreibung |
|---------|------|-------------|
| 1 | `extract_pdf` | Extrahiert Text aus dem hochgeladenen PDF mittels **Mistral OCR** |
| 2 | `fact_extraction` | Trennt **Fakten** von **Behauptungen/Wertungen** und extrahiert Entitaeten |
| 3 | `analyze_document` | Erstellt 3-7 Teilfragen plus `issues_to_check` (Widerspruch, Belegluecken, Spekulation, Prozessuales) |
| 4 | `process_sub_questions` | Fuehrt je Teilfrage parallel RAG + Websuche + Synthese mit Fakten/Behauptungs-Kontext aus |
| 5 | `allegation_validation` | Bewertet jede Behauptung gegen Fakten (`strong|medium|weak|none`) und erkennt Widersprueche |
| 6 | `final_synthesis` | Fuehrt alles zu Gutachtenstil + strukturiertem Fehler-/Widerspruchsbericht zusammen |
| 7 | `red_team` | Prueft finalen Text adversarial auf verpasste Punkte, fehlende Zitate, Ueberkonfidenz |
| 8 | `respond` | Formatiert finale Antwort inkl. compact citations + Sicherheits-Hinweis |

### Ausgabeformat

Die Endausgabe enthaelt immer:

1. Gutachtenstil-Wuerdigung (pro Tatkomplex)
2. Separaten **Fehler-/Widerspruchsbericht** mit:
- unbelegten Behauptungen
- inneren Widerspruechen
- Zirkelschluessen/Spekulationen
- prozessualen Auffaelligkeiten (wenn ableitbar)
3. Kompakte Citations-Sektion
4. Hinweis: **Keine Rechtsberatung, anwaltlich verifizieren**

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
│       ├── fact_extraction.py     # Fakten/Behauptungen + Entitaeten
│       ├── analyze_document.py    # Dokumentanalyse + Teilfragen-Zerlegung
│       ├── retrieve_rag.py        # RAG-Retrieval aus RAGIE
│       ├── search_case_law.py     # Rechtsprechungssuche (Google Search)
│       ├── synthesize_step.py     # Zwischenanalyse pro Teilfrage
│       ├── allegation_validation.py # Behauptungsvalidierung + Widerspruchsdetektion
│       ├── final_synthesis.py     # Gesamtgutachten-Erstellung
│       ├── red_team.py            # Adversarialer QA-Pass
│       └── respond.py             # Antwort-Formatierung
├── prompts/
│   ├── fact_extraction.py         # Prompt fuer Fakt/Behauptungs-Trennung
│   ├── allegation_validation.py   # Prompt fuer Behauptungsvalidierung
│   ├── analyze_document.py        # System-Prompts für Dokumentanalyse
│   ├── synthesize.py              # System-Prompt für Gutachtenstil
│   ├── final_analysis.py          # System-Prompt fuer Gesamtanalyse + Fehlerbericht
│   └── red_team.py                # Prompt fuer Red-Team-Addendum
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

Die Anwendung startet unter `http://localhost:8000`. Lade ein PDF hoch (z.B. eine Anklageschrift) und der Agent erstellt automatisch ein Gutachten plus Fehler-/Widerspruchsbericht.

## Deployment auf Railway

Dieses Repo ist fuer Railway vorbereitet (`Dockerfile`, `railway.toml`, `start.sh`).

1. Repository nach GitHub pushen.
2. In Railway: `New Project` -> `Deploy from GitHub repo`.
3. Unter `Variables` diese Secrets setzen:
   - `GEMINI_API_KEY`
   - `MISTRAL_API_KEY`
   - `RAGIE_API_KEY`
   - optional: `MODEL_NAME`
4. Deploy starten.
5. Railway vergibt die URL automatisch; `PORT` wird von Railway gesetzt und durch `start.sh` genutzt.

Hinweis: Upload-Dateien liegen waehrend der Laufzeit unter `.files`. Fuer persistente Uploads ueber Deploys hinweg einen Volume-Mount nutzen.

## Nutzung

1. Öffne die Chat-Oberfläche im Browser
2. Lade ein strafrechtliches PDF-Dokument hoch (Anklageschrift, Urteil, Beschluss, etc.)
3. **Optional**: Stelle eine konkrete Frage zum Dokument (z.B. "Liegt ein Rücktritt vom Versuch vor?")
4. Der Agent durchläuft die Pipeline und zeigt den Fortschritt in Echtzeit an
5. Am Ende erhaeltst du ein vollstaendiges Gutachten **und** einen separaten Fehler-/Widerspruchsbericht

## Test

Smoke-Test (ohne externe API-Aufrufe):

```bash
python3 -m unittest tests/test_anklageschrift_debugger_smoke.py
```

## Lizenz

Dieses Projekt ist für den privaten Gebrauch bestimmt.
