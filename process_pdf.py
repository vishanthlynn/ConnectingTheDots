import os
import json
import re
import statistics
import logging
import argparse
from datetime import datetime
import unicodedata
from typing import List, Dict, Tuple
from pathlib import Path

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBox, LTTextLine, LTChar

# Optional offline small embedding model (to be downloaded once and stored locally)
try:
    from sentence_transformers import SentenceTransformer, util
    model_path = os.path.expanduser("~/.cache/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2")
    if Path(model_path).exists():
        EMBEDDING_MODEL = SentenceTransformer(model_path)
    else:
        logging.warning("Model files not found; falling back to keyword scoring.")
        EMBEDDING_MODEL = None
except ImportError:
    EMBEDDING_MODEL = None

INPUT_DIR = "input"
OUTPUT_DIR = "output"
PERSONA_CONFIG_FILE = os.path.join(INPUT_DIR, "persona.json")

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class PDFProcessor:
    def process_pdf_outline(self, filename: str) -> dict:
        pdf_path = os.path.join(INPUT_DIR, filename)
        title = self._extract_title(pdf_path)
        headings = self._detect_headings(pdf_path)
        return {"title": title, "outline": headings}

    def process_pdf_analysis(self, filename: str) -> dict:
        pdf_path = os.path.join(INPUT_DIR, filename)
        persona_config = self._load_persona_config()
        title = self._extract_title(pdf_path)
        headings = self._detect_headings(pdf_path)
        sections = self._extract_section_content(pdf_path, headings)
        scored_sections = self._score_sections_for_persona(sections, persona_config)
        output = self._format_1b_output(scored_sections, title, filename, persona_config)
        return output

    def _extract_title(self, pdf_path: str) -> str:
        title_candidate = {"text": os.path.basename(pdf_path), "score": 0}
        for page in extract_pages(pdf_path, maxpages=1):
            for element in page:
                if isinstance(element, LTTextBox):
                    text = element.get_text().strip()
                    if 3 < len(text.split()) < 15 and text:
                        avg_size, is_bold = self._get_text_features(element)
                        score = avg_size + (5 if is_bold else 0)
                        if score > title_candidate["score"]:
                            title_candidate["score"] = score
                            title_candidate["text"] = text
        return title_candidate["text"]

    def _detect_headings(self, pdf_path: str) -> List[Dict]:
        candidates = []
        for page_num, page in enumerate(extract_pages(pdf_path), 1):
            for element in page:
                if isinstance(element, LTTextBox):
                    text = element.get_text().strip()
                    if self._is_likely_heading(text):
                        avg_size, is_bold = self._get_text_features(element)
                        if avg_size > 11:
                            candidates.append({
                                "text": text, "size": avg_size, "bold": is_bold,
                                "page": page_num, "y_pos": element.y1
                            })
        return self._classify_headings_by_font(candidates)

    def _extract_section_content(self, pdf_path: str, headings: list) -> list:
        if not headings: return []
        headings.sort(key=lambda h: (h['page'], -h['y_pos']))

        sections = []
        text_by_page = {}
        for page_num, page in enumerate(extract_pages(pdf_path), 1):
            text_by_page[page_num] = []
            for element in page:
                if isinstance(element, LTTextBox):
                    text_by_page[page_num].append({
                        "text": element.get_text().strip(),
                        "y_pos": element.y1
                    })

        for i, heading in enumerate(headings):
            start_page = heading['page']
            start_y = heading['y_pos']
            end_page, end_y = float('inf'), -1
            current_level = {'H1': 3, 'H2': 2, 'H3': 1}.get(heading['level'], 0)

            for next_heading in headings[i+1:]:
                next_level = {'H1': 3, 'H2': 2, 'H3': 1}.get(next_heading['level'], 0)
                if next_level >= current_level:
                    end_page, end_y = next_heading['page'], next_heading['y_pos']
                    break

            content = []
            for page_num in range(start_page, min(end_page+1, max(text_by_page.keys())+1)):
                for elem in text_by_page[page_num]:
                    if (page_num > start_page or elem['y_pos'] < start_y) and \
                       (page_num < end_page or elem['y_pos'] > end_y) and \
                       elem['text'] not in [h['text'] for h in headings]:
                        content.append(elem['text'])

            sections.append({**heading, "content": " ".join(content)})

        return sections

    def _load_persona_config(self) -> dict:
        try:
            with open(PERSONA_CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            logging.warning("Using default persona config.")
            return {
                "persona": "Default Analyst",
                "job_to_be_done": "Find key sections.",
                "keywords": {"introduction": 10, "methodology": 20, "result": 30, "conclusion": 40}
            }

    def _score_sections_for_persona(self, sections: list, config: dict) -> list:
        job_text = config.get("job_to_be_done", "")
        scored = []
        if EMBEDDING_MODEL:
            try:
                job_emb = EMBEDDING_MODEL.encode(job_text, convert_to_tensor=True)
                for sec in sections:
                    sec_emb = EMBEDDING_MODEL.encode(sec['content'], convert_to_tensor=True)
                    score = float(util.pytorch_cos_sim(job_emb, sec_emb))
                    sec['relevance'] = score
                    scored.append(sec)
            except Exception as e:
                logging.warning(f"Embedding model failed: {e}")
                EMBEDDING_MODEL = None
        if not EMBEDDING_MODEL:
            keywords = config.get("keywords", {})
            for sec in sections:
                score = sum(weight for kw, weight in keywords.items() if kw.lower() in sec['content'].lower())
                sec['relevance'] = score
                scored.append(sec)
        return sorted(scored, key=lambda x: x['relevance'], reverse=True)

    def _format_1b_output(self, sections: list, title: str, filename: str, config: dict) -> dict:
        output = {
            "metadata": {
                "input_documents": [filename],
                "persona": config.get("persona"),
                "job_to_be_done": config.get("job_to_be_done"),
                "processing_timestamp": datetime.utcnow().isoformat() + "Z"
            },
            "extracted_sections": [],
            "sub_section_analysis": []
        }

        for i, sec in enumerate(sections[:10]):
            output["extracted_sections"].append({
                "document": filename,
                "page_number": sec['page'],
                "section_title": sec['text'],
                "importance_rank": i + 1
            })

            sub_sentences = re.split(r'[.!?]', sec['content'])
            sub_sentences = [s.strip() for s in sub_sentences if len(s.strip().split()) > 3]
            if EMBEDDING_MODEL and sub_sentences:
                try:
                    job_emb = EMBEDDING_MODEL.encode(config.get("job_to_be_done", ""))
                    sent_emb = EMBEDDING_MODEL.encode(sub_sentences)
                    similarities = util.cos_sim(job_emb, sent_emb)[0].tolist()
                    scored_sentences = sorted(zip(sub_sentences, similarities), key=lambda x: x[1], reverse=True)
                    top_sub = [s[0] for s in scored_sentences[:3]]
                    refined_text = ". ".join(top_sub) + "."
                except Exception as e:
                    logging.warning(f"Sub-section embedding failed: {e}")
                    refined_text = "Error in generating refined text."
            else:
                top_sub = sub_sentences[:3]
                refined_text = ". ".join(top_sub) + "." if top_sub else "No detailed content available."

            output["sub_section_analysis"].append({
                "document": filename,
                "page_number": sec['page'],
                "refined_text": refined_text
            })
        return output

    def _is_likely_heading(self, text: str) -> bool:
        if not text or len(text.split()) > 20 or len(text) > 150: return False
        if text.endswith('.') or text.endswith(','): return False
        if re.match(r'^\d+$', text): return False
        if self._contains_non_latin(text): return True
        if re.match(r'^\d+(\.\d+)*\s+', text): return True
        return text.istitle() or text.isupper()

    def _contains_non_latin(self, text: str) -> bool:
        return any(unicodedata.name(char, '').startswith(('CJK', 'HIRAGANA', 'KATAKANA')) for char in text if char.isalpha())

    def _get_text_features(self, element: LTTextBox) -> Tuple[float, bool]:
        sizes, names = [], []
        for text_line in element:
            if isinstance(text_line, LTTextLine):
                for char in text_line:
                    if isinstance(char, LTChar) and char.get_text().strip():
                        sizes.append(char.size)
                        names.append(char.fontname)
        return (statistics.mean(sizes) if sizes else 0, any('bold' in n.lower() for n in names))

    def _classify_headings_by_font(self, candidates: list) -> list:
        if not candidates: return []
        sizes = [c['size'] for c in candidates]
        try:
            h1_threshold = statistics.quantiles(sizes, n=10)[-1]
            h2_threshold = statistics.quantiles(sizes, n=4)[-1]
        except statistics.StatisticsError:
            h1_threshold = h2_threshold = max(sizes)
        for c in candidates:
            if c['size'] >= h1_threshold * 0.99: c['level'] = 'H1'
            elif c['size'] >= h2_threshold: c['level'] = 'H2'
            else: c['level'] = 'H3'
        return [{"text": c['text'], "level": c['level'], "page": c['page'], "y_pos": c['y_pos']} for c in candidates]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['outline', 'analyze'], default='outline')
    args = parser.parse_args()

    processor = PDFProcessor()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]

    for pdf_file in pdf_files:
        try:
            logging.info(f"Processing '{pdf_file}' in '{args.mode}' mode...")
            if args.mode == 'outline':
                result = processor.process_pdf_outline(pdf_file)
                output_filename = pdf_file.replace('.pdf', '.json')
            else:
                result = processor.process_pdf_analysis(pdf_file)
                output_filename = pdf_file.replace('.pdf', '_analysis.json')

            with open(os.path.join(OUTPUT_DIR, output_filename), 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            logging.info(f"Successfully created output: {output_filename}")
        except Exception as e:
            logging.error(f"Failed to process {pdf_file}: {e}", exc_info=True)

if _name_ == "_main_":
    main()