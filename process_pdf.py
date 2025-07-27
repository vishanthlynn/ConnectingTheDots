import os
import json
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Tuple
import statistics
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBox, LTTextLine, LTChar

# Optional embedding model
try:
    from sentence_transformers import SentenceTransformer, util
    model_path = os.path.expanduser("~/.cache/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2")
    EMBEDDING_MODEL = SentenceTransformer(model_path) if os.path.exists(model_path) else None
except ImportError:
    EMBEDDING_MODEL = None

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


class PDFReader:
    def read_pdf(self, filepath: str) -> List[Dict]:
        sections = []
        for page_num, page in enumerate(extract_pages(filepath), 1):
            for element in page:
                if isinstance(element, LTTextBox):
                    text = element.get_text().strip()
                    if text:
                        avg_size, is_bold = self._get_text_features(element)
                        sections.append({
                            'text': text,
                            'page': page_num,
                            'size': avg_size,
                            'bold': is_bold,
                            'y_pos': element.y1
                        })
        return sections

    def _get_text_features(self, element: LTTextBox) -> Tuple[float, bool]:
        sizes, names = [], []
        for line in element:
            if isinstance(line, LTTextLine):
                for char in line:
                    if isinstance(char, LTChar) and char.get_text().strip():
                        sizes.append(char.size)
                        names.append(char.fontname)
        return (statistics.mean(sizes) if sizes else 0, any('bold' in n.lower() for n in names))


class SectionExtractor:
    def extract_sections(self, pdf_content: List[Dict]) -> List[Dict]:
        headings = self._identify_headings(pdf_content)
        return self._group_content_by_headings(pdf_content, headings)

    def _identify_headings(self, content: List[Dict]) -> List[Dict]:
        headings = []
        for item in content:
            if self._is_likely_heading(item['text']):
                headings.append({
                    'text': item['text'],
                    'page': item['page'],
                    'size': item['size'],
                    'bold': item['bold'],
                    'y_pos': item['y_pos']
                })
        return self._classify_headings(headings)

    def _is_likely_heading(self, text: str) -> bool:
        if not text or len(text.split()) > 20 or len(text) > 150:
            return False
        if text.endswith('.') or text.endswith(',') or text.isdigit():
            return False
        if any(char.isdigit() for char in text[:3]):
            return True
        if text.istitle() or text.isupper():
            return True
        keywords = ['introduction', 'conclusion', 'method', 'result', 'discussion', 'abstract']
        return any(k in text.lower() for k in keywords)

    def _classify_headings(self, headings: List[Dict]) -> List[Dict]:
        if not headings:
            return []
        sizes = [h['size'] for h in headings]
        try:
            h1_thresh = statistics.quantiles(sizes, n=10)[-1]
            h2_thresh = statistics.quantiles(sizes, n=4)[-1]
        except statistics.StatisticsError:
            h1_thresh = h2_thresh = max(sizes)
        for h in headings:
            if h['size'] >= h1_thresh * 0.99:
                h['level'] = 'H1'
            elif h['size'] >= h2_thresh:
                h['level'] = 'H2'
            else:
                h['level'] = 'H3'
        return headings

    def _group_content_by_headings(self, content: List[Dict], headings: List[Dict]) -> List[Dict]:
        if not headings:
            return []
        headings.sort(key=lambda h: (h['page'], -h['y_pos']))
        sections = []
        for i, heading in enumerate(headings):
            start_page, start_y = heading['page'], heading['y_pos']
            end_page, end_y = float('inf'), -1
            current_level = {'H1': 3, 'H2': 2, 'H3': 1}.get(heading['level'], 0)
            for nh in headings[i+1:]:
                if {'H1': 3, 'H2': 2, 'H3': 1}.get(nh['level'], 0) >= current_level:
                    end_page, end_y = nh['page'], nh['y_pos']
                    break
            body = []
            for item in content:
                if (item['page'] > start_page or item['y_pos'] < start_y) and \
                   (item['page'] < end_page or item['y_pos'] > end_y) and \
                   item['text'] not in [h['text'] for h in headings]:
                    body.append(item['text'])
            sections.append({
                'title': heading['text'],
                'level': heading['level'],
                'page': heading['page'],
                'content': ' '.join(body),
                'file': 'document.pdf'
            })
        return sections


class RelevanceEvaluator:
    def __init__(self, persona: str, job_to_be_done: str):
        self.persona = persona
        self.job_to_be_done = job_to_be_done

    def score_sections(self, sections: List[Dict]) -> List[Dict]:
        return self._score_with_embeddings(sections) if EMBEDDING_MODEL else self._score_with_keywords(sections)

    def _score_with_embeddings(self, sections: List[Dict]) -> List[Dict]:
        try:
            job_emb = EMBEDDING_MODEL.encode(self.job_to_be_done, convert_to_tensor=True)
            section_texts = [f"{s['title']} {s['content']}" for s in sections]
            section_embs = EMBEDDING_MODEL.encode(section_texts, convert_to_tensor=True)
            similarities = util.pytorch_cos_sim(job_emb, section_embs)[0]
            for i, s in enumerate(sections):
                s['importance_score'] = round(float(similarities[i]) * 10, 1)
            return sorted(sections, key=lambda x: x['importance_score'], reverse=True)
        except Exception as e:
            logging.warning(f"Embedding scoring failed: {e}")
            return self._score_with_keywords(sections)

    def _score_with_keywords(self, sections: List[Dict]) -> List[Dict]:
        keywords = self._extract_keywords(self.job_to_be_done)
        for s in sections:
            text = f"{s['title']} {s['content']}".lower()
            score = sum(2 for kw in keywords if kw in text)
            if self.job_to_be_done.lower() in text:
                score += 5
            score += {'H1': 3, 'H2': 2, 'H3': 1}.get(s['level'], 0)
            s['importance_score'] = min(score, 10)
        return sorted(sections, key=lambda x: x['importance_score'], reverse=True)

    def _extract_keywords(self, text: str) -> List[str]:
        stop_words = {'the', 'and', 'for', 'with', 'that', 'this', 'from', 'into', 'about', 'but'}
        words = text.lower().split()
        return [w for w in words if w not in stop_words and len(w) > 3][:10]


class OutputWriter:
    def write_output(self, sections: List[Dict], persona: str, job: str, filename: str, output_dir: str) -> None:
        metadata = {
            "file_processed": filename,
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": datetime.utcnow().isoformat() + "Z"
        }

        extracted_sections = []
        sub_section_analysis = []

        for rank, section in enumerate(sections[:10], start=1):
            extracted_sections.append({
                "document": section['file'],
                "page_number": section['page'],
                "section_title": section['title'],
                "importance_rank": rank
            })

            refined_text = self._generate_summary(section['content'], job)
            sub_section_analysis.append({
                "document": section['file'],
                "page_number": section['page'],
                "refined_text": refined_text
            })

        output = {
            "metadata": metadata,
            "extracted_sections": extracted_sections,
            "sub_section_analysis": sub_section_analysis
        }

        out_file = os.path.join(output_dir, filename.replace('.pdf', '_analysis.json'))
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logging.info(f"Output written to {out_file}")

    def _generate_summary(self, content: str, job: str) -> str:
        if not content:
            return "No content available."

        sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 10]
        if not sentences:
            return content[:200] + "..."

        if EMBEDDING_MODEL:
            try:
                job_emb = EMBEDDING_MODEL.encode(job, convert_to_tensor=True)
                sent_embs = EMBEDDING_MODEL.encode(sentences, convert_to_tensor=True)
                sims = util.pytorch_cos_sim(job_emb, sent_embs)[0]
                top_k = sims.topk(k=min(3, len(sentences))).indices
                top_sentences = [sentences[i] for i in top_k]
                return ' '.join(s + '.' for s in top_sentences)
            except Exception as e:
                logging.warning(f"Failed summary embedding: {e}")

        return '. '.join(sentences[:3]) + '.'


def run_analysis(input_dir: str, output_dir: str, persona: str, job: str):
    os.makedirs(output_dir, exist_ok=True)
    pdf_reader = PDFReader()
    extractor = SectionExtractor()
    evaluator = RelevanceEvaluator(persona, job)
    writer = OutputWriter()

    pdfs = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    if not pdfs:
        print("❌ No PDFs found in input directory.")
        return

    start = datetime.now()
    print(f"🕐 Started: {start.strftime('%H:%M:%S')}")

    for i, file in enumerate(pdfs, 1):
        print(f"📄 [{i}/{len(pdfs)}] Processing {file}...")
        try:
            content = pdf_reader.read_pdf(os.path.join(input_dir, file))
            sections = extractor.extract_sections(content)
            for s in sections:
                s['file'] = file
            scored = evaluator.score_sections(sections)
            writer.write_output(scored, persona, job, file, output_dir)
        except Exception as e:
            logging.error(f"Failed to process {file}: {e}")

    end = datetime.now()
    print(f"✅ Completed in {end - start}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart PDF Analyzer")
    parser.add_argument('--persona', type=str, default='Default User', help='User persona')
    parser.add_argument('--job', type=str, default='Analyze document content', help='Job to be done')
    parser.add_argument('--input_dir', type=str, default='input', help='Input PDF directory')
    parser.add_argument('--output_dir', type=str, default='output', help='Output JSON directory')
    args = parser.parse_args()

    run_analysis(args.input_dir, args.output_dir, args.persona, args.job)
