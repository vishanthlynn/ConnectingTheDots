import os
import json
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Tuple
import statistics
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBox, LTTextLine, LTChar

# Optional embedding model for relevance scoring
try:
    from sentence_transformers import SentenceTransformer, util
    model_path = os.path.expanduser("~/.cache/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2")
    if os.path.exists(model_path):
        EMBEDDING_MODEL = SentenceTransformer(model_path)
    else:
        logging.warning("Model not found locally; using keyword-based scoring")
        EMBEDDING_MODEL = None
except ImportError:
    EMBEDDING_MODEL = None

INPUT_DIR = "input"
OUTPUT_DIR = "output"

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class PDFReader:
    def read_pdf(self, filepath: str) -> Dict:
        """Read PDF and extract text content with layout information"""
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
        for text_line in element:
            if isinstance(text_line, LTTextLine):
                for char in text_line:
                    if isinstance(char, LTChar) and char.get_text().strip():
                        sizes.append(char.size)
                        names.append(char.fontname)
        return (statistics.mean(sizes) if sizes else 0, any('bold' in n.lower() for n in names))

class SectionExtractor:
    def extract_sections(self, pdf_content: List[Dict]) -> List[Dict]:
        """Extract meaningful sections from PDF content"""
        headings = self._identify_headings(pdf_content)
        sections = self._group_content_by_headings(pdf_content, headings)
        return sections

    def _identify_headings(self, content: List[Dict]) -> List[Dict]:
        """Identify potential headings based on font size and text patterns"""
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
        """Determine if text is likely a heading"""
        if not text or len(text.split()) > 20 or len(text) > 150:
            return False
        if text.endswith('.') or text.endswith(','):
            return False
        if text.isdigit():
            return False
        # Check for numbered headings
        if any(char.isdigit() for char in text[:3]):
            return True
        # Check for title case or all caps
        if text.istitle() or text.isupper():
            return True
        # Check for common heading patterns
        heading_keywords = ['introduction', 'conclusion', 'method', 'result', 'discussion', 'abstract']
        if any(keyword in text.lower() for keyword in heading_keywords):
            return True
        return False

    def _classify_headings(self, headings: List[Dict]) -> List[Dict]:
        """Classify headings into H1, H2, H3 based on font size"""
        if not headings:
            return []
        
        sizes = [h['size'] for h in headings]
        try:
            h1_threshold = statistics.quantiles(sizes, n=10)[-1]
            h2_threshold = statistics.quantiles(sizes, n=4)[-1]
        except statistics.StatisticsError:
            h1_threshold = h2_threshold = max(sizes)
        
        for heading in headings:
            if heading['size'] >= h1_threshold * 0.99:
                heading['level'] = 'H1'
            elif heading['size'] >= h2_threshold:
                heading['level'] = 'H2'
            else:
                heading['level'] = 'H3'
        
        return headings

    def _group_content_by_headings(self, content: List[Dict], headings: List[Dict]) -> List[Dict]:
        """Group content under their respective headings"""
        if not headings:
            return []
        
        sections = []
        headings.sort(key=lambda h: (h['page'], -h['y_pos']))
        
        for i, heading in enumerate(headings):
            start_page = heading['page']
            start_y = heading['y_pos']
            
            # Find end boundary
            end_page, end_y = float('inf'), -1
            current_level = {'H1': 3, 'H2': 2, 'H3': 1}.get(heading['level'], 0)
            
            for next_heading in headings[i+1:]:
                next_level = {'H1': 3, 'H2': 2, 'H3': 1}.get(next_heading['level'], 0)
                if next_level >= current_level:
                    end_page, end_y = next_heading['page'], next_heading['y_pos']
                    break
            
            # Collect content for this section
            section_content = []
            for item in content:
                if (item['page'] > start_page or item['y_pos'] < start_y) and \
                   (item['page'] < end_page or item['y_pos'] > end_y) and \
                   item['text'] not in [h['text'] for h in headings]:
                    section_content.append(item['text'])
            
            sections.append({
                'title': heading['text'],
                'level': heading['level'],
                'page': heading['page'],
                'content': ' '.join(section_content),
                'file': 'document.pdf'  # Will be set by main function
            })
        
        return sections

class RelevanceEvaluator:
    def __init__(self, persona: str, job_to_be_done: str):
        self.persona = persona
        self.job_to_be_done = job_to_be_done

    def score_sections(self, sections: List[Dict]) -> List[Dict]:
        """Score sections based on relevance to persona and job"""
        if EMBEDDING_MODEL:
            return self._score_with_embeddings(sections)
        else:
            return self._score_with_keywords(sections)

    def _score_with_embeddings(self, sections: List[Dict]) -> List[Dict]:
        """Score using sentence transformers"""
        try:
            job_embedding = EMBEDDING_MODEL.encode(self.job_to_be_done, convert_to_tensor=True)
            
            for section in sections:
                # Combine title and content for scoring
                section_text = f"{section['title']} {section['content']}"
                section_embedding = EMBEDDING_MODEL.encode(section_text, convert_to_tensor=True)
                similarity = float(util.pytorch_cos_sim(job_embedding, section_embedding))
                section['importance_score'] = round(similarity * 10, 1)
            
            return sorted(sections, key=lambda x: x['importance_score'], reverse=True)
        except Exception as e:
            logging.warning(f"Embedding scoring failed: {e}")
            return self._score_with_keywords(sections)

    def _score_with_keywords(self, sections: List[Dict]) -> List[Dict]:
        """Score using keyword matching"""
        keywords = self._extract_keywords(self.job_to_be_done)
        
        for section in sections:
            score = 0
            section_text = f"{section['title']} {section['content']}".lower()
            
            # Score based on keyword matches
            for keyword in keywords:
                if keyword in section_text:
                    score += 2
            
            # Bonus for exact phrase matches
            if self.job_to_be_done.lower() in section_text:
                score += 5
            
            # Bonus for heading level (H1 > H2 > H3)
            level_bonus = {'H1': 3, 'H2': 2, 'H3': 1}.get(section['level'], 0)
            score += level_bonus
            
            section['importance_score'] = min(score, 10)  # Cap at 10
        
        return sorted(sections, key=lambda x: x['importance_score'], reverse=True)

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        # Simple keyword extraction - can be enhanced
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = text.lower().split()
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        return keywords[:10]  # Top 10 keywords

class OutputWriter:
    def write_output(self, sections: List[Dict], persona: str, job_to_be_done: str, filename: str) -> None:
        """Write structured JSON output for a single PDF"""
        output = {
            "metadata": {
                "file_processed": filename,
                "persona": persona,
                "job_to_be_done": job_to_be_done,
                "processing_timestamp": datetime.utcnow().isoformat() + "Z"
            },
            "extracted_sections": []
        }
        
        for section in sections[:10]:  # Top 10 sections
            output["extracted_sections"].append({
                "file": section['file'],
                "section_title": section['title'],
                "content": section['content'][:500] + "..." if len(section['content']) > 500 else section['content'],
                "summary": self._generate_summary(section['content']),
                "importance_score": section['importance_score']
            })
        
        # Create output filename based on input PDF name
        output_filename = filename.replace('.pdf', '_analysis.json')
        output_file = os.path.join(OUTPUT_DIR, output_filename)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Output written to {output_file}")

    def _generate_summary(self, content: str) -> str:
        """Generate a simple summary of the content"""
        if not content:
            return "No content available"
        
        # Simple summary: first sentence or first 100 characters
        sentences = content.split('.')
        if sentences and sentences[0].strip():
            summary = sentences[0].strip() + "."
            return summary if len(summary) <= 200 else summary[:200] + "..."
        
        return content[:200] + "..." if len(content) > 200 else content

def get_user_input():
    """Get persona and job from user input"""
    print("\n" + "="*50)
    print("🤖 SMART DOCUMENT ANALYZER")
    print("="*50)
    
    print("\n👤 Please enter the user persona:")
    print("Examples: 'Research Analyst', 'Student', 'Business Manager', 'Academic Researcher'")
    persona = input("Persona: ").strip()
    
    if not persona:
        persona = "Default User"
    
    print("\n💼 Please enter the job to be done:")
    print("Examples: 'Find market trends', 'Identify key concepts for exam', 'Analyze competitive landscape'")
    job_to_be_done = input("Job to be done: ").strip()
    
    if not job_to_be_done:
        job_to_be_done = "Analyze document content"
    
    print(f"\n✅ Configuration:")
    print(f"   Persona: {persona}")
    print(f"   Job: {job_to_be_done}")
    print("="*50 + "\n")
    
    return persona, job_to_be_done

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get user input for persona and job
    persona, job_to_be_done = get_user_input()
    
    # Record start time
    start_time = datetime.now()
    print(f"🕐 Processing started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    logging.info(f"Persona: {persona}")
    logging.info(f"Job to be done: {job_to_be_done}")
    
    # Initialize components
    pdf_reader = PDFReader()
    section_extractor = SectionExtractor()
    relevance_evaluator = RelevanceEvaluator(persona, job_to_be_done)
    output_writer = OutputWriter()
    
    # Process all PDFs
    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("❌ No PDF files found in the 'input' folder!")
        print("Please place your PDF files in the 'input' folder and try again.")
        return
    
    print(f"📄 Found {len(pdf_files)} PDF file(s) to process:")
    for pdf_file in pdf_files:
        print(f"   - {pdf_file}")
    print()
    
    # Process each PDF separately
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"🔄 Processing {i}/{len(pdf_files)}: {pdf_file}")
        logging.info(f"Processing {pdf_file}")
        
        pdf_path = os.path.join(INPUT_DIR, pdf_file)
        
        try:
            # Read PDF
            pdf_content = pdf_reader.read_pdf(pdf_path)
            
            # Extract sections
            sections = section_extractor.extract_sections(pdf_content)
            
            # Add file information
            for section in sections:
                section['file'] = pdf_file
            
            # Score and rank sections
            scored_sections = relevance_evaluator.score_sections(sections)
            
            # Write output for this PDF
            output_writer.write_output(scored_sections, persona, job_to_be_done, pdf_file)
            
            print(f"✅ Completed: {pdf_file}")
            
        except Exception as e:
            logging.error(f"Failed to process {pdf_file}: {e}")
            print(f"❌ Error processing {pdf_file}: {e}")
    
    # Record end time and calculate duration
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n🎉 Analysis complete! Check the 'output' folder for results.")
    print(f"📊 Generated {len(pdf_files)} analysis file(s):")
    for pdf_file in pdf_files:
        output_file = pdf_file.replace('.pdf', '_analysis.json')
        print(f"   - {output_file}")
    
    print(f"\n⏱️ Processing Summary:")
    print(f"   Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Total duration: {duration}")
    print(f"   Files processed: {len(pdf_files)}")

if __name__ == "__main__":
    main()
