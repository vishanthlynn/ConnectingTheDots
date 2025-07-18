import os
import json
import re
import statistics
from datetime import datetime
from collections import defaultdict
from io import BytesIO # Needed for some pdfminer internals that might use BytesIO
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LTRect, LTFigure, LTTextBox, LTTextLine

INPUT_DIR = "/app/input"
OUTPUT_DIR = "/app/output"
CONFIG_DIR = "/app/config"
PERSONA_CONFIG_PATH = os.path.join(CONFIG_DIR, "persona_config.json")
MAX_PROCESSING_TIME = 10 # seconds per document (for 50-page PDF)

class PDFProcessor:
    def __init__(self):
        self.persona_config = self._load_persona_config()
        self.font_stats = defaultdict(list)
        self.line_heights = []
        
    def _load_persona_config(self):
        """Load and validate persona configuration"""
        default_config = {
            "persona": "researcher",
            "job_to_be_done": "information_extraction",
            "keyword_weights": {
                "summary": 30, "conclusion": 25, "result": 20,
                "method": 20, "finding": 18, "recommendation": 15,
                "introduction": 10, "background": 8, "reference": 5
            },
            "section_weights": {
                "conclusion": 35, "results": 30, "methodology": 25,
                "introduction": 15, "appendix": 5, "other": 10
            }
        }
        
        try:
            with open(PERSONA_CONFIG_PATH) as f:
                config = json.load(f)
                # Validate config structure
                if not all(k in config for k in ["persona", "job_to_be_done"]):
                    return default_config
                return {**default_config, **config}  # Merge with defaults
        except FileNotFoundError: # Handle case where config file might not exist initially
            print(f"Warning: Persona config file not found at {PERSONA_CONFIG_PATH}. Using default.")
            return default_config
        except json.JSONDecodeError:
            print(f"Warning: Error decoding persona config file at {PERSONA_CONFIG_PATH}. Using default.")
            return default_config
        except Exception as e:
            print(f"An unexpected error occurred loading persona config: {e}. Using default.")
            return default_config

    def process_all_pdfs(self):
        """Process all PDFs in input directory with timeout handling"""
        pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]
        
        for filename in pdf_files:
            start_time = datetime.now()
            print(f"Processing new test pdf for {filename}...")
            
            try:
                result = self.process_pdf(filename)
                processing_time = (datetime.now() - start_time).total_seconds()
                
                if processing_time > MAX_PROCESSING_TIME:
                    print(f"Warning: Processing {filename} took {processing_time:.2f}s (over limit)")
                
                yield result
            except Exception as e:
                print(f"Error processing new test pdf for {filename}: {str(e)}")
                yield self._create_error_output(filename)

    def process_pdf(self, filename):
        """Process single PDF with comprehensive analysis"""
        pdf_path = os.path.join(INPUT_DIR, filename)
        
        # Extract structure and content
        title, first_page = self._extract_title_page(pdf_path)
        headings = self._detect_headings(pdf_path)

        # --- DEBUGGING OUTPUT FOR HEADINGS ---
        print(f"DEBUG: Headings detected for {filename} (count: {len(headings)}):")
        for h in headings:
            print(f"  - Text: '{h.get('text', 'N/A')}', Size: {h.get('size', 'N/A')}, Level: {h.get('level', 'MISSING!')}")
        # -------------------------------------

        sections = self._analyze_sections(headings, pdf_path)
        
        return {
            "metadata": self._create_metadata(filename, title),
            "sections": self._rank_sections(sections)
        }

    def _extract_title_page(self, pdf_path):
        """Efficiently extract title and first page content"""
        first_page_elements = None
        title_candidates = []
        
        for page_num, page in enumerate(extract_pages(pdf_path)):
            if page_num > 0:
                break

            for element in page:
                if isinstance(element, LTTextBox):
                    text = element.get_text().strip()
                    if text and len(text.split()) > 1:
                        size = self._get_avg_font_size(element)
                        title_candidates.append({
                            "text": text,
                            "size": size,
                            "centered": self._is_centered(element, page.width)
                        })
            first_page_elements = page
            
        title = max(title_candidates, key=lambda x: (x['size'], x['centered']), 
                             default={"text": "Untitled Document", "size": 0, "centered": False})['text']
        
        return title, first_page_elements

    def _detect_headings(self, pdf_path):
        """Optimized heading detection with early termination"""
        headings = []
        font_sizes = []
        
        for page in extract_pages(pdf_path):
            for element in page:
                if not isinstance(element, LTTextBox):
                    continue
                    
                text = element.get_text().strip()
                if not self._is_heading_text(text):
                    continue
                    
                size = self._get_avg_font_size(element)
                if size == 0:
                    continue

                font_sizes.append(size)
                
                headings.append({
                    "text": text,
                    "page": page.pageid,
                    "size": size,
                    "bold": self._is_bold(element),
                    "position": element.bbox
                })
        
        if len(font_sizes) > 5:
            thresholds = self._calculate_font_thresholds(font_sizes)
            for h in headings:
                h["level"] = self._classify_heading(h["size"], thresholds)
        else:
            headings.sort(key=lambda x: -x['size'])
            for i, h in enumerate(headings):
                if i == 0: h["level"] = "H1"
                elif i == 1: h["level"] = "H2"
                elif i == 2: h["level"] = "H3"
                else: h["level"] = "Body"
        
        return headings

    def _analyze_sections(self, headings, pdf_path):
        """Analyze sections with optimized content extraction"""
        sections = []
        
        sorted_headings = sorted(headings, key=lambda h: (h['page'], -h['position'][1]))

        all_heading_texts = {h['text'] for h in headings}
        
        for i, heading in enumerate(sorted_headings):
            relevance = self._calculate_relevance(heading['text'], heading['level'])
            
            section_end_page = float('inf')
            section_end_y = float('inf')

            for next_heading in sorted_headings[i+1:]:
                level_map = {"H1": 3, "H2": 2, "H3": 1, "Body": 0}
                if level_map.get(next_heading['level'], 0) >= level_map.get(heading['level'], 0):
                    section_end_page = next_heading['page']
                    section_end_y = next_heading['position'][3]
                    break

            content = self._extract_section_content(pdf_path, heading, section_end_page, section_end_y, all_heading_texts)
            
            section = {
                "title": heading['text'],
                "page": heading['page'],
                "level": heading['level'],
                "relevance": relevance,
                "content": content,
                "position": heading['position']
            }
            sections.append(section)
        
        return sections

    def _extract_section_content(self, pdf_path, heading, end_page, end_y, all_heading_texts):
        """Efficient content extraction with paragraph awareness until next heading or page limit."""
        content = []
        found_heading = False
        
        max_content_pages = 5
        
        for page_obj in extract_pages(pdf_path):
            if page_obj.pageid < heading['page']:
                continue
            if page_obj.pageid > heading['page'] + max_content_pages or page_obj.pageid > end_page:
                break
                
            for element in page_obj:
                if not isinstance(element, LTTextBox):
                    continue
                    
                text = element.get_text().strip()
                if not text:
                    continue
                    
                if page_obj.pageid == heading['page'] and element.bbox[1] > heading['position'][1] + 5:
                    continue
                
                if not found_heading and heading['text'] in text:
                    found_heading = True
                    if text == heading['text']:
                        continue 
                
                if found_heading:
                    if page_obj.pageid == end_page and element.bbox[3] < end_y - 5:
                             break 

                    if text in all_heading_texts and text != heading['text']:
                        break
                        
                    if len(text.split()) > 5:
                        content.append(text)
        
        return " ".join(content[:5])[:750] 

    def _calculate_relevance(self, text, level):
        """Calculate semantic relevance score"""
        text_lower = text.lower()
        score = 0
        
        for keyword, weight in self.persona_config["keyword_weights"].items():
            if keyword.lower() in text_lower:
                score += weight
                
        section_type = self._identify_section_type(text)
        score += self.persona_config["section_weights"].get(section_type, 0)
        
        score *= {"H1": 1.3, "H2": 1.1, "H3": 1.0, "Body": 0.5}.get(level, 1.0)
        
        return min(100, max(0, int(score)))

    def _rank_sections(self, sections):
        """Rank sections by relevance and organize hierarchy"""
        sections.sort(key=lambda x: (x['page'], -x['position'][1]))

        ranked_sections = []
        
        hierarchy_stack = [] 
        
        level_to_int = {"H1": 3, "H2": 2, "H3": 1, "Body": 0}

        for s in sections:
            current_level_int = level_to_int.get(s['level'], 0)

            while hierarchy_stack and level_to_int.get(hierarchy_stack[-1].get('level', 'Body'), 0) >= current_level_int:
                hierarchy_stack.pop()

            sub_section_data = {
                "sub_section_title": s['title'],
                "sub_section_page": s['page'],
                "importance_rank": s['relevance'],
                "refined_text": s['content'],
                "level": s['level']
            }

            if s['level'] == 'H1':
                new_h1_section = {
                    "section_title": s['title'],
                    "section_page": s['page'],
                    "importance_rank": s['relevance'],
                    "sub_sections": [],
                    "level": s['level']
                }
                ranked_sections.append(new_h1_section)
                hierarchy_stack = [new_h1_section]
            elif s['level'] == 'H2':
                found_parent = False
                for parent_section in reversed(hierarchy_stack):
                    if parent_section.get('level') == 'H1':
                        if 'sub_sections' not in parent_section:
                            parent_section['sub_sections'] = []
                        parent_section['sub_sections'].append(sub_section_data)
                        hierarchy_stack.append(sub_section_data)
                        found_parent = True
                        break
                if not found_parent:
                    new_h2_section = {
                        "section_title": s['title'],
                        "section_page": s['page'],
                        "importance_rank": s['relevance'],
                        "sub_sections": [],
                        "level": s['level']
                    }
                    ranked_sections.append(new_h2_section)
                    hierarchy_stack = [new_h2_section]
            elif s['level'] == 'H3':
                found_parent = False
                for parent_section in reversed(hierarchy_stack):
                    if parent_section.get('level') in ['H1', 'H2']:
                        if 'sub_sections' not in parent_section:
                            parent_section['sub_sections'] = []
                        parent_section['sub_sections'].append(sub_section_data)
                        hierarchy_stack.append(sub_section_data)
                        found_parent = True
                        break
                if not found_parent:
                    new_h3_section = {
                        "section_title": s['title'],
                        "section_page": s['page'],
                        "importance_rank": s['relevance'],
                        "sub_sections": [],
                        "level": s['level']
                    }
                    ranked_sections.append(new_h3_section)
                    hierarchy_stack = [new_h3_section]
            else:
                if hierarchy_stack:
                    parent = hierarchy_stack[-1]
                    if 'sub_sections' not in parent:
                        parent['sub_sections'] = []
                    parent['sub_sections'].append(sub_section_data)
                else:
                    ranked_sections.append({
                        "section_title": s['title'],
                        "section_page": s['page'],
                        "importance_rank": s['relevance'],
                        "sub_sections": [],
                        "level": s['level']
                    })
        return ranked_sections

    def _get_avg_font_size(self, element):
        """Calculate average font size for a text element (LTTextBox or LTTextLine)."""
        sizes = []
        if isinstance(element, LTTextBox):
            for text_line in element:
                if isinstance(text_line, LTTextLine):
                    for char in text_line:
                        if isinstance(char, LTChar):
                            sizes.append(char.size)
        elif isinstance(element, LTTextLine):
            for char in element:
                if isinstance(char, LTChar):
                    sizes.append(char.size)
        
        if sizes:
            return sum(sizes) / len(sizes)
        return 0.0

    def _is_bold(self, element):
        """Determine if text is bold for a text element (LTTextBox or LTTextLine)."""
        if isinstance(element, LTTextBox):
            for text_line in element:
                if isinstance(text_line, LTTextLine):
                    for char in text_line:
                        if isinstance(char, LTChar) and 'Bold' in char.fontname:
                            return True
        elif isinstance(element, LTTextLine):
            for char in element:
                if isinstance(char, LTChar) and 'Bold' in char.fontname:
                    return True
        return False

    def _is_centered(self, element, page_width, tolerance=20):
        """Check if element is centered on page based on its x-coordinates and page width."""
        if not hasattr(element, 'x0') or not hasattr(element, 'x1'):
            return False

        element_center_x = (element.x0 + element.x1) / 2
        page_center_x = page_width / 2
        
        return abs(element_center_x - page_center_x) < tolerance

    def _is_heading_text(self, text):
        """Comprehensive heuristic to determine if text is likely a heading."""
        if not text or len(text.strip()) < 3:
            return False
            
        text = text.strip()
        
        # --- NEW EXCLUSION: Common bullet points and very short or non-semantic text ---
        bullet_patterns = [
            r'^\s*[\u2022\u2023\u25E6\u2043\u25AA\u25AB\u2023]\s*', # Common Unicode bullets
            r'^\s*[-*+]\s*', # ASCII bullets (e.g., '-', '*', '+')
            r'^\s*\uf0b7\s*' # Specific bullet character seen in your output
        ]
        if any(re.match(p, text) for p in bullet_patterns):
            return False

        # Common non-heading patterns (more robust)
        non_heading_patterns = [
            r'^\d{1,2}:\d{2}$',   # Time (e.g., 20:45)
            r'^\d+$',             # Pure numbers (e.g., page numbers, '12', '10', '8', '6', '4', '2', '0' from chart axis)
            r'^[A-Z]$',           # Single uppercase letters
            r'^[IVXLCDM]+$',      # Roman numerals (e.g., I, II, III)
            r'^\W+$',             # Only punctuation/symbols
            r'^.{1,2}$',          # Very short text (1 or 2 characters)
            r'.*\.{3,}$',         # Ellipses (often for truncated text)
            r'^\s*$',             # Empty or whitespace-only strings
            r'^[a-z].*$',         # Starts with lowercase (usually body text)
            # --- NEW EXCLUSION: Chart legends like "Column X" or "Row X" across multiple lines ---
            r'(?:^|\n)\s*(?:Column|Row)\s*\d+.*(?:$|\n)', 
            r'.*\\n.*',           # Contains newline within a line (often body text broken by layout)
            r'Fig(ure)?\.?\s*\d+', # Figure captions (e.g., "Figure 1.")
            r'Table\.?\s*\d+',     # Table captions (e.g., "Table 1.")
        ]
        
        if any(re.match(p, text, re.IGNORECASE) for p in non_heading_patterns):
            return False
            
        # Common heading patterns
        heading_patterns = [
            r'^(Chapter|Section|Article|Part)\s+[A-Z0-9\.]+', # E.g., Chapter 1, Section A
            r'^\d+(\.\d+)*(\s+[A-Za-z])',   # Numbered headings: 1.1 Introduction, 2. Methods
            r'^[A-Z][A-Za-z0-9]*(?:\s+[A-Za-z0-9]+)*$',   # Title Case (e.g., "Introduction to AI")
            r'^[A-Z\s]{5,}$',   # ALL CAPS with at least 5 characters (e.g., "ABSTRACT")
            r'^.{5,50}$',       # Text length heuristic (5 to 50 chars for typical headings)
        ]
        
        # Check if it looks like a heading based on pattern
        if any(re.match(p, text) for p in heading_patterns):
            return True

        # Fallback: simple check for text that could be a heading but doesn't fit patterns
        # e.g., "Discussion"
        if text.istitle() and len(text.split()) < 7: # Title case, short phrase
            return True
        if text.isupper() and len(text.split()) < 5 and len(text) > 4: # All caps, very short phrase
            return True

        return False

    def _calculate_font_thresholds(self, font_sizes):
        """Dynamic font threshold calculation using quartiles for robustness."""
        if len(font_sizes) < 3:
            return {"H1": 20, "H2": 16, "H3": 12}
            
        sorted_sizes = sorted(font_sizes, reverse=True)
        
        try:
            q25, q50, q75 = statistics.quantiles(font_sizes, n=4)
        except statistics.StatisticsError:
            if len(sorted_sizes) >= 3:
                q25 = sorted_sizes[len(sorted_sizes)//4*3]
                q50 = sorted_sizes[len(sorted_sizes)//2]
                q75 = sorted_sizes[len(sorted_sizes)//4]
            else:
                q25, q50, q75 = sorted_sizes[0], sorted_sizes[0], sorted_sizes[0]


        h1 = sorted_sizes[0]
        h2 = max(q75, h1 * 0.8) 
        h3 = max(q50, h2 * 0.8)

        if h2 >= h1: h2 = h1 * 0.9
        if h3 >= h2: h3 = h2 * 0.9

        h1 = max(h1, 16) 
        h2 = max(h2, 12)
        h3 = max(h3, 10)
        
        return {"H1": h1, "H2": h2, "H3": h3}

    def _classify_heading(self, size, thresholds):
        """Classify heading level based on size thresholds."""
        tolerance_factor = 0.95 
        if size >= thresholds["H1"] * tolerance_factor:
            return "H1"
        elif size >= thresholds["H2"] * tolerance_factor:
            return "H2"
        elif size >= thresholds["H3"] * tolerance_factor:
            return "H3"
        return "Body"

    def _identify_section_type(self, text):
        """Categorize section based on content using keywords."""
        text_lower = text.lower()
        
        section_types = {
            "introduction": ["intro", "overview", "background", "abstract", "scope"],
            "methodology": ["method", "approach", "process", "design", "experiment", "procedure"],
            "results": ["result", "finding", "analysis", "data", "outcome"],
            "conclusion": ["conclusion", "summary", "recommendation", "discussion", "implication"],
            "references": ["reference", "bibliography", "citation", "works cited"],
            "appendix": ["appendix", "supplement", "annex", "addendum"]
        }
        
        for section, keywords in section_types.items():
            if any(kw in text_lower for kw in keywords):
                return section
        return "other"

    def _get_heading_texts(self, page):
        """Extract all heading-like texts from a page (used for section termination)."""
        headings = []
        for element in page:
            if isinstance(element, LTTextBox):
                text = element.get_text().strip()
                if self._is_heading_text(text):
                    headings.append(text)
        return headings

    def _create_metadata(self, filename, title):
        """Generate standardized metadata dictionary."""
        return {
            "input_document": filename,
            "persona": self.persona_config.get("persona", "unknown"),
            "job_to_be_done": self.persona_config.get("job_to_be_done", "unknown"),
            "processing_timestamp": datetime.utcnow().isoformat() + "Z",
            "document_title": title
        }

    def _create_error_output(self, filename):
        """Generate error output while maintaining required JSON format."""
        return {
            "metadata": {
                "input_document": filename,
                "persona": self.persona_config.get("persona", "unknown"),
                "job_to_be_done": self.persona_config.get("job_to_be_done", "unknown"),
                "processing_timestamp": datetime.utcnow().isoformat() + "Z",
                "document_title": "Error Processing Document",
                "error": True,
                "message": f"Failed to process {filename}"
            },
            "sections": []
        }

# Template for persona_config.json
PERSONA_CONFIG_TEMPLATE = """{
    "persona": "researcher",
    "job_to_be_done": "information_extraction",
    "keyword_weights": {
        "summary": 30, "conclusion": 25, "result": 20,
        "method": 20, "finding": 18, "recommendation": 15,
        "introduction": 10, "background": 8, "reference": 5
    },
    "section_weights": {
        "conclusion": 35, "results": 30, "methodology": 25,
        "introduction": 15, "appendix": 5, "other": 10
    }
}"""

def ensure_persona_config():
    """Create default persona_config.json if it doesn't exist."""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    if not os.path.exists(PERSONA_CONFIG_PATH):
        with open(PERSONA_CONFIG_PATH, 'w') as f:
            f.write(PERSONA_CONFIG_TEMPLATE)
        print(f"Created default persona config at {PERSONA_CONFIG_PATH}")

def main():
    ensure_persona_config()
    processor = PDFProcessor()
    results = list(processor.process_all_pdfs())
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for result in results:
        input_filename = result['metadata']['input_document']
        output_filename = input_filename.replace('.pdf', '.json')
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved output for {input_filename} to {output_path}")

if __name__ == "__main__":
    main()