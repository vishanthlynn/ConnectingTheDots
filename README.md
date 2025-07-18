# Approach Explanation: Persona-Driven Document Intelligence

## Overview

Our solution, "Connecting the Dots," is designed to extract structured outlines and key insights from raw PDF documents, leveraging on-device intelligence to prioritize information based on a defined user persona and their specific "job-to-be-done." This system forms the foundational "brains" for smarter document experiences, enabling efficient information retrieval and analysis from unstructured data.

## Methodology

The core of our approach lies in a multi-stage PDF processing pipeline implemented in Python, utilizing the `pdfminer.six` library for robust text and layout extraction.

1.  **Document Parsing and Initial Extraction:**
    * The system iterates through all PDF files placed in a designated `/app/input` directory.
    * For each PDF, it first extracts the document title and processes its pages to identify potential text elements.

2.  **Sophisticated Heading Detection and Classification:**
    * This is a critical component for building structured outlines. We employ a comprehensive heuristic-based approach within the `_is_heading_text` method.
    * **Dynamic Font Thresholds:** Instead of fixed font sizes, we dynamically calculate font size thresholds (H1, H2, H3) based on the actual font sizes present in the document using statistical quantiles. This makes the system adaptable to various document styles.
    * **Robust Exclusion Patterns:** A key focus has been on improving the precision of heading detection by implementing extensive exclusion patterns. This prevents common non-heading elements from being misclassified:
        * **Bullet Points:** Specific Unicode and ASCII patterns are used to filter out list items (e.g., `•`, `*`, `-`).
        * **Chart Legends & Axes:** Regular expressions are deployed to exclude text commonly found in chart legends (e.g., "Column 1", "Row 2") and numerical axis labels, which often appear visually prominent but are not structural headings.
        * **Non-Semantic Text:** Filters are in place for short, purely numeric, or symbolic text, as well as text containing internal newlines (often indicating broken body text rather than a heading).
    * **Positive Identification:** Alongside exclusions, patterns for numbered headings, title-cased phrases, and all-caps sections are used to confirm genuine headings.
    * **Hierarchical Assignment:** Detected headings are then classified into H1, H2, or H3 levels based on their font size relative to the dynamically calculated thresholds. In cases with limited font variations, a fallback mechanism ensures consistent H1/H2/H3 assignment.

3.  **Persona-Driven Relevance Scoring:**
    * A `persona_config.json` file allows for configurable weights for keywords and section types (e.g., "conclusion", "methodology").
    * The `_calculate_relevance` method assigns an `importance_rank` to each detected section by matching keywords and identifying section types (e.g., "introduction", "results"). This score is further adjusted based on the heading's hierarchical level (H1, H2, H3), ensuring that higher-level headings generally receive higher relevance.

4.  **Content Extraction and Hierarchical Organization:**
    * The `_extract_section_content` method efficiently extracts relevant text paragraphs following a heading, stopping precisely before the next identified heading or a defined page limit.
    * The `_rank_sections` method organizes the extracted sections and their content into a hierarchical JSON structure, maintaining the document's logical flow and nesting sub-sections under their appropriate parent headings based on their assigned levels.

## Adherence to Constraints

The solution is designed to strictly adhere to the hackathon's technical constraints:
* **CPU-only:** All processing is performed on the CPU.
* **Model Size:** The solution uses lightweight heuristics and standard libraries, ensuring the model size remains well within the 1GB limit.
* **Offline Execution:** No external API calls or internet access are required during runtime.
* **Processing Time:** The pipeline is optimized for speed, aiming to process documents within the specified 10 seconds per 50-page PDF, and the overall collection within the 60-second limit.

## Conclusion

By combining robust PDF parsing, intelligent heuristic-based heading detection, and a configurable persona-driven relevance scoring system, our solution provides a powerful tool for transforming unstructured PDF data into actionable, prioritized insights, directly addressing the "Connecting the Dots" challenge.