"""
MedQuAD XML Parser

Parses MedQuAD XML files and extracts question-answer pairs.
Follows Single Responsibility Principle - handles only XML parsing logic.

Design decisions:
- Uses lxml for robust XML parsing (handles malformed XML better than xml.etree)
- Filters empty answers (copyright-removed collections)
- Preserves question types and focus metadata for future analysis
- Returns structured data (not formatting) - separation of concerns
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class QAPair:
    """
    Structured representation of a question-answer pair.

    Attributes:
        question: The medical question text
        answer: The answer text (may be empty for copyright-removed content)
        qid: Unique question ID
        qtype: Question type (e.g., 'treatment', 'symptoms', 'causes')
        focus: Medical entity the question focuses on (disease, drug, etc.)
        focus_category: Category of focus (Disease, Drug, Other)
        source: Source collection name
    """

    question: str
    answer: str
    qid: str
    qtype: str
    focus: str
    focus_category: Optional[str]
    source: str

    def has_answer(self) -> bool:
        """Check if this pair has a non-empty answer."""
        return bool(self.answer and self.answer.strip())

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "question": self.question,
            "answer": self.answer,
            "qid": self.qid,
            "qtype": self.qtype,
            "focus": self.focus,
            "focus_category": self.focus_category,
            "source": self.source,
        }


class MedQuADParser:
    """
    Parser for MedQuAD XML files.

    This class is responsible only for parsing XML structure.
    Formatting to causal text is handled separately (SRP).
    """

    # Collections known to have removed answers (MedlinePlus copyright)
    EMPTY_COLLECTIONS = {"10_MPlus_ADAM_QA", "11_MPlusDrugs_QA", "12_MPlusHerbsSupplements_QA"}

    def __init__(self, dataset_root: Path, skip_empty: bool = True):
        """
        Initialize parser.

        Args:
            dataset_root: Path to MedQuAD root directory
            skip_empty: Whether to skip collections with removed answers
        """
        self.dataset_root = Path(dataset_root)
        self.skip_empty = skip_empty

        if not self.dataset_root.exists():
            raise ValueError(f"Dataset root does not exist: {dataset_root}")

    def parse_xml_file(self, xml_path: Path) -> List[QAPair]:
        """
        Parse a single XML file and extract Q&A pairs.

        Args:
            xml_path: Path to XML file

        Returns:
            List of QAPair objects

        Raises:
            ET.ParseError: If XML is malformed
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Extract document-level metadata
            source = root.get("source", "Unknown")
            focus = root.findtext("Focus", default="Unknown")

            # Get focus category if available
            focus_category = None
            focus_annotations = root.find("FocusAnnotations")
            if focus_annotations is not None:
                category = focus_annotations.find("Category")
                if category is not None:
                    focus_category = category.text

            # Extract Q&A pairs
            qa_pairs = []
            qa_pairs_elem = root.find("QAPairs")

            if qa_pairs_elem is None:
                return qa_pairs

            for qa_elem in qa_pairs_elem.findall("QAPair"):
                question_elem = qa_elem.find("Question")
                answer_elem = qa_elem.find("Answer")

                if question_elem is None:
                    continue

                question = question_elem.text or ""
                answer = answer_elem.text if answer_elem is not None else ""
                qid = question_elem.get("qid", "")
                qtype = question_elem.get("qtype", "unknown")

                qa_pair = QAPair(
                    question=question.strip(),
                    answer=answer.strip() if answer else "",
                    qid=qid,
                    qtype=qtype,
                    focus=focus,
                    focus_category=focus_category,
                    source=source,
                )

                qa_pairs.append(qa_pair)

            return qa_pairs

        except ET.ParseError as e:
            raise ET.ParseError(f"Failed to parse {xml_path}: {e}")

    def get_collection_paths(self) -> List[Path]:
        """
        Get all collection directories to process.

        Returns:
            List of collection directory paths
        """
        collections = []

        for item in self.dataset_root.iterdir():
            if not item.is_dir():
                continue

            # Skip hidden directories and git
            if item.name.startswith("."):
                continue

            # Skip empty collections if requested
            if self.skip_empty and item.name in self.EMPTY_COLLECTIONS:
                continue

            collections.append(item)

        return sorted(collections)

    def parse_collection(self, collection_path: Path) -> List[QAPair]:
        """
        Parse all XML files in a collection directory.

        Args:
            collection_path: Path to collection directory

        Returns:
            List of all QAPair objects from this collection
        """
        qa_pairs = []
        xml_files = list(collection_path.glob("*.xml"))

        for xml_file in xml_files:
            try:
                pairs = self.parse_xml_file(xml_file)
                qa_pairs.extend(pairs)
            except ET.ParseError as e:
                print(f"Warning: Skipping malformed file {xml_file}: {e}")
                continue

        return qa_pairs

    def parse_all(self, filter_empty_answers: bool = True) -> List[QAPair]:
        """
        Parse all collections in the dataset.

        Args:
            filter_empty_answers: Whether to filter out pairs with empty answers

        Returns:
            List of all QAPair objects from all collections
        """
        all_pairs = []
        collections = self.get_collection_paths()

        print(f"Found {len(collections)} collections to process")

        for collection in collections:
            print(f"Processing {collection.name}...")
            pairs = self.parse_collection(collection)

            if filter_empty_answers:
                pairs = [p for p in pairs if p.has_answer()]

            all_pairs.extend(pairs)
            print(f"  Extracted {len(pairs)} Q&A pairs")

        print(f"\nTotal Q&A pairs: {len(all_pairs)}")
        return all_pairs

    def get_statistics(self, qa_pairs: List[QAPair]) -> Dict:
        """
        Get statistics about parsed Q&A pairs.

        Args:
            qa_pairs: List of QAPair objects

        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_pairs": len(qa_pairs),
            "pairs_with_answers": sum(1 for p in qa_pairs if p.has_answer()),
            "pairs_without_answers": sum(1 for p in qa_pairs if not p.has_answer()),
            "sources": {},
            "qtypes": {},
            "focus_categories": {},
        }

        for pair in qa_pairs:
            # Count by source
            stats["sources"][pair.source] = stats["sources"].get(pair.source, 0) + 1

            # Count by question type
            stats["qtypes"][pair.qtype] = stats["qtypes"].get(pair.qtype, 0) + 1

            # Count by focus category
            if pair.focus_category:
                cat = pair.focus_category
                stats["focus_categories"][cat] = stats["focus_categories"].get(cat, 0) + 1

        return stats
