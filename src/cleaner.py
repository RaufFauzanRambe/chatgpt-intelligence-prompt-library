
import re
import unicodedata
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
import pandas as pd
from collections import Counter


@dataclass
class CleaningConfig:
    """Configuration for prompt cleaning operations."""
    normalize_whitespace: bool = True
    remove_extra_newlines: bool = True
    normalize_quotes: bool = True
    fix_common_typos: bool = False
    lowercase: bool = False
    remove_emojis: bool = False
    normalize_unicode: bool = True
    trim_edges: bool = True
    preserve_placeholders: bool = True


class PromptCleaner:
    """
    A comprehensive cleaner for prompt text preprocessing.

    This class provides various text cleaning and normalization
    operations specifically designed for ChatGPT prompts.

    Attributes:
        config (CleaningConfig): Configuration for cleaning operations
    """

    # Common placeholder patterns
    PLACEHOLDER_PATTERN = r'\[([^\]]+)\]'

    # Quote normalization mappings
    QUOTE_MAPPINGS = {
        '"': '"',
        '"': '"',
        ''': "'",
        ''': "'",
        '«': '"',
        '»': '"',
        '„': '"',
        '‟': '"',
    }

    # Common typo corrections (expandable)
    COMMON_TYPOS = {
        'teh': 'the',
        'adn': 'and',
        'taht': 'that',
        'wiht': 'with',
        'thier': 'their',
        'recieve': 'receive',
        'occured': 'occurred',
        'seperate': 'separate',
    }

    def __init__(self, config: CleaningConfig = None):
        """
        Initialize the PromptCleaner.

        Args:
            config: Optional cleaning configuration
        """
        self.config = config or CleaningConfig()

    def clean(self, text: str) -> str:
        """
        Apply all cleaning operations to text.

        Args:
            text: Input text to clean

        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""

        # Store placeholders if preservation is enabled
        placeholders = {}
        if self.config.preserve_placeholders:
            placeholders = self._extract_and_mask_placeholders(text)
            text = placeholders['masked_text']

        # Apply cleaning operations
        if self.config.normalize_unicode:
            text = self._normalize_unicode(text)

        if self.config.normalize_quotes:
            text = self._normalize_quotes(text)

        if self.config.normalize_whitespace:
            text = self._normalize_whitespace(text)

        if self.config.remove_extra_newlines:
            text = self._remove_extra_newlines(text)

        if self.config.remove_emojis:
            text = self._remove_emojis(text)

        if self.config.fix_common_typos:
            text = self._fix_typos(text)

        if self.config.lowercase:
            text = text.lower()

        if self.config.trim_edges:
            text = text.strip()

        # Restore placeholders
        if self.config.preserve_placeholders:
            text = self._restore_placeholders(text, placeholders['masks'])

        return text

    def _extract_and_mask_placeholders(self, text: str) -> Dict:
        """Extract placeholders and replace with temporary masks."""
        masks = {}
        counter = 0

        def replace(match):
            nonlocal counter
            placeholder = match.group(0)
            mask = f"__PLACEHOLDER_{counter}__"
            masks[mask] = placeholder
            counter += 1
            return mask

        masked_text = re.sub(self.PLACEHOLDER_PATTERN, replace, text)

        return {'masked_text': masked_text, 'masks': masks}

    def _restore_placeholders(self, text: str, masks: Dict) -> str:
        """Restore masked placeholders."""
        for mask, placeholder in masks.items():
            text = text.replace(mask, placeholder)
        return text

    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters."""
        return unicodedata.normalize('NFKC', text)

    def _normalize_quotes(self, text: str) -> str:
        """Normalize various quote types to standard ASCII."""
        for old, new in self.QUOTE_MAPPINGS.items():
            text = text.replace(old, new)
        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace characters."""
        # Replace various whitespace with single space
        text = re.sub(r'[ \t]+', ' ', text)
        return text

    def _remove_extra_newlines(self, text: str) -> str:
        """Remove excessive newlines."""
        # Replace 3+ newlines with 2
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text

    def _remove_emojis(self, text: str) -> str:
        """Remove emoji characters."""
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        return emoji_pattern.sub('', text)

    def _fix_typos(self, text: str) -> str:
        """Fix common typos."""
        words = text.split()
        corrected = []

        for word in words:
            lower_word = word.lower()
            if lower_word in self.COMMON_TYPOS:
                # Preserve original case
                if word.isupper():
                    corrected.append(self.COMMON_TYPOS[lower_word].upper())
                elif word[0].isupper():
                    corrected.append(self.COMMON_TYPOS[lower_word].capitalize())
                else:
                    corrected.append(self.COMMON_TYPOS[lower_word])
            else:
                corrected.append(word)

        return ' '.join(corrected)

    def extract_placeholders(self, text: str) -> List[str]:
        """
        Extract all placeholders from text.

        Args:
            text: Input text

        Returns:
            List[str]: List of placeholder names (without brackets)
        """
        return re.findall(self.PLACEHOLDER_PATTERN, text)

    def get_placeholder_count(self, text: str) -> int:
        """
        Count the number of placeholders in text.

        Args:
            text: Input text

        Returns:
            int: Number of placeholders
        """
        return len(self.extract_placeholders(text))

    def replace_placeholders(self,
                             text: str,
                             replacements: Dict[str, str]) -> str:
        """
        Replace placeholders with provided values.

        Args:
            text: Input text with placeholders
            replacements: Dictionary mapping placeholder names to values

        Returns:
            str: Text with replaced placeholders
        """
        def replace(match):
            placeholder_name = match.group(1)
            if placeholder_name in replacements:
                return replacements[placeholder_name]
            return match.group(0)  # Keep original if no replacement

        return re.sub(self.PLACEHOLDER_PATTERN, replace, text)

    def remove_placeholders(self, text: str, replacement: str = '') -> str:
        """
        Remove all placeholders from text.

        Args:
            text: Input text
            replacement: Optional replacement string

        Returns:
            str: Text with placeholders removed
        """
        return re.sub(self.PLACEHOLDER_PATTERN, replacement, text)

    def clean_dataframe(self,
                        df: pd.DataFrame,
                        text_column: str = 'prompt',
                        output_column: str = None) -> pd.DataFrame:
        """
        Clean all prompts in a DataFrame.

        Args:
            df: Input DataFrame
            text_column: Column containing text to clean
            output_column: Column for cleaned text (default: overwrites original)

        Returns:
            pd.DataFrame: DataFrame with cleaned text
        """
        df = df.copy()

        if output_column is None:
            output_column = text_column

        df[output_column] = df[text_column].apply(self.clean)
        df['original_text'] = df[text_column]
        df['was_cleaned'] = df['original_text'] != df[output_column]

        return df

    def deduplicate(self,
                    texts: List[str],
                    similarity_threshold: float = 1.0) -> Tuple[List[str], List[int]]:
        """
        Remove duplicate texts.

        Args:
            texts: List of texts
            similarity_threshold: Threshold for exact matching (1.0 = exact)

        Returns:
            Tuple[List[str], List[int]]: Deduplicated texts and indices kept
        """
        seen = set()
        unique_texts = []
        kept_indices = []

        for i, text in enumerate(texts):
            normalized = self.clean(text)

            if similarity_threshold == 1.0:
                # Exact deduplication
                if normalized not in seen:
                    seen.add(normalized)
                    unique_texts.append(text)
                    kept_indices.append(i)
            else:
                # Would need fuzzy matching for non-exact threshold
                # For simplicity, using exact matching
                if normalized not in seen:
                    seen.add(normalized)
                    unique_texts.append(text)
                    kept_indices.append(i)

        return unique_texts, kept_indices


class PlaceholderAnalyzer:
    """
    Analyzer for placeholder patterns in prompts.
    """

    def __init__(self):
        self.pattern = re.compile(r'\[([^\]]+)\]')

    def analyze(self, text: str) -> Dict:
        """
        Analyze placeholders in text.

        Args:
            text: Input text

        Returns:
            Dict: Analysis results
        """
        placeholders = self.pattern.findall(text)

        return {
            'count': len(placeholders),
            'placeholders': placeholders,
            'unique_count': len(set(placeholders)),
            'unique_placeholders': list(set(placeholders)),
            'positions': [m.start() for m in self.pattern.finditer(text)]
        }

    def get_placeholder_types(self, texts: List[str]) -> Counter:
        """
        Get frequency of placeholder types across multiple texts.

        Args:
            texts: List of texts

        Returns:
            Counter: Placeholder frequencies
        """
        all_placeholders = []

        for text in texts:
            placeholders = self.pattern.findall(text)
            all_placeholders.extend(placeholders)

        return Counter(all_placeholders)

    def suggest_placeholders(self, text: str) -> List[Dict]:
        """
        Suggest potential placeholders for text that lacks them.

        Args:
            text: Input text

        Returns:
            List[Dict]: Suggested placeholders with positions
        """
        suggestions = []

        # Common patterns that could be placeholders
        patterns = [
            (r'\b\d+\b', 'number'),
            (r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', 'name'),
            (r'\bhttps?://\S+\b', 'url'),
            (r'\b[\w.]+@[\w.]+\b', 'email'),
        ]

        for pattern, placeholder_type in patterns:
            for match in re.finditer(pattern, text):
                suggestions.append({
                    'text': match.group(),
                    'type': placeholder_type,
                    'start': match.start(),
                    'end': match.end(),
                    'suggested_placeholder': f'[{placeholder_type}]'
                })

        return suggestions


def clean_prompt(text: str, **kwargs) -> str:
    """
    Convenience function for quick prompt cleaning.

    Args:
        text: Text to clean
        **kwargs: Configuration options

    Returns:
        str: Cleaned text
    """
    config = CleaningConfig(**kwargs)
    cleaner = PromptCleaner(config)
    return cleaner.clean(text)


def extract_placeholders(text: str) -> List[str]:
    """
    Convenience function for extracting placeholders.

    Args:
        text: Input text

    Returns:
        List[str]: List of placeholders
    """
    return re.findall(r'\[([^\]]+)\]', text)


if __name__ == "__main__":
    # Example usage
    print("Prompt Cleaner Module")
    print("=" * 50)

    cleaner = PromptCleaner()

    # Example prompts to clean
    examples = [
        "Write a  story  about [topic] and [character].",
        'She said "Hello" and  left.',
        "Create  \n\n\n  a list of [number] items.",
        "The prompt has   extra   whitespace.",
    ]

    print("\nCleaning Examples:")
    print("-" * 50)

    for example in examples:
        cleaned = cleaner.clean(example)
        print(f"\nOriginal: {repr(example)}")
        print(f"Cleaned:  {repr(cleaned)}")
        print(f"Placeholders: {cleaner.extract_placeholders(example)}")

    # Placeholder analyzer example
    print("\n" + "=" * 50)
    print("Placeholder Analysis:")
    print("-" * 50)

    analyzer = PlaceholderAnalyzer()
    sample = "Write a [type] about [topic] with [number] examples."
    analysis = analyzer.analyze(sample)
    print(f"\nText: {sample}")
    print(f"Analysis: {analysis}")
