import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import pandas as pd
from datetime import datetime


@dataclass
class DatasetInfo:
    """Information about a loaded dataset."""
    category: str
    level: str
    description: str
    prompt_count: int
    file_path: str
    loaded_at: datetime


class PromptLoader:
    """
    A comprehensive loader for ChatGPT prompt datasets.

    This class handles loading prompt datasets from JSON files, combining
    multiple datasets, and providing convenient access methods.

    Attributes:
        data_dir (Path): Path to the data directory
        datasets (Dict): Loaded datasets indexed by category
        df (pd.DataFrame): Combined DataFrame of all prompts
    """

    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize the PromptLoader.

        Args:
            data_dir: Path to the directory containing JSON prompt files
        """
        self.data_dir = Path(data_dir)
        self.datasets: Dict[str, dict] = {}
        self.dataset_info: Dict[str, DatasetInfo] = {}
        self._df: Optional[pd.DataFrame] = None
        self._loaded = False

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

    def load_file(self, file_path: Union[str, Path]) -> dict:
        """
        Load a single JSON prompt file.

        Args:
            file_path: Path to the JSON file

        Returns:
            dict: Loaded JSON data

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data

    def load_category(self, category: str) -> pd.DataFrame:
        """
        Load prompts for a specific category.

        Args:
            category: Category name (e.g., 'basic', 'coding', 'creative')

        Returns:
            pd.DataFrame: DataFrame containing prompts for the category

        Raises:
            ValueError: If category not found
        """
        if not self._loaded:
            self.load_all()

        if category not in self.datasets:
            available = list(self.datasets.keys())
            raise ValueError(f"Category '{category}' not found. Available: {available}")

        data = self.datasets[category]
        prompts = data.get('prompts', [])

        df = pd.DataFrame(prompts)
        df['category'] = category
        df['level'] = data.get('level', 'unknown')

        return df

    def load_all(self, reload: bool = False) -> pd.DataFrame:
        """
        Load all prompt datasets from the data directory.

        Args:
            reload: Force reload even if already loaded

        Returns:
            pd.DataFrame: Combined DataFrame of all prompts
        """
        if self._loaded and not reload:
            return self._df

        json_files = list(self.data_dir.glob('*.json'))

        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {self.data_dir}")

        all_prompts = []

        for file_path in json_files:
            try:
                data = self.load_file(file_path)
                category = data.get('category', file_path.stem)
                level = data.get('level', 'unknown')
                description = data.get('description', '')
                prompts = data.get('prompts', [])

                # Store dataset
                self.datasets[category] = data

                # Store info
                self.dataset_info[category] = DatasetInfo(
                    category=category,
                    level=level,
                    description=description,
                    prompt_count=len(prompts),
                    file_path=str(file_path),
                    loaded_at=datetime.now()
                )

                # Add metadata to each prompt
                for prompt in prompts:
                    prompt['category'] = category
                    prompt['level'] = level
                    all_prompts.append(prompt)

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Error loading {file_path}: {e}")
                continue

        self._df = pd.DataFrame(all_prompts)
        self._loaded = True

        return self._df

    @property
    def df(self) -> pd.DataFrame:
        """Get the combined DataFrame, loading if necessary."""
        if self._df is None:
            self.load_all()
        return self._df

    def get_categories(self) -> List[str]:
        """Get list of available categories."""
        if not self._loaded:
            self.load_all()
        return list(self.datasets.keys())

    def get_summary(self) -> pd.DataFrame:
        """
        Get a summary of loaded datasets.

        Returns:
            pd.DataFrame: Summary table with dataset information
        """
        if not self._loaded:
            self.load_all()

        summary_data = []
        for category, info in self.dataset_info.items():
            summary_data.append({
                'Category': info.category,
                'Level': info.level,
                'Description': info.description[:60] + '...' if len(info.description) > 60 else info.description,
                'Prompt Count': info.prompt_count,
                'File': Path(info.file_path).name
            })

        return pd.DataFrame(summary_data)

    def search_prompts(self,
                       query: str,
                       columns: List[str] = None,
                       case_sensitive: bool = False) -> pd.DataFrame:
        """
        Search for prompts containing specific text.

        Args:
            query: Search query string
            columns: Columns to search in (default: all text columns)
            case_sensitive: Whether search is case sensitive

        Returns:
            pd.DataFrame: Filtered DataFrame with matching prompts
        """
        if not self._loaded:
            self.load_all()

        if columns is None:
            columns = ['prompt', 'title', 'use_case']

        mask = pd.Series([False] * len(self._df))

        for col in columns:
            if col in self._df.columns:
                if case_sensitive:
                    mask |= self._df[col].astype(str).str.contains(query, na=False)
                else:
                    mask |= self._df[col].astype(str).str.contains(query, case=False, na=False)

        return self._df[mask]

    def filter_by_tags(self, tags: Union[str, List[str]]) -> pd.DataFrame:
        """
        Filter prompts by tags.

        Args:
            tags: Tag or list of tags to filter by

        Returns:
            pd.DataFrame: Filtered DataFrame with prompts containing any of the tags
        """
        if not self._loaded:
            self.load_all()

        if isinstance(tags, str):
            tags = [tags]

        def has_tag(prompt_tags):
            if not isinstance(prompt_tags, list):
                return False
            return any(tag in prompt_tags for tag in tags)

        mask = self._df['tags'].apply(has_tag)
        return self._df[mask]

    def filter_by_level(self, level: str) -> pd.DataFrame:
        """
        Filter prompts by difficulty level.

        Args:
            level: Level to filter by ('beginner', 'intermediate', 'advanced', 'all levels')

        Returns:
            pd.DataFrame: Filtered DataFrame
        """
        if not self._loaded:
            self.load_all()

        return self._df[self._df['level'] == level]

    def get_random_prompt(self,
                          category: str = None,
                          level: str = None,
                          n: int = 1) -> Union[pd.Series, pd.DataFrame]:
        """
        Get random prompt(s) from the dataset.

        Args:
            category: Optional category filter
            level: Optional level filter
            n: Number of prompts to return

        Returns:
            pd.Series if n=1, else pd.DataFrame
        """
        if not self._loaded:
            self.load_all()

        df = self._df

        if category:
            df = df[df['category'] == category]
        if level:
            df = df[df['level'] == level]

        sample = df.sample(min(n, len(df)))

        if n == 1:
            return sample.iloc[0]
        return sample

    def export_to_json(self,
                       output_path: Union[str, Path],
                       category: str = None,
                       indent: int = 2) -> None:
        """
        Export prompts to a JSON file.

        Args:
            output_path: Path for output file
            category: Optional category to export (default: all)
            indent: JSON indentation
        """
        if not self._loaded:
            self.load_all()

        output_path = Path(output_path)

        if category:
            data = self.datasets.get(category)
            if not data:
                raise ValueError(f"Category '{category}' not found")
        else:
            data = {
                'exported_at': datetime.now().isoformat(),
                'total_prompts': len(self._df),
                'categories': self.get_categories(),
                'prompts': self._df.to_dict('records')
            }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)

        print(f"Exported to {output_path}")

    def __len__(self) -> int:
        """Return total number of prompts."""
        return len(self.df)

    def __repr__(self) -> str:
        """String representation."""
        return f"PromptLoader(data_dir='{self.data_dir}', prompts={len(self)}, categories={len(self.datasets)})"


def load_prompts(data_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Convenience function to quickly load all prompts.

    Args:
        data_dir: Path to data directory

    Returns:
        pd.DataFrame: Combined DataFrame of all prompts
    """
    loader = PromptLoader(data_dir)
    return loader.load_all()


if __name__ == "__main__":
    # Example usage
    print("Prompt Loader Module")
    print("=" * 50)

    # Example: Load from relative path
    data_path = Path(__file__).parent.parent / "data"

    if data_path.exists():
        loader = PromptLoader(data_path)
        df = loader.load_all()

        print(f"\nLoaded {len(df)} prompts from {len(loader.get_categories())} categories")
        print(f"\nCategories: {loader.get_categories()}")

        print("\nDataset Summary:")
        print(loader.get_summary())

        # Example search
        print("\nSearching for 'analysis'...")
        results = loader.search_prompts('analysis')
        print(f"Found {len(results)} matching prompts")
    else:
        print(f"Data directory not found: {data_path}")
