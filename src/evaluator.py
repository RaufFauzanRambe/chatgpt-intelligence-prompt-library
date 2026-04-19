import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from collections import Counter


class QualityDimension(Enum):
    """Dimensions for prompt quality assessment."""
    CLARITY = "clarity"
    SPECIFICITY = "specificity"
    ACTIONABILITY = "actionability"
    COMPLETENESS = "completeness"
    REUSABILITY = "reusability"


@dataclass
class EvaluationResult:
    """Result of a single prompt evaluation."""
    prompt: str
    overall_score: float
    dimension_scores: Dict[str, float]
    complexity_score: float
    issues: List[str]
    suggestions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'prompt': self.prompt,
            'overall_score': self.overall_score,
            'dimension_scores': self.dimension_scores,
            'complexity_score': self.complexity_score,
            'issues': self.issues,
            'suggestions': self.suggestions,
            'metadata': self.metadata
        }


@dataclass
class BatchEvaluationResult:
    """Result of batch evaluation."""
    results: List[EvaluationResult]
    summary_stats: Dict[str, float]
    ranking: List[Tuple[int, float]]
    best_prompts: List[EvaluationResult]
    worst_prompts: List[EvaluationResult]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        data = [r.to_dict() for r in self.results]
        df = pd.DataFrame(data)
        return df


class PromptEvaluator:
    """
    Comprehensive evaluator for prompt quality assessment.

    This class evaluates prompts across multiple dimensions and provides
    actionable feedback for improvement.

    Attributes:
        weights (Dict): Weights for each quality dimension
    """

    # Default weights for quality dimensions
    DEFAULT_WEIGHTS = {
        QualityDimension.CLARITY: 0.25,
        QualityDimension.SPECIFICITY: 0.25,
        QualityDimension.ACTIONABILITY: 0.25,
        QualityDimension.COMPLETENESS: 0.15,
        QualityDimension.REUSABILITY: 0.10
    }

    # Action verbs that indicate clear instructions
    ACTION_VERBS = {
        'write', 'create', 'design', 'develop', 'analyze', 'explain',
        'generate', 'build', 'make', 'describe', 'compare', 'evaluate',
        'summarize', 'translate', 'solve', 'implement', 'construct',
        'compose', 'draft', 'prepare', 'formulate', 'outline', 'list'
    }

    # Words that indicate specificity
    SPECIFICITY_INDICATORS = {
        'specific', 'exactly', 'precisely', 'detailed', 'comprehensive',
        'step-by-step', 'including', 'must', 'should', 'at least',
        'minimum', 'maximum', 'following', 'format', 'structure'
    }

    # Common issues patterns
    ISSUE_PATTERNS = {
        'vague_request': r'^(help|assist|tell|give)\s+me',
        'missing_context': r'^what\s+is\s+\w+\??$',
        'no_action_verb': r'^[a-z]',
        'too_short': r'^.{1,20}$',
        'no_placeholder': r'^[^\[\]]+$',
    }

    def __init__(self,
                 weights: Dict[QualityDimension, float] = None,
                 custom_rules: Dict = None):
        """
        Initialize the PromptEvaluator.

        Args:
            weights: Custom weights for quality dimensions
            custom_rules: Additional custom evaluation rules
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.custom_rules = custom_rules or {}

        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}

    def evaluate(self,
                 prompt: str,
                 context: Dict = None) -> EvaluationResult:
        """
        Evaluate a single prompt.

        Args:
            prompt: Prompt text to evaluate
            context: Optional context for evaluation

        Returns:
            EvaluationResult: Complete evaluation result
        """
        context = context or {}

        # Calculate dimension scores
        dimension_scores = self._calculate_dimension_scores(prompt, context)

        # Calculate overall score
        overall_score = sum(
            dimension_scores[dim.value] * self.weights.get(dim, 0)
            for dim in QualityDimension
        )

        # Calculate complexity
        complexity_score = self._calculate_complexity(prompt)

        # Identify issues
        issues = self._identify_issues(prompt)

        # Generate suggestions
        suggestions = self._generate_suggestions(
            prompt, dimension_scores, issues
        )

        return EvaluationResult(
            prompt=prompt,
            overall_score=round(overall_score, 2),
            dimension_scores={k: round(v, 2) for k, v in dimension_scores.items()},
            complexity_score=round(complexity_score, 2),
            issues=issues,
            suggestions=suggestions,
            metadata={
                'word_count': len(prompt.split()),
                'char_count': len(prompt),
                'placeholder_count': len(re.findall(r'\[([^\]]+)\]', prompt)),
                'sentence_count': len(re.split(r'[.!?]+', prompt)) - 1
            }
        )

    def _calculate_dimension_scores(self,
                                    prompt: str,
                                    context: Dict) -> Dict[str, float]:
        """Calculate scores for each quality dimension."""
        return {
            QualityDimension.CLARITY.value: self._score_clarity(prompt),
            QualityDimension.SPECIFICITY.value: self._score_specificity(prompt),
            QualityDimension.ACTIONABILITY.value: self._score_actionability(prompt),
            QualityDimension.COMPLETENESS.value: self._score_completeness(prompt),
            QualityDimension.REUSABILITY.value: self._score_reusability(prompt)
        }

    def _score_clarity(self, prompt: str) -> float:
        """
        Score prompt clarity (0-10).

        Factors:
        - Clear action verb at start
        - Reasonable length
        - Grammatical structure
        """
        score = 5.0  # Base score

        # Check for clear action verb
        first_word = prompt.split()[0].lower() if prompt.split() else ''
        if first_word in self.ACTION_VERBS:
            score += 2.0

        # Check length appropriateness
        word_count = len(prompt.split())
        if 10 <= word_count <= 50:
            score += 1.5
        elif 50 < word_count <= 100:
            score += 1.0
        elif word_count < 5:
            score -= 2.0

        # Check for clear sentence structure
        if re.match(r'^[A-Z]', prompt):
            score += 0.5

        # Check for question mark (might indicate unclear instruction)
        if prompt.strip().endswith('?'):
            score -= 1.0

        return min(max(score, 0), 10)

    def _score_specificity(self, prompt: str) -> float:
        """
        Score prompt specificity (0-10).

        Factors:
        - Placeholder presence
        - Specificity indicators
        - Output format specification
        """
        score = 5.0  # Base score

        # Check for placeholders
        placeholders = re.findall(r'\[([^\]]+)\]', prompt)
        placeholder_count = len(placeholders)

        if placeholder_count >= 3:
            score += 2.5
        elif placeholder_count >= 1:
            score += 1.5 * placeholder_count

        # Check for specificity indicators
        prompt_lower = prompt.lower()
        specificity_count = sum(
            1 for word in self.SPECIFICITY_INDICATORS
            if word in prompt_lower
        )
        score += min(specificity_count * 0.5, 2.0)

        # Check for output format specification
        if any(kw in prompt_lower for kw in ['format', 'structure', 'list', 'include']):
            score += 1.0

        return min(max(score, 0), 10)

    def _score_actionability(self, prompt: str) -> float:
        """
        Score prompt actionability (0-10).

        Factors:
        - Clear action verb
        - Measurable output
        - Specific deliverables
        """
        score = 5.0  # Base score

        # Check for action verb
        first_word = prompt.split()[0].lower() if prompt.split() else ''
        if first_word in self.ACTION_VERBS:
            score += 2.0

        # Check for measurable output indicators
        prompt_lower = prompt.lower()
        measurable_indicators = [
            'list', 'number', 'count', 'example', 'step',
            'item', 'point', 'paragraph', 'sentence', 'word'
        ]

        for indicator in measurable_indicators:
            if indicator in prompt_lower:
                score += 0.5

        # Check for deliverable specifications
        if any(kw in prompt_lower for kw in ['create', 'generate', 'produce', 'output']):
            score += 1.0

        return min(max(score, 0), 10)

    def _score_completeness(self, prompt: str) -> float:
        """
        Score prompt completeness (0-10).

        Factors:
        - Context provision
        - Multiple instruction components
        - Edge case handling
        """
        score = 5.0  # Base score

        # Check for multiple components (comma-separated instructions)
        component_count = prompt.count(',') + 1
        if component_count >= 3:
            score += 2.0
        elif component_count >= 2:
            score += 1.0

        # Check for context words
        prompt_lower = prompt.lower()
        context_indicators = ['context', 'background', 'given', 'following', 'regarding']
        for indicator in context_indicators:
            if indicator in prompt_lower:
                score += 0.5

        # Check for constraint/specification words
        constraint_words = ['must', 'should', 'need to', 'require', 'ensure']
        for word in constraint_words:
            if word in prompt_lower:
                score += 0.3

        return min(max(score, 0), 10)

    def _score_reusability(self, prompt: str) -> float:
        """
        Score prompt reusability (0-10).

        Factors:
        - Placeholder presence
        - Generic vs specific language
        - Template quality
        """
        score = 5.0  # Base score

        # Check for placeholders (key for reusability)
        placeholders = re.findall(r'\[([^\]]+)\]', prompt)
        placeholder_count = len(placeholders)

        if placeholder_count >= 3:
            score += 3.0
        elif placeholder_count >= 1:
            score += 2.0

        # Check for generic language
        generic_indicators = ['a', 'an', 'the', 'any', 'some']
        word_count = len(prompt.split())
        generic_word_count = sum(
            1 for word in prompt.lower().split()
            if word in generic_indicators
        )

        if word_count > 0:
            generic_ratio = generic_word_count / word_count
            score += generic_ratio * 2

        return min(max(score, 0), 10)

    def _calculate_complexity(self, prompt: str) -> float:
        """
        Calculate overall complexity score.

        Factors:
        - Word count
        - Placeholder count
        - Structural complexity
        """
        score = 0.0

        # Word count contribution
        word_count = len(prompt.split())
        score += min(word_count / 10, 10)

        # Placeholder contribution
        placeholders = re.findall(r'\[([^\]]+)\]', prompt)
        score += len(placeholders) * 2

        # Structural complexity
        score += prompt.count(',') * 0.5
        score += prompt.count(';') * 1.0
        score += prompt.count('\n') * 0.5

        return score

    def _identify_issues(self, prompt: str) -> List[str]:
        """Identify potential issues with the prompt."""
        issues = []

        # Check for common issues
        for issue_name, pattern in self.ISSUE_PATTERNS.items():
            if re.search(pattern, prompt, re.IGNORECASE):
                issues.append(self._get_issue_description(issue_name))

        # Additional checks
        if len(prompt.split()) < 5:
            issues.append("Prompt is too short - may lack necessary detail")

        if not re.findall(r'\[([^\]]+)\]', prompt):
            issues.append("No placeholders found - prompt may not be reusable as a template")

        first_word = prompt.split()[0].lower() if prompt.split() else ''
        if first_word not in self.ACTION_VERBS:
            issues.append("Prompt doesn't start with a clear action verb")

        return issues

    def _get_issue_description(self, issue_name: str) -> str:
        """Get human-readable description for an issue."""
        descriptions = {
            'vague_request': "Request is vague - consider being more specific",
            'missing_context': "Missing context - provide more background",
            'no_action_verb': "No clear action verb at the start",
            'too_short': "Prompt is too short for effective use",
            'no_placeholder': "No placeholders - not usable as a template"
        }
        return descriptions.get(issue_name, f"Issue: {issue_name}")

    def _generate_suggestions(self,
                              prompt: str,
                              scores: Dict[str, float],
                              issues: List[str]) -> List[str]:
        """Generate improvement suggestions based on evaluation."""
        suggestions = []

        # Clarity suggestions
        if scores.get('clarity', 10) < 7:
            suggestions.append("Start with a clear action verb (e.g., Write, Create, Analyze)")

        # Specificity suggestions
        if scores.get('specificity', 10) < 7:
            suggestions.append("Add placeholders like [topic], [number], or [format] to make the prompt more specific")

        # Actionability suggestions
        if scores.get('actionability', 10) < 7:
            suggestions.append("Specify the expected output format or deliverable")

        # Completeness suggestions
        if scores.get('completeness', 10) < 7:
            suggestions.append("Add more context or constraints to guide the response")

        # Reusability suggestions
        if scores.get('reusability', 10) < 7:
            suggestions.append("Add placeholders to make the prompt reusable as a template")

        # Issue-based suggestions
        if "Prompt doesn't start with a clear action verb" in issues:
            suggestions.append(f"Consider starting with: {', '.join(list(self.ACTION_VERBS)[:5])}")

        return suggestions[:5]  # Limit to top 5 suggestions

    def evaluate_batch(self,
                       prompts: List[str],
                       contexts: List[Dict] = None) -> BatchEvaluationResult:
        """
        Evaluate multiple prompts.

        Args:
            prompts: List of prompts to evaluate
            contexts: Optional list of contexts for each prompt

        Returns:
            BatchEvaluationResult: Combined evaluation results
        """
        contexts = contexts or [{}] * len(prompts)

        results = [
            self.evaluate(prompt, context)
            for prompt, context in zip(prompts, contexts)
        ]

        # Calculate summary statistics
        all_scores = [r.overall_score for r in results]
        summary_stats = {
            'mean_score': np.mean(all_scores),
            'median_score': np.median(all_scores),
            'std_score': np.std(all_scores),
            'min_score': np.min(all_scores),
            'max_score': np.max(all_scores),
            'total_prompts': len(results)
        }

        # Create ranking
        ranking = sorted(
            enumerate(results),
            key=lambda x: x[1].overall_score,
            reverse=True
        )
        ranking = [(idx, r.overall_score) for idx, r in ranking]

        # Get best and worst prompts
        sorted_results = sorted(results, key=lambda x: x.overall_score, reverse=True)
        best_prompts = sorted_results[:5]
        worst_prompts = sorted_results[-5:] if len(sorted_results) >= 5 else sorted_results

        return BatchEvaluationResult(
            results=results,
            summary_stats=summary_stats,
            ranking=ranking,
            best_prompts=best_prompts,
            worst_prompts=worst_prompts
        )

    def evaluate_dataframe(self,
                           df: pd.DataFrame,
                           prompt_column: str = 'prompt') -> pd.DataFrame:
        """
        Evaluate prompts in a DataFrame.

        Args:
            df: Input DataFrame
            prompt_column: Column containing prompts

        Returns:
            pd.DataFrame: DataFrame with evaluation columns added
        """
        df = df.copy()

        results = []
        for prompt in df[prompt_column]:
            result = self.evaluate(prompt)
            results.append(result.to_dict())

        eval_df = pd.DataFrame(results)

        # Merge with original DataFrame
        for col in eval_df.columns:
            df[f'eval_{col}'] = eval_df[col].values

        return df

    def compare_prompts(self,
                        prompt1: str,
                        prompt2: str) -> Dict:
        """
        Compare two prompts side by side.

        Args:
            prompt1: First prompt
            prompt2: Second prompt

        Returns:
            Dict: Comparison results
        """
        result1 = self.evaluate(prompt1)
        result2 = self.evaluate(prompt2)

        return {
            'prompt1': result1.to_dict(),
            'prompt2': result2.to_dict(),
            'score_difference': result2.overall_score - result1.overall_score,
            'better_prompt': 'prompt2' if result2.overall_score > result1.overall_score else 'prompt1',
            'dimension_comparison': {
                dim: {
                    'prompt1': result1.dimension_scores.get(dim),
                    'prompt2': result2.dimension_scores.get(dim),
                    'difference': result2.dimension_scores.get(dim, 0) - result1.dimension_scores.get(dim, 0)
                }
                for dim in result1.dimension_scores.keys()
            }
        }


class ComplexityAnalyzer:
    """
    Standalone analyzer for prompt complexity.
    """

    def analyze(self, prompt: str) -> Dict:
        """
        Analyze complexity of a prompt.

        Args:
            prompt: Prompt to analyze

        Returns:
            Dict: Complexity analysis results
        """
        word_count = len(prompt.split())
        char_count = len(prompt)
        placeholders = re.findall(r'\[([^\]]+)\]', prompt)
        sentences = len(re.split(r'[.!?]+', prompt)) - 1

        return {
            'word_count': word_count,
            'char_count': char_count,
            'placeholder_count': len(placeholders),
            'placeholders': placeholders,
            'sentence_count': max(sentences, 1),
            'avg_word_length': char_count / max(word_count, 1),
            'structural_complexity': self._calculate_structural_complexity(prompt)
        }

    def _calculate_structural_complexity(self, prompt: str) -> float:
        """Calculate structural complexity score."""
        score = 0

        # Punctuation complexity
        score += prompt.count(',') * 0.5
        score += prompt.count(';') * 1.0
        score += prompt.count(':') * 0.5

        # Nested structures
        score += prompt.count('(') * 1.0
        score += prompt.count('[') * 1.5

        # Multi-line complexity
        score += prompt.count('\n') * 0.5

        return score


def evaluate_prompt(prompt: str) -> Dict:
    """
    Convenience function for quick prompt evaluation.

    Args:
        prompt: Prompt to evaluate

    Returns:
        Dict: Evaluation results
    """
    evaluator = PromptEvaluator()
    result = evaluator.evaluate(prompt)
    return result.to_dict()


if __name__ == "__main__":
    # Example usage
    print("Prompt Evaluator Module")
    print("=" * 50)

    evaluator = PromptEvaluator()

    # Test prompts
    test_prompts = [
        "Write a story about [topic] with [number] characters.",
        "What is AI?",
        "Create a comprehensive analysis of [topic] including at least [number] key points, pros and cons, and a summary.",
        "Help me with something.",
        "Design a [type] application for [platform] that handles [feature] with support for [requirement]."
    ]

    print("\nIndividual Evaluations:")
    print("-" * 50)

    for prompt in test_prompts[:3]:
        result = evaluator.evaluate(prompt)
        print(f"\nPrompt: {prompt[:50]}...")
        print(f"Overall Score: {result.overall_score}/10")
        print(f"Dimension Scores: {result.dimension_scores}")
        print(f"Complexity: {result.complexity_score}")
        if result.issues:
            print(f"Issues: {result.issues}")
        if result.suggestions:
            print(f"Suggestions: {result.suggestions}")

    # Batch evaluation
    print("\n" + "=" * 50)
    print("Batch Evaluation:")
    print("-" * 50)

    batch_result = evaluator.evaluate_batch(test_prompts)
    print(f"\nSummary Stats: {batch_result.summary_stats}")
    print(f"\nTop 3 Prompts by Score:")
    for idx, score in batch_result.ranking[:3]:
        print(f"  {idx+1}. Score: {score} - {test_prompts[idx][:40]}...")
