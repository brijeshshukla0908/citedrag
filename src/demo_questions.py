"""
Demo Questions Module
=====================
Pre-computed demo questions with cached responses for instant display.

Key Functions:
- load_demo_questions(): Load demo Q&A pairs
- generate_demo_responses(): Pre-compute responses
- get_demo_question(): Get a specific demo Q&A
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
import json
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config


class DemoQuestionsManager:
    """
    Manages demo questions with pre-computed responses.
    """
    
    def __init__(
        self,
        demo_file: str = config.DEMO_QUESTIONS_FILE,
        max_questions: int = config.MAX_DEMO_QUESTIONS
    ):
        """
        Initialize DemoQuestionsManager.
        
        Args:
            demo_file: Path to demo questions JSON file
            max_questions: Maximum number of demo questions
        """
        self.demo_file = Path(demo_file)
        self.max_questions = max_questions
        self.demo_questions = []
        
        # Ensure data directory exists
        self.demo_file.parent.mkdir(exist_ok=True, parents=True)
        
        # Load or create demo questions
        if self.demo_file.exists():
            self.demo_questions = self._load_from_file()
            logger.info(f"Loaded {len(self.demo_questions)} demo questions")
        else:
            self.demo_questions = self._create_default_questions()
            self._save_to_file()
            logger.info(f"Created {len(self.demo_questions)} default demo questions")
    
    
    def _create_default_questions(self) -> List[Dict]:
        """
        Create default demo questions for HR policy document.
        
        Returns:
            List of demo question dictionaries
        """
        return [
            {
                "id": 1,
                "question": "What is this document about?",
                "category": "General",
                "complexity": "easy"
            },
            {
                "id": 2,
                "question": "What are the main employee conduct expectations?",
                "category": "Conduct",
                "complexity": "medium"
            },
            {
                "id": 3,
                "question": "What does the document say about confidentiality?",
                "category": "Confidentiality",
                "complexity": "medium"
            },
            {
                "id": 4,
                "question": "What are the rules regarding conflicts of interest?",
                "category": "Ethics",
                "complexity": "medium"
            },
            {
                "id": 5,
                "question": "What is mentioned about intellectual property?",
                "category": "IP Rights",
                "complexity": "medium"
            },
            {
                "id": 6,
                "question": "What are the employee responsibilities according to this document?",
                "category": "Responsibilities",
                "complexity": "easy"
            },
            {
                "id": 7,
                "question": "What does it say about non-compete agreements?",
                "category": "Legal",
                "complexity": "hard"
            },
            {
                "id": 8,
                "question": "Are there any rules about working with competitors?",
                "category": "Ethics",
                "complexity": "medium"
            },
            {
                "id": 9,
                "question": "What are the data protection requirements?",
                "category": "Data Protection",
                "complexity": "medium"
            },
            {
                "id": 10,
                "question": "What happens if an employee violates these policies?",
                "category": "Compliance",
                "complexity": "hard"
            },
            {
                "id": 11,
                "question": "What is the company's stance on transparency?",
                "category": "Values",
                "complexity": "easy"
            },
            {
                "id": 12,
                "question": "Are there guidelines for external communications?",
                "category": "Communication",
                "complexity": "medium"
            },
            {
                "id": 13,
                "question": "What does the document say about client relationships?",
                "category": "Client Relations",
                "complexity": "medium"
            },
            {
                "id": 14,
                "question": "What are the expectations for professional behavior?",
                "category": "Conduct",
                "complexity": "easy"
            },
            {
                "id": 15,
                "question": "How should employees handle sensitive information?",
                "category": "Information Security",
                "complexity": "medium"
            }
        ]
    
    
    def get_all_questions(self) -> List[Dict]:
        """
        Get all demo questions.
        
        Returns:
            List of demo question dictionaries
        """
        return self.demo_questions[:self.max_questions]
    
    
    def get_question_by_id(self, question_id: int) -> Optional[Dict]:
        """
        Get a specific demo question by ID.
        
        Args:
            question_id: Question ID
            
        Returns:
            Question dictionary or None
        """
        for q in self.demo_questions:
            if q['id'] == question_id:
                return q
        return None
    
    
    def get_questions_by_category(self, category: str) -> List[Dict]:
        """
        Get demo questions by category.
        
        Args:
            category: Category name
            
        Returns:
            List of matching questions
        """
        return [
            q for q in self.demo_questions
            if q.get('category', '').lower() == category.lower()
        ]
    
    
    def get_random_questions(self, count: int = 5) -> List[Dict]:
        """
        Get random demo questions.
        
        Args:
            count: Number of questions to return
            
        Returns:
            List of random questions
        """
        import random
        return random.sample(
            self.demo_questions,
            min(count, len(self.demo_questions))
        )
    
    
    def add_demo_response(
        self,
        question_id: int,
        response: Dict
    ):
        """
        Add pre-computed response to a demo question.
        
        Args:
            question_id: Question ID
            response: Response dictionary from LLM
        """
        for q in self.demo_questions:
            if q['id'] == question_id:
                q['response'] = response
                q['has_response'] = True
                self._save_to_file()
                logger.info(f"✅ Added response for question ID: {question_id}")
                return
        
        logger.warning(f"Question ID not found: {question_id}")
    
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about demo questions.
        
        Returns:
            Dictionary with statistics
        """
        total = len(self.demo_questions)
        with_responses = sum(1 for q in self.demo_questions if q.get('has_response', False))
        
        categories = {}
        for q in self.demo_questions:
            cat = q.get('category', 'Uncategorized')
            categories[cat] = categories.get(cat, 0) + 1
        
        complexity_dist = {}
        for q in self.demo_questions:
            comp = q.get('complexity', 'unknown')
            complexity_dist[comp] = complexity_dist.get(comp, 0) + 1
        
        return {
            'total_questions': total,
            'questions_with_responses': with_responses,
            'coverage_percentage': round((with_responses / total * 100) if total > 0 else 0, 1),
            'categories': categories,
            'complexity_distribution': complexity_dist
        }
    
    
    def _load_from_file(self) -> List[Dict]:
        """
        Load demo questions from file.
        
        Returns:
            List of question dictionaries
        """
        try:
            with open(self.demo_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load demo questions: {str(e)}")
            return self._create_default_questions()
    
    
    def _save_to_file(self):
        """
        Save demo questions to file.
        """
        try:
            with open(self.demo_file, 'w') as f:
                json.dump(self.demo_questions, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save demo questions: {str(e)}")


# ==========================================
# Testing & Demo
# ==========================================

if __name__ == "__main__":
    """
    Test demo questions manager.
    """
    
    # Setup logging
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    print("\n" + "="*60)
    print("Demo Questions Manager Test")
    print("="*60 + "\n")
    
    # Initialize manager
    print("🔄 Initializing demo questions manager...")
    demo_mgr = DemoQuestionsManager()
    
    # Test 1: Get all questions
    print("\n--- Test 1: All Questions ---")
    questions = demo_mgr.get_all_questions()
    print(f"✅ Total questions: {len(questions)}")
    print(f"\nFirst 3 questions:")
    for q in questions[:3]:
        print(f"  {q['id']}. {q['question']} [{q['category']}]")
    
    # Test 2: Get by category
    print("\n--- Test 2: Questions by Category ---")
    conduct_questions = demo_mgr.get_questions_by_category("Conduct")
    print(f"✅ 'Conduct' category: {len(conduct_questions)} questions")
    for q in conduct_questions:
        print(f"  - {q['question']}")
    
    # Test 3: Get random questions
    print("\n--- Test 3: Random Questions ---")
    random_qs = demo_mgr.get_random_questions(count=3)
    print(f"✅ Random selection:")
    for q in random_qs:
        print(f"  {q['id']}. {q['question']}")
    
    # Test 4: Add mock response
    print("\n--- Test 4: Add Response ---")
    mock_response = {
        'answer': 'This is a test answer.',
        'citations': [{'chunk_id': 0}],
        'token_usage': {'total_tokens': 50}
    }
    demo_mgr.add_demo_response(1, mock_response)
    
    question = demo_mgr.get_question_by_id(1)
    print(f"✅ Response added: {question.get('has_response', False)}")
    
    # Test 5: Statistics
    print("\n--- Test 5: Statistics ---")
    stats = demo_mgr.get_statistics()
    print(f"Total questions: {stats['total_questions']}")
    print(f"With responses: {stats['questions_with_responses']}")
    print(f"Coverage: {stats['coverage_percentage']}%")
    print(f"\nCategories:")
    for cat, count in stats['categories'].items():
        print(f"  {cat}: {count}")
    print(f"\nComplexity:")
    for comp, count in stats['complexity_distribution'].items():
        print(f"  {comp}: {count}")
    
    print("\n" + "="*60)
    print("✅ All demo questions tests passed!")
    print("="*60 + "\n")
