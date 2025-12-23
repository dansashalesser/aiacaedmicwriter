"""Topic segmentation logic using LLM to extract research topics from paper ideas."""

import os
from typing import List
from pydantic import BaseModel, Field, field_validator
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langfuse import observe

# Load environment variables
load_dotenv()


# Pydantic model for structured LLM output
class TopicSegmentation(BaseModel):
    """Structured output for topic segmentation."""
    topics: List[str] = Field(
        description="List of 1-5 general research topics extracted from the user's idea",
        min_length=1,
        max_length=5
    )

    @field_validator('topics')
    @classmethod
    def validate_topics(cls, v):
        # Remove empty strings and strip whitespace
        v = [topic.strip() for topic in v if topic.strip()]
        # Ensure at least 1 topic
        if len(v) == 0:
            raise ValueError("Must have at least 1 topic")
        # Truncate to max 5 topics
        return v[:5]


# System prompt for topic extraction
TOPIC_EXTRACTION_PROMPT = """You are an academic research assistant. Your task is to analyze a user's
paper idea and extract 1-5 general research topics that should be searched in academic databases.

Guidelines:
- If a topic doesn't need segmenting, you will **NOT SEGMENT BY FORCE** - the goal is to have CONCRETE academic topics
- If a topic does require segmenting, extract broad, searchable topics (not full sentences)
- Focus on key concepts and domains
- Use standard academic terminology
- You will return one to five topics MAX (fewer is better if they cover the idea well)
- Topics should be complementary, not overlapping

Examples:
Input: "An LLM that helps in predicting natural disasters"
Output: ["large language models", "natural disaster prediction", "environmental forecasting"]

Input: "Machine Learning"
Output: ["machine learning"]"""


@observe(name="segment-topics", as_type="generation")
def segment_topics(user_input: str, max_topics: int = 5) -> List[str]:
    """
    Use LLM to segment a user's paper idea into research topics.

    Args:
        user_input: Detailed paper idea from the user
        max_topics: Maximum number of topics to return (default 5)

    Returns:
        List of 1-5 research topics

    Raises:
        Exception: If LLM call fails (caller should handle with fallback)
    """
    user_prompt = f"Extract research topics from this paper idea:\n\n{user_input}"

    # Initialize LLM with structured output
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    structured_llm = llm.with_structured_output(TopicSegmentation)

    # Make LLM call
    result = structured_llm.invoke([
        {"role": "system", "content": TOPIC_EXTRACTION_PROMPT},
        {"role": "user", "content": user_prompt}
    ])

    # Handle both dict and Pydantic model responses
    if isinstance(result, dict):
        topics = result.get('topics', [])
    else:
        topics = result.topics

    # Respect max_topics limit
    if len(topics) > max_topics:
        topics = topics[:max_topics]

    return topics
