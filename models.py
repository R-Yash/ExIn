from enum import Enum
from typing import List, Optional, TypedDict
from pydantic import BaseModel, Field

class InterviewPhase(str, Enum):
    """Discrete phases of the interview lifecycle."""

    WELCOME = "welcome"
    QUESTIONING = "questioning"
    EVALUATION = "evaluation"
    SUMMARY = "summary"
    COMPLETED = "completed"

class SkillLevel(str, Enum):
    """Estimated Excel proficiency level for adaptive questioning."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

class QuestionOutput(BaseModel):
    """Output of the question-generation tool."""
    question: str = Field(..., description="A question to ask the candidate.")
    topics: List[str] = Field(...,description="A list of excel topics covered in the question.")

class InterviewResponse(BaseModel):
    """
    Structured response from the Interview Coordinator agent.
    """

    message: Optional[str] = Field(..., description="A Message to send to the candidate. Does not include question.")
    question: Optional[QuestionOutput] = Field(default=None, description="Current question asked by the interviewer, along with the topics.")
    
class CandidateInput(BaseModel):
    """
    Latest input message received from the candidate.
    """

    response: str = Field(..., description="The candidate's response to the current question or instruction")

# Graph State for LangGraph workflow
class GraphState(TypedDict):
    """Mutable state shared across the interview workflow.

    Keys:
        interviewer_messages: All messages sent by the coordinator.
        candidate_messages: All messages from the candidate.
        current_phase: Current `InterviewPhase`.
        total_questions: Target number of questions.
        skill_level: Estimated `SkillLevel`, if known.
        done: Whether the interview is completed.
    """
    interviewer_messages: List[str]
    candidate_messages: List[str]
    current_phase: InterviewPhase
    total_questions: int
    skill_level: Optional[SkillLevel]
    done: bool


class EvaluationInput(BaseModel):
    """Payload for the evaluator: lists of questions and responses."""
    questions: List[str] = Field(..., description="Ordered list of asked questions.")
    responses: List[str] = Field(..., description="Ordered list of candidate responses.")


class ScoreBreakdown(BaseModel):
    """Per-question scoring details."""
    question: str
    response: str
    difficulty: str
    accuracy_score: float
    relevance_score: float
    weighted_points: float


class EvaluationResult(BaseModel):
    """Overall evaluation result with breakdown and totals."""
    total_points: float
    max_points: float
    percentage: float
    breakdown: List[ScoreBreakdown]
