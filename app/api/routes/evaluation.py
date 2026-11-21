"""Evaluation API routes for RAG system."""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.document import EvaluationMetrics
from app.services.evaluation import EvaluationService

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])


@router.get("/metrics", response_model=EvaluationMetrics)
async def get_evaluation_metrics(db: AsyncSession = Depends(get_db)):
    """Get current evaluation metrics for RAG system."""
    evaluation_service = EvaluationService()
    metrics = await evaluation_service.get_current_metrics()

    return EvaluationMetrics(
        faithfulness_score=metrics.get("faithfulness_score", 0.0),
        answer_relevance_score=metrics.get("answer_relevance_score", 0.0),
        context_precision_score=metrics.get("context_precision_score", 0.0),
        context_recall_score=metrics.get("context_recall_score", 0.0),
        overall_score=metrics.get("overall_score", 0.0),
    )


@router.post("/report", response_model=EvaluationMetrics)
async def generate_evaluation_report(db: AsyncSession = Depends(get_db)):
    """Generate comprehensive evaluation report for RAG system."""
    evaluation_service = EvaluationService()
    report = await evaluation_service.generate_report()

    return EvaluationMetrics(**report)
