"""HTTP request/response DTOs for the FastAPI layer."""

from pydantic import BaseModel, Field

from domain import Citation


class QueryRequest(BaseModel):
    query: str = Field(
        min_length=1,
        description="Natural-language engineering question (Dutch or English).",
        examples=["Wat zijn de eisen voor brandwerende deuren?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of reranked parent chunks to include in the context.",
        examples=[5],
    )
    project: str | None = Field(
        default=None,
        description="Optional exact project/dossier code filter (e.g. '12345.001').",
        examples=["12345.001"],
    )
    building: str | None = Field(
        default=None,
        description="Optional building/wing filter matched against multi-value `buildings` metadata.",
        examples=["Blok A"],
    )
    phase: str | None = Field(
        default=None,
        description="Optional construction phase filter (e.g. 'Voorontwerp', 'Definitief Ontwerp', 'Uitvoering').",
        examples=["Uitvoering"],
    )
    auto_filter: bool = Field(
        default=False,
        description="Automatically infer project, building, and phase filters from the query text.",
        examples=[False],
    )


class QueryResponse(BaseModel):
    answer: str = Field(
        ...,
        description="Grounded answer synthesized from the top reranked parent chunks.",
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Ordered list of source document citations supporting the answer.",
    )
