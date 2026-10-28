"""Context and badge formatters for retrieved multimodal evidence."""

from typing import Dict, List
from src.models import RetrievedEvidence


def format_context_for_llm(evidence_list: List[RetrievedEvidence]) -> str:
    """Format retrieved evidence into a structured context block for the LLM prompt.

    Enforces unambiguous [evidence_id] tagging to enable the LLM to cite evidence
    accurately in its response using [claim_id | evidence_id].
    """
    if not evidence_list:
        return "No relevant course evidence found."

    blocks = []
    for idx, ev in enumerate(evidence_list, start=1):
        lines = [f"--- Evidence Item {idx} ---"]
        lines.append(f"Evidence ID: [{ev.evidence_id}]")
        lines.append(f"Source: {ev.source_id}")
        lines.append(f"Modality: {ev.modality}")

        if ev.modality == "transcript":
            start_s = ev.start_ms / 1000.0
            end_s = ev.end_ms / 1000.0
            lines.append(f"Timestamp: {start_s:.2f}s - {end_s:.2f}s ({ev.start_ms}ms - {ev.end_ms}ms)")
            lines.append(f"Verbatim Lecture Audio:\n\"{ev.transcript_span}\"")
        elif ev.modality == "slide":
            lines.append(f"Slide Page: {ev.page_number}")
            lines.append(f"Slide Notes Content:\n\"{ev.transcript_span}\"")

        if ev.visual_caption:
            lines.append(f"Visual Diagram Description: {ev.visual_caption}")

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def format_evidence_badges(evidence_list: List[RetrievedEvidence]) -> List[Dict]:
    """Format evidence into UI badge representations for the Gradio frontend."""
    badges = []
    for ev in evidence_list:
        badge = {
            "evidence_id": ev.evidence_id,
            "modality": ev.modality,
            "score": round(ev.score, 4),
            "start_ms": ev.start_ms,
            "end_ms": ev.end_ms,
            "frame_path": ev.frame_asset_path,
            "slide_path": ev.slide_asset_path,
            "page_number": ev.page_number,
            "snippet": (ev.transcript_span[:120] + "...") if len(ev.transcript_span) > 120 else ev.transcript_span,
        }
        badges.append(badge)
    return badges

