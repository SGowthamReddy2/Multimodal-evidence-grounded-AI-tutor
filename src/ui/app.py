"""Minimal 3-Pane Gradio Interface for the Multimodal AI Tutor.

Implements the specification:
1. Left Pane: Interactive Chatbot Workspace with verified citation badges
2. Right Top Pane: Synchronized Media Viewer (Video + Slide/Keyframe Image)
3. Right Bottom Pane: Learner Memory Governance Panel (Student CRUD)
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from src.config import DATA_DIR, FRAMES_DIR, PROJECT_ROOT, RAW_DIR, SLIDES_DIR
from src.service import TutorService

logger = logging.getLogger(__name__)

SEEK_VIDEO_JS = """
(seconds, display_str) => {
    let video = document.querySelector("#lecture_video_player video") || document.querySelector("video");
    if (!video) {
        console.warn("No video element found in DOM to seek");
        return;
    }
    let t = parseFloat(seconds);
    if ((isNaN(t) || t <= 0) && display_str && display_str.includes(":")) {
        let clean = display_str.split(" ")[0].trim();
        let parts = clean.split(":");
        if (parts.length === 2) {
            let m = parseFloat(parts[0]);
            let s = parseFloat(parts[1]);
            if (!isNaN(m) && !isNaN(s)) {
                t = m * 60 + s;
            }
        }
    }
    if (!isNaN(t) && t >= 0) {
        video.currentTime = t;
        let playPromise = video.play();
        if (playPromise !== undefined) {
            playPromise.catch(err => {
                console.log("Video playback notice:", err);
            });
        }
    }
}
"""


def format_timestamp_min_sec(ts_ms: Optional[int]) -> Tuple[str, float]:
    """Format millisecond timestamp into 'MM:SS (X min Y sec)' and float seconds."""
    if ts_ms is None or ts_ms < 0:
        return "N/A", 0.0
    total_sec = ts_ms / 1000.0
    minutes = int(total_sec // 60)
    seconds = int(total_sec % 60)
    return f"{minutes:02d}:{seconds:02d} ({minutes}m {seconds:02d}s)", total_sec


def build_app(service: Optional[TutorService] = None) -> gr.Blocks:
    """Construct the Gradio Blocks web application."""
    tutor_service = service or TutorService()

    # Find primary video file in data/raw/lecture-03/
    lec3_videos = list((RAW_DIR / "lecture-03").glob("*.mp4"))
    default_video_path = str(lec3_videos[0]) if lec3_videos else None

    with gr.Blocks(title="MIT 6.006 Multimodal AI Tutor") as demo:
        gr.Markdown(
            """
            # 🎓 Grounded Multimodal AI Tutor (MIT 6.006)
            **Interactive Algorithm Tutor with Timestamp Citations, LEARNet Keyframes, and Selective Learner Memory**
            """
        )

        with gr.Row():
            # ==========================================================
            # Left Pane: Chat Workspace (scale=6)
            # ==========================================================
            with gr.Column(scale=6):
                gr.Markdown("### 💬 Tutor Dialogue")
                chatbot = gr.Chatbot(
                    label="Tutor Conversation",
                    height=520,
                    avatar_images=(None, "https://upload.wikimedia.org/wikipedia/commons/0/0c/MIT_logo.svg"),
                )

                with gr.Row():
                    query_input = gr.Textbox(
                        placeholder="Ask about Lecture 3 (e.g. 'How does merge sort recursion tree calculate work at each level?')",
                        label="Student Question",
                        scale=8,
                        lines=2,
                    )
                    submit_btn = gr.Button("Submit", variant="primary", scale=2)

                with gr.Row():
                    config_selector = gr.Radio(
                        choices=["Config A (Quality)", "Config B (Speed)"],
                        value="Config A (Quality)",
                        label="Operational Mode (M8 Pareto Benchmark)",
                    )
                    clear_btn = gr.ClearButton([chatbot, query_input])

                decision_badge = gr.Markdown("🟢 **Tutor Status**: Ready")

            # ==========================================================
            # Right Pane: Evidence & Memory Tabs (scale=6)
            # ==========================================================
            with gr.Column(scale=6):
                with gr.Tabs():
                    # Tab 1: Synchronized Media & Evidence
                    with gr.Tab("📺 Media & Evidence Viewer"):
                        gr.Markdown("#### 🎥 Synchronized Lecture Player")
                        video_player = gr.Video(
                            value=default_video_path,
                            label="MIT 6.006 Lecture Video",
                            autoplay=False,
                            height=260,
                            elem_id="lecture_video_player",
                        )

                        target_timestamp_sec = gr.Number(value=0.0, visible=False)

                        with gr.Row():
                            timestamp_display = gr.Textbox(
                                label="Aligned Timestamp (min:sec)",
                                value="00:00 (0m 00s)",
                                interactive=False,
                                scale=4,
                            )
                            jump_btn = gr.Button("⚡ Play Cited Moment", scale=3, variant="secondary")

                        gr.Markdown("#### 🖼️ Visual Evidence (Slide / Board Keyframe)")
                        evidence_image = gr.Image(
                            label="Current Grounding Asset",
                            type="filepath",
                            height=280,
                        )

                        with gr.Accordion("🔍 Top-K Retrieved Evidence Candidates", open=False):
                            evidence_json = gr.JSON(label="Retrieved Evidence Chunks & Provenance")

                    # Tab 2: Learner Memory Governance Panel
                    with gr.Tab("🧠 Learner Memory Governance"):
                        gr.Markdown(
                            """
                            #### Student Profile Governance
                            Inspect, confirm, edit, or delete durable memory records.
                            """
                        )
                        memory_table = gr.Dataframe(
                            headers=["Memory ID", "Type", "Value", "Authority", "Confidence", "Created At"],
                            datatype=["str", "str", "str", "str", "number", "str"],
                            label="Active Memory Records",
                            interactive=False,
                        )
                        refresh_mem_btn = gr.Button("🔄 Refresh Memory Table")

                        gr.Markdown("##### Add / Confirm Profile Attribute")
                        with gr.Row():
                            mem_type = gr.Dropdown(
                                choices=["goal", "mastery", "misconception", "preference"],
                                value="goal",
                                label="Attribute Type",
                            )
                            mem_value = gr.Textbox(
                                placeholder="e.g., 'Targeting mastery of divide-and-conquer recurrences'",
                                label="Profile Value",
                            )
                        add_mem_btn = gr.Button("Save to Profile", variant="secondary")

                        gr.Markdown("##### Delete Profile Record")
                        with gr.Row():
                            del_id_input = gr.Textbox(placeholder="mem-xxxx", label="Memory ID to Delete")
                            del_btn = gr.Button("Delete Record", variant="stop")

        # --------------------------------------------------------------
        # Reactive Event Handlers
        # --------------------------------------------------------------

        def handle_query(user_query: str, chat_history: List[Dict], config_choice: str):
            if not user_query.strip():
                return (
                    chat_history,
                    "⚪ **Tutor Status**: Empty query",
                    None,
                    "00:00 (0m 00s)",
                    0.0,
                    [],
                )

            config_code = "A" if "Config A" in config_choice else "B"
            chat_history = chat_history or []
            chat_history.append({"role": "user", "content": user_query})

            # Call TutorService
            res = tutor_service.answer_query(
                query=user_query,
                learner_id="student_01",
                config=config_code,
            )

            bot_reply = res["answer"]
            chat_history.append({"role": "assistant", "content": bot_reply})

            decision = res["decision"]
            top_score = res["top_score"]
            if decision == "FULL":
                status_text = f"🟢 **Decision**: FULL ANSWER (Confidence: {top_score:.3f})"
            elif decision == "QUALIFIED":
                status_text = f"🟡 **Decision**: QUALIFIED HEDGE (Confidence: {top_score:.3f})"
            else:
                status_text = f"🔴 **Decision**: EXPLICIT ABSTENTION (Confidence: {top_score:.3f})"

            # Timestamp & asset navigation
            ts_ms = res.get("primary_video_timestamp_ms")
            ts_str, ts_sec = format_timestamp_min_sec(ts_ms)
            asset_path = res.get("primary_asset_path")
            if asset_path:
                asset_path_obj = Path(asset_path).resolve()
                asset_path = str(asset_path_obj) if asset_path_obj.exists() else None

            return (
                chat_history,
                status_text,
                asset_path,
                ts_str,
                ts_sec,
                res.get("evidence_badges", []),
            )

        def load_memory_table():
            records = tutor_service.memory.list_all_records("student_01")
            data = [
                [
                    r["memory_id"],
                    r["type"],
                    r["value"],
                    r["write_authority"],
                    round(r["confidence"], 2),
                    r["created_at"],
                ]
                for r in records
            ]
            return data

        def add_memory(m_type: str, m_val: str):
            if m_val.strip():
                tutor_service.memory.add_student_memory("student_01", m_type, m_val.strip())
            return load_memory_table()

        def delete_memory(m_id: str):
            if m_id.strip():
                tutor_service.memory.delete_memory(m_id.strip())
            return load_memory_table()

        # Connect events
        submit_btn.click(
            handle_query,
            inputs=[query_input, chatbot, config_selector],
            outputs=[
                chatbot,
                decision_badge,
                evidence_image,
                timestamp_display,
                target_timestamp_sec,
                evidence_json,
            ],
        )
        query_input.submit(
            handle_query,
            inputs=[query_input, chatbot, config_selector],
            outputs=[
                chatbot,
                decision_badge,
                evidence_image,
                timestamp_display,
                target_timestamp_sec,
                evidence_json,
            ],
        )

        jump_btn.click(
            fn=None,
            inputs=[target_timestamp_sec, timestamp_display],
            outputs=None,
            js=SEEK_VIDEO_JS,
        )

        refresh_mem_btn.click(load_memory_table, outputs=[memory_table])
        add_mem_btn.click(add_memory, inputs=[mem_type, mem_value], outputs=[memory_table])
        del_btn.click(delete_memory, inputs=[del_id_input], outputs=[memory_table])

        # Load initial memory table on launch
        demo.load(load_memory_table, outputs=[memory_table])

    return demo


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = build_app()
    allowed_dirs = [
        str(PROJECT_ROOT),
        str(DATA_DIR),
        str(FRAMES_DIR),
        str(SLIDES_DIR),
        str(RAW_DIR),
    ]
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        allowed_paths=allowed_dirs,
    )

