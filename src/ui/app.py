"""Minimal 3-Pane Gradio Interface for the Multimodal AI Tutor.

Implements:
1. Multi-Course & Multi-Lecture Dynamic Selection
2. Left Pane: Interactive Chatbot Workspace with verified citation badges
3. Right Top Pane: Synchronized Media Viewer (Scoped Video + Slide/Keyframe Image)
4. Right Bottom Pane: Learner Memory Governance Panel (Student CRUD)
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from src.config import DATA_DIR, DEFAULT_COURSE_ID, FRAMES_DIR, PROJECT_ROOT, RAW_DIR, SLIDES_DIR
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


def find_lecture_video(course_id: str, lecture_id: str) -> Optional[str]:
    """Locate the MP4 video file for a specific course and lecture."""
    # 1. Check hierarchical directory: data/raw/<course_id>/<lecture_id>/*.mp4
    p1 = RAW_DIR / course_id / lecture_id
    if p1.exists():
        vids = list(p1.glob("*.mp4"))
        if vids:
            return str(vids[0])

    # 2. Check legacy flat directory: data/raw/<lecture_id>/*.mp4
    p2 = RAW_DIR / lecture_id
    if p2.exists():
        vids = list(p2.glob("*.mp4"))
        if vids:
            return str(vids[0])

    # 3. Fallback: check all mp4 in RAW_DIR
    all_vids = list(RAW_DIR.rglob("*.mp4"))
    if all_vids:
        return str(all_vids[0])

    return None


def build_app(service: Optional[TutorService] = None) -> gr.Blocks:
    """Construct the Gradio Blocks web application."""
    tutor_service = service or TutorService()

    # Discover available courses and lectures
    available_courses_map = tutor_service.list_available_courses_and_lectures()
    course_choices = sorted(list(available_courses_map.keys())) or [DEFAULT_COURSE_ID]
    default_course = course_choices[0]
    lecture_choices = available_courses_map.get(default_course, ["lecture-03"])
    default_lecture = lecture_choices[0] if lecture_choices else "lecture-03"

    initial_video_path = find_lecture_video(default_course, default_lecture)

    with gr.Blocks(title="Multimodal Evidence-Grounded AI Tutor") as demo:
        gr.Markdown(
            """
            # 🎓 Multimodal Evidence-Grounded AI Tutor
            **Interactive Algorithm Tutor with Timestamp Citations, VLM Blackboard Extraction, and Selective Learner Memory**
            """
        )

        # Multi-Course & Multi-Lecture Scope Bar
        with gr.Row():
            course_selector = gr.Dropdown(
                choices=course_choices,
                value=default_course,
                label="📚 Course",
                scale=4,
            )
            lecture_selector = gr.Dropdown(
                choices=lecture_choices,
                value=default_lecture,
                label="🎥 Lecture",
                scale=4,
            )
            scope_selector = gr.Radio(
                choices=["Current Lecture", "Entire Course", "All Courses"],
                value="Current Lecture",
                label="🔍 Retrieval Scope",
                scale=4,
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
                        placeholder="Ask about the lecture (e.g. 'How does merge sort recursion tree calculate work at each level?')",
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
                            value=initial_video_path,
                            label="Lecture Video",
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

        def on_course_change(selected_course: str):
            mapping = tutor_service.list_available_courses_and_lectures()
            lectures = mapping.get(selected_course, ["lecture-03"])
            first_lec = lectures[0] if lectures else "lecture-03"
            vid = find_lecture_video(selected_course, first_lec)
            return gr.update(choices=lectures, value=first_lec), vid

        def on_lecture_change(selected_course: str, selected_lecture: str):
            return find_lecture_video(selected_course, selected_lecture)

        course_selector.change(
            on_course_change,
            inputs=[course_selector],
            outputs=[lecture_selector, video_player],
        )
        lecture_selector.change(
            on_lecture_change,
            inputs=[course_selector, lecture_selector],
            outputs=[video_player],
        )

        def handle_query(
            user_query: str,
            chat_history: List[Dict],
            config_choice: str,
            course_choice: str,
            lecture_choice: str,
            scope_choice: str,
        ):
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

            # Determine course/lecture filtering scope
            target_course = course_choice if scope_choice in ("Current Lecture", "Entire Course") else None
            target_lecture = lecture_choice if scope_choice == "Current Lecture" else None

            # Call TutorService
            res = tutor_service.answer_query(
                query=user_query,
                learner_id="student_01",
                config=config_code,
                course_id=target_course,
                lecture_id=target_lecture,
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
            inputs=[
                query_input,
                chatbot,
                config_selector,
                course_selector,
                lecture_selector,
                scope_selector,
            ],
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
            inputs=[
                query_input,
                chatbot,
                config_selector,
                course_selector,
                lecture_selector,
                scope_selector,
            ],
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
