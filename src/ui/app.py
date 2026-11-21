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

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

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


def find_lecture_video(course_id: Optional[str], lecture_id: Optional[str]) -> Optional[str]:
    """Locate the MP4 video file for a specific course and lecture."""
    if not course_id or course_id == "All Courses":
        course_id = DEFAULT_COURSE_ID

    if not lecture_id or lecture_id == "All Lectures":
        p = RAW_DIR / course_id
        if p.exists():
            vids = list(p.rglob("*.mp4"))
            if vids:
                return str(vids[0])
        all_vids = list(RAW_DIR.rglob("*.mp4"))
        return str(all_vids[0]) if all_vids else None

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

    # 3. Search for lecture substring in any mp4 in RAW_DIR
    lec_clean = lecture_id.replace("lecture-", "").replace("lec", "").strip()
    try:
        lec_int = int(lec_clean)
        for mp4 in RAW_DIR.rglob("*.mp4"):
            if f"Lecture_{lec_int}" in mp4.name or f"Lecture_{lec_int:02d}" in mp4.name:
                return str(mp4)
    except Exception:
        pass

    all_vids = list(RAW_DIR.rglob("*.mp4"))
    if all_vids:
        return str(all_vids[0])

    return None


def get_session_choices(tutor_service: TutorService, learner_id: str = "student_01") -> Tuple[List[Tuple[str, str]], Optional[str]]:
    """Retrieve formatted session choices (display_label, session_id) and the active session ID."""
    sessions = tutor_service.memory.list_sessions(learner_id)
    if not sessions:
        first_id = tutor_service.memory.create_session(learner_id, "Interactive Session")
        sessions = tutor_service.memory.list_sessions(learner_id)

    choices = []
    for s in sessions:
        title = (s.get("title") or "Untitled Session").strip()
        sid = s["session_id"]
        short_sid = sid[5:13] if sid.startswith("sess-") else sid[:8]
        display_label = f"{title[:28]} ({short_sid})"
        choices.append((display_label, sid))
    active_id = sessions[0]["session_id"] if sessions else None
    return choices, active_id


def load_session_messages_for_chatbot(tutor_service: TutorService, session_id: Optional[str]) -> List[Dict[str, str]]:
    """Load dialogue messages from SQLite for a session and format for gr.Chatbot."""
    if not session_id:
        return []
    raw_msgs = tutor_service.memory.get_session_messages(session_id, limit=100)
    formatted = []
    for m in raw_msgs:
        role = m.get("role", "user")
        if role in ("user", "assistant"):
            formatted.append({"role": role, "content": m.get("content", "")})
    return formatted


def load_memory_table_data(tutor_service: TutorService, learner_id: str = "student_01") -> List[List[Any]]:
    """Retrieve and format learner memory records for the governance dataframe."""
    records = tutor_service.memory.list_all_records(learner_id)
    return [
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


def load_core_memory_markdown(tutor_service: TutorService, learner_id: str = "student_01") -> str:
    """Return formatted markdown profile card of active core memory blocks."""
    if not hasattr(tutor_service, "memory") or not hasattr(tutor_service.memory, "get_active_core_memory_blocks"):
        return "### 🧠 Active Student Profile: Initializing..."

    blocks = tutor_service.memory.get_active_core_memory_blocks(learner_id)
    goals = blocks.get("goals", [])
    mastery = blocks.get("mastery", [])
    misconceptions = blocks.get("misconceptions", [])
    preferences = blocks.get("preferences", [])

    total_active = len(goals) + len(mastery) + len(misconceptions) + len(preferences)

    lines = [
        "### 🧠 Active Student Learning Profile",
        f"*Durable working memory actively maintained across all chat sessions ({total_active} active trait(s))*",
        "",
    ]

    # Goals section
    lines.append("#### 🎯 Goals & Deadlines")
    if goals:
        for g in goals:
            lines.append(f"- **{g['value']}** *(Noted: {g['created_at'][:10]})*")
    else:
        lines.append("- *No specific assessment deadlines or goals recorded yet.*")
    lines.append("")

    # Knowledge & Mastery section
    lines.append("#### 📈 Knowledge State & Mastery")
    if mastery or misconceptions:
        for m in mastery:
            lines.append(f"- **{m['value']}**")
        for misc in misconceptions:
            lines.append(f"- ⚠️ **{misc['value']}** *(Focus area)*")
    else:
        lines.append("- *General MIT 6.006 student baseline.*")
    lines.append("")

    # Preferences section
    lines.append("#### 🎨 Learning Preferences")
    if preferences:
        for p in preferences:
            lines.append(f"- **{p['value']}**")
    else:
        lines.append("- *Balanced pedagogical explanations (step-by-step mathematical & visual).*")

    return "\n".join(lines)


def handle_app_init(tutor_service: TutorService, learner_id: str = "student_01"):
    """Restore active session, chat history, and learner profile on launch or browser refresh."""
    choices, active_id = get_session_choices(tutor_service, learner_id)
    history = load_session_messages_for_chatbot(tutor_service, active_id)
    mem_table = load_memory_table_data(tutor_service, learner_id)
    badge = (
        f"🟢 **Tutor Status**: Restored session with {len(history)} message(s)"
        if history
        else "🟢 **Tutor Status**: Ready (Ephemeral Memory Active)"
    )
    return (
        gr.Dropdown(choices=choices, value=active_id),
        history,
        active_id,
        mem_table,
        badge,
    )


def handle_session_select(tutor_service: TutorService, selected_session_id: Optional[str]):
    """Handle switching between sessions from the dropdown."""
    if not selected_session_id:
        return [], None, "⚪ **Tutor Status**: No Session Selected"
    history = load_session_messages_for_chatbot(tutor_service, selected_session_id)
    badge = f"🟢 **Tutor Status**: Active Session Loaded ({len(history)} message(s))"
    return history, selected_session_id, badge


def handle_new_session_click(tutor_service: TutorService, learner_id: str = "student_01"):
    """Create a brand new conversation session and clear the chatbot workspace."""
    new_sess_id = tutor_service.memory.create_session(learner_id, "New Session")
    choices, _ = get_session_choices(tutor_service, learner_id)
    return (
        gr.Dropdown(choices=choices, value=new_sess_id),
        [],
        "",
        new_sess_id,
        "🟢 **Tutor Status**: Started New Chat Session",
    )


def handle_delete_session_click(tutor_service: TutorService, active_session_id: Optional[str], learner_id: str = "student_01"):
    """Delete the current chat session and load the next available session."""
    if active_session_id:
        tutor_service.memory.delete_session(active_session_id)
    choices, next_id = get_session_choices(tutor_service, learner_id)
    history = load_session_messages_for_chatbot(tutor_service, next_id)
    badge = (
        f"🟡 **Tutor Status**: Session deleted. Active: {next_id[:8]} ({len(history)} message(s))"
        if next_id
        else "⚪ **Tutor Status**: No Active Sessions"
    )
    return (
        gr.Dropdown(choices=choices, value=next_id),
        history,
        next_id,
        badge,
    )


def build_app(service: Optional[TutorService] = None) -> gr.Blocks:
    """Construct the Gradio Blocks web application."""
    tutor_service = service or TutorService()

    ALL_COURSES = "All Courses"
    ALL_LECTURES = "All Lectures"

    # Discover available courses and lectures
    available_courses_map = tutor_service.list_available_courses_and_lectures()
    course_choices = sorted(list(available_courses_map.keys())) or [DEFAULT_COURSE_ID]
    default_course = course_choices[0]
    all_known_lectures = sorted(list({lec for lecs in available_courses_map.values() for lec in lecs})) or ["lecture-03", "lecture-04", "lecture-06", "lecture-07"]

    initial_video_path = find_lecture_video(default_course, all_known_lectures[0])

    with gr.Blocks(
        title="Multimodal Evidence-Grounded AI Tutor",
        analytics_enabled=False,
    ) as demo:
        gr.Markdown(
            """
            # 🎓 Multimodal Evidence-Grounded AI Tutor
            **Interactive Algorithm Tutor with Timestamp Citations, VLM Blackboard Extraction, and Selective Learner Memory**
            """
        )

        # Multi-Course & Multi-Lecture Scope Bar with Checkboxes
        with gr.Group():
            with gr.Row():
                with gr.Column(scale=6):
                    with gr.Row():
                        course_selector = gr.Dropdown(
                            choices=[ALL_COURSES] + course_choices,
                            value=[ALL_COURSES],
                            multiselect=True,
                            label="📚 Courses (Check options in dropdown)",
                            info="Select one or more courses, or 'All Courses'",
                            scale=9,
                        )
                        all_courses_btn = gr.Button("All Courses", scale=3, size="sm", variant="secondary")
                with gr.Column(scale=6):
                    with gr.Row():
                        lecture_selector = gr.Dropdown(
                            choices=[ALL_LECTURES] + all_known_lectures,
                            value=[ALL_LECTURES],
                            multiselect=True,
                            label="🎥 Lectures (Check options in dropdown)",
                            info="Select one or more lectures, or 'All Lectures'",
                            scale=9,
                        )
                        all_lectures_btn = gr.Button("All Lectures", scale=3, size="sm", variant="secondary")
            scope_badge = gr.Markdown(
                "🌐 **Retrieval Scope**: **All Courses** | **All Lectures** *(Searching across all available course materials)*"
            )

        with gr.Row():
            # ==========================================================
            # Left Pane: Chat Workspace (scale=6)
            # ==========================================================
            with gr.Column(scale=6):
                gr.Markdown("### 💬 Tutor Dialogue")
                with gr.Row():
                    session_selector = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="🗂️ Active Chat Session",
                        scale=8,
                        interactive=True,
                    )
                    new_session_btn = gr.Button("➕ New", scale=2, variant="secondary")
                    del_session_btn = gr.Button("🗑️", scale=1, variant="stop", min_width=45)

                chatbot = gr.Chatbot(
                    label="Tutor Conversation",
                    height=480,
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
                        scale=7,
                    )
                    clear_input_btn = gr.ClearButton([query_input], value="🧹 Clear Input", scale=3)

                session_state = gr.State(value=None)
                decision_badge = gr.Markdown("🟢 **Tutor Status**: Ready (Ephemeral Memory Active)")

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

                    # Tab 2: Student Learning Profile & Memory Manager
                    with gr.Tab("🧠 Student Learning Profile"):
                        core_memory_md = gr.Markdown(
                            value=load_core_memory_markdown(tutor_service, "student_01"),
                        )

                        gr.Markdown("---")
                        gr.Markdown("##### ⚙️ Manage Active Memories")
                        with gr.Row():
                            active_items_dropdown = gr.Dropdown(
                                choices=tutor_service.memory.get_active_memory_choices("student_01") if hasattr(tutor_service, "memory") and hasattr(tutor_service.memory, "get_active_memory_choices") else [],
                                value=None,
                                label="Select an Active Memory to Remove",
                                scale=8,
                                interactive=True,
                            )
                            forget_btn = gr.Button("🗑️ Forget Selected", variant="stop", scale=4)

                        with gr.Row():
                            clear_all_mem_btn = gr.Button("🧹 Clear All Profile Memories", variant="secondary", scale=6)
                            refresh_mem_btn = gr.Button("🔄 Refresh Profile", variant="secondary", scale=6)

                        gr.Markdown("---")
                        gr.Markdown("##### ➕ Manually Add Goal or Preference")
                        with gr.Row():
                            mem_type = gr.Dropdown(
                                choices=[
                                    ("🎯 Goal & Deadline", "goals"),
                                    ("📈 Concept Mastery / Focus", "mastery"),
                                    ("🎨 Learning Preference", "preferences"),
                                ],
                                value="goals",
                                label="Category",
                                scale=4,
                            )
                            mem_value = gr.Textbox(
                                placeholder="e.g. 'Midterm exam tomorrow' or 'Prefers step-by-step visual proofs'",
                                label="Content",
                                scale=8,
                            )
                        add_mem_btn = gr.Button("➕ Save to Profile", variant="primary")

        # --------------------------------------------------------------
        # Scope and Dynamic Selection Event Handlers
        # --------------------------------------------------------------

        def format_scope_badge_text(courses: Optional[List[str]], lectures: Optional[List[str]]) -> str:
            courses = courses or []
            lectures = lectures or []
            is_all_c = not courses or ALL_COURSES in courses
            is_all_l = not lectures or ALL_LECTURES in lectures

            if is_all_c and is_all_l:
                return "🌐 **Retrieval Scope**: **All Courses** | **All Lectures** *(Searching across all available course materials)*"

            clean_c = [c for c in courses if c != ALL_COURSES]
            clean_l = [l for l in lectures if l != ALL_LECTURES]

            if is_all_c:
                c_str = "**All Courses**"
            elif len(clean_c) == 1:
                c_str = f"Course: `{clean_c[0]}`"
            else:
                c_str = f"Courses: `{', '.join(clean_c)}`"

            if is_all_l:
                l_str = "**All Lectures**"
            elif len(clean_l) == 1:
                l_str = f"Lecture: `{clean_l[0]}`"
            else:
                l_str = f"Lectures: `{', '.join(clean_l)}` ({len(clean_l)} selected)"

            return f"🔍 **Retrieval Scope**: {c_str} | {l_str}"

        def on_all_courses_btn_click(current_lectures: List[str]):
            mapping = tutor_service.list_available_courses_and_lectures()
            all_c = sorted(list(mapping.keys())) or [DEFAULT_COURSE_ID]
            all_l = sorted(list({l for lecs in mapping.values() for l in lecs})) or ["lecture-03"]

            new_c = [ALL_COURSES]
            lec_choices = [ALL_LECTURES] + all_l
            current_lectures = current_lectures or [ALL_LECTURES]
            target_l = current_lectures[0] if current_lectures[0] != ALL_LECTURES else all_l[0]
            vid = find_lecture_video(all_c[0], target_l)
            badge = format_scope_badge_text(new_c, current_lectures)
            return gr.update(value=new_c), gr.update(choices=lec_choices), vid, badge

        def on_all_lectures_btn_click(selected_courses: List[str]):
            mapping = tutor_service.list_available_courses_and_lectures()
            all_c = sorted(list(mapping.keys())) or [DEFAULT_COURSE_ID]

            selected_courses = selected_courses or []
            specific_c = [c for c in selected_courses if c != ALL_COURSES]
            target_c = specific_c[0] if specific_c else all_c[0]

            new_l = [ALL_LECTURES]
            first_lec = mapping.get(target_c, ["lecture-03"])[0]
            vid = find_lecture_video(target_c, first_lec)
            badge = format_scope_badge_text(selected_courses, new_l)
            return gr.update(value=new_l), vid, badge

        def on_course_change(selected_courses: List[str], current_lectures: List[str]):
            mapping = tutor_service.list_available_courses_and_lectures()
            all_c = sorted(list(mapping.keys())) or [DEFAULT_COURSE_ID]
            all_l = sorted(list({l for lecs in mapping.values() for l in lecs})) or ["lecture-03"]

            selected_courses = selected_courses or []
            specific_c = [c for c in selected_courses if c != ALL_COURSES]

            if not specific_c:
                valid_lecture_choices = [ALL_LECTURES] + all_l
                target_c = all_c[0]
            else:
                sub_l = set()
                for c in specific_c:
                    sub_l.update(mapping.get(c, []))
                valid_lecture_choices = [ALL_LECTURES] + sorted(list(sub_l))
                target_c = specific_c[0]

            current_lectures = current_lectures or [ALL_LECTURES]
            if ALL_LECTURES in current_lectures:
                updated_lectures = [ALL_LECTURES]
            else:
                updated_lectures = [l for l in current_lectures if l in valid_lecture_choices]
                if not updated_lectures:
                    updated_lectures = [ALL_LECTURES]

            target_l = updated_lectures[0] if updated_lectures[0] != ALL_LECTURES else mapping.get(target_c, ["lecture-03"])[0]
            vid = find_lecture_video(target_c, target_l)

            badge = format_scope_badge_text(selected_courses, updated_lectures)
            return gr.update(choices=valid_lecture_choices, value=updated_lectures), vid, badge

        def on_lecture_change(selected_lectures: List[str], selected_courses: List[str]):
            mapping = tutor_service.list_available_courses_and_lectures()
            all_c = sorted(list(mapping.keys())) or [DEFAULT_COURSE_ID]

            selected_courses = selected_courses or []
            specific_c = [c for c in selected_courses if c != ALL_COURSES]
            target_c = specific_c[0] if specific_c else all_c[0]

            selected_lectures = selected_lectures or []
            specific_l = [l for l in selected_lectures if l != ALL_LECTURES]
            target_l = specific_l[0] if specific_l else mapping.get(target_c, ["lecture-03"])[0]
            vid = find_lecture_video(target_c, target_l)

            badge = format_scope_badge_text(selected_courses, selected_lectures)
            return vid, badge

        all_courses_btn.click(
            on_all_courses_btn_click,
            inputs=[lecture_selector],
            outputs=[course_selector, lecture_selector, video_player, scope_badge],
        )

        all_lectures_btn.click(
            on_all_lectures_btn_click,
            inputs=[course_selector],
            outputs=[lecture_selector, video_player, scope_badge],
        )

        course_selector.change(
            on_course_change,
            inputs=[course_selector, lecture_selector],
            outputs=[lecture_selector, video_player, scope_badge],
        )

        lecture_selector.change(
            on_lecture_change,
            inputs=[lecture_selector, course_selector],
            outputs=[video_player, scope_badge],
        )

        def on_app_init():
            return handle_app_init(tutor_service, "student_01")

        def on_session_select(selected_session_id: Optional[str]):
            return handle_session_select(tutor_service, selected_session_id)

        def on_new_session_click():
            return handle_new_session_click(tutor_service, "student_01")

        def on_delete_session_click(active_session_id: Optional[str]):
            return handle_delete_session_click(tutor_service, active_session_id, "student_01")

        def handle_query(
            user_query: str,
            chat_history: List[Dict],
            config_choice: str,
            selected_courses: List[str],
            selected_lectures: List[str],
            current_session_id: Optional[str],
        ):
            if not user_query.strip():
                choices, _ = get_session_choices(tutor_service, "student_01")
                return (
                    chat_history,
                    user_query,
                    "⚪ **Tutor Status**: Empty query",
                    None,
                    "00:00 (0m 00s)",
                    0.0,
                    [],
                    current_session_id,
                    gr.update(choices=load_active_choices(), value=None),
                    gr.Dropdown(choices=choices, value=current_session_id),
                    load_core_md(),
                    gr.skip(),
                )

            config_code = "A" if "Config A" in config_choice else "B"
            chat_history = chat_history or []
            chat_history.append({"role": "user", "content": user_query})

            # Determine course/lecture filtering scope
            mapping = tutor_service.list_available_courses_and_lectures()
            all_c = sorted(list(mapping.keys())) or [DEFAULT_COURSE_ID]

            # If "All Courses" selected or empty or all individual courses selected -> None (search all)
            if not selected_courses or ALL_COURSES in selected_courses:
                target_courses = None
            else:
                clean_c = [c for c in selected_courses if c != ALL_COURSES]
                target_courses = None if set(clean_c) == set(all_c) else clean_c

            # If "All Lectures" selected or empty -> None (search all lectures in scope)
            if not selected_lectures or ALL_LECTURES in selected_lectures:
                target_lectures = None
            else:
                clean_l = [l for l in selected_lectures if l != ALL_LECTURES]
                target_lectures = clean_l if clean_l else None

            # Ensure we have a valid active session
            if not current_session_id:
                current_session_id = tutor_service.memory.get_or_create_session("student_01")

            # Call TutorService with session_id and multi-course/lecture scope
            res = tutor_service.answer_query(
                query=user_query,
                learner_id="student_01",
                session_id=current_session_id,
                config=config_code,
                course_id=target_courses,
                lecture_id=target_lectures,
            )

            bot_reply = res["answer"]
            chat_history.append({"role": "assistant", "content": bot_reply})

            decision = res.get("decision", "FULL")
            top_score = res.get("top_score", 0.0)
            active_session = res.get("session_id", current_session_id)
            memories_added = res.get("proposed_memories", [])

            # Check if session needs auto-titling based on first query
            sessions = tutor_service.memory.list_sessions("student_01")
            for s in sessions:
                if s["session_id"] == active_session:
                    if s.get("title") in ("New Session", "Interactive Session", "Interactive Tutor Session", "Untitled Session"):
                        clean_title = user_query.strip().replace("\n", " ")
                        if len(clean_title) > 35:
                            clean_title = clean_title[:32] + "..."
                        tutor_service.memory.update_session_title(active_session, clean_title)
                    break

            choices, _ = get_session_choices(tutor_service, "student_01")

            if decision == "CONVERSATIONAL":
                status_text = "💬 **Decision**: CONVERSATIONAL (Humane Response)"
            elif decision == "PROFILE_UPDATED":
                status_text = f"🧠 **Decision**: LEARNER PROFILE UPDATED ({len(memories_added)} trait(s) saved)"
            elif decision == "FULL":
                status_text = f"🟢 **Decision**: FULL ANSWER (Confidence: {top_score:.3f})"
            elif decision == "QUALIFIED":
                status_text = f"🟡 **Decision**: QUALIFIED HEDGE (Confidence: {top_score:.3f})"
            else:
                status_text = f"🔴 **Decision**: EXPLICIT ABSTENTION (Confidence: {top_score:.3f})"

            if memories_added and decision not in ("PROFILE_UPDATED",):
                status_text += f" | 💡 Noted {len(memories_added)} profile trait(s)"

            # Timestamp & asset navigation
            ts_ms = res.get("primary_video_timestamp_ms")
            ts_str, ts_sec = format_timestamp_min_sec(ts_ms)
            asset_path = res.get("primary_asset_path")
            if asset_path:
                asset_path_obj = Path(asset_path).resolve()
                asset_path = str(asset_path_obj) if asset_path_obj.exists() else None

            # Video update if cited lecture has video
            ev_lec = res.get("primary_lecture")
            ev_c = res.get("primary_course") or DEFAULT_COURSE_ID
            cited_vid = find_lecture_video(ev_c, ev_lec) if ev_lec else None
            vid_out = cited_vid if cited_vid else gr.skip()

            return (
                chat_history,
                "",  # Clear query input box
                status_text,
                asset_path,
                ts_str,
                ts_sec,
                res.get("evidence_badges", []),
                active_session,
                gr.update(choices=load_active_choices(), value=None),
                gr.Dropdown(choices=choices, value=active_session),
                load_core_md(),
                vid_out,
            )

        def load_active_choices():
            if hasattr(tutor_service, "memory") and hasattr(tutor_service.memory, "get_active_memory_choices"):
                return tutor_service.memory.get_active_memory_choices("student_01")
            return []

        def load_core_md():
            return load_core_memory_markdown(tutor_service, "student_01")

        def forget_selected_memory(m_id: Optional[str]):
            if m_id and str(m_id).strip():
                tutor_service.memory.core_memory_delete(str(m_id).strip(), "Removed by student in profile")
            choices = load_active_choices()
            return load_core_md(), gr.update(choices=choices, value=None)

        def clear_all_memory():
            tutor_service.memory.clear_all_learner_memory("student_01", "Student cleared all profile memories")
            choices = load_active_choices()
            return load_core_md(), gr.update(choices=choices, value=None)

        def refresh_memory():
            choices = load_active_choices()
            return load_core_md(), gr.update(choices=choices, value=None)

        def add_memory(m_type: str, m_val: str):
            if m_val and m_val.strip():
                tutor_service.memory.core_memory_append("student_01", m_type, m_val.strip(), authority="student_authored")
            choices = load_active_choices()
            return load_core_md(), gr.update(choices=choices, value=None), ""

        # Connect events
        query_outputs = [
            chatbot,
            query_input,
            decision_badge,
            evidence_image,
            timestamp_display,
            target_timestamp_sec,
            evidence_json,
            session_state,
            active_items_dropdown,
            session_selector,
            core_memory_md,
            video_player,
        ]

        query_inputs = [
            query_input,
            chatbot,
            config_selector,
            course_selector,
            lecture_selector,
            session_state,
        ]

        submit_btn.click(
            handle_query,
            inputs=query_inputs,
            outputs=query_outputs,
        )
        query_input.submit(
            handle_query,
            inputs=query_inputs,
            outputs=query_outputs,
        )

        session_selector.change(
            on_session_select,
            inputs=[session_selector],
            outputs=[chatbot, session_state, decision_badge],
        )

        new_session_btn.click(
            on_new_session_click,
            inputs=None,
            outputs=[session_selector, chatbot, query_input, session_state, decision_badge],
        )

        del_session_btn.click(
            on_delete_session_click,
            inputs=[session_state],
            outputs=[session_selector, chatbot, session_state, decision_badge],
        )

        jump_btn.click(
            fn=None,
            inputs=[target_timestamp_sec, timestamp_display],
            outputs=None,
            js=SEEK_VIDEO_JS,
        )

        forget_btn.click(
            forget_selected_memory,
            inputs=[active_items_dropdown],
            outputs=[core_memory_md, active_items_dropdown],
        )
        clear_all_mem_btn.click(
            clear_all_memory,
            inputs=None,
            outputs=[core_memory_md, active_items_dropdown],
        )
        refresh_mem_btn.click(
            refresh_memory,
            inputs=None,
            outputs=[core_memory_md, active_items_dropdown],
        )
        add_mem_btn.click(
            add_memory,
            inputs=[mem_type, mem_value],
            outputs=[core_memory_md, active_items_dropdown, mem_value],
        )

        def on_app_init():
            dd_choices, history, active_id, mem_table, badge = handle_app_init(tutor_service, "student_01")
            return (
                dd_choices,
                history,
                active_id,
                load_core_md(),
                gr.update(choices=load_active_choices(), value=None),
                badge,
            )

        # Restore active session, chat history, and memory profile on launch / refresh
        demo.load(
            on_app_init,
            outputs=[
                session_selector,
                chatbot,
                session_state,
                core_memory_md,
                active_items_dropdown,
                decision_badge,
            ],
        )

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
