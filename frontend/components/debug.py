import streamlit as st
import json
import re
from utils.helpers import display_source_content
from utils.performance import display_performance_stats, clear_performance_data
from components.knowledge_graph import display_knowledge_graph_tab
from components.knowledge_graph.management import display_kg_management_tab
from components.styles import KG_MANAGEMENT_CSS

def display_source_content_tab(tabs):
    """Display the source content tab"""
    with tabs[2]:
        if st.session_state.source_content:
            st.markdown('<div class="source-content-container">', unsafe_allow_html=True)
            display_source_content(st.session_state.source_content)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            if st.session_state.agent_type == "deep_research_agent":
                st.info("Deep Research Agent does not provide source content viewing. See the Execution Trace tab for detailed reasoning.")
            else:
                st.info("Click the 'View Source Content' button on an AI answer to view the source text")

def display_execution_trace_tab(tabs):
    """Display the execution trace tab"""
    with tabs[0]:
        if st.session_state.agent_type == "deep_research_agent":
            st.markdown("""
            <div style="padding:10px 0px; margin:15px 0; border-bottom:1px solid #eee;">
                <h2 style="margin:0; color:#333333;">Deep Research Execution Process</h2>
            </div>
            """, unsafe_allow_html=True)

            tool_type = "Enhanced (DeeperResearch)" if st.session_state.get("use_deeper_tool", True) else "Standard (DeepResearch)"
            st.markdown(f"""
            <div style="background-color:#f0f7ff; padding:8px 15px; border-radius:5px; margin-bottom:15px; border-left:4px solid #4285F4;">
                <span style="font-weight:500;">Current tool:</span> {tool_type}
            </div>
            """, unsafe_allow_html=True)

            if st.session_state.get("use_deeper_tool", True):
                with st.expander("Enhanced Feature Details", expanded=False):
                    st.markdown("""
                    #### Community-Aware Enhancement
                    Intelligently identifies relevant knowledge communities and automatically extracts valuable background knowledge and related information.

                    #### Knowledge Graph Enhancement
                    Builds a query-relevant knowledge graph in real time, providing structured reasoning and relationship discovery.

                    #### Evidence Chain Tracking
                    Records the full reasoning path and evidence sources, delivering an explainable conclusion process.
                    """)

            execution_logs = []

            if hasattr(st.session_state, 'execution_logs') and st.session_state.execution_logs:
                execution_logs = st.session_state.execution_logs

            elif hasattr(st.session_state, 'execution_log') and st.session_state.execution_log:
                for entry in st.session_state.execution_log:
                    if entry.get("node") == "deep_research" and entry.get("output"):
                        output = entry.get("output")
                        if isinstance(output, str):
                            execution_logs = output.strip().split('\n')

            if not execution_logs and len(st.session_state.messages) > 0:
                for msg in reversed(st.session_state.messages):
                    if msg.get("role") == "assistant" and "raw_thinking" in msg:
                        thinking_text = msg["raw_thinking"]
                        if "[Deep Research]" in thinking_text or "[KB Search]" in thinking_text:
                            execution_logs = thinking_text.strip().split('\n')
                            break

            if not execution_logs and 'raw_thinking' in st.session_state:
                thinking_text = st.session_state.raw_thinking
                if thinking_text and ("[Deep Research]" in thinking_text or "[KB Search]" in thinking_text):
                    execution_logs = thinking_text.strip().split('\n')

            if st.session_state.get("use_deeper_tool", True) and "reasoning_chain" in st.session_state:
                reasoning_chain = st.session_state.reasoning_chain

                if reasoning_chain:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("#### Community Analysis")
                        steps = reasoning_chain.get("steps", [])
                        community_step = next((s for s in steps if "knowledge_community_analysis" in s.get("search_query", "")), None)

                        if community_step:
                            st.success("✓ Relevant community identified")
                            evidence = community_step.get("evidence", [])

                            for ev in evidence:
                                if ev.get("source_type") == "community_knowledge":
                                    with st.expander(f"Community Knowledge {ev.get('evidence_id', '')}"):
                                        st.write(ev.get("content", ""))
                        else:
                            st.info("No community analysis was performed")

                    with col2:
                        st.markdown("#### Knowledge Graph")
                        if "knowledge_graph" in st.session_state:
                            kg = st.session_state.knowledge_graph
                            st.metric("Entity Count", kg.get("entity_count", 0))
                            st.metric("Relation Count", kg.get("relation_count", 0))

                            central_entities = kg.get("central_entities", [])
                            if central_entities:
                                st.write("**Core entities:**")
                                for entity in central_entities[:5]:
                                    entity_id = entity.get("id", "")
                                    entity_type = entity.get("type", "Unknown")
                                    st.markdown(f"- **{entity_id}** ({entity_type})")
                        else:
                            st.info("No knowledge graph data available")

            if not execution_logs:
                st.info("Waiting for execution logs. Send a new query to generate an execution trace. If you already sent a query and see this message, try again.")
            else:
                display_formatted_logs(execution_logs)
        else:
            if st.session_state.execution_log:
                for entry in st.session_state.execution_log:
                    with st.expander(f"Node: {entry['node']}", expanded=False):
                        st.markdown("**Input:**")
                        st.code(json.dumps(entry["input"], ensure_ascii=False, indent=2), language="json")
                        st.markdown("**Output:**")
                        st.code(json.dumps(entry["output"], ensure_ascii=False, indent=2), language="json")
            else:
                st.info("Send a query to display the execution trace here.")

def display_formatted_logs(log_lines):
    """Format and display log lines"""
    if not log_lines:
        st.warning("No execution logs available")
        return

    has_deep_research_markers = any("[Deep Research]" in line for line in log_lines)
    has_kb_search_markers = any("[KB Search]" in line for line in log_lines)

    if has_deep_research_markers or has_kb_search_markers:
        current_round = None
        in_search_results = False

        current_iteration = None
        current_iteration_content = []
        iterations = []
        current_round = None

        for line in log_lines:
            if "[Deep Research] Starting round" in line:
                if current_iteration_content:
                    iterations.append({
                        "round": current_round,
                        "content": current_iteration_content
                    })

                round_match = re.search(r'Starting round (\d+)', line)
                if round_match:
                    current_round = int(round_match.group(1))
                    current_iteration_content = [line]
            elif current_round is not None:
                if current_iteration_content is not None:
                    current_iteration_content.append(line)

            elif "[Deep Research] Executing query:" in line:
                if current_iteration_content is not None:
                    current_iteration_content.append(line)

            elif "[KB Search] Starting search:" in line:
                in_search_results = True
                if current_iteration_content is not None:
                    current_iteration_content.append(line)

            elif "[KB Search]" in line:
                if current_iteration_content is not None:
                    current_iteration_content.append(line)

            elif "[Deep Research] Found useful info:" in line:
                if current_iteration_content is not None:
                    current_iteration_content.append(line)

            elif "[Deep Research] No new queries generated, sufficient info available, ending iteration" in line:
                if current_iteration_content is not None:
                    current_iteration_content.append(line)

            elif current_iteration_content is not None:
                current_iteration_content.append(line)

        if current_iteration_content:
            iterations.append({
                "round": current_round,
                "content": current_iteration_content
            })

        if iterations:
            st.markdown("#### Select Iteration Round")

            valid_iterations = [it for it in iterations if it["round"] is not None]
            if not valid_iterations:
                st.warning("No valid iteration rounds found")
                return

            iteration_options = {f"Round {it['round']}": it for it in valid_iterations}

            default_key = next((k for k in iteration_options.keys() if k == "Round 1"), list(iteration_options.keys())[0])

            selected_round_key = st.selectbox(
                "Select Iteration Round",
                list(iteration_options.keys()),
                index=list(iteration_options.keys()).index(default_key)
            )

            iteration = iteration_options[selected_round_key]

            st.markdown("""
            <div style="padding:10px 0; margin:10px 0; border-bottom:1px solid #eee;">
                <h4 style="margin:0;">Iteration Details</h4>
            </div>
            """, unsafe_allow_html=True)

            queries = []
            kb_searches = []
            kb_results = []
            useful_info = None
            other_lines = []

            for line in iteration.get("content", []):
                if "[Deep Research] Executing query:" in line:
                    query = re.sub(r'\[Deep Research\] Executing query:', '', line).strip()
                    queries.append(query)
                elif "[KB Search] Starting search:" in line:
                    search = re.sub(r'\[KB Search\] Starting search:', '', line).strip()
                    kb_searches.append(search)
                elif "[KB Search] Results:" in line:
                    kb_results.append(line)
                elif "[Deep Research] Found useful info:" in line:
                    useful_info = re.sub(r'\[Deep Research\] Found useful info:', '', line).strip()
                else:
                    other_lines.append(line)

            if queries:
                st.markdown("##### Queries Executed")
                for query in queries:
                    st.markdown(f"""
                    <div style="background-color:#f5f5f5; padding:8px; border-left:4px solid #4CAF50; margin:8px 0; border-radius:3px;">
                        {query}
                    </div>
                    """, unsafe_allow_html=True)

            if useful_info:
                st.markdown("##### Useful Information Found")
                st.markdown(f"""
                <div style="background-color:#E8F5E9; padding:10px; border-left:4px solid #4CAF50; margin:10px 0; border-radius:4px;">
                    {useful_info}
                </div>
                """, unsafe_allow_html=True)

            if kb_searches or kb_results:
                st.markdown("##### Knowledge Base Search")
                col1, col2 = st.columns(2)

                with col1:
                    if kb_searches:
                        st.markdown("**Search Queries**")
                        for search in kb_searches:
                            st.markdown(f"""
                            <div style="background-color:#FFF8E1; padding:8px; border-left:4px solid #FFA000; margin:8px 0; border-radius:3px;">
                                {search}
                            </div>
                            """, unsafe_allow_html=True)

                with col2:
                    if kb_results:
                        st.markdown("**Search Results**")
                        st.code("\n".join(kb_results), language="text")

            if other_lines:
                with st.expander("Detailed Logs", expanded=False):
                    st.markdown("""
                    <div style="background-color:#f8f9fa; padding:10px; border-radius:5px; margin:10px 0; font-family:monospace;">
                    """, unsafe_allow_html=True)

                    for line in other_lines:
                        if "[KB Search]" in line:
                            st.markdown(f'<div style="padding:2px 0; color:#f57c00;">{line}</div>', unsafe_allow_html=True)
                        elif "[Deep Research]" in line:
                            st.markdown(f'<div style="padding:2px 0; color:#1976d2;">{line}</div>', unsafe_allow_html=True)
                        elif "[Dual Path Search]" in line:
                            st.markdown(f'<div style="padding:2px 0; color:#7b1fa2;">{line}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div style="padding:2px 0; color:#666;">{line}</div>', unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            # No iteration rounds detected — display by log type
            deep_research_logs = [line for line in log_lines if "[Deep Research]" in line]
            kb_search_logs = [line for line in log_lines if "[KB Search]" in line]
            other_logs = [line for line in log_lines if "[Deep Research]" not in line and "[KB Search]" not in line]

            log_tabs = st.tabs(["Deep Research Logs", "Knowledge Base Search Logs", "Other Logs"])

            with log_tabs[0]:
                for line in deep_research_logs:
                    if "Found useful info" in line:
                        useful_info = re.sub(r'\[Deep Research\] Found useful info:', '', line).strip()
                        st.markdown(f"""
                        <div style="background-color:#E8F5E9; padding:10px; border-left:4px solid #4CAF50; margin:10px 0; border-radius:4px;">
                            <span style="color:#4CAF50; font-weight:bold;">Useful information found:</span><br>{useful_info}
                        </div>
                        """, unsafe_allow_html=True)
                    elif "Executing query" in line:
                        query = re.sub(r'\[Deep Research\] Executing query:', '', line).strip()
                        st.markdown(f"""
                        <div style="background-color:#f5f5f5; padding:8px; border-left:4px solid #4CAF50; margin:8px 0; border-radius:3px;">
                            <span style="color:#4CAF50; font-weight:bold;">Executing query:</span> {query}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"<span style='color:#1976d2;'>{line}</span>", unsafe_allow_html=True)

            with log_tabs[1]:
                for line in kb_search_logs:
                    if "Starting search" in line:
                        search = re.sub(r'\[KB Search\] Starting search:', '', line).strip()
                        st.markdown(f"""
                        <div style="background-color:#FFF8E1; padding:8px; border-left:4px solid #FFA000; margin:8px 0; border-radius:3px;">
                            <span style="color:#FFA000; font-weight:bold;">Starting search:</span> {search}
                        </div>
                        """, unsafe_allow_html=True)
                    elif "Results" in line:
                        st.markdown(f"<span style='color:#f57c00; font-weight:bold;'>{line}</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<span style='color:#f57c00;'>{line}</span>", unsafe_allow_html=True)

            with log_tabs[2]:
                if other_logs:
                    st.markdown("""
                    <div style="background-color:#f8f9fa; padding:10px; border-radius:5px; font-family:monospace;">
                    """, unsafe_allow_html=True)

                    for line in other_logs:
                        if "[Dual Path Search]" in line:
                            st.markdown(f'<div style="padding:2px 0; color:#7b1fa2;">{line}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div style="padding:2px 0; color:#666;">{line}</div>', unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.info("No other logs")
    else:
        st.code("\n".join(log_lines), language="text")

def add_performance_tab(tabs):
    """Add the performance monitoring tab"""
    with tabs[4]:
        st.markdown('<div class="debug-header">Performance Statistics</div>', unsafe_allow_html=True)
        display_performance_stats()

        if st.button("Clear Performance Data"):
            clear_performance_data()
            st.rerun()

def display_debug_panel():
    """Display the debug panel"""
    st.subheader("🔍 Debug Information")

    tabs = st.tabs(["Execution Trace", "Knowledge Graph", "Source Content", "KG Management", "Performance Monitor"])

    display_execution_trace_tab(tabs)
    display_knowledge_graph_tab(tabs)
    display_source_content_tab(tabs)

    if st.session_state.current_tab == "KG Management":
        display_kg_management_tab(tabs)
    else:
        with tabs[3]:
            if st.button("Load KG Management Panel", key="load_kg_management"):
                st.session_state.current_tab = "KG Management"
                st.rerun()
            else:
                st.info("Click the button above to load the KG management panel")

    add_performance_tab(tabs)

    tab_index = 0

    if st.session_state.current_tab == "Execution Trace":
        tab_index = 0
    elif st.session_state.current_tab == "Knowledge Graph":
        tab_index = 1
    elif st.session_state.current_tab == "Source Content":
        tab_index = 2
    elif st.session_state.current_tab == "KG Management":
        tab_index = 3
    elif st.session_state.current_tab == "Performance Monitor":
        tab_index = 4

    st.markdown(KG_MANAGEMENT_CSS, unsafe_allow_html=True)

    tab_js = f"""
    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            setTimeout(function() {{
                const tabs = document.querySelectorAll('[data-baseweb="tab"]');
                if (tabs.length > {tab_index}) {{
                    tabs[{tab_index}].click();
                }}
            }}, 100);
        }});
    </script>
    """

    if "current_tab" in st.session_state:
        st.markdown(tab_js, unsafe_allow_html=True)
