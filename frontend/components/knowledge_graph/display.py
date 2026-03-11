import streamlit as st
from utils.api import get_knowledge_graph, get_kg_reasoning
from .visualization import visualize_knowledge_graph
import re

def display_knowledge_graph_tab(tabs):
    """Display knowledge graph tab content - lazy-loaded"""
    with tabs[1]:
        st.markdown('<div class="kg-controls">', unsafe_allow_html=True)

        # Check current agent type
        if st.session_state.agent_type == "deep_research_agent":
            st.info("Deep Research Agent focuses on deep reasoning and does not support knowledge graph visualization. See the Execution Trace tab for detailed reasoning.")
            return

        # Sub-tabs: graph display vs. reasoning Q&A
        kg_tabs = st.tabs(["Graph Display", "Reasoning Q&A"])

        with kg_tabs[0]:
            kg_display_mode = st.radio(
                "Display Mode:",
                ["Answer-Related Graph", "Global Knowledge Graph"],
                key="kg_display_mode",
                horizontal=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

            # Track whether the graph needs to be reloaded
            should_load_kg = False

            if "current_tab" in st.session_state and st.session_state.current_tab == "Knowledge Graph":
                if "last_kg_mode" not in st.session_state or st.session_state.last_kg_mode != kg_display_mode:
                    should_load_kg = True
                    st.session_state.last_kg_mode = kg_display_mode

            if kg_display_mode == "Answer-Related Graph":
                if "current_kg_message" in st.session_state and st.session_state.current_kg_message is not None:
                    msg_idx = st.session_state.current_kg_message

                    if (0 <= msg_idx < len(st.session_state.messages) and
                        "kg_data" in st.session_state.messages[msg_idx] and
                        st.session_state.messages[msg_idx]["kg_data"] is not None and
                        len(st.session_state.messages[msg_idx]["kg_data"].get("nodes", [])) > 0):

                        msg_preview = st.session_state.messages[msg_idx]["content"][:20] + "..."
                        st.success(f"Showing knowledge graph related to answer \"{msg_preview}\"")

                        visualize_knowledge_graph(st.session_state.messages[msg_idx]["kg_data"])
                    else:
                        st.info("No knowledge graph data found for the current answer")
                        # Commented out: automatic fallback to global KG on every render (requires backend)
                        # st.warning("Trying to load the global knowledge graph...")
                        # with st.spinner("Loading global knowledge graph..."):
                        #     kg_data = get_knowledge_graph(limit=100)
                        #     if kg_data and len(kg_data.get("nodes", [])) > 0:
                        #         visualize_knowledge_graph(kg_data)
                else:
                    st.info("Send a query in debug mode to view the related knowledge graph")
            else:
                # Global knowledge graph
                # Commented out: automatic load on every render (requires backend)
                # with st.spinner("Loading global knowledge graph..."):
                #     kg_data = get_knowledge_graph(limit=100)
                #     if kg_data and len(kg_data.get("nodes", [])) > 0:
                #         visualize_knowledge_graph(kg_data)
                #     else:
                #         st.warning("Failed to load the global knowledge graph data")
                if st.button("Load Global Knowledge Graph", key="load_global_kg"):
                    with st.spinner("Loading global knowledge graph..."):
                        kg_data = get_knowledge_graph(limit=100)
                        if kg_data and len(kg_data.get("nodes", [])) > 0:
                            visualize_knowledge_graph(kg_data)
                        else:
                            st.warning("Failed to load the global knowledge graph data")

        with kg_tabs[1]:
            # Knowledge graph reasoning Q&A interface
            st.markdown("## Knowledge Graph Reasoning Q&A")
            st.markdown("Explore relationships and paths between entities to discover deeper connections in the knowledge graph.")

            # Select reasoning type
            reasoning_type = st.selectbox(
                "Select Reasoning Type",
                options=[
                    "shortest_path",
                    "one_two_hop",
                    "common_neighbors",
                    "all_paths",
                    "entity_cycles",
                    "entity_influence",
                    "entity_community"
                ],
                format_func=lambda x: {
                    "shortest_path": "Shortest Path Query",
                    "one_two_hop": "1–2 Hop Relationship Path",
                    "common_neighbors": "Common Neighbors Query",
                    "all_paths": "All Relationship Paths",
                    "entity_cycles": "Entity Cycle Detection",
                    "entity_influence": "Influence Analysis",
                    "entity_community": "Community Detection"
                }.get(x, x),
                key="kg_reasoning_type"
            )

            # Show description for each reasoning type
            if reasoning_type == "shortest_path":
                st.info("Find the shortest connection path between two entities to understand how they are related.")
            elif reasoning_type == "one_two_hop":
                st.info("Find direct relationships or indirect relationships through a single intermediate node between two entities.")
            elif reasoning_type == "common_neighbors":
                st.info("Discover entities that are related to both given entities simultaneously (common neighbors).")
            elif reasoning_type == "all_paths":
                st.info("Explore all possible paths between two entities to understand the full range of connections.")
            elif reasoning_type == "entity_cycles":
                st.info("Detect cycles for an entity — paths that start from and return to the same entity.")
            elif reasoning_type == "entity_influence":
                st.info("Analyze the reach of an entity, finding all directly and indirectly connected entities.")
            elif reasoning_type == "entity_community":
                st.info("Discover which community or cluster an entity belongs to, and analyze its position in the broader knowledge network.")
                algorithm = st.selectbox(
                    "Community Detection Algorithm",
                    options=["leiden", "sllpa"],
                    format_func=lambda x: {
                        "leiden": "Leiden Algorithm",
                        "sllpa": "SLLPA Algorithm"
                    }.get(x, x),
                    key="community_algorithm"
                )

                if algorithm == "leiden":
                    st.markdown("""
                    **Leiden Algorithm** is an optimized community detection method similar to Louvain, but better at avoiding isolated communities.
                    Suitable for larger graphs with higher quality output but greater computation cost.
                    """)
                else:
                    st.markdown("""
                    **SLLPA** (Speaker-Listener Label Propagation Algorithm) is a label propagation algorithm
                    that rapidly detects overlapping communities. Suitable for small-to-medium graphs with faster execution.
                    """)

            # Input forms based on reasoning type
            if reasoning_type in ["shortest_path", "one_two_hop", "common_neighbors", "all_paths"]:
                # Two-entity reasoning types
                col1, col2 = st.columns(2)

                with col1:
                    entity_a = st.text_input("Entity A", key="kg_entity_a",
                                            help="Enter the name of the first entity")

                with col2:
                    entity_b = st.text_input("Entity B", key="kg_entity_b",
                                            help="Enter the name of the second entity")

                if reasoning_type in ["shortest_path", "all_paths"]:
                    max_depth = st.slider("Max Depth / Hops", 1, 5, 3, key="kg_max_depth",
                                        help="Limit the maximum search depth")
                else:
                    max_depth = 1  # Default

                if st.button("Run Reasoning", key="kg_reasoning_button",
                            help="Click to run knowledge graph reasoning"):
                    if not entity_a or not entity_b:
                        st.error("Please enter names for both entities")
                    else:
                        with st.spinner("Running knowledge graph reasoning..."):
                            process_info = st.empty()
                            process_info.info(f"Processing: {reasoning_type} query (this may take a few seconds...)")

                            try:
                                result = get_kg_reasoning(
                                    reasoning_type=reasoning_type,
                                    entity_a=entity_a,
                                    entity_b=entity_b,
                                    max_depth=max_depth
                                )

                                process_info.empty()

                                if "error" in result and result["error"]:
                                    st.error(f"Reasoning failed: {result['error']}")
                                    return

                                if len(result.get("nodes", [])) == 0:
                                    st.warning("No reasoning results found")
                                    return

                                display_reasoning_result(reasoning_type, result, entity_a, entity_b)
                                visualize_knowledge_graph(result)
                            except Exception as e:
                                process_info.empty()
                                st.error(f"Error processing request: {str(e)}")
                                import traceback
                                st.error(traceback.format_exc())
            else:
                # Single-entity reasoning types (entity_cycles, entity_influence, entity_community)
                entity_id = st.text_input("Entity Name", key="kg_entity_single",
                                        help="Enter the name of the entity")

                max_depth = st.slider("Max Depth", 1, 4, 2, key="kg_max_depth_single",
                                    help="Limit the maximum search depth")

                algorithm = st.session_state.get("community_algorithm", "leiden") if reasoning_type == "entity_community" else None

                if st.button("Run Reasoning", key="kg_reasoning_button_single",
                           help="Click to run knowledge graph reasoning"):
                    if not entity_id:
                        st.error("Please enter an entity name")
                    else:
                        with st.spinner("Running knowledge graph reasoning..."):
                            process_info = st.empty()
                            process_info.info(f"Processing: {reasoning_type} query (this may take a few seconds...)")

                            try:
                                result = get_kg_reasoning(
                                    reasoning_type=reasoning_type,
                                    entity_a=entity_id,
                                    max_depth=max_depth,
                                    algorithm=algorithm
                                )

                                process_info.empty()

                                if "error" in result and result["error"]:
                                    st.error(f"Reasoning failed: {result['error']}")
                                    return

                                if len(result.get("nodes", [])) == 0:
                                    st.warning("No reasoning results found")
                                    return

                                display_reasoning_result(reasoning_type, result, entity_id)
                                visualize_knowledge_graph(result)
                            except Exception as e:
                                process_info.empty()
                                st.error(f"Error processing request: {str(e)}")
                                import traceback
                                st.error(traceback.format_exc())

            # Usage guide
            with st.expander("📖 Reasoning Q&A Usage Guide", expanded=False):
                st.markdown("""
                ### Knowledge Graph Reasoning Feature Guide

                This feature allows you to explore relationships and structures between entities in the knowledge graph.

                #### 1. Shortest Path Query
                Find the shortest connection path between two entities to understand how they are related.
                - **Input**: Entity A and Entity B names
                - **Parameter**: Max hops (limits search depth)
                - **Output**: Shortest path visualization and path length

                #### 2. 1–2 Hop Relationship Path
                Find direct relationships or indirect ones through a single intermediate node.
                - **Input**: Entity A and Entity B names
                - **Output**: List and visualization of all 1-hop or 2-hop paths

                #### 3. Common Neighbors Query
                Discover entities related to both given entities simultaneously.
                - **Input**: Entity A and Entity B names
                - **Output**: Common neighbor list and network visualization

                #### 4. All Relationship Paths
                Explore all possible paths between two entities, not just the shortest.
                - **Input**: Entity A and Entity B names
                - **Parameter**: Max depth (limits search depth)
                - **Output**: All discovered paths and visualization

                #### 5. Entity Cycle Detection
                Detect cycles for an entity — paths that start and return to the same entity.
                - **Input**: Entity name
                - **Parameter**: Max cycle length
                - **Output**: Cycle list and visualization

                #### 6. Influence Analysis
                Analyze the influence scope of an entity and find all directly and indirectly connected entities.
                - **Input**: Entity name
                - **Parameter**: Max depth
                - **Output**: Influence statistics and network visualization

                #### 7. Community Detection
                Discover which community or cluster an entity belongs to.
                - **Input**: Entity name
                - **Parameters**: Max depth (defines community scope) and algorithm choice
                - **Output**: Community statistics and visualization
                - **Algorithms**:
                  - Leiden Algorithm - higher precision, good for complex graphs
                  - SLLPA Algorithm - faster, good for small-to-medium graphs

                ### Usage Tips

                - For large knowledge graphs, start with a small search depth and increase as needed
                - In the graph visualization, double-click a node to focus on it; right-click for more options
                - Click on a blank area to reset the graph view
                - Use the control panel in the top-right for graph navigation
                """)

            # Legend
            with st.expander("🎨 Graph Visualization Legend", expanded=False):
                st.markdown("""
                ### Node Color Reference

                - **Blue**: Source entity / query entity
                - **Red**: Target entity
                - **Green**: Intermediate node / common neighbor
                - **Purple**: Community 1 member
                - **Cyan**: Community 2 member
                - **Yellow**: Other community members

                ### Graph Interaction Guide

                - **Double-click node**: Focus view on that node and its direct connections
                - **Right-click node**: Open context menu for more actions
                - **Click blank area**: Reset view to show all nodes
                - **Drag node**: Manually adjust layout
                - **Scroll wheel**: Zoom in or out
                - **Top-right control panel**: Additional functions like reset and back
                """)

def display_reasoning_result(reasoning_type, result, entity_a=None, entity_b=None):
    """Display reasoning results based on type, using entity names instead of IDs"""
    if reasoning_type == "shortest_path":
        if "path_info" in result:
            path_info = result["path_info"]
            if entity_a and entity_b:
                path_info = path_info.replace(entity_a, f"'{entity_a}'")
                path_info = path_info.replace(entity_b, f"'{entity_b}'")
            st.success(f"{path_info} (length: {result['path_length']})")

    elif reasoning_type == "one_two_hop":
        if "paths_info" in result:
            st.success(f"Found {result['path_count']} path(s)")
            if result["path_count"] > 0:
                with st.expander("View Detailed Paths", expanded=True):
                    for i, path in enumerate(result["paths_info"]):
                        formatted_path = format_path_with_names(path)
                        st.markdown(f"**Path {i+1}**: {formatted_path}")

    elif reasoning_type == "common_neighbors":
        if "common_neighbors" in result:
            st.success(f"Found {result['neighbor_count']} common neighbor(s)")
            if result["neighbor_count"] > 0:
                neighbors = [format_entity_name(neighbor) for neighbor in result["common_neighbors"]]
                neighbors_str = ", ".join(neighbors)
                if len(neighbors_str) > 200:
                    neighbors_str = neighbors_str[:200] + "..."
                st.write(f"Common neighbors: {neighbors_str}")

                if len(result["common_neighbors"]) > 5:
                    with st.expander("View All Common Neighbors", expanded=False):
                        for i, neighbor in enumerate(result["common_neighbors"]):
                            st.markdown(f"- {format_entity_name(neighbor)}")

    elif reasoning_type == "all_paths":
        if "paths_info" in result:
            st.success(f"Found {result['path_count']} path(s)")
            if result["path_count"] > 0:
                with st.expander("View Detailed Paths", expanded=True):
                    for i, path in enumerate(result["paths_info"]):
                        formatted_path = format_path_with_names(path)
                        st.markdown(f"**Path {i+1}**: {formatted_path}")

    elif reasoning_type == "entity_cycles":
        if "cycles_info" in result:
            st.success(f"Found {result['cycle_count']} cycle(s)")
            if result["cycle_count"] > 0:
                with st.expander("View Cycle Details", expanded=True):
                    for i, cycle in enumerate(result["cycles_info"]):
                        formatted_desc = format_path_with_names(cycle["description"])
                        st.markdown(f"**Cycle {i+1}** (length: {cycle['length']}): {formatted_desc}")

    elif reasoning_type == "entity_influence":
        if "influence_stats" in result:
            stats = result["influence_stats"]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Direct Connections", stats["direct_connections"])
            with col2:
                st.metric("Total Connections", stats["total_connections"])
            with col3:
                st.metric("Relationship Types", len(stats["connection_types"]))

            if stats["connection_types"]:
                st.subheader("Relationship Type Distribution")
                for rel_type in stats["connection_types"]:
                    st.markdown(f"- **{rel_type['type']}**: {rel_type['count']} occurrence(s)")

    elif reasoning_type == "entity_community":
        if "communities" in result:
            st.success(f"Detected {result['community_count']} community/communities")

            if "entity_community" in result:
                entity_name = entity_a if entity_a else "current entity"
                st.info(f"Entity '{entity_name}' belongs to community: {result['entity_community']}")

            if result["communities"]:
                with st.expander("View Community Details", expanded=True):
                    for comm in result["communities"]:
                        contains = "✓" if comm["contains_center"] else "✗"
                        st.markdown(f"**Community {comm['id']}** (contains center entity: {contains})")
                        st.markdown(f"- Member count: {comm['size']}")
                        st.markdown(f"- Connection density: {comm['density']:.2f}")

                        if "sample_members" in comm and comm["sample_members"]:
                            sample_members = [format_entity_name(member) for member in comm["sample_members"]]
                            sample_str = ", ".join(sample_members)
                            if len(sample_str) > 100:
                                sample_str = sample_str[:100] + "..."
                            st.markdown(f"- Sample members: {sample_str}")

                        st.markdown("---")

        if "community_info" in result and isinstance(result["community_info"], dict):
            info = result["community_info"]
            if "summary" in info and info["summary"]:
                with st.expander("Community Summary", expanded=True):
                    st.markdown(f"""
                    **Community ID**: {info.get('id', 'N/A')}

                    **Entity Count**: {info.get('entity_count', 0)}

                    **Relation Count**: {info.get('relation_count', 0)}

                    **Summary**:
                    {info.get('summary', 'No summary available')}
                    """)

def format_entity_name(entity_id):
    """Format an entity ID as a friendly display name"""
    if not entity_id:
        return "Unknown entity"

    if isinstance(entity_id, (int, float)) or (isinstance(entity_id, str) and entity_id.isdigit()):
        return str(entity_id)

    return f"'{entity_id}'"

def format_path_with_names(path):
    """Format entity IDs in a path as friendly display names"""
    if not path:
        return ""

    formatted = path

    entity_pattern = r'\b([a-zA-Z0-9_\u4e00-\u9fa5]+)\b'

    def replace_entity(match):
        entity = match.group(1)

        if "-[" in match.string[max(0, match.start()-2):match.start()]:
            return entity

        if match.start() > 0 and match.string[match.start()-1:match.start()+len(entity)+1] == f"[{entity}]":
            return entity

        return format_entity_name(entity)

    formatted = re.sub(entity_pattern, replace_entity, formatted)

    return formatted

def get_node_color(node_type, is_center=False):
    """Return color based on node type and whether it is the center node"""
    from frontend_config.settings import NODE_TYPE_COLORS, KG_COLOR_PALETTE

    if is_center:
        return NODE_TYPE_COLORS["Center"]

    if node_type in NODE_TYPE_COLORS:
        return NODE_TYPE_COLORS[node_type]

    if isinstance(node_type, str) and "Community" in node_type:
        try:
            comm_id_str = node_type.replace("Community", "")
            if not comm_id_str:
                comm_id = 0
            else:
                comm_id = int(comm_id_str)

            color_index = (comm_id - 1) % len(KG_COLOR_PALETTE) if comm_id > 0 else 0
            return KG_COLOR_PALETTE[color_index]
        except (ValueError, TypeError):
            return "#757575"  # Gray fallback

    return "#757575"  # Gray default
