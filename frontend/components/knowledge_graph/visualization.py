import tempfile
import os
import streamlit as st
from pyvis.network import Network
import streamlit.components.v1 as components
from frontend_config.settings import KG_COLOR_PALETTE, NODE_TYPE_COLORS

def visualize_knowledge_graph(kg_data):
    """Visualize a knowledge graph using pyvis - dynamic node types and colors with Neo4j-style interaction"""
    if not kg_data or "nodes" not in kg_data or "links" not in kg_data:
        st.warning("Unable to retrieve knowledge graph data")
        return

    if len(kg_data["nodes"]) == 0:
        st.info("No entities or relationships found")
        return

    # Graph display settings and interaction guide
    with st.expander("Graph Display Settings & Interaction Guide", expanded=False):
        st.markdown("""
        ### Interaction Guide
        - **Double-click node**: Focus on that node and its direct connections and relationships
        - **Right-click node**: Open context menu for more actions
        - **Click blank area**: Reset the graph to show all nodes
        - **Control panel**: The top-right control panel provides Reset and Back functions

        ### Display Settings
        """)

        import hashlib
        import time
        timestamp = str(time.time())
        node_count = str(len(kg_data["nodes"]))
        base_key = hashlib.md5((node_count + timestamp).encode()).hexdigest()[:8]

        col1, col2 = st.columns(2)
        with col1:
            physics_enabled = st.checkbox("Enable Physics Engine",
                                       value=st.session_state.kg_display_settings["physics_enabled"],
                                       key=f"physics_enabled_{base_key}",
                                       help="Controls whether nodes can move dynamically")
            node_size = st.slider("Node Size", 10, 50,
                                st.session_state.kg_display_settings["node_size"],
                                key=f"node_size_{base_key}",
                                help="Adjust node size")

        with col2:
            edge_width = st.slider("Edge Width", 1, 10,
                                 st.session_state.kg_display_settings["edge_width"],
                                 key=f"edge_width_{base_key}",
                                 help="Adjust edge line width")
            spring_length = st.slider("Spring Length", 50, 300,
                                    st.session_state.kg_display_settings["spring_length"],
                                    key=f"spring_length_{base_key}",
                                    help="Adjust spacing between nodes")

        # Update settings
        st.session_state.kg_display_settings = {
            "physics_enabled": physics_enabled,
            "node_size": node_size,
            "edge_width": edge_width,
            "spring_length": spring_length,
            "gravity": st.session_state.kg_display_settings["gravity"]
        }

    # Create network with white background
    net = Network(height="600px", width="100%", bgcolor="#FFFFFF", font_color="#333333", directed=True)

    # Enhanced configuration for Neo4j-style interaction
    net.set_options("""
    {
      "physics": {
        "enabled": %s,
        "barnesHut": {
          "gravitationalConstant": %d,
          "centralGravity": 0.5,
          "springLength": %d,
          "springConstant": 0.04,
          "damping": 0.15,
          "avoidOverlap": 0.1
        },
        "solver": "barnesHut",
        "stabilization": {
          "enabled": true,
          "iterations": 1000,
          "updateInterval": 100,
          "onlyDynamicEdges": false,
          "fit": true
        }
      },
      "interaction": {
        "navigationButtons": true,
        "keyboard": {
          "enabled": true,
          "bindToWindow": true
        },
        "hover": true,
        "multiselect": true,
        "tooltipDelay": 200
      },
      "layout": {
        "improvedLayout": true,
        "hierarchical": {
          "enabled": false
        }
      }
    }
    """ % (str(physics_enabled).lower(), st.session_state.kg_display_settings["gravity"], spring_length))

    # Collect all unique group types
    group_types = set()
    for node in kg_data["nodes"]:
        group = node.get("group", "Unknown")
        if group:
            group_types.add(group)

    # Assign colors to each group
    group_colors = {}

    # First assign predefined colors
    for group in group_types:
        if group in NODE_TYPE_COLORS:
            group_colors[group] = NODE_TYPE_COLORS[group]

    # Then assign palette colors to remaining groups
    palette_index = 0
    for group in sorted(group_types):
        if group not in group_colors:
            if isinstance(group, str) and "Community" in group:
                try:
                    comm_id_str = group.replace("Community", "")
                    if not comm_id_str:
                        comm_id = 0
                    else:
                        comm_id = int(comm_id_str)

                    color_index = (comm_id - 1) % len(KG_COLOR_PALETTE) if comm_id > 0 else 0
                    group_colors[group] = KG_COLOR_PALETTE[color_index]
                except (ValueError, TypeError):
                    group_colors[group] = KG_COLOR_PALETTE[palette_index % len(KG_COLOR_PALETTE)]
                    palette_index += 1
            else:
                group_colors[group] = KG_COLOR_PALETTE[palette_index % len(KG_COLOR_PALETTE)]
                palette_index += 1

    # Add nodes with modern styles and enhanced interaction
    for node in kg_data["nodes"]:
        node_id = node["id"]
        label = node.get("label", node_id)
        group = node.get("group", "Unknown")
        description = node.get("description", "")

        color = group_colors.get(group, KG_COLOR_PALETTE[0])

        title = f"{label}" + (f": {description}" if description else "")

        net.add_node(
            node_id,
            label=label,
            title=title,
            color={
                "background": color,
                "border": "#ffffff",
                "highlight": {
                    "background": color,
                    "border": "#000000"
                },
                "hover": {
                    "background": color,
                    "border": "#000000"
                }
            },
            size=node_size,
            font={"color": "#ffffff", "size": 14, "face": "Arial"},
            shadow={"enabled": True, "color": "rgba(0,0,0,0.2)", "size": 3},
            borderWidth=2,
            group=group,
            description=description
        )

    # Add edges with modern styles and enhanced interaction
    for link in kg_data["links"]:
        source = link["source"]
        target = link["target"]
        label = link.get("label", "")
        weight = link.get("weight", 1)

        # Scale width based on weight
        width = edge_width * min(1 + (weight * 0.2), 3)

        smooth = {"enabled": True, "type": "dynamic", "roundness": 0.5}

        title = label

        net.add_edge(
            source,
            target,
            title=title,
            label=label,
            width=width,
            smooth=smooth,
            color={
                "color": "#999999",
                "highlight": "#666666",
                "hover": "#666666"
            },
            shadow={"enabled": True, "color": "rgba(0,0,0,0.1)"},
            selectionWidth=2,
            weight=weight,
            arrowStrikethrough=False
        )

    # Save to temp file and render
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
        net.save_graph(tmp.name)
        with open(tmp.name, 'r', encoding='utf-8') as f:
            html_content = f.read()

            # Inject custom styles
            from .kg_styles import KG_STYLES
            html_content = html_content.replace('</head>', KG_STYLES + '</head>')

            # Inject interaction script
            from .interaction import KG_INTERACTION_SCRIPT
            html_content = html_content.replace('</body>', KG_INTERACTION_SCRIPT + '</body>')

            components.html(html_content, height=600)

        # Clean up temp file
        try:
            os.unlink(tmp.name)
        except:
            pass

    # Display legend
    st.write("### Legend")

    priority_groups = ["Center", "Source", "Target", "Common"]
    community_groups = []
    other_groups = []

    for group in group_colors.keys():
        if group in priority_groups:
            continue
        elif isinstance(group, str) and "Community" in group:
            community_groups.append(group)
        else:
            other_groups.append(group)

    community_groups.sort()
    other_groups.sort()

    all_groups = []
    for group in priority_groups:
        if group in group_colors:
            all_groups.append(group)
    all_groups.extend(other_groups)
    all_groups.extend(community_groups)

    # Multi-column legend display
    cols = st.columns(3)
    for i, group in enumerate(all_groups):
        if group in group_colors:
            color = group_colors[group]
            col_idx = i % 3
            with cols[col_idx]:
                group_display_name = group
                if group == "Center":
                    group_display_name = "Center Node"
                elif group == "Source":
                    group_display_name = "Source Node"
                elif group == "Target":
                    group_display_name = "Target Node"
                elif group == "Common":
                    group_display_name = "Common Neighbor"

                st.markdown(
                    f'<div style="display:flex;align-items:center;margin-bottom:12px">'
                    f'<div style="width:20px;height:20px;border-radius:50%;background-color:{color};margin-right:10px;box-shadow:0 2px 4px rgba(0,0,0,0.1);"></div>'
                    f'<span style="font-family:sans-serif;color:#333;">{group_display_name}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
