import streamlit as st
import pandas as pd
from utils.api import (
    get_entity_types, get_relation_types, create_entity, create_relation,
    update_entity, update_relation, delete_entity, delete_relation,
    get_entities, get_relations
)

def display_kg_management_tab(tabs):
    """Display the knowledge graph management tab"""
    with tabs[3]:
        st.markdown('<div class="debug-header">Knowledge Graph Management</div>', unsafe_allow_html=True)

        management_tabs = st.tabs(["Entity Management", "Relation Management"])

        with management_tabs[0]:
            display_entity_management()

        with management_tabs[1]:
            display_relation_management()

def display_entity_management():
    """Display the entity management interface"""
    st.subheader("Entity Management")

    operation_tabs = st.tabs(["Query Entities", "Create Entity", "Update Entity", "Delete Entity"])

    # Query entities
    with operation_tabs[0]:
        st.markdown("#### Query Entities")

        col1, col2 = st.columns([1, 1])

        with col1:
            search_term = st.text_input("Entity Name / ID", key="entity_search_term", placeholder="Enter entity name or ID")

        with col2:
            try:
                entity_types = get_entity_types() or []
            except Exception as e:
                st.error(f"Failed to fetch entity types: {str(e)}")
                entity_types = []

            selected_type = st.selectbox(
                "Entity Type",
                options=["All"] + entity_types,
                key="entity_search_type"
            )

        if st.button("Query", key="entity_search_button"):
            with st.spinner("Querying entities..."):
                try:
                    filters = {}
                    if search_term:
                        filters["term"] = search_term
                    if selected_type and selected_type != "All":
                        filters["type"] = selected_type

                    entities = get_entities(filters)

                    if entities is not None and len(entities) > 0:
                        df = pd.DataFrame(entities)
                        st.dataframe(df, use_container_width=True)
                        st.success(f"Found {len(entities)} entity/entities")
                    else:
                        st.info("No matching entities found")
                except Exception as e:
                    st.error(f"Query failed: {str(e)}")

    # Create entity
    with operation_tabs[1]:
        st.markdown("#### Create Entity")

        try:
            entity_types = get_entity_types() or []
        except Exception as e:
            st.error(f"Failed to fetch entity types: {str(e)}")
            entity_types = []

        with st.form("create_entity_form"):
            entity_id = st.text_input("Entity ID *", placeholder="Enter a unique entity ID")
            entity_name = st.text_input("Entity Name *", placeholder="Enter entity name")
            entity_type = st.selectbox("Entity Type *", options=entity_types) if entity_types else st.text_input("Entity Type *", placeholder="Enter entity type")
            entity_description = st.text_area("Entity Description", placeholder="Enter entity description")

            st.markdown("##### Custom Properties (Optional)")

            prop_key_prefix = "create_entity_prop"
            property_keys = []
            property_values = []

            for i in range(5):
                col1, col2 = st.columns([1, 2])
                with col1:
                    key = st.text_input(f"Property Name {i+1}", key=f"{prop_key_prefix}_key_{i}")
                    property_keys.append(key)
                with col2:
                    value = st.text_input(f"Property Value {i+1}", key=f"{prop_key_prefix}_value_{i}")
                    property_values.append(value)

            submitted = st.form_submit_button("Create Entity")

            if submitted:
                if not entity_id or not entity_name or not entity_type:
                    st.error("Please fill in the required fields: Entity ID, Entity Name, and Entity Type")
                else:
                    properties = {}
                    for i in range(len(property_keys)):
                        if property_keys[i] and property_values[i]:
                            properties[property_keys[i]] = property_values[i]

                    entity_data = {
                        "id": entity_id,
                        "name": entity_name,
                        "type": entity_type,
                        "description": entity_description,
                        "properties": properties
                    }

                    try:
                        with st.spinner("Creating entity..."):
                            result = create_entity(entity_data)
                            if result is not None and result.get("success", False):
                                st.success(f"Entity created successfully: {entity_id}")
                            else:
                                error_message = "Unknown error"
                                if result is not None:
                                    error_message = result.get("message", "Unknown error")
                                st.error(f"Creation failed: {error_message}")
                    except Exception as e:
                        st.error(f"Creation failed: {str(e)}")

    # Update entity
    with operation_tabs[2]:
        st.markdown("#### Update Entity")

        entity_id_to_update = st.text_input("Enter the entity ID to update", key="update_entity_id")

        if entity_id_to_update:
            lookup_button = st.button("Find Entity", key="lookup_entity_button")

            if lookup_button or "entity_to_update" in st.session_state:
                try:
                    if ("entity_to_update" not in st.session_state or
                        st.session_state.entity_to_update is None or
                        st.session_state.entity_to_update.get("id") != entity_id_to_update):
                        with st.spinner("Looking up entity..."):
                            entities = get_entities({"term": entity_id_to_update})
                            if entities is not None and len(entities) > 0:
                                entity = next((e for e in entities if e.get("id") == entity_id_to_update), entities[0])
                                st.session_state.entity_to_update = entity
                            else:
                                st.error(f"No entity found with ID: {entity_id_to_update}")
                                if "entity_to_update" in st.session_state:
                                    del st.session_state.entity_to_update
                                return

                    if "entity_to_update" in st.session_state and st.session_state.entity_to_update is not None:
                        entity = st.session_state.entity_to_update

                        with st.form("update_entity_form"):
                            st.markdown(f"##### Update Entity: {entity.get('id')}")

                            st.info(f"Entity ID: {entity.get('id')} (cannot be changed)")

                            current_name = entity.get("name", "")
                            current_type = entity.get("type", "")
                            current_description = entity.get("description", "")
                            current_properties = entity.get("properties", {}) or {}

                            try:
                                entity_types_list = get_entity_types() or []
                                type_index = entity_types_list.index(current_type) if current_type in entity_types_list else 0
                            except Exception as e:
                                st.warning(f"Failed to fetch entity types, using current type: {str(e)}")
                                entity_types_list = [current_type] if current_type else ["Unknown"]
                                type_index = 0

                            new_name = st.text_input("Entity Name", value=current_name)
                            new_type = st.selectbox("Entity Type", options=entity_types_list, index=type_index)
                            new_description = st.text_area("Entity Description", value=current_description)

                            st.markdown("##### Edit Properties")

                            st.markdown("Existing properties:")
                            prop_updates = {}
                            prop_to_delete = []

                            for key, value in current_properties.items():
                                col1, col2, col3 = st.columns([1, 2, 0.5])
                                with col1:
                                    st.text(key)
                                with col2:
                                    new_value = st.text_input(f"Value: {key}", value=value, key=f"prop_{key}")
                                    prop_updates[key] = new_value
                                with col3:
                                    if st.checkbox("Delete", key=f"delete_prop_{key}"):
                                        prop_to_delete.append(key)

                            st.markdown("Add new properties:")
                            new_props = {}

                            for i in range(3):
                                col1, col2 = st.columns([1, 2])
                                with col1:
                                    new_key = st.text_input(f"New Property Name {i+1}", key=f"new_prop_key_{i}")
                                with col2:
                                    new_val = st.text_input(f"New Property Value {i+1}", key=f"new_prop_val_{i}")

                                if new_key and new_val:
                                    new_props[new_key] = new_val

                            submitted = st.form_submit_button("Update Entity")

                            if submitted:
                                try:
                                    update_data = {
                                        "id": entity.get("id"),
                                        "name": new_name,
                                        "type": new_type,
                                        "description": new_description,
                                    }

                                    properties = {**current_properties}
                                    for key in prop_to_delete:
                                        if key in properties:
                                            del properties[key]

                                    for key, value in prop_updates.items():
                                        if key not in prop_to_delete:
                                            properties[key] = value

                                    properties.update(new_props)
                                    update_data["properties"] = properties

                                    with st.spinner("Updating entity..."):
                                        result = update_entity(update_data)
                                        if result is not None and result.get("success", False):
                                            st.success(f"Entity updated successfully: {entity.get('id')}")
                                            st.session_state.entity_to_update = {**update_data}
                                        else:
                                            error_message = "Unknown error"
                                            if result is not None:
                                                error_message = result.get("message", "Unknown error")
                                            st.error(f"Update failed: {error_message}")
                                except Exception as e:
                                    st.error(f"Update failed: {str(e)}")

                except Exception as e:
                    st.error(f"Error looking up entity: {str(e)}")

    # Delete entity
    with operation_tabs[3]:
        st.markdown("#### Delete Entity")

        delete_id = st.text_input("Enter the entity ID to delete", key="delete_entity_id")

        col1, col2 = st.columns([1, 3])

        with col1:
            confirm = st.checkbox("Confirm Deletion", key="confirm_entity_delete")

        with col2:
            if delete_id and confirm:
                if st.button("Delete Entity", key="delete_entity_button"):
                    try:
                        with st.spinner("Deleting entity..."):
                            result = delete_entity(delete_id)
                            if result is not None and result.get("success", False):
                                st.success(f"Entity deleted successfully: {delete_id}")
                                if ("entity_to_update" in st.session_state and
                                    st.session_state.entity_to_update is not None and
                                    st.session_state.entity_to_update.get("id") == delete_id):
                                    del st.session_state.entity_to_update
                            else:
                                error_message = "Unknown error"
                                if result is not None:
                                    error_message = result.get("message", "Unknown error")
                                st.error(f"Deletion failed: {error_message}")
                    except Exception as e:
                        st.error(f"Deletion failed: {str(e)}")
            else:
                st.info("Enter an entity ID and check the confirmation box to enable deletion")

def display_relation_management():
    """Display the relation management interface"""
    st.subheader("Relation Management")

    operation_tabs = st.tabs(["Query Relations", "Create Relation", "Update Relation", "Delete Relation"])

    # Query relations
    with operation_tabs[0]:
        st.markdown("#### Query Relations")

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            source_entity = st.text_input("Source Entity ID", key="relation_search_source", placeholder="Source entity ID (optional)")

        with col2:
            target_entity = st.text_input("Target Entity ID", key="relation_search_target", placeholder="Target entity ID (optional)")

        with col3:
            try:
                relation_types = get_relation_types() or []
            except Exception as e:
                st.error(f"Failed to fetch relation types: {str(e)}")
                relation_types = []

            selected_rel_type = st.selectbox(
                "Relation Type",
                options=["All"] + relation_types,
                key="relation_search_type"
            )

        if st.button("Query", key="relation_search_button"):
            with st.spinner("Querying relations..."):
                try:
                    filters = {}
                    if source_entity:
                        filters["source"] = source_entity
                    if target_entity:
                        filters["target"] = target_entity
                    if selected_rel_type and selected_rel_type != "All":
                        filters["type"] = selected_rel_type

                    relations = get_relations(filters)

                    if relations is not None and len(relations) > 0:
                        df = pd.DataFrame(relations)
                        st.dataframe(df, use_container_width=True)
                        st.success(f"Found {len(relations)} relation(s)")
                    else:
                        st.info("No matching relations found")
                except Exception as e:
                    st.error(f"Query failed: {str(e)}")

    # Create relation
    with operation_tabs[1]:
        st.markdown("#### Create Relation")

        try:
            relation_types = get_relation_types() or []
        except Exception as e:
            st.error(f"Failed to fetch relation types: {str(e)}")
            relation_types = []

        with st.form("create_relation_form"):
            source_id = st.text_input("Source Entity ID *", placeholder="Enter source entity ID")
            relation_type = st.selectbox("Relation Type *", options=relation_types) if relation_types else st.text_input("Relation Type *", placeholder="Enter relation type")
            target_id = st.text_input("Target Entity ID *", placeholder="Enter target entity ID")

            relation_description = st.text_area("Relation Description", placeholder="Enter relation description")

            weight = st.slider("Relation Weight", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

            st.markdown("##### Custom Properties (Optional)")

            prop_key_prefix = "create_relation_prop"
            property_keys = []
            property_values = []

            for i in range(3):
                col1, col2 = st.columns([1, 2])
                with col1:
                    key = st.text_input(f"Property Name {i+1}", key=f"{prop_key_prefix}_key_{i}")
                    property_keys.append(key)
                with col2:
                    value = st.text_input(f"Property Value {i+1}", key=f"{prop_key_prefix}_value_{i}")
                    property_values.append(value)

            submitted = st.form_submit_button("Create Relation")

            if submitted:
                if not source_id or not target_id or not relation_type:
                    st.error("Please fill in the required fields: Source Entity ID, Relation Type, and Target Entity ID")
                else:
                    properties = {}
                    for i in range(len(property_keys)):
                        if property_keys[i] and property_values[i]:
                            properties[property_keys[i]] = property_values[i]

                    relation_data = {
                        "source": source_id,
                        "type": relation_type,
                        "target": target_id,
                        "description": relation_description,
                        "weight": weight,
                        "properties": properties
                    }

                    try:
                        with st.spinner("Creating relation..."):
                            result = create_relation(relation_data)
                            if result is not None and result.get("success", False):
                                st.success(f"Relation created successfully: {source_id} -[{relation_type}]-> {target_id}")
                            else:
                                error_message = "Unknown error"
                                if result is not None:
                                    error_message = result.get("message", "Unknown error")
                                st.error(f"Creation failed: {error_message}")
                    except Exception as e:
                        st.error(f"Creation failed: {str(e)}")

    # Update relation
    with operation_tabs[2]:
        st.markdown("#### Update Relation")

        st.markdown("##### Find the relation to update")

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            source_filter = st.text_input("Source Entity ID", key="update_relation_source")

        with col2:
            try:
                relation_types_list = get_relation_types() or []
            except Exception as e:
                st.error(f"Failed to fetch relation types: {str(e)}")
                relation_types_list = []

            relation_type_filter = st.selectbox(
                "Relation Type",
                options=["All"] + relation_types_list,
                key="update_relation_type"
            )

        with col3:
            target_filter = st.text_input("Target Entity ID", key="update_relation_target")

        lookup_button = st.button("Find Relation", key="lookup_relation_button")

        if lookup_button:
            try:
                with st.spinner("Looking up relation..."):
                    filters = {}
                    if source_filter:
                        filters["source"] = source_filter
                    if target_filter:
                        filters["target"] = target_filter
                    if relation_type_filter and relation_type_filter != "All":
                        filters["type"] = relation_type_filter

                    relations = get_relations(filters)

                    if relations is not None and len(relations) > 0:
                        st.session_state.found_relations = relations

                        df = pd.DataFrame(relations)
                        st.dataframe(df, use_container_width=True)

                        relation_ids = [f"{r.get('source')} -[{r.get('type')}]-> {r.get('target')}" for r in relations]
                        selected_relation = st.selectbox(
                            "Select relation to update",
                            options=relation_ids,
                            key="relation_to_update_id"
                        )

                        if selected_relation:
                            parts = selected_relation.split(" -[")
                            source = parts[0]
                            type_target = parts[1].split("]-> ")
                            rel_type = type_target[0]
                            target = type_target[1]

                            relation = next((r for r in relations if r.get("source") == source and r.get("type") == rel_type and r.get("target") == target), None)

                            if relation is not None:
                                st.session_state.relation_to_update = relation

                                with st.form("update_relation_form"):
                                    st.markdown(f"##### Update Relation: {source} -[{rel_type}]-> {target}")

                                    try:
                                        rel_types = get_relation_types() or []
                                        type_index = rel_types.index(rel_type) if rel_type in rel_types else 0
                                    except Exception as e:
                                        st.warning(f"Failed to fetch relation types, using current type: {str(e)}")
                                        rel_types = [rel_type] if rel_type else ["Unknown"]
                                        type_index = 0

                                    new_type = st.selectbox(
                                        "Relation Type",
                                        options=rel_types,
                                        index=type_index
                                    )

                                    current_description = relation.get("description", "")
                                    new_description = st.text_area("Relation Description", value=current_description)

                                    current_weight = relation.get("weight", 0.5)
                                    new_weight = st.slider("Relation Weight", min_value=0.0, max_value=1.0, value=current_weight, step=0.01)

                                    st.markdown("##### Edit Properties")

                                    current_properties = relation.get("properties", {}) or {}
                                    st.markdown("Existing properties:")
                                    prop_updates = {}
                                    prop_to_delete = []

                                    for key, value in current_properties.items():
                                        col1, col2, col3 = st.columns([1, 2, 0.5])
                                        with col1:
                                            st.text(key)
                                        with col2:
                                            new_value = st.text_input(f"Value: {key}", value=value, key=f"rel_prop_{key}")
                                            prop_updates[key] = new_value
                                        with col3:
                                            if st.checkbox("Delete", key=f"delete_rel_prop_{key}"):
                                                prop_to_delete.append(key)

                                    st.markdown("Add new properties:")
                                    new_props = {}

                                    for i in range(2):
                                        col1, col2 = st.columns([1, 2])
                                        with col1:
                                            new_key = st.text_input(f"New Property Name {i+1}", key=f"new_rel_prop_key_{i}")
                                        with col2:
                                            new_val = st.text_input(f"New Property Value {i+1}", key=f"new_rel_prop_val_{i}")

                                        if new_key and new_val:
                                            new_props[new_key] = new_val

                                    submitted = st.form_submit_button("Update Relation")

                                    if submitted:
                                        try:
                                            update_data = {
                                                "source": source,
                                                "original_type": rel_type,
                                                "target": target,
                                                "new_type": new_type,
                                                "description": new_description,
                                                "weight": new_weight
                                            }

                                            properties = {**current_properties}
                                            for key in prop_to_delete:
                                                if key in properties:
                                                    del properties[key]

                                            for key, value in prop_updates.items():
                                                if key not in prop_to_delete:
                                                    properties[key] = value

                                            properties.update(new_props)
                                            update_data["properties"] = properties

                                            with st.spinner("Updating relation..."):
                                                result = update_relation(update_data)
                                                if result is not None and result.get("success", False):
                                                    st.success("Relation updated successfully")
                                                    if "relation_to_update" in st.session_state:
                                                        del st.session_state.relation_to_update
                                                else:
                                                    error_message = "Unknown error"
                                                    if result is not None:
                                                        error_message = result.get("message", "Unknown error")
                                                    st.error(f"Update failed: {error_message}")
                                        except Exception as e:
                                            st.error(f"Update failed: {str(e)}")
                    else:
                        st.warning("No matching relations found")
            except Exception as e:
                st.error(f"Error looking up relation: {str(e)}")

    # Delete relation
    with operation_tabs[3]:
        st.markdown("#### Delete Relation")

        st.markdown("##### Specify the relation to delete")

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            delete_source = st.text_input("Source Entity ID *", key="delete_relation_source")

        with col2:
            try:
                relation_types = get_relation_types() or []
            except Exception as e:
                st.error(f"Failed to fetch relation types: {str(e)}")
                relation_types = []

            delete_type = st.selectbox("Relation Type *", options=relation_types, key="delete_relation_type") if relation_types else st.text_input("Relation Type *", key="delete_relation_type_input")

        with col3:
            delete_target = st.text_input("Target Entity ID *", key="delete_relation_target")

        col1, col2 = st.columns([1, 3])

        with col1:
            confirm_delete = st.checkbox("Confirm Deletion", key="confirm_relation_delete")

        with col2:
            if delete_source and delete_type and delete_target and confirm_delete:
                if st.button("Delete Relation", key="delete_relation_button"):
                    try:
                        with st.spinner("Deleting relation..."):
                            delete_data = {
                                "source": delete_source,
                                "type": delete_type,
                                "target": delete_target
                            }

                            result = delete_relation(delete_data)
                            if result is not None and result.get("success", False):
                                st.success(f"Relation deleted successfully: {delete_source} -[{delete_type}]-> {delete_target}")
                            else:
                                error_message = "Unknown error"
                                if result is not None:
                                    error_message = result.get("message", "Unknown error")
                                st.error(f"Deletion failed: {error_message}")
                    except Exception as e:
                        st.error(f"Deletion failed: {str(e)}")
            else:
                st.info("Please fill in all relation fields and check the confirmation box to enable deletion")
