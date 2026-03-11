# Knowledge Graph Q&A System — Frontend

This project is the frontend interface for a knowledge graph-based intelligent Q&A system, built with Streamlit. It provides a user-friendly interaction experience and works closely with the backend to support multiple agent types, knowledge graph visualization, and real-time streaming responses.

## Project Structure

```
frontend/
├── app.py                          # Application entry point; initializes the Streamlit app
├── components/                     # UI component modules
│   ├── __init__.py
│   ├── chat.py                     # Chat interface component
│   ├── debug.py                    # Debug panel component
│   ├── knowledge_graph/            # Knowledge graph components
│   │   ├── __init__.py
│   │   ├── display.py              # Knowledge graph display component
│   │   ├── interaction.py          # Knowledge graph interaction scripts
│   │   ├── kg_styles.py            # Knowledge graph styles
│   │   ├── management.py           # Knowledge graph management component
│   │   └── visualization.py        # Knowledge graph visualization component
│   ├── sidebar.py                  # Sidebar component
│   └── styles.py                   # Global style definitions
├── frontend_config/                # Frontend configuration
│   ├── __init__.py
│   └── settings.py                 # Frontend settings file
└── utils/                          # Utility functions
    ├── __init__.py
    ├── api.py                      # API call functions
    ├── helpers.py                  # Helper functions
    ├── performance.py              # Performance monitoring tools
    └── state.py                    # Session state management
```

## Core Design

### 1. Multi-Mode UI

The system implements two primary modes:
- **Standard mode**: Focused chat interface for a smooth Q&A experience
- **Debug mode**: Adds a debug panel alongside the chat to display execution traces, knowledge graph data, and source content

### 2. Agent Type Selection

Supports multiple agent types that users can select based on their needs:
- `graph_agent`: Uses local and global graph search
- `hybrid_agent`: Hybrid search approach
- `naive_rag_agent`: Traditional vector retrieval-augmented generation
- `deep_research_agent`: Deep research agent supporting multi-round reasoning
- `fusion_agent`: Advanced agent fusing knowledge graph and RAG

### 3. Streaming Response Design

Implements streaming response functionality to display AI-generated content in real time:
- Supports streaming display of standard answers
- Supports streaming display of the thinking process (in `deep_research_agent`)
- Communicates with the backend via SSE (Server-Sent Events)

### 4. Knowledge Graph Visualization

Uses interactive knowledge graph visualization:
- Implements Neo4j-style graph interaction using pyvis
- Supports double-click to focus on a node, right-click context menus
- Provides community detection and relationship reasoning features

### 5. Session State Management

Uses Streamlit's `session_state` to manage application state:
- Maintains user session ID and message history
- Handles debug mode and caching
- Manages graph display settings and current view

## Core Functions

### Chat Interface Management

- `display_chat_interface`: Renders the main chat interface; handles message input and display
- `send_message` / `send_message_stream`: Sends messages to the backend and processes responses
- `send_feedback`: Sends user feedback on responses

### Knowledge Graph Features

- `visualize_knowledge_graph`: Visualizes knowledge graph data as an interactive chart
- `display_knowledge_graph_tab`: Displays the knowledge graph tab, including graph display and reasoning features
- `get_kg_reasoning`: Retrieves reasoning results between entities

### Debug Features

- `display_debug_panel`: Displays a debug panel with multiple tabs
- `display_execution_trace_tab`: Displays the execution trace, with specialized handling for different agent types
- `display_formatted_logs`: Formats and displays the deep research agent's iteration process

### Session State Management

- `init_session_state`: Initializes session state variables with default values
- `clear_chat`: Clears chat history and related state; resets the session

### Performance Monitoring

- `monitor_performance`: Performance monitoring decorator that tracks API call latency
- `display_performance_stats`: Displays performance statistics and charts
- `PerformanceCollector`: Class for collecting and analyzing performance data

## Key Features

1. **Multi-mode agent selection**: Supports different agent types suited to different query scenarios
2. **Interactive knowledge graph**: Provides Neo4j-style graph interaction with node focus and community detection
3. **Deep research mode**: Supports thinking process visualization, showing the AI's reasoning trajectory
4. **Streaming response**: Displays generated content in real time for a better user experience
5. **Feedback mechanism**: Allows users to provide feedback on responses to improve system quality
6. **Knowledge graph management**: Manage entities and relationships directly in the UI
7. **Performance monitoring**: Provides detailed API performance monitoring and analysis tools
