KG_INTERACTION_SCRIPT = """
<script>
    // Global variables for graph state
    var originalNodes = [];
    var originalEdges = [];
    var isFiltered = false;
    var lastSelectedNode = null;
    var nodeHistory = [];

    // Wait for network initialization
    document.addEventListener('DOMContentLoaded', function() {
        // Save original graph data
        setTimeout(function() {
            try {
                originalNodes = new vis.DataSet(network.body.data.nodes.get());
                originalEdges = new vis.DataSet(network.body.data.edges.get());
                console.log("Graph data saved:", originalNodes.length, "nodes,", originalEdges.length, "edges");
            } catch(e) {
                console.error("Error saving graph data:", e);
            }
        }, 500);
    });

    // Slight animation effect on initial load
    setTimeout(function() {
        try {
            network.once("stabilizationIterationsDone", function() {
                network.setOptions({
                    physics: {
                        stabilization: false,
                        barnesHut: {
                            gravitationalConstant: -2000,
                            springConstant: 0.04,
                            damping: 0.2,
                        }
                    }
                });
            });
            network.stabilize(200);
        } catch(e) {
            console.error("Error setting physics engine:", e);
        }
    }, 1000);

    // Create floating control panel
    setTimeout(createControlPanel, 800);

    // Add basic event handlers
    try {
        // Hover cursor effect
        network.on("hoverNode", function(params) {
            document.body.style.cursor = 'pointer';
        });

        network.on("blurNode", function(params) {
            document.body.style.cursor = 'default';
        });

        // Double-click node - Neo4j-style neighbor view
        network.on("doubleClick", function(params) {
            if (params.nodes.length > 0) {
                var nodeId = params.nodes[0];
                focusOnNode(nodeId);
            }
        });

        // Click background - restore full graph
        network.on("click", function(params) {
            if (params.nodes.length === 0 && params.edges.length === 0) {
                resetGraph();
            }
        });

        // Right-click context menu
        network.on("oncontext", function(params) {
            params.event.preventDefault();
            var nodeId = network.getNodeAt(params.pointer.DOM);

            if (nodeId) {
                showContextMenu(nodeId, params);
            }
        });
    } catch(e) {
        console.error("Error adding event handlers:", e);
    }

    // Create floating control panel
    function createControlPanel() {
        try {
            var controlPanel = document.createElement('div');
            controlPanel.id = 'graph-control-panel';
            controlPanel.className = 'graph-control-panel';

            var panelTitle = document.createElement('div');
            panelTitle.style.fontWeight = 'bold';
            panelTitle.style.marginBottom = '8px';
            panelTitle.style.borderBottom = '1px solid #eee';
            panelTitle.style.paddingBottom = '5px';
            panelTitle.textContent = 'Graph Controls';
            controlPanel.appendChild(panelTitle);

            var resetButton = document.createElement('button');
            resetButton.textContent = 'Reset Graph';
            resetButton.className = 'graph-control-button';
            resetButton.onclick = resetGraph;
            controlPanel.appendChild(resetButton);

            var backButton = document.createElement('button');
            backButton.textContent = 'Back';
            backButton.className = 'graph-control-button';
            backButton.onclick = goBack;
            controlPanel.appendChild(backButton);

            var infoDiv = document.createElement('div');
            infoDiv.id = 'graph-info';
            infoDiv.className = 'graph-info';
            controlPanel.appendChild(infoDiv);

            var networkContainer = document.querySelector('.vis-network');
            if (networkContainer && networkContainer.parentNode) {
                networkContainer.parentNode.appendChild(controlPanel);
                console.log("Control panel created");
            } else {
                console.error("Network container not found");
            }
        } catch(e) {
            console.error("Error creating control panel:", e);
        }
    }

    // Show context menu on right-click
    function showContextMenu(nodeId, params) {
        try {
            var nodeInfo = network.body.data.nodes.get(nodeId);

            var contextMenu = document.getElementById('node-context-menu');
            if (!contextMenu) {
                contextMenu = document.createElement('div');
                contextMenu.id = 'node-context-menu';
                contextMenu.className = 'node-context-menu';
                document.body.appendChild(contextMenu);

                document.addEventListener('click', function() {
                    if (contextMenu) contextMenu.style.display = 'none';
                });
            }

            var canvasRect = params.event.srcElement.getBoundingClientRect();
            contextMenu.style.left = (canvasRect.left + params.pointer.DOM.x) + 'px';
            contextMenu.style.top = (canvasRect.top + params.pointer.DOM.y) + 'px';

            var label = nodeInfo.label || nodeId;
            var group = nodeInfo.group || "Unknown type";

            contextMenu.innerHTML = `
                <div class="node-context-menu-header">
                    ${label}
                </div>
                <div class="node-context-menu-item" id="focus-node">
                    🔍 Focus this node
                </div>
                <div class="node-context-menu-item" id="hide-node">
                    🚫 Hide this node
                </div>
                <div class="node-context-menu-item" id="show-info">
                    ℹ️ View details
                </div>
                <div class="node-context-menu-header" style="margin-top:5px;font-size:11px;color:#666;border-bottom:none;">
                    Type: ${group}
                </div>
            `;

            contextMenu.style.display = 'block';

            document.getElementById('focus-node').onclick = function(e) {
                e.stopPropagation();
                focusOnNode(nodeId);
                contextMenu.style.display = 'none';
            };

            document.getElementById('hide-node').onclick = function(e) {
                e.stopPropagation();
                network.body.data.nodes.remove(nodeId);
                contextMenu.style.display = 'none';
            };

            document.getElementById('show-info').onclick = function(e) {
                e.stopPropagation();
                showNodeDetails(nodeId);
                contextMenu.style.display = 'none';
            };
        } catch(e) {
            console.error("Error showing context menu:", e);
        }
    }

    // Show detailed node information
    function showNodeDetails(nodeId) {
        try {
            var node = network.body.data.nodes.get(nodeId);
            if (!node) return;

            var details = '';
            details += `<div style="font-weight:bold;margin-bottom:5px;">Node ID: ${node.id}</div>`;
            details += `<div style="margin-bottom:5px;">Label: ${node.label || 'None'}</div>`;
            details += `<div style="margin-bottom:5px;">Type: ${node.group || 'Unknown'}</div>`;
            details += `<div>Description: ${node.description || 'No description'}</div>`;

            var connectedNodes = [];
            var connectedEdges = [];

            var edges = network.body.data.edges.get();
            edges.forEach(function(edge) {
                if (edge.from === nodeId || edge.to === nodeId) {
                    connectedEdges.push(edge);
                    var connectedNodeId = edge.from === nodeId ? edge.to : edge.from;
                    if (!connectedNodes.includes(connectedNodeId)) {
                        connectedNodes.push(connectedNodeId);
                    }
                }
            });

            details += `<div style="margin-top:10px;"><strong>Connected nodes:</strong> ${connectedNodes.length}</div>`;
            details += `<div><strong>Relationship count:</strong> ${connectedEdges.length}</div>`;

            var infoDiv = document.getElementById('graph-info');
            if (infoDiv) {
                infoDiv.innerHTML = details;
            }

            network.body.data.nodes.update([{
                id: nodeId,
                borderWidth: 3,
                borderColor: '#FF5733',
                size: 35
            }]);
        } catch(e) {
            console.error("Error showing node details:", e);
        }
    }

    // Get connected nodes and edges for a given node
    function getConnectedNodes(nodeId) {
        try {
            var connectedNodes = [nodeId];
            var connectedEdges = [];

            var edges = network.body.data.edges.get();
            for (var i = 0; i < edges.length; i++) {
                var edge = edges[i];
                if (edge.from === nodeId || edge.to === nodeId) {
                    connectedEdges.push(edge.id);

                    var connectedNodeId = edge.from === nodeId ? edge.to : edge.from;
                    if (!connectedNodes.includes(connectedNodeId)) {
                        connectedNodes.push(connectedNodeId);
                    }
                }
            }

            return {
                nodes: connectedNodes,
                edges: connectedEdges
            };
        } catch(e) {
            console.error("Error getting connected nodes:", e);
            return { nodes: [nodeId], edges: [] };
        }
    }

    // Focus on a specific node
    function focusOnNode(nodeId) {
        try {
            if (lastSelectedNode !== nodeId) {
                nodeHistory.push({
                    nodeId: lastSelectedNode,
                    isFiltered: isFiltered
                });
            }

            lastSelectedNode = nodeId;
            isFiltered = true;

            var nodeInfo = network.body.data.nodes.get(nodeId);
            var nodeLabel = nodeInfo.label || nodeId;

            var connected = getConnectedNodes(nodeId);

            var connectedNodes = network.body.data.nodes.get(connected.nodes);
            var connectedEdges = network.body.data.edges.get(connected.edges);

            network.body.data.nodes.clear();
            network.body.data.edges.clear();

            network.body.data.nodes.add(connectedNodes);
            network.body.data.edges.add(connectedEdges);

            updateInfoPanel(nodeLabel, connected.nodes.length - 1, connected.edges.length);

            network.body.data.nodes.update([{
                id: nodeId,
                borderWidth: 3,
                borderColor: '#FF5733',
                size: 35
            }]);

            network.focus(nodeId, {
                scale: 1.2,
                animation: true
            });

            console.log("Focused on node:", nodeId);
        } catch(e) {
            console.error("Error focusing node:", e);
        }
    }

    // Reset the graph to its original state
    function resetGraph() {
        try {
            if (!isFiltered || !originalNodes || originalNodes.length === 0) return;

            nodeHistory = [];
            lastSelectedNode = null;
            isFiltered = false;

            network.body.data.nodes.clear();
            network.body.data.edges.clear();

            network.body.data.nodes.add(originalNodes.get());
            network.body.data.edges.add(originalEdges.get());

            network.fit({
                animation: true
            });

            var infoDiv = document.getElementById('graph-info');
            if (infoDiv) infoDiv.innerHTML = '';

            console.log("Graph reset");
        } catch(e) {
            console.error("Error resetting graph:", e);
        }
    }

    // Go back to the previous state
    function goBack() {
        try {
            if (nodeHistory.length === 0) {
                resetGraph();
                return;
            }

            var prevState = nodeHistory.pop();

            if (prevState.isFiltered && prevState.nodeId !== null) {
                focusOnNode(prevState.nodeId);
                nodeHistory.pop();
            } else {
                resetGraph();
            }

            console.log("Went back");
        } catch(e) {
            console.error("Error going back:", e);
        }
    }

    // Update the info panel
    function updateInfoPanel(nodeLabel, connectedCount, edgesCount) {
        try {
            var infoDiv = document.getElementById('graph-info');
            if (!infoDiv) return;

            infoDiv.innerHTML = `
                <div style="margin-bottom:5px;"><strong>Current node:</strong> ${nodeLabel}</div>
                <div><strong>Connected nodes:</strong> ${connectedCount}</div>
                <div><strong>Relationship count:</strong> ${edgesCount}</div>
                <div style="margin-top:8px;font-style:italic;font-size:11px;">Double-click a node to view its connections</div>
                <div style="font-style:italic;font-size:11px;">Click blank area to reset graph</div>
            `;
        } catch(e) {
            console.error("Error updating info panel:", e);
        }
    }
</script>
"""
