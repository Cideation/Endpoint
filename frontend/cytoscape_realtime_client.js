/**
 * Cytoscape.js Real-Time Client - No Cosmetic Delays
 * ⚡ Immediate graph updates directly from backend state changes
 * Stack: Apollo Client + GraphQL Subscriptions + Cytoscape.js
 */

class CytoscapeRealTimeClient {
    constructor(config = {}) {
        this.config = {
            graphqlEndpoint: config.graphqlEndpoint || 'ws://localhost:8004/graphql',
            wsEndpoint: config.wsEndpoint || 'ws://localhost:8004/ws/realtime',
            containerId: config.containerId || 'cytoscape-container',
            enableDebugLog: config.enableDebugLog || true,
            ...config
        };
        
        this.cy = null;
        this.apolloClient = null;
        this.wsConnection = null;
        this.subscriptions = new Map();
        this.graphVersion = 0;
        this.isConnected = false;
        
        this.init();
    }
    
    async init() {
        try {
            this.log('🚀 Initializing Real-Time Cytoscape Client');
            
            // Initialize Cytoscape
            await this.initCytoscape();
            
            // Initialize GraphQL Apollo Client
            await this.initApolloClient();
            
            // Initialize WebSocket connection
            await this.initWebSocket();
            
            // Subscribe to real-time updates
            await this.subscribeToUpdates();
            
            // Load initial graph state
            await this.loadInitialState();
            
            this.log('✅ Real-Time Client initialized successfully');
            
        } catch (error) {
            this.log('❌ Initialization failed:', error);
            throw error;
        }
    }
    
    async initCytoscape() {
        this.log('🎨 Initializing Cytoscape.js');
        
        this.cy = cytoscape({
            container: document.getElementById(this.config.containerId),
            
            style: [
                // Node styles based on phase
                {
                    selector: 'node[phase="alpha"]',
                    style: {
                        'background-color': '#3498db',
                        'border-color': '#2980b9',
                        'border-width': 2,
                        'label': 'data(node_id)',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'font-size': '10px',
                        'width': 40,
                        'height': 40
                    }
                },
                {
                    selector: 'node[phase="beta"]',
                    style: {
                        'background-color': '#f39c12',
                        'border-color': '#e67e22',
                        'border-width': 2,
                        'label': 'data(node_id)',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'font-size': '10px',
                        'width': 40,
                        'height': 40
                    }
                },
                {
                    selector: 'node[phase="gamma"]',
                    style: {
                        'background-color': '#27ae60',
                        'border-color': '#229954',
                        'border-width': 2,
                        'label': 'data(node_id)',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'font-size': '10px',
                        'width': 40,
                        'height': 40
                    }
                },
                // Node status indicators
                {
                    selector: 'node[status="executing"]',
                    style: {
                        'border-color': '#e74c3c',
                        'border-width': 4,
                        'background-color': '#ffebee'
                    }
                },
                {
                    selector: 'node[status="completed"]',
                    style: {
                        'border-color': '#27ae60',
                        'border-width': 3
                    }
                },
                // Edge styles
                {
                    selector: 'edge',
                    style: {
                        'width': 'data(weight)',
                        'line-color': '#95a5a6',
                        'target-arrow-color': '#95a5a6',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier',
                        'arrow-scale': 1.2
                    }
                },
                {
                    selector: 'edge[edge_type="alpha_edges"]',
                    style: {
                        'line-color': '#3498db',
                        'target-arrow-color': '#3498db'
                    }
                },
                {
                    selector: 'edge[edge_type="beta_relationships"]',
                    style: {
                        'line-color': '#f39c12',
                        'target-arrow-color': '#f39c12'
                    }
                },
                {
                    selector: 'edge[edge_type="gamma_edges"]',
                    style: {
                        'line-color': '#27ae60',
                        'target-arrow-color': '#27ae60'
                    }
                },
                {
                    selector: 'edge[edge_type="cross_phase_edges"]',
                    style: {
                        'line-color': '#9b59b6',
                        'target-arrow-color': '#9b59b6',
                        'line-style': 'dashed'
                    }
                }
            ],
            
            layout: {
                name: 'cose',
                idealEdgeLength: 100,
                nodeOverlap: 20,
                refresh: 20,
                fit: true,
                padding: 30,
                randomize: false,
                componentSpacing: 100,
                nodeRepulsion: 400000,
                edgeElasticity: 100,
                nestingFactor: 5,
                gravity: 80,
                numIter: 1000,
                initialTemp: 200,
                coolingFactor: 0.95,
                minTemp: 1.0
            },
            
            // Disable animations for immediate updates
            animationDuration: 0,
            animationEasing: 'linear'
        });
        
        // Add event handlers
        this.setupEventHandlers();
    }
    
    setupEventHandlers() {
        // Node click handler
        this.cy.on('tap', 'node', (event) => {
            const node = event.target;
            this.log('🔍 Node clicked:', node.data());
            this.showNodeDetails(node);
        });
        
        // Node drag handler for position updates
        this.cy.on('dragfree', 'node', (event) => {
            const node = event.target;
            const position = node.position();
            
            // Send position update via GraphQL mutation
            this.updateNodePosition(node.data('node_id'), position.x, position.y);
        });
        
        // Edge click handler
        this.cy.on('tap', 'edge', (event) => {
            const edge = event.target;
            this.log('🔗 Edge clicked:', edge.data());
            this.showEdgeDetails(edge);
        });
    }
    
    async initApolloClient() {
        this.log('📡 Initializing Apollo GraphQL Client');
        
        // Note: In a real implementation, you would use Apollo Client
        // For this example, we'll use a simplified GraphQL client
        this.graphqlClient = {
            query: async (query, variables = {}) => {
                const response = await fetch(this.config.graphqlEndpoint.replace('ws://', 'http://'), {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ query, variables })
                });
                return response.json();
            },
            
            mutate: async (mutation, variables = {}) => {
                const response = await fetch(this.config.graphqlEndpoint.replace('ws://', 'http://'), {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ query: mutation, variables })
                });
                return response.json();
            }
        };
    }
    
    async initWebSocket() {
        this.log('🔌 Initializing WebSocket connection');
        
        return new Promise((resolve, reject) => {
            this.wsConnection = new WebSocket(this.config.wsEndpoint);
            
            this.wsConnection.onopen = () => {
                this.log('✅ WebSocket connected');
                this.isConnected = true;
                
                // Send ping to test connection
                this.wsConnection.send(JSON.stringify({
                    type: 'ping',
                    timestamp: new Date().toISOString()
                }));
                
                resolve();
            };
            
            this.wsConnection.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    this.handleRealtimeUpdate(message);
                } catch (error) {
                    this.log('❌ Error parsing WebSocket message:', error);
                }
            };
            
            this.wsConnection.onclose = () => {
                this.log('❌ WebSocket disconnected');
                this.isConnected = false;
                
                // Attempt to reconnect
                setTimeout(() => {
                    this.log('🔄 Attempting to reconnect...');
                    this.initWebSocket();
                }, 3000);
            };
            
            this.wsConnection.onerror = (error) => {
                this.log('❌ WebSocket error:', error);
                reject(error);
            };
        });
    }
    
    async subscribeToUpdates() {
        this.log('📡 Subscribing to GraphQL real-time updates');
        
        // Note: In a real implementation with Apollo Client:
        // const subscription = gql`
        //     subscription GraphUpdates {
        //         graphUpdates {
        //             updateId
        //             updateType
        //             timestamp
        //             node { ... }
        //             edge { ... }
        //             functorExecution { ... }
        //             graphVersion
        //         }
        //     }
        // `;
        // 
        // this.apolloClient.subscribe({
        //     query: subscription
        // }).subscribe({
        //     next: (data) => this.handleGraphQLUpdate(data),
        //     error: (error) => this.log('❌ Subscription error:', error)
        // });
        
        // For now, we rely on WebSocket updates
        this.log('✅ Subscribed to real-time updates via WebSocket');
    }
    
    async loadInitialState() {
        this.log('📥 Loading initial graph state');
        
        try {
            // Query current nodes
            const nodesQuery = `
                query GetNodes {
                    nodes {
                        nodeId
                        functorType
                        phase
                        status
                        inputs
                        outputs
                        positionX
                        positionY
                        lastUpdate
                        version
                    }
                }
            `;
            
            const edgesQuery = `
                query GetEdges {
                    edges {
                        edgeId
                        sourceNode
                        targetNode
                        edgeType
                        weight
                        metadata
                        lastUpdate
                        version
                    }
                }
            `;
            
            const [nodesResult, edgesResult] = await Promise.all([
                this.graphqlClient.query(nodesQuery),
                this.graphqlClient.query(edgesQuery)
            ]);
            
            // Add nodes to graph
            if (nodesResult.data && nodesResult.data.nodes) {
                const nodes = nodesResult.data.nodes.map(node => ({
                    data: {
                        id: node.nodeId,
                        node_id: node.nodeId,
                        functor_type: node.functorType,
                        phase: node.phase,
                        status: node.status,
                        inputs: node.inputs,
                        outputs: node.outputs,
                        last_update: node.lastUpdate,
                        version: node.version
                    },
                    position: {
                        x: node.positionX || Math.random() * 400,
                        y: node.positionY || Math.random() * 400
                    }
                }));
                
                this.cy.add(nodes);
                this.log(`📊 Loaded ${nodes.length} nodes`);
            }
            
            // Add edges to graph
            if (edgesResult.data && edgesResult.data.edges) {
                const edges = edgesResult.data.edges.map(edge => ({
                    data: {
                        id: edge.edgeId,
                        source: edge.sourceNode,
                        target: edge.targetNode,
                        edge_type: edge.edgeType,
                        weight: edge.weight,
                        metadata: edge.metadata,
                        last_update: edge.lastUpdate,
                        version: edge.version
                    }
                }));
                
                this.cy.add(edges);
                this.log(`🔗 Loaded ${edges.length} edges`);
            }
            
            // Apply layout
            this.cy.layout({ name: 'cose' }).run();
            
        } catch (error) {
            this.log('❌ Error loading initial state:', error);
        }
    }
    
    handleRealtimeUpdate(message) {
        this.log('⚡ Real-time update received:', message.type);
        
        switch (message.type) {
            case 'INITIAL_STATE':
                this.handleInitialState(message.payload);
                break;
                
            case 'NODE_UPDATE':
                this.handleNodeUpdate(message.payload);
                break;
                
            case 'EDGE_UPDATE':
                this.handleEdgeUpdate(message.payload);
                break;
                
            case 'FUNCTOR_EXECUTION':
                this.handleFunctorExecution(message.payload);
                break;
                
            case 'pong':
                this.log('🏓 Pong received');
                break;
                
            default:
                this.log('❓ Unknown message type:', message.type);
        }
    }
    
    handleNodeUpdate(payload) {
        const { node, graph_version } = payload;
        this.graphVersion = graph_version;
        
        // Check if node exists
        const existingNode = this.cy.getElementById(node.node_id);
        
        if (existingNode.length > 0) {
            // Update existing node - NO ANIMATION, immediate update
            existingNode.data({
                functor_type: node.functor_type,
                phase: node.phase,
                status: node.status,
                inputs: JSON.stringify(node.inputs || {}),
                outputs: JSON.stringify(node.outputs || {}),
                last_update: node.last_update,
                version: node.version
            });
            
            // Update position if provided
            if (node.position_x !== undefined && node.position_y !== undefined) {
                existingNode.position({
                    x: node.position_x,
                    y: node.position_y
                });
            }
            
            this.log(`⚡ Node updated immediately: ${node.node_id}`);
        } else {
            // Add new node - NO ANIMATION, immediate add
            this.cy.add({
                data: {
                    id: node.node_id,
                    node_id: node.node_id,
                    functor_type: node.functor_type,
                    phase: node.phase,
                    status: node.status,
                    inputs: JSON.stringify(node.inputs || {}),
                    outputs: JSON.stringify(node.outputs || {}),
                    last_update: node.last_update,
                    version: node.version
                },
                position: {
                    x: node.position_x || Math.random() * 400,
                    y: node.position_y || Math.random() * 400
                }
            });
            
            this.log(`⚡ Node added immediately: ${node.node_id}`);
        }
        
        // Trigger any callbacks
        this.triggerUpdateCallbacks('node', node);
    }
    
    handleEdgeUpdate(payload) {
        const { edge, graph_version } = payload;
        this.graphVersion = graph_version;
        
        const edgeId = `${edge.source_node}-${edge.target_node}`;
        const existingEdge = this.cy.getElementById(edgeId);
        
        if (existingEdge.length > 0) {
            // Update existing edge - NO ANIMATION, immediate update
            existingEdge.data({
                edge_type: edge.edge_type,
                weight: edge.weight,
                metadata: JSON.stringify(edge.metadata || {}),
                last_update: edge.last_update,
                version: edge.version
            });
            
            this.log(`⚡ Edge updated immediately: ${edgeId}`);
        } else {
            // Add new edge - NO ANIMATION, immediate add
            this.cy.add({
                data: {
                    id: edgeId,
                    source: edge.source_node,
                    target: edge.target_node,
                    edge_type: edge.edge_type,
                    weight: edge.weight,
                    metadata: JSON.stringify(edge.metadata || {}),
                    last_update: edge.last_update,
                    version: edge.version
                }
            });
            
            this.log(`⚡ Edge added immediately: ${edgeId}`);
        }
        
        // Trigger any callbacks
        this.triggerUpdateCallbacks('edge', edge);
    }
    
    handleFunctorExecution(payload) {
        const { functor_result, graph_version } = payload;
        this.graphVersion = graph_version;
        
        this.log(`⚡ Functor execution: ${functor_result.functor_type} on ${functor_result.node_id}`);
        
        // Update affected nodes immediately
        const affected_nodes = JSON.parse(functor_result.affected_nodes || '{}');
        
        for (const [nodeId, nodeChanges] of Object.entries(affected_nodes)) {
            const node = this.cy.getElementById(nodeId);
            if (node.length > 0) {
                // Apply changes immediately - NO ANIMATION
                node.data(nodeChanges);
                this.log(`⚡ Node ${nodeId} updated from functor execution`);
            }
        }
        
        // Trigger callbacks
        this.triggerUpdateCallbacks('functor_execution', functor_result);
    }
    
    async updateNodePosition(nodeId, x, y) {
        const mutation = `
            mutation UpdateNodePosition($nodeId: String!, $x: Float!, $y: Float!) {
                updateNodePosition(nodeId: $nodeId, x: $x, y: $y)
            }
        `;
        
        try {
            await this.graphqlClient.mutate(mutation, { nodeId, x, y });
            this.log(`📍 Position updated: ${nodeId} (${x.toFixed(1)}, ${y.toFixed(1)})`);
        } catch (error) {
            this.log('❌ Error updating position:', error);
        }
    }
    
    async executeFunctor(nodeId, functorType, inputs = {}) {
        const mutation = `
            mutation ExecuteFunctor($nodeId: String!, $functorType: String!, $inputs: String!) {
                executeFunctor(nodeId: $nodeId, functorType: $functorType, inputs: $inputs) {
                    executionId
                    success
                    executionTimeMs
                    outputs
                }
            }
        `;
        
        try {
            const result = await this.graphqlClient.mutate(mutation, {
                nodeId,
                functorType,
                inputs: JSON.stringify(inputs)
            });
            
            this.log(`⚙️ Functor executed: ${functorType} on ${nodeId}`);
            return result.data.executeFunctor;
        } catch (error) {
            this.log('❌ Error executing functor:', error);
            throw error;
        }
    }
    
    triggerUpdateCallbacks(updateType, data) {
        // Trigger any registered callbacks
        if (this.config.onUpdate) {
            this.config.onUpdate(updateType, data, this.graphVersion);
        }
        
        // Dispatch custom event
        const event = new CustomEvent('graph_update', {
            detail: { updateType, data, graphVersion: this.graphVersion }
        });
        document.dispatchEvent(event);
    }
    
    showNodeDetails(node) {
        const data = node.data();
        const details = {
            'Node ID': data.node_id,
            'Functor Type': data.functor_type,
            'Phase': data.phase,
            'Status': data.status,
            'Last Update': data.last_update,
            'Version': data.version,
            'Inputs': data.inputs,
            'Outputs': data.outputs
        };
        
        console.table(details);
        
        // You can implement a custom modal or tooltip here
        this.log('📋 Node details logged to console');
    }
    
    showEdgeDetails(edge) {
        const data = edge.data();
        const details = {
            'Edge ID': data.id,
            'Source': data.source,
            'Target': data.target,
            'Type': data.edge_type,
            'Weight': data.weight,
            'Last Update': data.last_update,
            'Metadata': data.metadata
        };
        
        console.table(details);
        this.log('📋 Edge details logged to console');
    }
    
    log(...args) {
        if (this.config.enableDebugLog) {
            console.log('[RealTime Client]', ...args);
        }
    }
    
    // Public API methods
    getGraphVersion() {
        return this.graphVersion;
    }
    
    isRealTimeConnected() {
        return this.isConnected;
    }
    
    getCytoscapeInstance() {
        return this.cy;
    }
    
    disconnect() {
        if (this.wsConnection) {
            this.wsConnection.close();
        }
        this.log('🔌 Disconnected from real-time updates');
    }
}

// Usage example:
/*
const realTimeClient = new CytoscapeRealTimeClient({
    containerId: 'graph-container',
    graphqlEndpoint: 'ws://localhost:8004/graphql',
    wsEndpoint: 'ws://localhost:8004/ws/realtime',
    enableDebugLog: true,
    onUpdate: (updateType, data, version) => {
        console.log(`Graph updated: ${updateType}, Version: ${version}`);
    }
});

// Execute a functor
realTimeClient.executeFunctor('V01_ProductComponent', 'MaterialSpecification', {
    material_type: 'Steel_A36',
    strength_requirement: 36000
});
*/ 