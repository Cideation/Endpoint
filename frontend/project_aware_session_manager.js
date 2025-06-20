/**
 * Project-Aware Session Manager
 * Handles agent_project_tag propagation across all frontend operations
 */

class ProjectAwareSessionManager {
    constructor() {
        this.currentSession = null;
        this.activeProjects = new Map();
        this.sessionStorage = window.sessionStorage;
        this.listeners = new Set();
        
        // Initialize from stored session
        this.loadStoredSession();
    }

    /**
     * Assign agent project tag based on node context
     * Mirrors the backend assign_agent_project_tag function
     */
    assignAgentProjectTag(agentId, nodeContext) {
        const project = nodeContext.project_id || 'default';
        const phase = nodeContext.phase || 'phase_1';
        const nodes = nodeContext.node_ids || [];

        const tag = `${agentId}::${project}::${phase}`;

        const sessionData = {
            agent_id: agentId,
            agent_project_tag: tag,
            project_id: project,
            phase: phase,
            active_nodes: nodes,
            timestamp: new Date().toISOString(),
            session_id: this.generateSessionId()
        };

        this.currentSession = sessionData;
        this.storeSession();
        this.notifyListeners('session_assigned', sessionData);

        console.log('🎯 Agent Project Tag Assigned:', tag);
        return sessionData;
    }

    /**
     * Get current agent project tag for GraphQL operations
     */
    getCurrentProjectTag() {
        return this.currentSession?.agent_project_tag || null;
    }

    /**
     * Get current session metadata for GraphQL requests
     */
    getSessionMetadata() {
        if (!this.currentSession) {
            return {};
        }

        return {
            agentProjectTag: this.currentSession.agent_project_tag,
            agentId: this.currentSession.agent_id,
            projectId: this.currentSession.project_id,
            phase: this.currentSession.phase,
            sessionId: this.currentSession.session_id
        };
    }

    /**
     * Switch to a different project/phase context
     */
    switchProject(projectId, phase = null, nodeIds = []) {
        if (!this.currentSession) {
            console.warn('No active session to switch project');
            return null;
        }

        const nodeContext = {
            project_id: projectId,
            phase: phase || this.currentSession.phase,
            node_ids: nodeIds
        };

        return this.assignAgentProjectTag(this.currentSession.agent_id, nodeContext);
    }

    /**
     * Initialize session for new agent
     */
    initializeSession(agentId, projectId = 'default', phase = 'phase_1') {
        const nodeContext = {
            project_id: projectId,
            phase: phase,
            node_ids: []
        };

        return this.assignAgentProjectTag(agentId, nodeContext);
    }

    /**
     * Add project tag metadata to GraphQL variables
     */
    addProjectTagToVariables(variables = {}) {
        const metadata = this.getSessionMetadata();
        
        return {
            ...variables,
            projectContext: {
                agentProjectTag: metadata.agentProjectTag,
                projectId: metadata.projectId,
                phase: metadata.phase,
                agentId: metadata.agentId
            }
        };
    }

    /**
     * Create project-aware GraphQL client wrapper
     */
    createProjectAwareGraphQLClient(baseClient) {
        const sessionManager = this;
        
        return {
            query: async (query, variables = {}) => {
                const enhancedVariables = sessionManager.addProjectTagToVariables(variables);
                return baseClient.query(query, enhancedVariables);
            },
            
            mutate: async (mutation, variables = {}) => {
                const enhancedVariables = sessionManager.addProjectTagToVariables(variables);
                return baseClient.mutate(mutation, enhancedVariables);
            },
            
            subscribe: (subscription, variables = {}) => {
                const enhancedVariables = sessionManager.addProjectTagToVariables(variables);
                return baseClient.subscribe(subscription, enhancedVariables);
            }
        };
    }

    /**
     * Get available projects for current agent
     */
    async getAvailableProjects() {
        // This would typically come from a GraphQL query
        // For now, return stored projects or defaults
        const stored = this.sessionStorage.getItem('bem_available_projects');
        if (stored) {
            return JSON.parse(stored);
        }

        return [
            { id: 'default', name: 'Default Project', phases: ['phase_1', 'phase_2', 'phase_3'] },
            { id: 'demo', name: 'Demo Project', phases: ['phase_1', 'phase_2'] },
            { id: 'production', name: 'Production Project', phases: ['phase_2', 'phase_3'] }
        ];
    }

    /**
     * Filter nodes by current project tag
     */
    filterNodesByProject(nodes) {
        if (!this.currentSession) {
            return nodes;
        }

        const { project_id, phase } = this.currentSession;
        
        return nodes.filter(node => {
            const nodeProject = node.project_id || node.projectId || 'default';
            const nodePhase = node.phase || 'phase_1';
            
            return nodeProject === project_id && nodePhase === phase;
        });
    }

    /**
     * Filter dashboard data by project tag
     */
    filterDashboardByProject(dashboardData) {
        if (!this.currentSession) {
            return dashboardData;
        }

        const { project_id, phase } = this.currentSession;

        return {
            ...dashboardData,
            nodes: this.filterNodesByProject(dashboardData.nodes || []),
            metrics: this.filterMetricsByProject(dashboardData.metrics || {}),
            pulses: this.filterPulsesByProject(dashboardData.pulses || [])
        };
    }

    /**
     * Filter metrics by project
     */
    filterMetricsByProject(metrics) {
        const { project_id } = this.currentSession;
        
        // Filter metrics that have project context
        const filtered = {};
        for (const [key, value] of Object.entries(metrics)) {
            if (typeof value === 'object' && value.project_id) {
                if (value.project_id === project_id) {
                    filtered[key] = value;
                }
            } else {
                // Include metrics without project context
                filtered[key] = value;
            }
        }
        
        return filtered;
    }

    /**
     * Filter pulse events by project
     */
    filterPulsesByProject(pulses) {
        const { project_id, phase } = this.currentSession;
        
        return pulses.filter(pulse => {
            const pulseProject = pulse.project_id || pulse.metadata?.project_id || 'default';
            const pulsePhase = pulse.phase || pulse.metadata?.phase;
            
            return pulseProject === project_id && (!pulsePhase || pulsePhase === phase);
        });
    }

    /**
     * Generate unique session ID
     */
    generateSessionId() {
        return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    /**
     * Store session to localStorage
     */
    storeSession() {
        if (this.currentSession) {
            this.sessionStorage.setItem('bem_current_session', JSON.stringify(this.currentSession));
        }
    }

    /**
     * Load session from localStorage
     */
    loadStoredSession() {
        const stored = this.sessionStorage.getItem('bem_current_session');
        if (stored) {
            try {
                this.currentSession = JSON.parse(stored);
                console.log('🔄 Restored session:', this.currentSession.agent_project_tag);
            } catch (e) {
                console.warn('Failed to restore session:', e);
                this.sessionStorage.removeItem('bem_current_session');
            }
        }
    }

    /**
     * Clear current session
     */
    clearSession() {
        this.currentSession = null;
        this.sessionStorage.removeItem('bem_current_session');
        this.notifyListeners('session_cleared', null);
        console.log('🗑️ Session cleared');
    }

    /**
     * Add listener for session changes
     */
    addListener(callback) {
        this.listeners.add(callback);
    }

    /**
     * Remove listener
     */
    removeListener(callback) {
        this.listeners.delete(callback);
    }

    /**
     * Notify all listeners of session changes
     */
    notifyListeners(event, data) {
        this.listeners.forEach(callback => {
            try {
                callback(event, data);
            } catch (e) {
                console.error('Session listener error:', e);
            }
        });
    }

    /**
     * Get session status for UI display
     */
    getSessionStatus() {
        if (!this.currentSession) {
            return {
                active: false,
                message: 'No active session'
            };
        }

        return {
            active: true,
            agent_id: this.currentSession.agent_id,
            project_id: this.currentSession.project_id,
            phase: this.currentSession.phase,
            tag: this.currentSession.agent_project_tag,
            node_count: this.currentSession.active_nodes.length,
            session_time: this.getSessionDuration()
        };
    }

    /**
     * Get session duration
     */
    getSessionDuration() {
        if (!this.currentSession?.timestamp) {
            return '0m';
        }

        const start = new Date(this.currentSession.timestamp);
        const now = new Date();
        const diffMinutes = Math.floor((now - start) / (1000 * 60));
        
        if (diffMinutes < 60) {
            return `${diffMinutes}m`;
        } else {
            const hours = Math.floor(diffMinutes / 60);
            const minutes = diffMinutes % 60;
            return `${hours}h ${minutes}m`;
        }
    }

    /**
     * Update active nodes in current session
     */
    updateActiveNodes(nodeIds) {
        if (this.currentSession) {
            this.currentSession.active_nodes = nodeIds;
            this.storeSession();
            this.notifyListeners('nodes_updated', nodeIds);
        }
    }

    /**
     * Add node to active session
     */
    addNodeToSession(nodeId) {
        if (this.currentSession) {
            if (!this.currentSession.active_nodes.includes(nodeId)) {
                this.currentSession.active_nodes.push(nodeId);
                this.storeSession();
                this.notifyListeners('node_added', nodeId);
            }
        }
    }

    /**
     * Remove node from active session
     */
    removeNodeFromSession(nodeId) {
        if (this.currentSession) {
            const index = this.currentSession.active_nodes.indexOf(nodeId);
            if (index > -1) {
                this.currentSession.active_nodes.splice(index, 1);
                this.storeSession();
                this.notifyListeners('node_removed', nodeId);
            }
        }
    }
}

// Global session manager instance
window.projectSessionManager = new ProjectAwareSessionManager();

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ProjectAwareSessionManager;
} 