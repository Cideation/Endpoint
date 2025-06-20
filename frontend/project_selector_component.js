/**
 * Project Selector Component
 * UI component for selecting/switching agent project tags
 */

class ProjectSelectorComponent {
    constructor(containerId, sessionManager) {
        this.container = document.getElementById(containerId);
        this.sessionManager = sessionManager;
        this.isInitialized = false;
        
        if (!this.container) {
            console.error(`Project selector container '${containerId}' not found`);
            return;
        }

        this.init();
    }

    async init() {
        await this.render();
        this.attachEventListeners();
        this.isInitialized = true;

        // Listen for session changes
        this.sessionManager.addListener((event, data) => {
            if (event === 'session_assigned' || event === 'session_cleared') {
                this.updateSessionDisplay();
            }
        });

        console.log('🎯 Project Selector Component initialized');
    }

    async render() {
        const projects = await this.sessionManager.getAvailableProjects();
        const sessionStatus = this.sessionManager.getSessionStatus();

        this.container.innerHTML = `
            <div class="project-selector-panel">
                <!-- Session Status Display -->
                <div class="session-status ${sessionStatus.active ? 'active' : 'inactive'}">
                    <div class="session-header">
                        <span class="session-indicator"></span>
                        <h3>Agent Session</h3>
                        ${sessionStatus.active ? `<span class="session-time">${sessionStatus.session_time}</span>` : ''}
                    </div>
                    
                    ${sessionStatus.active ? `
                        <div class="current-session">
                            <div class="session-info">
                                <strong>Tag:</strong> 
                                <code class="agent-tag">${sessionStatus.tag}</code>
                            </div>
                            <div class="session-details">
                                <span><strong>Agent:</strong> ${sessionStatus.agent_id}</span>
                                <span><strong>Project:</strong> ${sessionStatus.project_id}</span>
                                <span><strong>Phase:</strong> ${sessionStatus.phase}</span>
                                <span><strong>Nodes:</strong> ${sessionStatus.node_count}</span>
                            </div>
                        </div>
                    ` : `
                        <div class="no-session">
                            <p>No active agent session</p>
                        </div>
                    `}
                </div>

                <!-- Project Selection Form -->
                <div class="project-selection-form">
                    <h4>Initialize/Switch Project</h4>
                    
                    <div class="form-group">
                        <label for="agent-id-input">Agent ID:</label>
                        <input type="text" 
                               id="agent-id-input" 
                               placeholder="Enter agent identifier"
                               value="${sessionStatus.active ? sessionStatus.agent_id : 'agent_001'}"
                               ${sessionStatus.active ? 'readonly' : ''}>
                    </div>

                    <div class="form-group">
                        <label for="project-select">Project:</label>
                        <select id="project-select">
                            <option value="">Select Project...</option>
                            ${projects.map(project => `
                                <option value="${project.id}" 
                                        ${sessionStatus.active && sessionStatus.project_id === project.id ? 'selected' : ''}>
                                    ${project.name}
                                </option>
                            `).join('')}
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="phase-select">Phase:</label>
                        <select id="phase-select">
                            <option value="">Select Phase...</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="node-ids-input">Node IDs (optional):</label>
                        <input type="text" 
                               id="node-ids-input" 
                               placeholder="node_1,node_2,node_3"
                               value="${sessionStatus.active ? sessionStatus.active_nodes?.join(',') || '' : ''}">
                        <small>Comma-separated node IDs</small>
                    </div>

                    <div class="form-actions">
                        ${sessionStatus.active ? `
                            <button id="switch-project-btn" class="btn btn-primary">
                                Switch Project
                            </button>
                            <button id="clear-session-btn" class="btn btn-secondary">
                                Clear Session
                            </button>
                        ` : `
                            <button id="initialize-session-btn" class="btn btn-primary">
                                Initialize Session
                            </button>
                        `}
                    </div>
                </div>

                <!-- Quick Project Actions -->
                <div class="quick-actions">
                    <h4>Quick Actions</h4>
                    <div class="quick-action-buttons">
                        ${projects.map(project => `
                            <button class="quick-project-btn" 
                                    data-project="${project.id}"
                                    data-phase="phase_1">
                                📋 ${project.name}
                            </button>
                        `).join('')}
                    </div>
                </div>

                <!-- Project Tag Preview -->
                <div class="tag-preview">
                    <h4>Preview Agent Tag</h4>
                    <div class="preview-display">
                        <code id="tag-preview">No project selected</code>
                    </div>
                </div>
            </div>
        `;

        // Update phases when project changes
        this.updatePhaseOptions();
    }

    attachEventListeners() {
        // Project selection change
        const projectSelect = this.container.querySelector('#project-select');
        const phaseSelect = this.container.querySelector('#phase-select');
        const agentInput = this.container.querySelector('#agent-id-input');
        const nodeInput = this.container.querySelector('#node-ids-input');

        if (projectSelect) {
            projectSelect.addEventListener('change', () => {
                this.updatePhaseOptions();
                this.updateTagPreview();
            });
        }

        if (phaseSelect) {
            phaseSelect.addEventListener('change', () => {
                this.updateTagPreview();
            });
        }

        if (agentInput) {
            agentInput.addEventListener('input', () => {
                this.updateTagPreview();
            });
        }

        // Initialize session button
        const initBtn = this.container.querySelector('#initialize-session-btn');
        if (initBtn) {
            initBtn.addEventListener('click', () => {
                this.initializeSession();
            });
        }

        // Switch project button
        const switchBtn = this.container.querySelector('#switch-project-btn');
        if (switchBtn) {
            switchBtn.addEventListener('click', () => {
                this.switchProject();
            });
        }

        // Clear session button
        const clearBtn = this.container.querySelector('#clear-session-btn');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                this.clearSession();
            });
        }

        // Quick action buttons
        const quickBtns = this.container.querySelectorAll('.quick-project-btn');
        quickBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const projectId = btn.dataset.project;
                const phase = btn.dataset.phase;
                this.quickInitialize(projectId, phase);
            });
        });
    }

    async updatePhaseOptions() {
        const projectSelect = this.container.querySelector('#project-select');
        const phaseSelect = this.container.querySelector('#phase-select');
        
        if (!projectSelect || !phaseSelect) return;

        const selectedProjectId = projectSelect.value;
        const projects = await this.sessionManager.getAvailableProjects();
        const selectedProject = projects.find(p => p.id === selectedProjectId);

        phaseSelect.innerHTML = '<option value="">Select Phase...</option>';

        if (selectedProject) {
            selectedProject.phases.forEach(phase => {
                const option = document.createElement('option');
                option.value = phase;
                option.textContent = phase.replace('_', ' ').toUpperCase();
                phaseSelect.appendChild(option);
            });

            // Auto-select first phase
            if (selectedProject.phases.length > 0) {
                phaseSelect.value = selectedProject.phases[0];
            }
        }

        this.updateTagPreview();
    }

    updateTagPreview() {
        const agentInput = this.container.querySelector('#agent-id-input');
        const projectSelect = this.container.querySelector('#project-select');
        const phaseSelect = this.container.querySelector('#phase-select');
        const tagPreview = this.container.querySelector('#tag-preview');

        if (!agentInput || !projectSelect || !phaseSelect || !tagPreview) return;

        const agentId = agentInput.value.trim();
        const projectId = projectSelect.value;
        const phase = phaseSelect.value;

        if (agentId && projectId && phase) {
            const tag = `${agentId}::${projectId}::${phase}`;
            tagPreview.textContent = tag;
            tagPreview.className = 'tag-valid';
        } else {
            tagPreview.textContent = 'Incomplete - fill all fields';
            tagPreview.className = 'tag-invalid';
        }
    }

    initializeSession() {
        const agentInput = this.container.querySelector('#agent-id-input');
        const projectSelect = this.container.querySelector('#project-select');
        const phaseSelect = this.container.querySelector('#phase-select');
        const nodeInput = this.container.querySelector('#node-ids-input');

        const agentId = agentInput.value.trim();
        const projectId = projectSelect.value;
        const phase = phaseSelect.value;
        const nodeIds = this.parseNodeIds(nodeInput.value);

        if (!agentId || !projectId || !phase) {
            this.showError('Please fill in all required fields');
            return;
        }

        const nodeContext = {
            project_id: projectId,
            phase: phase,
            node_ids: nodeIds
        };

        const session = this.sessionManager.assignAgentProjectTag(agentId, nodeContext);
        this.showSuccess(`Session initialized: ${session.agent_project_tag}`);
        
        // Re-render to show active session
        setTimeout(() => this.render(), 100);
    }

    switchProject() {
        const projectSelect = this.container.querySelector('#project-select');
        const phaseSelect = this.container.querySelector('#phase-select');
        const nodeInput = this.container.querySelector('#node-ids-input');

        const projectId = projectSelect.value;
        const phase = phaseSelect.value;
        const nodeIds = this.parseNodeIds(nodeInput.value);

        if (!projectId || !phase) {
            this.showError('Please select project and phase');
            return;
        }

        const session = this.sessionManager.switchProject(projectId, phase, nodeIds);
        if (session) {
            this.showSuccess(`Switched to: ${session.agent_project_tag}`);
            setTimeout(() => this.render(), 100);
        }
    }

    clearSession() {
        this.sessionManager.clearSession();
        this.showSuccess('Session cleared');
        setTimeout(() => this.render(), 100);
    }

    quickInitialize(projectId, phase) {
        const agentId = `agent_${Date.now().toString().slice(-4)}`;
        
        const nodeContext = {
            project_id: projectId,
            phase: phase,
            node_ids: []
        };

        const session = this.sessionManager.assignAgentProjectTag(agentId, nodeContext);
        this.showSuccess(`Quick session: ${session.agent_project_tag}`);
        
        setTimeout(() => this.render(), 100);
    }

    parseNodeIds(nodeString) {
        if (!nodeString || !nodeString.trim()) {
            return [];
        }

        return nodeString
            .split(',')
            .map(id => id.trim())
            .filter(id => id.length > 0);
    }

    updateSessionDisplay() {
        if (this.isInitialized) {
            setTimeout(() => this.render(), 100);
        }
    }

    showSuccess(message) {
        this.showNotification(message, 'success');
    }

    showError(message) {
        this.showNotification(message, 'error');
    }

    showNotification(message, type) {
        // Remove existing notifications
        const existing = this.container.querySelectorAll('.notification');
        existing.forEach(n => n.remove());

        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;

        this.container.prepend(notification);

        // Auto-remove after 3 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 3000);
    }

    getSelectedProjectContext() {
        const sessionStatus = this.sessionManager.getSessionStatus();
        if (sessionStatus.active) {
            return {
                projectId: sessionStatus.project_id,
                phase: sessionStatus.phase,
                agentId: sessionStatus.agent_id,
                tag: sessionStatus.tag
            };
        }
        return null;
    }

    // Public method to refresh the component
    async refresh() {
        await this.render();
    }
}

// Auto-initialize if container exists
document.addEventListener('DOMContentLoaded', () => {
    if (typeof window.projectSessionManager !== 'undefined') {
        const container = document.getElementById('project-selector-container');
        if (container) {
            window.projectSelector = new ProjectSelectorComponent(
                'project-selector-container', 
                window.projectSessionManager
            );
        }
    }
});

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ProjectSelectorComponent;
} 