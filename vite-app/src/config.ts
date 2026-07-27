// Configuration for the application
export const config = {
	// WebSocket connection settings
	websocket: {
		host: 'localhost', // Will be discovered at runtime
		port: '8000', // Will be discovered at runtime
		protocol: 'ws',
	},
	// API settings
	api: {
		host: 'localhost', // Will be discovered at runtime
		port: '8000', // Will be discovered at runtime
		protocol: 'http',
	},
};

/**
 * Extract invocation_ids from URL filterConfig parameter.
 * Looks for filters on $.execution_metadata.invocation_id field.
 */
export const extractInvocationIdsFromUrl = (): string[] => {
	try {
		const urlParams = new URLSearchParams(window.location.search);
		const filterConfigStr = urlParams.get('filterConfig');
		if (!filterConfigStr) {
			return [];
		}

		// Parse the filter config JSON
		const filterConfig = JSON.parse(filterConfigStr);
		const invocationIds: string[] = [];

		// filterConfig is an array of FilterGroups
		if (Array.isArray(filterConfig)) {
			for (const group of filterConfig) {
				if (group.filters && Array.isArray(group.filters)) {
					for (const filter of group.filters) {
						// Check if this filter is on invocation_id field
						if (
							filter.field === '$.execution_metadata.invocation_id' &&
							(filter.operator === '==' || filter.operator === 'equals') &&
							filter.value
						) {
							invocationIds.push(filter.value);
						}
					}
				}
			}
		}

		return invocationIds;
	} catch (error) {
		console.warn('Failed to extract invocation_ids from URL:', error);
		return [];
	}
};

// Helper function to build WebSocket URL with optional invocation_ids filter
export const getWebSocketUrl = (invocationIds?: string[]): string => {
	const { protocol, host, port } = config.websocket;
	const baseUrl = `${protocol}://${host}:${port}/ws`;

	// If invocation_ids provided, add as query param
	if (invocationIds && invocationIds.length > 0) {
		const params = new URLSearchParams();
		params.set('invocation_ids', invocationIds.join(','));
		return `${baseUrl}?${params.toString()}`;
	}

	return baseUrl;
};

// Helper function to build API URL
export const getApiUrl = (): string => {
	const { protocol, host, port } = config.api;
	return `${protocol}://${host}:${port}`;
};

// Runtime configuration discovery
export const discoverServerConfig = async (): Promise<void> => {
	try {
		// First, check if server injected configuration is available
		if (window.SERVER_CONFIG) {
			const serverConfig = window.SERVER_CONFIG;
			config.websocket.host = serverConfig.host;
			config.websocket.port = serverConfig.port;
			config.websocket.protocol = serverConfig.protocol;
			config.api.host = serverConfig.host;
			config.api.port = serverConfig.port;
			config.api.protocol = serverConfig.apiProtocol;
			console.log('Using server-injected config:', config);
			return;
		}

		// Check if we're in Vite development mode
		if (import.meta.env.DEV) {
			// In dev mode, use localhost:8000
			config.websocket.host = 'localhost';
			config.websocket.port = '8000';
			config.websocket.protocol = 'ws';

			config.api.host = 'localhost';
			config.api.port = '8000';
			config.api.protocol = 'http';

			console.log('Using Vite dev config (localhost:8000):', config);
			return;
		}

		// Fallback: Try to discover server configuration from the current location
		const currentHost = window.location.hostname;
		const currentPort = window.location.port;
		const currentProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';

		// Update config with discovered values
		config.websocket.host = currentHost;
		config.websocket.port = currentPort || (currentProtocol === 'wss:' ? '443' : '80');
		config.websocket.protocol = currentProtocol;

		config.api.host = currentHost;
		config.api.port = currentPort || (currentProtocol === 'wss:' ? '443' : '80');
		config.api.protocol = window.location.protocol === 'https:' ? 'https:' : 'http:';

		console.log('Using discovered config from location:', config);
	} catch (error) {
		console.warn('Failed to discover server config, using defaults:', error);
	}
};
