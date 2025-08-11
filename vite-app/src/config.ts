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

// Helper function to build WebSocket URL
export const getWebSocketUrl = (): string => {
	const { protocol, host, port } = config.websocket;
	return `${protocol}://${host}:${port}/ws`;
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
