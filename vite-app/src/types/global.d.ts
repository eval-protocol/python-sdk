// Global type declarations
declare global {
	interface Window {
		SERVER_CONFIG?: {
			host: string;
			port: string;
			protocol: string;
			apiProtocol: string;
		};
	}
}

export {};
