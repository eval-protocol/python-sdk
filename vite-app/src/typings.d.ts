declare module '*.css' {
    const content: Record<string, string>;
    export default content;
}

declare module '*.png' {
    const content: string;
    export default content;
}
/// <reference types="vite/client" />

interface ImportMetaEnv {
    readonly DEV: boolean
    readonly PROD: boolean
    readonly MODE: string
}

interface ImportMeta {
    readonly env: ImportMetaEnv
}
