/*
 Flattens a deeply nested JSON value into a single-level object with JSONPath keys.
 Prefers dot-notation for identifier-safe property names and uses brackets for
 array indices and special keys:
   $.foo.bar[0].baz
   $['weird.key'][2]['spaced name']
 Only leaf primitives (string, number, boolean, null) are emitted.
 Date objects are preserved as Date objects for filtering capabilities.
*/

export type FlatJson = Record<string, unknown>;

function isPlainObject(value: unknown): value is Record<string, unknown> {
	return (
		value !== null &&
		typeof value === "object" &&
		!Array.isArray(value)
	);
}

function joinPath(parent: string, segment: string | number): string {
	if (typeof segment === "number") {
		return `${parent}[${segment}]`;
	}
	// Use dot-notation when the key is identifier-safe, otherwise bracket-notation
	const identifierSafe = /^[A-Za-z_][A-Za-z0-9_]*$/.test(segment);
	if (identifierSafe) {
		return `${parent}.${segment}`;
	}
	const escaped = segment.replace(/'/g, "\\'");
	return `${parent}['${escaped}']`;
}

export function flattenJson(input: unknown, root: string = "$" ): FlatJson {
	const out: FlatJson = {};

	const walk = (value: unknown, path: string) => {
		if (
			value === null ||
			typeof value === "string" ||
			typeof value === "number" ||
			typeof value === "boolean" ||
			value === undefined
		) {
			out[path] = value;
			return;
		}

		// Handle Date objects by preserving them for filtering
		if (value instanceof Date) {
			out[path] = value;
			return;
		}

		if (Array.isArray(value)) {
			for (let i = 0; i < value.length; i++) {
				walk(value[i], joinPath(path, i));
			}
			return;
		}

		if (isPlainObject(value)) {
			for (const key of Object.keys(value)) {
				walk((value as Record<string, unknown>)[key], joinPath(path, key));
			}
			return;
		}

		// For unsupported types (e.g., functions, symbols), coerce to string
		out[path] = String(value);
	};

	walk(input, root);
	return out;
}

export default flattenJson;
