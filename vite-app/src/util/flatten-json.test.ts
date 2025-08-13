import { describe, it, expect } from 'vitest'
import { flattenJson } from './flatten-json'
import { readFileSync } from 'node:fs'

describe('flattenJson', () => {
	describe('basic functionality', () => {
		it('flattens simple primitives', () => {
			const input = {
				name: 'John',
				age: 30,
				active: true,
				score: null
			}
			const result = flattenJson(input)

			expect(result).toEqual({
				'$.name': 'John',
				'$.age': 30,
				'$.active': true,
				'$.score': null
			})
		})

		it('flattens nested objects', () => {
			const input = {
				user: {
					profile: {
						firstName: 'John',
						lastName: 'Doe'
					}
				}
			}
			const result = flattenJson(input)

			expect(result).toEqual({
				'$.user.profile.firstName': 'John',
				'$.user.profile.lastName': 'Doe'
			})
		})

		it('flattens arrays', () => {
			const input = {
				tags: ['javascript', 'typescript'],
				numbers: [1, 2, 3]
			}
			const result = flattenJson(input)

			expect(result).toEqual({
				'$.tags[0]': 'javascript',
				'$.tags[1]': 'typescript',
				'$.numbers[0]': 1,
				'$.numbers[1]': 2,
				'$.numbers[2]': 3
			})
		})

		it('handles mixed nested structures', () => {
			const input = {
				users: [
					{ name: 'Alice', scores: [85, 90] },
					{ name: 'Bob', scores: [92, 88] }
				]
			}
			const result = flattenJson(input)

			expect(result).toEqual({
				'$.users[0].name': 'Alice',
				'$.users[0].scores[0]': 85,
				'$.users[0].scores[1]': 90,
				'$.users[1].name': 'Bob',
				'$.users[1].scores[0]': 92,
				'$.users[1].scores[1]': 88
			})
		})

		it('handles special characters in keys', () => {
			const input = {
				'weird.key': 'value',
				'spaced name': 'another value',
				'key-with-dash': 'dash value'
			}
			const result = flattenJson(input)

			expect(result).toEqual({
				"$['weird.key']": 'value',
				"$['spaced name']": 'another value',
				"$['key-with-dash']": 'dash value'
			})
		})
	})

	describe('Date handling', () => {
		it('preserves Date objects as Date objects', () => {
			const now = new Date('2024-01-15T10:30:00Z')
			const input = {
				createdAt: now,
				updatedAt: new Date('2024-01-16T14:45:00Z')
			}
			const result = flattenJson(input)

			expect(result['$.createdAt']).toBeInstanceOf(Date)
			expect(result['$.updatedAt']).toBeInstanceOf(Date)
			expect(result['$.createdAt']).toEqual(now)
		})

		it('handles Date objects in arrays', () => {
			const dates = [
				new Date('2024-01-01T00:00:00Z'),
				new Date('2024-01-02T00:00:00Z')
			]
			const input = { dates }
			const result = flattenJson(input)

			expect(result['$.dates[0]']).toBeInstanceOf(Date)
			expect(result['$.dates[1]']).toBeInstanceOf(Date)
			expect(result['$.dates[0]']).toEqual(dates[0])
			expect(result['$.dates[1]']).toEqual(dates[1])
		})

		it('handles Date objects in nested objects', () => {
			const input = {
				metadata: {
					created: new Date('2024-01-01T00:00:00Z'),
					modified: new Date('2024-01-15T12:00:00Z')
				}
			}
			const result = flattenJson(input)

			expect(result['$.metadata.created']).toBeInstanceOf(Date)
			expect(result['$.metadata.modified']).toBeInstanceOf(Date)
		})

		it('handles mixed Date and primitive values', () => {
			const input = {
				name: 'Event',
				startTime: new Date('2024-01-01T09:00:00Z'),
				endTime: new Date('2024-01-01T17:00:00Z'),
				attendees: 25,
				confirmed: true
			}
			const result = flattenJson(input)

			expect(result['$.name']).toBe('Event')
			expect(result['$.startTime']).toBeInstanceOf(Date)
			expect(result['$.endTime']).toBeInstanceOf(Date)
			expect(result['$.attendees']).toBe(25)
			expect(result['$.confirmed']).toBe(true)
		})

		it('handles Date objects in complex nested structures', () => {
			const input = {
				events: [
					{
						name: 'Meeting 1',
						schedule: {
							start: new Date('2024-01-01T10:00:00Z'),
							end: new Date('2024-01-01T11:00:00Z')
						}
					},
					{
						name: 'Meeting 2',
						schedule: {
							start: new Date('2024-01-02T14:00:00Z'),
							end: new Date('2024-01-02T15:00:00Z')
						}
					}
				]
			}
			const result = flattenJson(input)

			expect(result['$.events[0].name']).toBe('Meeting 1')
			expect(result['$.events[0].schedule.start']).toBeInstanceOf(Date)
			expect(result['$.events[0].schedule.end']).toBeInstanceOf(Date)
			expect(result['$.events[1].name']).toBe('Meeting 2')
			expect(result['$.events[1].schedule.start']).toBeInstanceOf(Date)
			expect(result['$.events[1].schedule.end']).toBeInstanceOf(Date)
		})
	})

	describe('edge cases', () => {
		it('handles empty objects', () => {
			const result = flattenJson({})
			expect(result).toEqual({})
		})

		it('handles empty arrays', () => {
			const result = flattenJson({ items: [] })
			expect(result).toEqual({})
		})

		it('handles null values', () => {
			const input = { value: null }
			const result = flattenJson(input)
			expect(result).toEqual({ '$.value': null })
		})

		it('handles undefined values', () => {
			const input = { value: undefined }
			const result = flattenJson(input)
			expect(result).toEqual({ '$.value': undefined })
		})

		it('handles functions by converting to string', () => {
			const input = { fn: () => 'test' }
			const result = flattenJson(input)
			// String() can produce different quote styles, so check it contains the function content
			expect(result['$.fn']).toContain('test')
			expect(typeof result['$.fn']).toBe('string')
		})

		it('handles symbols by converting to string', () => {
			const input = { sym: Symbol('test') }
			const result = flattenJson(input)
			expect(result['$.sym']).toBe('Symbol(test)')
		})
	})

	describe('against logs.json', () => {
		it('flattens each entry in logs.json.logs and matches snapshot', () => {
			const logsUrl = new URL('../../data/logs.json', import.meta.url)
			const raw = readFileSync(logsUrl, 'utf-8')
			const parsed = JSON.parse(raw) as { logs?: unknown[] }

			expect(Array.isArray(parsed.logs)).toBe(true)
			const flattened = (parsed.logs ?? []).map((entry) => flattenJson(entry))

			expect(flattened).toMatchSnapshot()
		})
	})
})
