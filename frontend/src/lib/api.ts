const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export interface FilterOptions {
  tournaments: string[]
  tours: string[]
  years: number[]
  player_statuses: string[]
  defaults: {
    min_points_per_year: Record<string, number>
    high_pressure_percentile: number
  }
}

export interface PlayerClutch {
  player: string
  tour: string
  country: string
  clutch_score: number
  total_points: number
  avg_wp_delta: number
}

export interface PlayerConsistency {
  player: string
  tour: string
  country: string
  avg_delta: number
  delta_std: number
  total_points: number
  consistency_index: number
}

export interface PlayerHighPressure {
  player: string
  tour: string
  country: string
  total_points: number
  hp_points: number
  overall_win_rate: number
  hp_win_rate: number
  clutch_delta: number
}

export interface UnlikelyMatch {
  match_id: string
  player1: string
  player2: string
  match_winner: string
  tournament_name: string
  year: number
  lowest_win_prob: number
}

export interface TopPoint {
  match_id: string
  player1: string
  player2: string
  tournament_name: string
  year: number
  point_number: number
  score: string
  point_winner: number
  importance: number
  swing: number
}

export interface PlayerScatterData {
  player: string
  country: string
  tour: string
  total_points: number
  hp_points: number
  clutch_score: number
  consistency_index: number
  consistency_percentile: number
  clutch_percentile: number
  overall_win_rate: number
  hp_win_rate: number
  clutch_delta: number
}

async function fetchApi<T>(endpoint: string, params?: Record<string, string | number>): Promise<T> {
  const url = new URL(`${API_BASE}${endpoint}`)
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      url.searchParams.append(key, String(value))
    })
  }

  const response = await fetch(url.toString())
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`)
  }
  return response.json()
}

export const api = {
  getFilters: () => fetchApi<FilterOptions>('/api/filters'),

  getClutchRankings: (params: {
    years: string
    tour?: string
    tournament?: string
    min_points?: number
  }) => fetchApi<{
    players: PlayerClutch[]
    top_10: PlayerClutch[]
    bottom_10: PlayerClutch[]
  }>('/api/players/clutch', params),

  getConsistency: (params: {
    years: string
    tour?: string
    tournament?: string
    min_points?: number
  }) => fetchApi<{
    players: PlayerConsistency[]
    most_consistent: PlayerConsistency[]
    least_consistent: PlayerConsistency[]
  }>('/api/players/consistency', params),

  getHighPressure: (params: {
    years: string
    tour?: string
    tournament?: string
    pressure_percentile?: number
    min_hp_points?: number
  }) => fetchApi<{
    players: PlayerHighPressure[]
    threshold: number
    best_under_pressure: PlayerHighPressure[]
    worst_under_pressure: PlayerHighPressure[]
  }>('/api/players/high-pressure', params),

  getUnlikelyMatches: (params: {
    years: string
    tour?: string
    unlikely_threshold?: number
  }) => fetchApi<{ matches: UnlikelyMatch[] }>('/api/matches/unlikely', params),

  getTopPoints: (params: {
    years: string
    tour?: string
    limit?: number
  }) => fetchApi<{ points: TopPoint[] }>('/api/points/top', params),

  getScatterData: (params: {
    years: string
    tour?: string
    tournament?: string
    min_points?: number
    pressure_percentile?: number
  }) => fetchApi<{
    players: PlayerScatterData[]
    threshold: number
  }>('/api/players/scatter', params),
}
