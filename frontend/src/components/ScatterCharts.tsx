"use client"

import { useEffect, useState, useRef } from "react"
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Cell,
  ZAxis
} from "recharts"
import { getFlagUrl, formatNumber, formatPercent } from "@/lib/utils"

interface PlayerScatterData {
  player: string
  country?: string
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

interface ScatterChartsProps {
  data: PlayerScatterData[]
  activeChart: "consistency-clutch" | "pressure-performance"
}

// Tennis ball gradient colors
const getTennisBallColor = (value: number, min: number, max: number): string => {
  const ratio = (value - min) / (max - min)
  // Interpolate from pale yellow (#ffffcc) to bright yellow-green (#ccff00)
  const r = Math.round(255 - ratio * 51)  // 255 to 204
  const g = 255
  const b = Math.round(204 - ratio * 204) // 204 to 0
  return `rgb(${r}, ${g}, ${b})`
}

// Clutch delta color gradient: red → orange → yellow → lime → green
const getClutchDeltaColor = (delta: number, minDelta: number, maxDelta: number): string => {
  // Normalize delta to 0-1 range
  const range = Math.max(Math.abs(minDelta), Math.abs(maxDelta))
  const normalized = (delta + range) / (2 * range) // 0 = most negative, 0.5 = neutral, 1 = most positive

  // Color stops: red (0) → orange (0.25) → yellow (0.5) → lime (0.75) → green (1)
  if (normalized <= 0.25) {
    // Red to orange
    const t = normalized / 0.25
    return `rgb(239, ${Math.round(68 + t * 85)}, 68)` // #ef4444 to #f97316
  } else if (normalized <= 0.5) {
    // Orange to yellow
    const t = (normalized - 0.25) / 0.25
    return `rgb(${Math.round(249 - t * 15)}, ${Math.round(115 + t * 90)}, ${Math.round(22 + t * 38)})` // #f97316 to #eab308
  } else if (normalized <= 0.75) {
    // Yellow to lime
    const t = (normalized - 0.5) / 0.25
    return `rgb(${Math.round(234 - t * 82)}, ${Math.round(179 + t * 17)}, ${Math.round(8 + t * 14)})` // #eab308 to #84cc16
  } else {
    // Lime to green
    const t = (normalized - 0.75) / 0.25
    return `rgb(${Math.round(132 - t * 88)}, ${Math.round(204 - t * 17)}, ${Math.round(22 + t * 72)})` // #84cc16 to #22c55e
  }
}

const CustomTooltip = ({ active, payload, chartType }: any) => {
  if (!active || !payload || !payload[0]) return null

  const data = payload[0].payload as PlayerScatterData

  return (
    <div className="bg-zinc-900 border border-zinc-700 rounded-lg p-3 shadow-xl">
      <div className="flex items-center gap-2 mb-2">
        <img
          src={getFlagUrl(data.country || "USA")}
          alt=""
          className="w-5 h-3 object-cover rounded-sm"
          onError={(e) => { e.currentTarget.style.display = 'none' }}
        />
        <span className="font-bold text-white">{data.player}</span>
      </div>
      <div className="space-y-1 text-sm text-zinc-300">
        <div>Total Points: <span className="text-white font-medium">{formatNumber(data.total_points, 0)}</span></div>
        {chartType === "consistency-clutch" ? (
          <>
            <div>Consistency: <span className="text-white font-medium">{formatPercent(data.consistency_percentile)}</span></div>
            <div>Clutch: <span className="text-white font-medium">{formatPercent(data.clutch_percentile)}</span></div>
            <div>EPA: <span className={`font-bold ${data.clutch_score >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
              {data.clutch_score >= 0 ? '+' : ''}{formatNumber(data.clutch_score, 0)}
            </span></div>
          </>
        ) : (
          <>
            <div>Overall Win %: <span className="text-white font-medium">{formatPercent(data.overall_win_rate)}</span></div>
            <div>HP Win %: <span className="text-white font-medium">{formatPercent(data.hp_win_rate)}</span></div>
            <div>Clutch Delta: <span className={`font-bold ${data.clutch_delta >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
              {data.clutch_delta >= 0 ? '+' : ''}{formatPercent(data.clutch_delta)}
            </span></div>
          </>
        )}
      </div>
    </div>
  )
}

export function ConsistencyClutchChart({ data }: { data: PlayerScatterData[] }) {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted || !data.length) return null

  const minPoints = Math.min(...data.map(d => d.total_points))
  const maxPoints = Math.max(...data.map(d => d.total_points))

  return (
    <div className="w-full h-[550px] bg-zinc-900/50 rounded-xl p-4 relative">
      <h3 className="text-lg font-bold text-zinc-100 mb-2 text-center">
        Player Consistency vs Clutchness
      </h3>
      <p className="text-xs text-zinc-500 text-center mb-2">
        Dashed lines = 50th percentile (average)
      </p>

      {/* Quadrant labels */}
      <div className="absolute top-20 left-16 text-xs text-zinc-600 opacity-60">
        Inconsistent<br/>Clutch
      </div>
      <div className="absolute top-20 right-12 text-xs text-zinc-600 opacity-60 text-right">
        Consistent<br/>Clutch ⭐
      </div>
      <div className="absolute bottom-24 left-16 text-xs text-zinc-600 opacity-60">
        Inconsistent<br/>Choker
      </div>
      <div className="absolute bottom-24 right-12 text-xs text-zinc-600 opacity-60 text-right">
        Consistent<br/>Choker
      </div>

      <ResponsiveContainer width="100%" height="80%">
        <ScatterChart margin={{ top: 20, right: 40, bottom: 20, left: 40 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#3f3f46" />
          <XAxis
            type="number"
            dataKey="consistency_percentile"
            name="Consistency"
            domain={[0, 1]}
            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
            stroke="#71717a"
            label={{ value: 'Consistency Percentile →', position: 'bottom', fill: '#71717a', offset: 0 }}
          />
          <YAxis
            type="number"
            dataKey="clutch_percentile"
            name="Clutch"
            domain={[0, 1]}
            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
            stroke="#71717a"
            label={{ value: '← Clutch Percentile', angle: -90, position: 'left', fill: '#71717a' }}
          />
          <ZAxis
            type="number"
            dataKey="total_points"
            range={[100, 800]}
            name="Total Points"
          />
          <Tooltip content={<CustomTooltip chartType="consistency-clutch" />} />
          <ReferenceLine x={0.5} stroke="#52525b" strokeDasharray="4 4" />
          <ReferenceLine y={0.5} stroke="#52525b" strokeDasharray="4 4" />
          <Scatter name="Players" data={data}>
            {data.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={getTennisBallColor(entry.total_points, minPoints, maxPoints)}
                stroke="#000"
                strokeWidth={1}
              />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div className="flex justify-center items-center gap-6 mt-2 text-xs text-zinc-400">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full" style={{ background: '#ffffcc' }} />
          <span>Fewer Points</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full" style={{ background: '#ccff00' }} />
          <span>More Points</span>
        </div>
        <span className="text-zinc-500">|</span>
        <span>Bubble size = Total Points</span>
      </div>
    </div>
  )
}

export function PressurePerformanceChart({ data }: { data: PlayerScatterData[] }) {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted || !data.length) return null

  // Get domain ranges
  const xMin = Math.min(...data.map(d => d.overall_win_rate)) - 0.02
  const xMax = Math.max(...data.map(d => d.overall_win_rate)) + 0.02
  const yMin = Math.min(...data.map(d => d.hp_win_rate)) - 0.02
  const yMax = Math.max(...data.map(d => d.hp_win_rate)) + 0.02

  // Get clutch delta range for color gradient
  const minDelta = Math.min(...data.map(d => d.clutch_delta))
  const maxDelta = Math.max(...data.map(d => d.clutch_delta))

  return (
    <div className="w-full h-[500px] bg-zinc-900/50 rounded-xl p-4">
      <h3 className="text-lg font-bold text-zinc-100 mb-4 text-center">
        Overall Win % vs High Pressure Win %
      </h3>
      <ResponsiveContainer width="100%" height="90%">
        <ScatterChart margin={{ top: 20, right: 30, bottom: 20, left: 30 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#3f3f46" />
          <XAxis
            type="number"
            dataKey="overall_win_rate"
            name="Overall Win %"
            domain={[xMin, xMax]}
            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
            stroke="#71717a"
            label={{ value: 'Overall Win %', position: 'bottom', fill: '#71717a', offset: 0 }}
          />
          <YAxis
            type="number"
            dataKey="hp_win_rate"
            name="HP Win %"
            domain={[yMin, yMax]}
            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
            stroke="#71717a"
            label={{ value: 'High Pressure Win %', angle: -90, position: 'left', fill: '#71717a' }}
          />
          <ZAxis
            type="number"
            dataKey="hp_points"
            range={[100, 600]}
            name="HP Points"
          />
          <Tooltip content={<CustomTooltip chartType="pressure-performance" />} />
          {/* Diagonal reference line (y = x, meaning no clutch advantage) */}
          <ReferenceLine
            segment={[{ x: xMin, y: xMin }, { x: xMax, y: xMax }]}
            stroke="#52525b"
            strokeDasharray="4 4"
          />
          <Scatter name="Players" data={data}>
            {data.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={getClutchDeltaColor(entry.clutch_delta, minDelta, maxDelta)}
                stroke="#000"
                strokeWidth={0.5}
              />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div className="flex justify-center items-center gap-6 mt-2 text-xs text-zinc-400">
        <div className="flex items-center gap-2">
          <div className="w-24 h-3 rounded-full" style={{
            background: 'linear-gradient(to right, #ef4444, #f97316, #eab308, #84cc16, #22c55e)'
          }} />
          <span>Clutch Delta (red = chokes, green = thrives)</span>
        </div>
        <span className="text-zinc-500">|</span>
        <span>Above diagonal = clutch player</span>
      </div>
    </div>
  )
}
