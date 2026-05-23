"use client"

import { useState, useEffect } from "react"
import { PlayerLeaderboard, ButterflyChart } from "@/components/PlayerLeaderboard"
import { ConsistencyClutchChart, PressurePerformanceChart } from "@/components/ScatterCharts"
import { api, FilterOptions, PlayerClutch, PlayerConsistency, PlayerHighPressure, PlayerScatterData } from "@/lib/api"
import { formatNumber, formatPercent } from "@/lib/utils"
import { Trophy, Target, Zap, Loader2 } from "lucide-react"

export default function Home() {
  const [filters, setFilters] = useState<FilterOptions | null>(null)
  const [selectedTour, setSelectedTour] = useState("ATP")
  const [selectedYears, setSelectedYears] = useState<number[]>([])
  const [minPoints, setMinPoints] = useState(400)

  const [clutchData, setClutchData] = useState<{ top: PlayerClutch[], bottom: PlayerClutch[] } | null>(null)
  const [consistencyData, setConsistencyData] = useState<{ top: PlayerConsistency[], bottom: PlayerConsistency[] } | null>(null)
  const [pressureData, setPressureData] = useState<{ best: PlayerHighPressure[], worst: PlayerHighPressure[] } | null>(null)
  const [scatterData, setScatterData] = useState<PlayerScatterData[] | null>(null)

  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState<"clutch" | "consistency" | "pressure">("clutch")

  // Load filter options
  useEffect(() => {
    api.getFilters().then(data => {
      setFilters(data)
      setSelectedYears(data.years)
      setMinPoints(data.defaults.min_points_per_year["ATP"] * data.years.length)
    }).catch(console.error)
  }, [])

  // Load data when filters change
  useEffect(() => {
    if (!filters || selectedYears.length === 0) return

    setLoading(true)
    const yearsStr = selectedYears.join(",")

    Promise.all([
      api.getClutchRankings({ years: yearsStr, tour: selectedTour, min_points: minPoints }),
      api.getConsistency({ years: yearsStr, tour: selectedTour, min_points: minPoints }),
      api.getHighPressure({ years: yearsStr, tour: selectedTour, min_hp_points: Math.floor(minPoints / 4) }),
      api.getScatterData({ years: yearsStr, tour: selectedTour, min_points: minPoints })
    ]).then(([clutch, consistency, pressure, scatter]) => {
      setClutchData({ top: clutch.top_10, bottom: clutch.bottom_10 })
      setConsistencyData({ top: consistency.most_consistent, bottom: consistency.least_consistent })
      setPressureData({ best: pressure.best_under_pressure, worst: pressure.worst_under_pressure })
      setScatterData(scatter.players)
      setLoading(false)
    }).catch(err => {
      console.error(err)
      setLoading(false)
    })
  }, [filters, selectedTour, selectedYears, minPoints])

  const updateMinPoints = (tour: string) => {
    if (!filters) return
    const base = filters.defaults.min_points_per_year[tour] || 400
    setMinPoints(base * selectedYears.length)
  }

  return (
    <div className="min-h-screen bg-zinc-950">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-zinc-950/90 backdrop-blur-lg border-b border-zinc-800">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-3xl">🎾</span>
            <div>
              <h1 className="text-xl font-bold text-white">
                Tennis Pressure Analysis
              </h1>
              <p className="text-sm text-zinc-500">Grand Slam Performance • {selectedYears.length > 0 ? `${Math.min(...selectedYears)}-${Math.max(...selectedYears)}` : ''}</p>
            </div>
          </div>

          {/* Tour Toggle */}
          <div className="flex items-center gap-1 bg-zinc-800 rounded-full p-1">
            {["ATP", "WTA"].map(tour => (
              <button
                key={tour}
                onClick={() => {
                  setSelectedTour(tour)
                  updateMinPoints(tour)
                }}
                className={`px-5 py-2 rounded-full text-sm font-semibold transition-all ${
                  selectedTour === tour
                    ? "bg-emerald-500 text-white shadow-lg shadow-emerald-500/25"
                    : "text-zinc-400 hover:text-white"
                }`}
              >
                {tour}
              </button>
            ))}
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-8">
        {/* Tab Navigation */}
        <div className="flex gap-2 mb-8">
          {[
            { id: "clutch", label: "Clutch", icon: Trophy, color: "emerald" },
            { id: "consistency", label: "Consistency", icon: Target, color: "blue" },
            { id: "pressure", label: "High Pressure", icon: Zap, color: "amber" }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as typeof activeTab)}
              className={`flex items-center gap-2 px-6 py-3 rounded-xl text-sm font-semibold transition-all ${
                activeTab === tab.id
                  ? `bg-${tab.color}-500/20 text-${tab.color}-400 ring-1 ring-${tab.color}-500/50`
                  : "bg-zinc-800/50 text-zinc-400 hover:bg-zinc-800 hover:text-white"
              }`}
              style={activeTab === tab.id ? {
                backgroundColor: tab.color === 'emerald' ? 'rgba(16, 185, 129, 0.2)' :
                                tab.color === 'blue' ? 'rgba(59, 130, 246, 0.2)' :
                                'rgba(245, 158, 11, 0.2)',
                color: tab.color === 'emerald' ? '#34d399' :
                       tab.color === 'blue' ? '#60a5fa' : '#fbbf24'
              } : {}}
            >
              <tab.icon className="w-4 h-4" />
              {tab.label}
            </button>
          ))}
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-32">
            <Loader2 className="w-10 h-10 animate-spin text-emerald-500" />
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Main Leaderboard */}
            {activeTab === "clutch" && clutchData && (
              <>
                {/* Scatter Chart */}
                {scatterData && scatterData.length > 0 && (
                  <div className="lg:col-span-2">
                    <ConsistencyClutchChart data={scatterData} />
                  </div>
                )}
                {/* Leaderboard */}
                <div className="lg:col-span-2">
                  <PlayerLeaderboard
                    title="Expected Points Added (EPA)"
                    topPlayers={clutchData.top}
                    bottomPlayers={clutchData.bottom}
                    valueKey="clutch_score"
                    valueLabel="EPA"
                    formatValue={(v) => formatNumber(v, 0)}
                  />
                </div>
              </>
            )}

            {activeTab === "consistency" && consistencyData && (
              <>
                <div className="lg:col-span-2">
                  <PlayerLeaderboard
                    title="Consistency Index"
                    topPlayers={consistencyData.top}
                    bottomPlayers={consistencyData.bottom}
                    valueKey="consistency_index"
                    valueLabel="Consistency"
                    formatValue={(v) => formatNumber(v, 2)}
                  />
                </div>
              </>
            )}

            {activeTab === "pressure" && pressureData && (
              <>
                {/* Scatter Chart */}
                {scatterData && scatterData.length > 0 && (
                  <div className="lg:col-span-2">
                    <PressurePerformanceChart data={scatterData} />
                  </div>
                )}
                {/* Leaderboard */}
                <div className="lg:col-span-2">
                  <PlayerLeaderboard
                    title="High Pressure Win Rate"
                    topPlayers={pressureData.best}
                    bottomPlayers={pressureData.worst}
                    valueKey="hp_win_rate"
                    valueLabel="Win %"
                    formatValue={(v) => `${(v * 100).toFixed(1)}%`}
                  />
                </div>
              </>
            )}
          </div>
        )}

        {/* Methodology note */}
        <div className="mt-12 p-6 bg-zinc-900 rounded-2xl border border-zinc-800">
          <h4 className="font-semibold text-white mb-2">Methodology</h4>
          <p className="text-sm text-zinc-400 leading-relaxed">
            Each point in Grand Slam matches was simulated <strong className="text-zinc-200">3,000 times</strong> using Monte Carlo methods
            to estimate win probability changes. <strong className="text-zinc-200">EPA (Expected Points Added)</strong> measures
            how much a player's clutch performance exceeds or falls below expectations in high-leverage situations.
            High Pressure points are defined as the <strong className="text-zinc-200">top 25%</strong> most important points by win probability swing.
          </p>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-zinc-800 py-6 mt-8">
        <div className="max-w-6xl mx-auto px-6 text-center text-sm text-zinc-500">
          Built with Next.js + FastAPI • Data from {selectedYears.length > 0 ? selectedYears.length : '—'} seasons of Grand Slam tennis
        </div>
      </footer>
    </div>
  )
}
