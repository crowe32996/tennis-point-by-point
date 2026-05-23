"use client"

import { useState } from "react"
import { getFlagUrl, formatNumber } from "@/lib/utils"
import { TrendingUp, TrendingDown, ChevronLeft, ChevronRight } from "lucide-react"

interface Player {
  player: string
  country?: string
  headshot_url?: string
  [key: string]: string | number | undefined
}

interface PlayerLeaderboardProps {
  title: string
  topPlayers: Player[]
  bottomPlayers: Player[]
  valueKey: string
  valueLabel: string
  formatValue?: (value: number) => string
  maxValue?: number
}

// Headshot URLs for top players - keys normalized to lowercase for matching
const HEADSHOTS_RAW: Record<string, string> = {
  // ATP Players
  "n. djokovic": "https://r2.thesportsdb.com/images/media/player/cutout/h6od2i1748970226.png",
  "c. alcaraz": "https://r2.thesportsdb.com/images/media/player/cutout/ybvci41748969943.png",
  "a. zverev": "https://r2.thesportsdb.com/images/media/player/cutout/c8fy2l1748969907.png",
  "j. sinner": "https://r2.thesportsdb.com/images/media/player/cutout/3ava8z1748970589.png",
  "d. medvedev": "https://r2.thesportsdb.com/images/media/player/cutout/bk4gah1675267907.png",
  "s. tsitsipas": "https://r2.thesportsdb.com/images/media/player/cutout/tb56ow1675267667.png",
  "a. rublev": "https://r2.thesportsdb.com/images/media/player/cutout/jp6sam1675268673.png",
  "r. nadal": "https://r2.thesportsdb.com/images/media/player/cutout/nzkxsm1709318598.png",
  "h. hurkacz": "https://r2.thesportsdb.com/images/media/player/cutout/b7okgh1674812969.png",
  "t. fritz": "https://r2.thesportsdb.com/images/media/player/cutout/k6f2ic1675267263.png",
  "c. ruud": "https://r2.thesportsdb.com/images/media/player/cutout/36r02q1748970858.png",
  "h. rune": "https://r2.thesportsdb.com/images/media/player/cutout/yw26pu1675268826.png",
  "f. auger-aliassime": "https://r2.thesportsdb.com/images/media/player/cutout/h0yqcx1675268125.png",
  "a. de minaur": "https://r2.thesportsdb.com/images/media/player/cutout/ec7ymd1675267811.png",
  "g. dimitrov": "https://r2.thesportsdb.com/images/media/player/cutout/a7nsr21675268026.png",
  "d. thiem": "https://r2.thesportsdb.com/images/media/player/cutout/einc3u1675267575.png",
  "f. tiafoe": "https://r2.thesportsdb.com/images/media/player/cutout/lmppui1675267619.png",
  "a. murray": "https://r2.thesportsdb.com/images/media/player/cutout/cgg5861675267346.png",
  "n. kyrgios": "https://r2.thesportsdb.com/images/media/player/cutout/11po941675267958.png",
  "d. shapovalov": "https://r2.thesportsdb.com/images/media/player/cutout/un30ms1675358146.png",
  "m. berrettini": "https://r2.thesportsdb.com/images/media/player/cutout/y9ut4z1675268167.png",
  // WTA Players
  "i. swiatek": "https://r2.thesportsdb.com/images/media/player/cutout/k10lhh1748965462.png",
  "a. sabalenka": "https://r2.thesportsdb.com/images/media/player/cutout/4knki51748965857.png",
  "c. gauff": "https://r2.thesportsdb.com/images/media/player/cutout/k0rslz1748966077.png",
  "e. rybakina": "https://r2.thesportsdb.com/images/media/player/cutout/3ibxe01749009271.png",
  "j. pegula": "https://r2.thesportsdb.com/images/media/player/cutout/sq7ahe1749014407.png",
  "o. jabeur": "https://r2.thesportsdb.com/images/media/player/cutout/htue0g1675262452.png",
  "m. sakkari": "https://r2.thesportsdb.com/images/media/player/cutout/g7tomm1675266335.png",
  "m. keys": "https://r2.thesportsdb.com/images/media/player/cutout/snjegu1748966783.png",
  "p. kvitova": "https://r2.thesportsdb.com/images/media/player/cutout/r6bt8q1723379145.png",
  "b. haddad maia": "https://r2.thesportsdb.com/images/media/player/cutout/honejy1675265171.png",
}

// Helper to get headshot with case-insensitive matching
function getHeadshot(playerName: string): string | undefined {
  return HEADSHOTS_RAW[playerName.toLowerCase()]
}

export function PlayerLeaderboard({
  title,
  topPlayers,
  bottomPlayers,
  valueKey,
  valueLabel,
  formatValue = (v) => formatNumber(v, 0),
  maxValue,
}: PlayerLeaderboardProps) {
  const [showBest, setShowBest] = useState(true)

  const players = showBest ? topPlayers : bottomPlayers
  const allValues = [...topPlayers, ...bottomPlayers].map(p => Math.abs(p[valueKey] as number))
  const max = maxValue || Math.max(...allValues)

  return (
    <div className="space-y-4">
      {/* Header with toggle */}
      <div className="flex items-center justify-between">
        <h3 className="text-xl font-bold text-zinc-100">{title}</h3>

        <div className="flex items-center gap-2 bg-zinc-800 rounded-full p-1">
          <button
            onClick={() => setShowBest(true)}
            className={`flex items-center gap-1.5 px-4 py-2 rounded-full text-sm font-medium transition-all ${
              showBest
                ? "bg-emerald-500 text-white shadow-lg"
                : "text-zinc-400 hover:text-white"
            }`}
          >
            <TrendingUp className="w-4 h-4" />
            Best
          </button>
          <button
            onClick={() => setShowBest(false)}
            className={`flex items-center gap-1.5 px-4 py-2 rounded-full text-sm font-medium transition-all ${
              !showBest
                ? "bg-red-500 text-white shadow-lg"
                : "text-zinc-400 hover:text-white"
            }`}
          >
            <TrendingDown className="w-4 h-4" />
            Worst
          </button>
        </div>
      </div>

      {/* Leaderboard */}
      <div className="space-y-3">
        {players.slice(0, 10).map((player, index) => {
          const value = player[valueKey] as number
          const percentage = (Math.abs(value) / max) * 100
          const headshot = getHeadshot(player.player)
          const isPositive = value >= 0

          return (
            <div
              key={player.player}
              className="group relative flex items-center gap-4 p-3 rounded-xl bg-zinc-800/50 hover:bg-zinc-800 transition-all"
            >
              {/* Rank */}
              <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${
                index === 0
                  ? (showBest ? "bg-yellow-500 text-black" : "bg-red-500 text-white")
                  : index === 1
                  ? "bg-zinc-400 text-black"
                  : index === 2
                  ? "bg-amber-700 text-white"
                  : "bg-zinc-700 text-zinc-300"
              }`}>
                {index + 1}
              </div>

              {/* Headshot */}
              <div className="relative w-12 h-12 rounded-full overflow-hidden bg-zinc-700 flex-shrink-0">
                {headshot ? (
                  <img
                    src={headshot}
                    alt={player.player}
                    className="w-full h-full object-cover object-top"
                    onError={(e) => { e.currentTarget.style.display = 'none' }}
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-zinc-500 text-lg font-bold">
                    {player.player.split(' ').pop()?.[0]}
                  </div>
                )}
                {/* Flag overlay */}
                <img
                  src={getFlagUrl(player.country || "USA")}
                  alt=""
                  className="absolute bottom-0 right-0 w-5 h-3 object-cover rounded-sm border border-zinc-800"
                  onError={(e) => { e.currentTarget.style.display = 'none' }}
                />
              </div>

              {/* Name & Bar */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between mb-1">
                  <span className="font-semibold text-white truncate">
                    {player.player}
                  </span>
                  <span className={`font-bold text-lg ${
                    showBest ? "text-emerald-400" : "text-red-400"
                  }`}>
                    {isPositive ? "+" : ""}{formatValue(value)}
                  </span>
                </div>

                {/* Progress bar */}
                <div className="h-2 bg-zinc-700 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      showBest
                        ? "bg-gradient-to-r from-emerald-600 to-emerald-400"
                        : "bg-gradient-to-r from-red-600 to-red-400"
                    }`}
                    style={{ width: `${percentage}%` }}
                  />
                </div>
              </div>
            </div>
          )
        })}
      </div>

      {/* Stats summary */}
      <div className="flex items-center justify-center gap-6 pt-2 text-sm text-zinc-500">
        <span>
          {showBest ? "Top" : "Bottom"} performers by {valueLabel}
        </span>
      </div>
    </div>
  )
}

// Butterfly chart showing both extremes
export function ButterflyChart({
  title,
  topPlayers,
  bottomPlayers,
  valueKey,
  formatValue = (v) => formatNumber(v, 0),
}: {
  title: string
  topPlayers: Player[]
  bottomPlayers: Player[]
  valueKey: string
  formatValue?: (value: number) => string
}) {
  const combined = [
    ...topPlayers.slice(0, 5).map(p => ({ ...p, side: 'top' as const })),
    ...bottomPlayers.slice(0, 5).map(p => ({ ...p, side: 'bottom' as const })),
  ]

  const maxValue = Math.max(
    ...topPlayers.map(p => Math.abs(p[valueKey] as number)),
    ...bottomPlayers.map(p => Math.abs(p[valueKey] as number))
  )

  return (
    <div className="space-y-4">
      <h3 className="text-xl font-bold text-zinc-100 text-center">{title}</h3>

      <div className="space-y-2">
        {/* Best players - bars extend right */}
        <div className="text-xs text-emerald-400 font-medium mb-2">CLUTCH PERFORMERS</div>
        {topPlayers.slice(0, 5).map((player, index) => {
          const value = player[valueKey] as number
          const percentage = (Math.abs(value) / maxValue) * 100
          const headshot = getHeadshot(player.player)

          return (
            <div key={player.player} className="flex items-center gap-3">
              <div className="w-24 text-right text-sm text-zinc-300 truncate">
                {player.player}
              </div>
              <div className="flex-1 h-8 bg-zinc-800 rounded relative overflow-hidden">
                <div
                  className="absolute left-0 top-0 h-full bg-gradient-to-r from-emerald-600 to-emerald-400 rounded transition-all duration-700"
                  style={{ width: `${percentage}%` }}
                />
                <div className="absolute right-2 top-1/2 -translate-y-1/2 text-xs font-bold text-white">
                  +{formatValue(value)}
                </div>
              </div>
            </div>
          )
        })}

        {/* Divider */}
        <div className="flex items-center gap-4 py-3">
          <div className="flex-1 h-px bg-zinc-700" />
          <span className="text-xs text-zinc-500">vs</span>
          <div className="flex-1 h-px bg-zinc-700" />
        </div>

        {/* Worst players - bars extend right but red */}
        <div className="text-xs text-red-400 font-medium mb-2">STRUGGLING UNDER PRESSURE</div>
        {bottomPlayers.slice(0, 5).map((player, index) => {
          const value = player[valueKey] as number
          const percentage = (Math.abs(value) / maxValue) * 100

          return (
            <div key={player.player} className="flex items-center gap-3">
              <div className="w-24 text-right text-sm text-zinc-300 truncate">
                {player.player}
              </div>
              <div className="flex-1 h-8 bg-zinc-800 rounded relative overflow-hidden">
                <div
                  className="absolute left-0 top-0 h-full bg-gradient-to-r from-red-600 to-red-400 rounded transition-all duration-700"
                  style={{ width: `${percentage}%` }}
                />
                <div className="absolute right-2 top-1/2 -translate-y-1/2 text-xs font-bold text-white">
                  {formatValue(value)}
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
