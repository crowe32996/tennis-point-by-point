"use client"

import { getFlagUrl, formatNumber, formatPercent } from "@/lib/utils"
import { Trophy, TrendingUp, TrendingDown } from "lucide-react"

interface Player {
  player: string
  tour?: string
  country?: string
  [key: string]: string | number | undefined
}

interface PlayerRankingProps {
  title: string
  icon?: "trophy" | "up" | "down"
  players: Player[]
  valueKey: string
  valueLabel: string
  formatValue?: (value: number) => string
  isPercent?: boolean
  showRank?: boolean
  highlightColor?: "green" | "red" | "blue"
}

export function PlayerRanking({
  title,
  icon = "trophy",
  players,
  valueKey,
  valueLabel,
  formatValue,
  isPercent = false,
  showRank = true,
  highlightColor = "green"
}: PlayerRankingProps) {
  const Icon = icon === "trophy" ? Trophy : icon === "up" ? TrendingUp : TrendingDown

  const colorClasses = {
    green: "text-emerald-600 dark:text-emerald-400",
    red: "text-red-600 dark:text-red-400",
    blue: "text-blue-600 dark:text-blue-400"
  }

  const bgColorClasses = {
    green: "bg-emerald-50 dark:bg-emerald-950/30",
    red: "bg-red-50 dark:bg-red-950/30",
    blue: "bg-blue-50 dark:bg-blue-950/30"
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <Icon className={`w-5 h-5 ${colorClasses[highlightColor]}`} />
        <h4 className="font-semibold text-zinc-900 dark:text-zinc-100">{title}</h4>
      </div>

      <div className="space-y-2">
        {players.map((player, index) => {
          const value = player[valueKey] as number
          const formattedValue = formatValue
            ? formatValue(value)
            : isPercent
            ? formatPercent(value)
            : formatNumber(value)

          return (
            <div
              key={player.player}
              className={`flex items-center gap-3 p-3 rounded-lg transition-colors
                ${index === 0 ? bgColorClasses[highlightColor] : "hover:bg-zinc-50 dark:hover:bg-zinc-800/50"}`}
            >
              {showRank && (
                <span className={`w-6 text-center font-bold ${index < 3 ? colorClasses[highlightColor] : "text-zinc-400"}`}>
                  {index + 1}
                </span>
              )}

              <img
                src={getFlagUrl(player.country || "USA")}
                alt={player.country || ""}
                className="w-6 h-4 object-cover rounded-sm shadow-sm"
                onError={(e) => { e.currentTarget.style.display = 'none' }}
              />

              <span className="flex-1 font-medium text-zinc-900 dark:text-zinc-100 truncate">
                {player.player}
              </span>

              <div className="text-right">
                <span className={`font-semibold ${colorClasses[highlightColor]}`}>
                  {formattedValue}
                </span>
                <span className="text-xs text-zinc-500 ml-1 hidden sm:inline">
                  {valueLabel}
                </span>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
