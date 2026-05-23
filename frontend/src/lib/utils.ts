import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`
}

export function formatNumber(value: number, decimals: number = 2): string {
  return value.toFixed(decimals)
}

// IOC country codes to ISO 3166-1 alpha-2 (for flag images)
const IOC_TO_ISO2: Record<string, string> = {
  // Common tennis nations
  "USA": "US", "GBR": "GB", "GER": "DE", "SUI": "CH", "NED": "NL",
  "ESP": "ES", "FRA": "FR", "ITA": "IT", "AUS": "AU", "ARG": "AR",
  "RUS": "RU", "CZE": "CZ", "ROU": "RO", "CRO": "HR", "SRB": "RS",
  "POL": "PL", "UKR": "UA", "BEL": "BE", "AUT": "AT", "SLO": "SI",
  "SVK": "SK", "HUN": "HU", "BUL": "BG", "GRE": "GR", "POR": "PT",
  "DEN": "DK", "SWE": "SE", "NOR": "NO", "FIN": "FI", "IRL": "IE",
  "CAN": "CA", "BRA": "BR", "CHI": "CL", "COL": "CO", "MEX": "MX",
  "PER": "PE", "ECU": "EC", "URU": "UY", "VEN": "VE", "BOL": "BO",
  "JPN": "JP", "CHN": "CN", "KOR": "KR", "TPE": "TW", "THA": "TH",
  "IND": "IN", "KAZ": "KZ", "UZB": "UZ", "GEO": "GE", "ARM": "AM",
  "RSA": "ZA", "TUN": "TN", "EGY": "EG", "MAR": "MA", "ZIM": "ZW",
  "NZL": "NZ", "BLR": "BY", "LAT": "LV", "LTU": "LT", "EST": "EE",
  "MDA": "MD", "BIH": "BA", "MNE": "ME", "MKD": "MK", "ALB": "AL",
  "CYP": "CY", "LUX": "LU", "MON": "MC", "ISR": "IL", "LIB": "LB",
  "PUR": "PR", "ESA": "SV", "INA": "ID", "IRI": "IR", "PAK": "PK",
  "PHI": "PH", "MAS": "MY", "SIN": "SG", "VIE": "VN", "HKG": "HK"
}

export function getCountryFlag(iocCode: string): string {
  const iso2 = IOC_TO_ISO2[iocCode] || iocCode.slice(0, 2).toUpperCase()
  const OFFSET = 127397
  return [...iso2].map(c => String.fromCodePoint(c.charCodeAt(0) + OFFSET)).join('')
}

export function getFlagUrl(iocCode: string): string {
  const iso2 = IOC_TO_ISO2[iocCode] || iocCode.slice(0, 2).toUpperCase()
  return `https://flagcdn.com/w20/${iso2.toLowerCase()}.png`
}
