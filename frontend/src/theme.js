import { useEffect, useState } from "react";

// Reference data-viz palette, stepped separately for light and dark surfaces.
const LIGHT = {
  series1: "#2a78d6", series2: "#eb6834", grid: "#e1e0d9", axis: "#c3c2b7",
  muted: "#898781", text: "#0b0b0b", textSecondary: "#52514e", surface: "#fcfcfb",
  seq: ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"],
};
const DARK = {
  series1: "#3987e5", series2: "#d95926", grid: "#2c2c2a", axis: "#383835",
  muted: "#898781", text: "#f0efec", textSecondary: "#c3c2b7", surface: "#1a1a19",
  seq: ["#0d366b", "#104281", "#184f95", "#256abf", "#3987e5", "#6da7ec"],
};

export function useChartTheme() {
  const query = "(prefers-color-scheme: dark)";
  const [dark, setDark] = useState(() => window.matchMedia(query).matches);
  useEffect(() => {
    const mq = window.matchMedia(query);
    const onChange = (e) => setDark(e.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);
  return dark ? DARK : LIGHT;
}

export const MODEL_COLOR_KEY = { logistic_regression: "series1", random_forest: "series2" };
