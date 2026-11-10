// Paletas por nivel: misma estética de garaje con neón, distinto color en cada nivel.
// Los colores de estado (éxito, choque, sensores) no cambian para que se lean igual en todos.
export const THEMES = {
  N1: {   // lavanda y rosa
    background: 0x2a1a52, fog: 0x3b2a6e, hemiSky: 0xe0d2ff, hemiGround: 0x5a4680, keyLight: 0xfff0fa,
    floor: "#a99ecf", floorJoint: "rgba(90,70,140,0.12)", mirror: 0x6a6190,
    wall: 0xbfaee6, wallBase: 0x8f7cc4, ceiling: 0x8475b3, beam: 0x6c5d9c, pipes: [0xd0587e, 0xd8d2e8],
    lamp: 0xfff4fb, lampHousing: 0xe9e4f5, pointLight: 0xffeaf6, paintEmissive: 0x3a3450,
    pillarBands: ["#f2a7c9", "#86b6f2"], pillarCap: 0xf2a7c9,
    sky: ["#3b2a78", "#8a5bc4", "#f193b8", "#ffc29b"], buildings: ["#54408a", "#2a1d55"],
    windows: ["#ffd98a", "#f9a8d4"], windowFrame: "#efe7ff",
    neon: { stroke: "#ff7ac8", fill: "#ffd3ee", shadow: "#ff4fb4" },
    ego: 0x7c3aed, egoGlow: 0xa855f7, egoArrow: 0xf5d0fe, headlights: 0xf4f0ff,
    trail: [0x5b2a8c, 0xf0abfc], label: "#efe7ff",
    spark: ["#f5a3c7", "#86efac"], sparkAxis: "rgba(221,208,255,0.3)",
    parked: [0xf5f5f7, 0xd81e2c, 0x1f4fd8, 0xf5c518, 0xc8ccd6, 0x18161f],
  },
  N2: {   // menta y coral
    background: 0x0d3b3e, fog: 0x175458, hemiSky: 0xd4fff6, hemiGround: 0x3c7d77, keyLight: 0xf2fffb,
    floor: "#79aea6", floorJoint: "rgba(20,80,72,0.14)", mirror: 0x4b7f79,
    wall: 0x8fd0c4, wallBase: 0x4a9e92, ceiling: 0x6fb3a9, beam: 0x4f938a, pipes: [0xff8a5c, 0xd3ece8],
    lamp: 0xf4fffc, lampHousing: 0xe4f5f2, pointLight: 0xe6fff8, paintEmissive: 0x2e4a47,
    pillarBands: ["#ffa98a", "#7fdccc"], pillarCap: 0xffa98a,
    sky: ["#0f3d54", "#2b8b99", "#ffab85", "#ffe0a0"], buildings: ["#2b6a72", "#0f3340"],
    windows: ["#fff2a8", "#ffc2a3"], windowFrame: "#e6fffa",
    neon: { stroke: "#ff8a65", fill: "#ffe1d6", shadow: "#ff5a36" },
    ego: 0xff4f6d, egoGlow: 0xff6b85, egoArrow: 0xffd1da, headlights: 0xf0fffb,
    trail: [0x0f766e, 0x99f6e4], label: "#e6fffa",
    spark: ["#ffa98a", "#86efac"], sparkAxis: "rgba(204,251,241,0.3)",
    parked: [0xf5f5f7, 0xff7a1a, 0x0ea5e9, 0x84cc16, 0xc8ccd6, 0x18161f],
  },
};

export const themeFor = (level) => ({ id: THEMES[level] ? level : "N1", ...(THEMES[level] ?? THEMES.N1) });
