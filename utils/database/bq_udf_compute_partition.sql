CREATE TEMP FUNCTION compute_partition(
  sequence STRING,
  patternList ARRAY<STRUCT<pattern STRING, starting_positions ARRAY<INT64>>>)
RETURNS ARRAY<STRING>
LANGUAGE js
AS """
  if (patternList.length === 1 && patternList[0].starting_positions.length === 0) {
    return [sequence];
  }

  const entries = [];
  for (const pp of patternList) {
    for (const pos of pp.starting_positions) {
      entries.push({ pos, pattern: pp.pattern });
    }
  }
  entries.sort((a, b) => a.pos - b.pos);
  return entries.map(e => e.pattern);
""";