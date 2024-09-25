CREATE TEMP FUNCTION
  compute_partition(sequence STRING,
    patternList ARRAY<STRUCT< pattern STRING,
    starting_positions ARRAY<INT64>>>)
  RETURNS ARRAY<STRING>
  LANGUAGE js AS """
    sequenceLength = sequence.length
    const partitionMatrix = Array.from({ length: sequenceLength }, () => []);
    for (const patternPosition of patternList) {
      const pattern = patternPosition.pattern;
      for (const position of patternPosition.starting_positions) {
        partitionMatrix[position].push(pattern);
      }
    }

    if (patternList.length === 1 && patternList[0].starting_positions.length == 0) {
      partitionMatrix[0].push(sequence);
    }
    return partitionMatrix.flat()
""";