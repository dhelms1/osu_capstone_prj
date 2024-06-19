# Validation

The air temperature validation process on WSU-DAS consists in two steps:
- check the **absolute value**, where the temperature is validated against a range of max/min values depending on the season
 (100F is plausible in summer, but not in winter) 
- check **relative values**, that is the variation occurred compared to previous/following readings, identifying 
 anomalous spikes/dips of one or few consecutive records

#### Flags

- **missing**: temperature missing for the current timestamp
- **too hot**: temperature exceeds an absolute max value, regardless of the date or time
- **too cold**: temperature is lower than a min value (the threshold adjusts in summer) 
- **too warm at night**: temperature exceeds a certain value between 22:00 and 10:00
- **too hot early**: temperature exceeds a certain value in January-April
- **too hot late**: temperature exceeds a certain value from mid-September to end of year
- **no change**: no noticeable temperature change for 2 consecutive hours
- **too small variation**: no significant variation in 2 hours time during the day in spring and summer
- **spike/dip**: identifies dubious variations greater than a certain value (currently ~5˚F) between contiguous readings;
 e.g. one or more (up to 5) observations jumping up/down compared to the previous or the following records

## Validation parameters

Parameters to define for a more accurate, station-specific validation:
- overall range: absolute max/min values (over the last 10 years?)
- range: daily (weekly/monthly?) max/min values during the day and night
- positive variation: max allowed increase between two consecutive observations
- negative variation: max allowed decrease between two consecutive observations

