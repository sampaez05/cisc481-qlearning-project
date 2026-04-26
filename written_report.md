B. [15 points] On average, how long (how many runs) does it take to learn a deterministic
4x4 and 8x8 environment (you should report average and standard deviation)?
    The average for deterministic 4x4 and 8x8 environments combined is: 1209.1
    The standard deviation for the combined is: 643.44
    The average for deterministic 4x4 environments is: 574.8
    The standard deviation is: 6.794
    The average for deterministic 8x8 environments is: 1843.4
    The standard deviation is: 152.690
    Note: this was calculated using 5 4x4 determinstic random maps and 5 8x8 determinstic random maps 

D. [15 points] On average, how long (how many runs) does it take to learn a stochastic 4x4
and 8x8 environment with p=¾ ? (you should report average and standard deviation)?
    The average for stochastic 4x4 and 8x8 environments combined is: 5000 episodes
    The standard deviation for the combined is: unknown since they were not converging within 5000 episodes
    The average for stochastic 4x4 environments is: 5000 episodes
    The standard deviation for the combined is: unknown since they were not converging within 5000 episodes
    The average for stochastic 8x8 environments is: 5000 episodes
    The standard deviation for the combined is: unknown since they were not converging within 5000 episodes
    Note: this was calculated using 6 4x4 stochastic random maps and 6 8x8 stochastic random maps 

E. [10 points] How did you determine that you have “learned” a stochastic environment in
part D?
    I considered it to be a learned environment when the Q table stablized after at least 500 episodes, with at least 300 of which were successful