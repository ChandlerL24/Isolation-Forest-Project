# Parallel Isolation Forest

This is an anomaly detection system written in Go using the Isolation Forest algorithm. I wrote a plain sequential version first to get the algorithm right, then built three parallel versions on top of it (channels, BSP, and work-stealing) so I could compare how they scale.

If you want the full experiment write-up with the speedup graphs and cluster results, that's in [`proj3/REPORT.md`](proj3/REPORT.md). This file is just about how the thing works and how to run it.

## The basic idea

The goal is to find the data points that don't fit in with everything else. Isolation Forest does this with a pretty neat trick: anomalies are rare and they look different, so they're easy to "isolate." If you keep chopping the data up with random splits, an oddball point gets cut off from the rest after only a few cuts, while a normal point buried in the crowd takes a lot more.

So the algorithm basically measures how fast each point gets isolated:

1. Build a bunch of random trees, each on a small random sample of the data.
2. At each node, pick a random feature and a random split value, then split the points.
3. Count how many splits it takes to isolate a point (that's its path length).
4. Average that path length over all the trees. Short average = anomaly.
5. Turn the average into a score between 0 and 1, where higher means more suspicious.

The reason this is worth parallelizing is that the trees don't depend on each other at all. You can build all 100 of them at the same time and nobody needs to talk to anybody.

## How the code is laid out

Everything is under `proj3/` (that's where the `go.mod` lives).

```
proj3/
├── main.go                 # CLI, picks a mode, runs it, prints/saves results
├── go.mod
│
├── data/
│   └── data.go             # reads CSVs into Points and a Dataset
│
├── iforest/
│   ├── tree.go             # the tree itself: building, path length, scoring
│   ├── sequential.go       # the plain single-threaded version
│   ├── parallel.go         # channel-based version + BSP version
│   └── workstealing.go     # the work-stealing version
│
├── workstealing/
│   ├── deque.go            # lock-free deque (the CAS stuff)
│   └── scheduler.go        # the worker pool + stealing logic
│
├── scripts/generate_data.go  # makes fake datasets to test with
├── datasets/                 # some pre-made datasets (small/medium/large)
├── benchmark/                # benchmark scripts, the SLURM job, the graphs
├── run_benchmark.sh          # runs the whole benchmark in one go
└── REPORT.md                 # the detailed write-up
```

I split it into three packages on purpose. The `data` package only knows about CSVs and points, it has no clue what a forest is. The `iforest` package is the algorithm and the four ways of running it. And `workstealing` is a generic scheduler that doesn't know anything about isolation forests at all — you just hand it a function and it runs your tasks. The forest just plugs its tree-building into it, but you could reuse it for something else.

## Walking through what happens

**Loading the data.** `LoadCSV` in `data/data.go` reads the file and turns each row into a `Point`:

```go
type Point struct {
    Features []float64 // the numbers we actually use
    Label    string    // optional, only used to check accuracy
    ID       int        // which row it came from
}
```

You tell it which columns are features with `-features 0,1,2,...`, which column has the label (if any) with `-label`, and whether there's a header row to skip. Anything with missing or non-numeric values just gets skipped.

**Setting up the forest.** `NewIsolationForest` figures out two values from the original paper. The max depth is `ceil(log2(sampleSize))` — there's no point growing trees deeper than that. And `C` is the average path length you'd expect in a random binary tree, which is used to normalize the scores so they're comparable.

**Building a tree.** `BuildTree` recurses down: grab a random feature, find its min and max in the current points, pick a random split somewhere in between, send the points left or right, repeat. It stops and makes a leaf when it hits max depth, gets down to one point, or all the points are identical. (There's also a small guard for features that don't vary — it'll rotate to another feature if the one it picked is constant.)

The important thing: this part is identical across all four modes. The *only* difference between sequential, parallel, BSP, and work-stealing is how the work of building all the trees gets handed out to threads.

**Scoring.** Once the trees are built, every point gets walked down every tree to find its path length. Average those, then:

```
score = 2 ^ ( -avgPathLength / C )
```

Short path → exponent near zero → score near 1 → probably an anomaly.

**Output.** It prints the top N anomalies, dumps everything to a CSV sorted by score, and if you gave it labels it'll also print precision and recall on the top N.

## The four modes

You pick one with `-mode`. They all give you the same kind of answer — they just schedule the work differently.

**Sequential** (`sequential.go`) is the baseline. One tree at a time in a for loop, then score the points one at a time. Everything else gets compared against this.

**Parallel** (`parallel.go`) is the normal Go way of doing it. There's a task channel with one tree per task, a handful of worker goroutines pull from it, build their tree, and send it back on a result channel. Channels handle all the locking for you, and load balancing just happens naturally since a worker grabs the next task the moment it's free. Scoring works the same way but in batches.

**BSP** (also in `parallel.go`) does it in supersteps. Each superstep handles `numWorkers` trees, and there's a barrier (built with a `sync.Cond`) where everyone waits for each other before and after. Honestly this is overkill for this problem — the trees don't depend on each other so the barriers are just wasted waiting — but it's there to show the pattern, and it makes a nice point about when barriers actually cost you.

**Work-stealing** (`workstealing.go` plus the `workstealing` package) is the interesting one. Every worker has its own deque. It pushes and pops its own work from the bottom, and when it runs dry it steals from the top of some random other worker's deque. Owner works one end, thieves work the other, so they rarely collide. The deque is lock-free — it uses atomic compare-and-swap loops instead of mutexes.

To actually show stealing doing something, I hand the first worker half of all the tasks on purpose. Without stealing it would be stuck grinding while everyone else sat around. With stealing, the idle workers come grab its work and even things out. If you run it with `-verbose` you can see exactly how many tasks each worker did and how many it stole.

The scheduler itself (`scheduler.go`) is straightforward: each worker loops doing its own work, tries to steal when empty, and once every deque is empty it flips a done flag and everyone quits. There's a tiny exponential backoff so idle workers don't spin the CPU while they wait.

## Running it

Build it:

```bash
cd proj3
go build -o iforest_detector .
```

Run one mode:

```bash
# sequential
./iforest_detector -input datasets/small_gaussian.csv -features 0,1,2,3,4,5,6,7,8,9 -mode sequential

# parallel, 8 workers
./iforest_detector -input datasets/small_gaussian.csv -features 0,1,2,3,4,5,6,7,8,9 -mode parallel -workers 8

# bsp
./iforest_detector -input datasets/small_gaussian.csv -features 0,1,2,3,4,5,6,7,8,9 -mode bsp -workers 8

# work-stealing (add -verbose to see the steal counts)
./iforest_detector -input datasets/small_gaussian.csv -features 0,1,2,3,4,5,6,7,8,9 -mode workstealing -workers 8 -verbose
```

Compare all of them at once:

```bash
./iforest_detector -input datasets/small_gaussian.csv -features 0,1,2,3,4,5,6,7,8,9 -benchmark -workers 12
```

Or just run the whole benchmark, which builds everything, makes fresh datasets, and runs every mode:

```bash
cd proj3
./run_benchmark.sh
```

Tests:

```bash
cd proj3
go test ./...
```

## The flags

| Flag | Default | What it does |
|------|---------|--------------|
| `-input` | required | the input CSV |
| `-features` | required | which columns are features (e.g. `0,1,2`) |
| `-mode` | sequential | `sequential`, `parallel`, `bsp`, or `workstealing` |
| `-workers` | 4 | number of worker goroutines |
| `-trees` | 100 | how many trees in the forest |
| `-sample` | 256 | sample size per tree |
| `-label` | -1 | column with labels, or -1 if there aren't any |
| `-top` | 10 | how many top anomalies to print |
| `-seed` | 42 | random seed so runs are repeatable |
| `-output` | results.csv | where to write the scored output |
| `-header` | true | skip the first row of the CSV |
| `-benchmark` | false | run every mode and compare |
| `-verbose` | false | print extra timing and steal stats |

## Reading the scores

Each point ends up with a score between 0 and 1:

- above ~0.6: probably an anomaly
- 0.5 to 0.6: borderline, could go either way
- below 0.5: probably fine

Higher score means the point got isolated faster, which means it's more of an outlier.

## More detail

[`proj3/REPORT.md`](proj3/REPORT.md) has the real write-up — the design reasoning, the speedup graphs, the bottleneck analysis, and the results from running it on the UChicago cluster. The graphs themselves are in `proj3/benchmark/`.
