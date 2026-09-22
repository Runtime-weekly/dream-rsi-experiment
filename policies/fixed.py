"""Fixed parallel-refine baseline. Open up to W branches, then extend leaves."""
def choose(observation):
    nodes = observation["nodes"]
    workers = observation["max_parallelism"]
    leaves = [n["id"] for n in nodes if n["id"] != "root" and n["id"] in observation["legal"]]
    roots = sum(n["parent"] == "root" for n in nodes)
    if roots < workers:
        return ["root"] + leaves[:workers - 1]
    return leaves[:workers]

