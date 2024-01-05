from connectivity_ import if_connected, connect_datasets_generation
from cycle_ import if_cyclic, cycle_datasets_generation
from shortest import shortest_path, shortest_datasets_generation
from topology_ import topological_sort, topology_datasets_generation
from bipartite import if_bipartite, bipartite_datasets_generation
from diameter import diameter, diameter_datasets_generation
from max_triplet import maximum_triplet_sum, triplet_datasets_generation
from max_flow import find_maximum_flow, flow_datasets_generation
from hamilton_ import hamiltonian_path, hamiltonian_datasets_generation
from substructure import count_subgraph_occurrences, substruc_datasets_generation
from utils import load_yaml, build_args, write_to_file
import os


def main():
    task_config_lookup = load_yaml(os.path.join(os.path.dirname(__file__), "task_config.yaml"))
    args = build_args()
    task_config = task_config_lookup[args.task]
    print(task_config)
    if "cycle" in args.task:
        cycle_datasets_generation(task_config)
    elif "connectivity" in args.task:
        connect_datasets_generation(task_config)
    elif "shortest" in args.task:
        shortest_datasets_generation(task_config)
    elif "bipartite" in args.task:
        bipartite_datasets_generation(task_config)
    elif "diameter" in args.task:
        diameter_datasets_generation(task_config)
    elif "flow" in args.task:
        flow_datasets_generation(task_config)
    elif "hamilton" in args.task:
        hamiltonian_datasets_generation(task_config)
    elif "triplet" in args.task:
        triplet_datasets_generation(task_config)
    elif "topology" in args.task:
        topology_datasets_generation(task_config)
    elif "substructure" in args.task:
        substruc_datasets_generation(task_config)

if __name__ == "__main__":
    main()