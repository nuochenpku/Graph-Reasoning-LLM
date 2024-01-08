tasks=("connectivity_train" "cycle_train" "shortest_train" "bipartite_train" "diameter_train" "flow_train" "hamilton_train" "triplet_train" "topology_train" "substructure_train")

for task in "${tasks[@]}"
do
python ../generation/main.py --task $task
echo "Done with $task"
done