tasks=("connectivity_test" "cycle_test" "shortest_test" "bipartite_test" "flow_test" "topology_test" "triplet_test" "hamilton_test")
for task in "${tasks[@]}"
do
python ../generation/main.py --task $task
echo "Done with $task"
done