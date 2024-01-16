tasks=("connectivity" "cycle" "shortest" "bipartite" "flow" "topology" "triplet" "hamilton" "substructure")
for task in "${tasks[@]}"
do
python split_test_set.py --task $task
echo "Done with $task"
done