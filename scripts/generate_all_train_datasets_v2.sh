tasks=("connectivity_train_v2" "cycle_train_v2" "flow_train_v2" "triplet_train_v2" "substructure_train_v2")
for task in "${tasks[@]}"
do
python ../generation/main.py --task $task
echo "Done with $task"
done