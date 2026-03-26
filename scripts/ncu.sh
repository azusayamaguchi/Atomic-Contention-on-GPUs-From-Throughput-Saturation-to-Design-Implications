for K in {1,256}; do
    for i in {1..5}; do
	ncu --kernel-name atomic_stress --metrics "lts__d_atomic_input_cycles_active,lts__t_requests_op_atom,lts__t_requests_op_atom_lookup_hit,lts__t_requests_op_atom_lookup_miss,smsp__inst_executed_pipe_lsu.sum,smsp__warp_issue_stalled_membar_per_warp_active.pct,smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct,smsp__warp_issue_stalled_selected_per_warp_active.pct,dram__throughput.avg.pct_of_peak_sustained_elapsed" --csv --log-file ncu_resultsK${K}_${i}.csv ./atomic_stress_v4 ${K} 256 32768
    done
done
