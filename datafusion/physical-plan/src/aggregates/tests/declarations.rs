macro_rules! declare_aggregate_tests {
    (aggregate_exec) => {
        #[path = "tests"]
        mod paired_tests {
            #[path = "aggregate_exec.rs"]
            pub(in crate::aggregates) mod aggregate_exec;
        }
    };
    (row_hash) => {
        #[path = "tests"]
        mod paired_tests {
            #[path = "row_hash.rs"]
            mod row_hash;
        }
    };
}
pub(in crate::aggregates) use declare_aggregate_tests;
