macro_rules! declare_physical_optimizer_tests {
    (combine_partial_final_agg) => {
        #[path = "tests"]
        pub(crate) mod tests {
            #[path = "combine_partial_final_agg.rs"]
            pub(crate) mod combine_partial_final_agg;
        }
    };
    (enforce_distribution) => {
        #[path = "tests"]
        mod tests {
            #[path = "enforce_distribution.rs"]
            mod enforce_distribution;
        }
    };
}
pub(crate) use declare_physical_optimizer_tests;
